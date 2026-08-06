
module carbon_allocation_offline_kernel

!==========================================================================
! Carbon allocation
!==========================================================================
! This file is intentionally verbose. The comments are part of the model
! documentation and are meant to help a future reader understand why each
! equation appears in the code. 
!
! Purpose
! -------
! This module implements a self-contained woody-plant carbon allocation
! kernel based on the LPJ-style allometric allocation logic. BUT important:
! Not using "abnormal allocation" as LPJ-style. If there is no feasible solution 
! to the normal allometric problem, the carbon goes to storage instead of being 
! forced into a non-allometric solution. This is a more
! conservative approach that avoids unrealistic jumps in plant structure
!
! Scope of this first version
! ---------------------------
! This kernel solves allocation for ONE average woody individual over ONE
! allocation period. The allocation period can be annual, monthly, seasonal,
! or any other interval. The equations do not know the calendar frequency.
! The caller only needs to provide the carbon available over that period (here
! we are using daily because CAETE works on a daily basis).
!
! Key idea for the gradual storage-based allocation scheme
! -------------------------------------------------------
! Daily carbon input is first added to a labile storage pool. Structural
! growth is then paid from storage only when there is positive allocation
! demand and enough available carbon. If negative NPP exhausts storage, the
! remaining deficit is treated as a starvation pressure on living tissues:
! one third is assigned to leaves, one third to fine roots, and one third to
! sapwood. Leaf and fine-root losses remove carbon from the plant, whereas
! sapwood loss is converted into heartwood.
!
! In contrast to the legacy rigid allocation scheme (strictly based in LPJ), this routine does not
! force the plant to satisfy allometric constraints exactly at each daily
! timestep. Instead, the leaf-root relationship and the pipe-model relationship
! are used to compute structural demand. This demand gradually moves the plant
! toward allometric consistency over the timescale defined by
! allometric_adjustment_days.
!
! The structural increment is:
!
!     structural_growth = dL + dR + dS
!
! where:
!
!     dL = increment to leaf carbon mass
!     dR = increment to fine-root carbon mass
!     dS = increment to sapwood carbon mass
!
! If there is no structural demand, or if daily allocation is limited by the
! maximum allocation fraction, carbon remains in storage instead of being
! forced into an abnormal allocation pathway.
!
! Height is updated from total stem carbon rather than by forcing the
! pipe-model residual to zero at each daily timestep. The pipe model therefore
! acts as a gradual demand signal, not as an instantaneous constraint.
  

   use, intrinsic :: iso_fortran_env, only: real64

   implicit none

   private   

   public :: Parameters
   public :: PlantCarbonState
   public :: AllocationOutput
   public :: leaf_requirement
   public :: ControlsParam
   public :: allocate_gradual_with_storage


  !--------------------------------------------------------------------------
  ! Numerical constants
  !--------------------------------------------------------------------------

   real(real64), parameter :: pi = 3.1416_real64

   ! Tolerance used only for diagnostic carbon-accounting checks.
   real(real64), parameter :: carbon_accounting_tolerance = 1.0e-10_real64

   ! Annual turnover rates for plant carbon compartments.
   ! Leaf, fine-root, labile storage, and heartwood turnover remove carbon
   ! from the plant. Sapwood turnover converts living sapwood into heartwood.

   ! ATTENTION: Leaf turnover will be calculated from SLA
   real(real64), parameter :: l_turnover   = 1.0_real64 / 4.0_real64

   real(real64), parameter :: r_turnover   = 1.0_real64 / 4.0_real64
   real(real64), parameter :: s_turnover   = 1.0_real64 / 20.0_real64
   real(real64), parameter :: sto_turnover = 1.0_real64 / 20.0_real64
   real(real64), parameter :: h_turnover   = 1.0_real64 / 150.0_real64

  !--------------------------------------------------------------------------
  ! Input parameters
  !--------------------------------------------------------------------------

   type :: Parameters

      !!! PLS-specific parameters !!!
      !-----------------------------!

      ! Specific leaf area (PLS specific)
      ! Units must be consistent with leaf_mass and area.
      ! Example: if leaf_mass is gC individual-1 and SLA is m2 gC-1,
      ! then leaf_area = leaf_mass * sla is m2 individual-1.
      real(real64) :: sla

      ! Wood density (PLS specific)
      ! Units must be consistent with carbon mass and volume.
      ! Example: gC m-3 if carbon pools are in gC.
      real(real64) :: wood_density


     !!! Global parameters !!!
     !-----------------------!
      ! Leaf area to sapwood cross-sectional area ratio (global value)
      ! This is the pipe-model coefficient.
      ! Pipe model:
      !     leaf_area = latosa * sapwood_cross_sectional_area
      real(real64) :: latosa

     ! Leaf-to-fine-root mass ratio for the allocation period.
     ! Functional balance:
     !     leaf_mass = leaf_to_root_ratio * root_mass
     ! In LPJ-style code this is often computed as:
     !     leaf_to_root_ratio = max(lm2rm_max * water_scalar, 0.1)
     real(real64) :: leaf_to_root_ratio

     ! Height-diameter allometry coefficient.
     ! Stem mechanics:
     !     height = allom2 * stem_diameter**allom3
     real(real64) :: allom2

     ! Height-diameter allometry exponent.
     ! Stem mechanics:
     !     height = allom2 * stem_diameter**allom3
     real(real64) :: allom3

   end type Parameters


  !--------------------------------------------------------------------------
  ! Controls for the gradual daily allocation routine with labile storage
  !--------------------------------------------------------------------------

   type :: ControlsParam

      ! Length of one model time step expressed in years.
      ! For a daily time step, use 1/365.
      real(real64) :: dt_years = 1.0_real64 / 365.0_real64

      ! Time scale used to relax allometric imbalances.
      ! A value of 365 means that a full allometric deficit is not corrected
      ! instantly; instead, roughly 1/365 of the deficit becomes demand per day.
      real(real64) :: allometric_adjustment_days = 365.0_real64

      ! Maximum relative structural growth allowed in one time step.
      ! This parameter is a daily growth-rate cap. It prevents the plant from
      ! converting a very large amount of storage carbon into new structural biomass
      ! in a single time step, even when storage and allometric demand are both high.
      !
      ! This is important because the original LPJ allocation logic was annual,
      ! whereas this routine is applied at a daily time step. Without this cap, large
      ! allometric deficits or large storage pools could produce unrealistic daily
      ! jumps in leaf, fine-root, or sapwood biomass.
      !
      ! For example, max_allocation_fraction = 0.005 means that daily structural
      ! growth cannot exceed 0.5% of current living structural carbon:
      !
      !     max_daily_allocation = 0.005 * (leaf + root + sapwood)
      !
      ! This parameter should be interpreted as a maximum tissue-construction
      ! capacity, not as a carbon-availability term.
      
      ! ATTENTION: test different values for this parameter
      real(real64) :: max_allocation_fraction = 0.005_real64 ! This value can be adjusted to represent fast/slow growth strategies. 


      !!!! TO BE READJUSTED (can express fast/slow growth strategies)
      !---------------------
      ! Background demand time scale for leaves, in years.
      ! This is not an allometric target. It is a small baseline structural demand
      ! used when the plant is close to its allometric constraints.
      real(real64) :: leaf_background_timescale_years = 3.0_real64

      ! Background demand time scale for fine roots, in years.
      real(real64) :: root_background_timescale_years = 3.0_real64

      ! Background demand time scale for sapwood, in years.
      ! This should usually be longer than leaf and fine-root time scales.
      real(real64) :: sapwood_background_timescale_years = 15.0_real64

   end type ControlsParam



  !--------------------------------------------------------------------------
  ! Carbon state of one average woody individual
  !
  ! These variables represent the current carbon pools at the moment when the
  ! allocation routine is called.  
  !--------------------------------------------------------------------------

   type :: PlantCarbonState

     ! Leaf carbon mass 
     real(real64) :: leaf_mass

     ! Fine-root carbon mass 
     real(real64) :: root_mass

     ! Sapwood carbon mass 
     real(real64) :: sapwood_mass

     ! Heartwood carbon mass 
     real(real64) :: heartwood_mass

     ! Plant height.
     ! The variable comes from the current state, before allocation
     ! Required for detecting the minimum leaf mass needed to maintain the
     ! currently existing sapwood mass before normal allocation.
     real(real64) :: height

   end type PlantCarbonState



  !--------------------------------------------------------------------------
  ! Output from the allocation routine
  !--------------------------------------------------------------------------

   type :: AllocationOutput


      ! Carbon increments of the average individual over the allocation period.
      !! _real64 is used for precision safety 
      real(real64) :: delta_leaf    = 0.0_real64 
      real(real64) :: delta_root    = 0.0_real64
      real(real64) :: delta_sapwood = 0.0_real64

      ! New carbon pools after allocation.
      real(real64) :: leaf_mass_new      = 0.0_real64
      real(real64) :: root_mass_new      = 0.0_real64
      real(real64) :: sapwood_mass_new   = 0.0_real64
      real(real64) :: heartwood_mass_new = 0.0_real64

      ! New structural variables inferred from the final allometric state.
      real(real64) :: sapwood_area_new = 0.0_real64
      real(real64) :: height_new       = 0.0_real64
      real(real64) :: stem_diameter_new = 0.0_real64

            !! Diagnostic residuals and balance checks.
      ! These variables are numerical diagnostics used to evaluate the gradual
      ! storage-based allocation result.
      !
      ! In this allocation scheme, the plant is not forced to satisfy allometric
      ! constraints exactly at each daily timestep. Instead, the leaf-root
      ! relationship and the pipe-model relationship are used to define
      ! structural growth demands that gradually move the plant toward
      ! allometric consistency.
      !
      ! Therefore, leaf_root_residual and pipe_model_residual are not expected
      ! to be exactly zero after every allocation step. They should be interpreted
      ! as diagnostic measures of the current allometric imbalance. Under
      ! adequate carbon supply and reasonable parameter values, these residuals
      ! should generally move toward smaller absolute values over time.
      !
      ! carbon_balance_error is kept as a backward-compatible diagnostic name.
      ! In the gradual allocation scheme, it is equivalent to the structural
      ! balance error:
      !
      !     carbon_balance_error =
      !         carbon_to_allocate - (delta_leaf + delta_root + delta_sapwood)
      !
      ! This value should remain close to zero, because all carbon assigned to
      ! structural growth must be distributed among leaf, fine-root, and sapwood
      ! increments.
      !
      ! leaf_root_residual measures the mismatch between final leaf and fine-root
      ! pools:
      !
      !     leaf_root_residual =
      !         leaf_mass_new - leaf_to_root_ratio * root_mass_new
      !
      ! A positive value indicates relatively more leaf carbon than expected from
      ! the target leaf-root relationship. A negative value indicates relatively
      ! more fine-root carbon.
      !
      ! pipe_model_residual measures the mismatch between final leaf area and
      ! sapwood conductive area:
      !
      !     pipe_model_residual =
      !         leaf_area_new - latosa * sapwood_area_new
      !
      ! A negative value indicates that leaf area is low relative to sapwood area.
      ! A positive value indicates that leaf area is high relative to sapwood area.
      !
      ! final_root_residual is reserved for additional root-related diagnostics
      ! during integration with the full model. It can be removed if it remains
      ! redundant with leaf_root_residual.

      real(real64) :: carbon_balance_error = 0.0_real64
      real(real64) :: leaf_root_residual   = 0.0_real64
      real(real64) :: pipe_model_residual  = 0.0_real64
      real(real64) :: final_root_residual  = 0.0_real64

      ! Residual of the nonlinear equation f(dL) = 0 at the selected solution.
      real(real64) :: allocation_residual_final = 0.0_real64

      !! Bounds used by the normal-allocation solver 
      ! Inside this interval, the solver searches for the
      ! delta_leaf value that also makes the stem geometry and pipe-model
      ! constraints mutually consistent (more detailed explanations below)
      real(real64) :: lower_bound_delta_leaf = 0.0_real64
      real(real64) :: upper_bound_delta_leaf = 0.0_real64


      ! Diagnostics specific to gradual allocation with labile carbon storage.
      ! These fields remain zero when the original rigid allocate() routine is used.
      real(real64) :: npp_daily = 0.0_real64
      real(real64) :: carbon_storage_before = 0.0_real64
      real(real64) :: carbon_storage_after = 0.0_real64
      real(real64) :: carbon_to_allocate = 0.0_real64
      real(real64) :: total_demand_daily = 0.0_real64
      real(real64) :: leaf_demand_daily = 0.0_real64
      real(real64) :: root_demand_daily = 0.0_real64
      real(real64) :: sapwood_demand_daily = 0.0_real64

      ! Carbon-accounting diagnostics for gradual allocation.
      ! storage_after_npp_unclamped stores the raw storage value after adding NPP.
      ! If this value is negative, storage is clamped to zero and the missing
      ! carbon is first reported as unmet_storage_deficit.
      !
      ! The starvation rule then tries to represent the biological consequence
      ! of this deficit on living tissues. The deficit is split equally among
      ! leaves, fine roots, and sapwood. Leaf and fine-root losses remove carbon
      ! from the plant. Sapwood loss is treated as sapwood-to-heartwood
      ! conversion, so it reduces living sapwood but does not remove carbon from
      ! total plant biomass.
      !
      ! Because sapwood-to-heartwood conversion conserves total plant carbon,
      ! only leaf_starvation_loss + root_starvation_loss pay part of the carbon
      ! deficit. Any remaining deficit is reported as unpaid_carbon_deficit.
      real(real64) :: storage_after_npp_unclamped = 0.0_real64
      real(real64) :: unmet_storage_deficit = 0.0_real64
      real(real64) :: leaf_starvation_loss = 0.0_real64
      real(real64) :: root_starvation_loss = 0.0_real64
      real(real64) :: sapwood_starvation_loss = 0.0_real64
      real(real64) :: sapwood_to_heartwood = 0.0_real64
      real(real64) :: starvation_carbon_loss = 0.0_real64
      real(real64) :: unpaid_carbon_deficit = 0.0_real64

      ! Turnover diagnostics. Leaf, fine-root, storage, and heartwood turnover
      ! are carbon losses from the plant. Sapwood turnover is a conversion from
      ! living sapwood to heartwood and therefore does not directly remove total
      ! plant carbon.
      real(real64) :: leaf_turnover_loss = 0.0_real64
      real(real64) :: root_turnover_loss = 0.0_real64
      real(real64) :: sapwood_turnover_loss = 0.0_real64
      real(real64) :: storage_turnover_loss = 0.0_real64
      real(real64) :: heartwood_turnover_loss = 0.0_real64
      real(real64) :: sapwood_to_heartwood_turnover = 0.0_real64
      real(real64) :: total_sapwood_to_heartwood = 0.0_real64
      real(real64) :: turnover_carbon_loss = 0.0_real64

      ! The structural balance checks whether all carbon assigned to growth was
      ! actually distributed among leaf, fine-root, and sapwood increments.
      real(real64) :: structural_increment_sum = 0.0_real64
      real(real64) :: structural_balance_error = 0.0_real64

      ! The storage balance checks whether the labile storage pool was updated
      ! consistently after adding NPP and subtracting structural allocation.
      real(real64) :: storage_balance_error = 0.0_real64

      ! The whole-plant balance checks the combined structural + storage carbon.
      ! If unpaid_carbon_deficit is zero and turnover is zero, the whole plant
      ! should change by NPP. With turnover, carbon losses from leaf, fine root,
      ! storage, and heartwood are subtracted. Sapwood turnover is excluded from
      ! this loss term because it is converted into heartwood.
      real(real64) :: whole_plant_balance_error = 0.0_real64

      ! Maximum structural carbon allocation allowed by the daily limiter.
      real(real64) :: max_daily_allocation = 0.0_real64

      ! True when the numerical carbon-accounting equations close within the
      ! tolerance defined by carbon_accounting_tolerance.
      logical :: carbon_accounting_ok = .false.

      ! Diagnostic message
      character(len=160) :: message = ""

   end type AllocationOutput


   contains

   !==========================================================================
   !> Compute the leaf mass required to maintain the existing sapwood mass.
   !==========================================================================
      function leaf_requirement(state, params) result(leaf_required)

         type(PlantCarbonState), intent(in) :: state
         type(Parameters), intent(in) :: params

         real(real64) :: leaf_required

         !-----------------------------------------------------------------------
         ! Derivation
         ! ----------
         ! The pipe model states:
         !     leaf_area = latosa * sapwood_area
         !
         ! Leaf area is:
         !     leaf_area = leaf_mass * SLA
         !
         ! Sapwood volume is:
         !     sapwood_volume = height * sapwood_area
         !
         ! Sapwood mass is:
         !     sapwood_mass = wood_density * sapwood_volume
         !                  = wood_density * height * sapwood_area
         !
         ! Therefore:
         !     sapwood_area = sapwood_mass / (wood_density * height)
         !
         ! Substitute this into the pipe model:
         !     leaf_mass * SLA = latosa * sapwood_mass / (wood_density * height)
         !
         ! Solve for leaf_mass:
         !     leaf_mass_required = latosa * sapwood_mass /
         !                          (wood_density * height * SLA)
         !
         ! This is the minimum leaf mass needed to keep the current sapwood mass
         ! consistent with the pipe model, assuming no new sapwood is produced.
         !-----------------------------------------------------------------------

         if (state%height <= 0.0_real64) then
            leaf_required = 0.0_real64
         else
            leaf_required = params%latosa * state%sapwood_mass / &
                           (params%wood_density * state%height * params%sla)
         end if

      end function leaf_requirement


   !==========================================================================
   !> Gradual daily allocation with labile carbon storage.
   !==========================================================================
      subroutine allocate_gradual_with_storage(state, params, controls, npp_rate, &
                                               carbon_storage, result)

         type(PlantCarbonState),          intent(in)    :: state
         type(Parameters),                intent(in)    :: params
         type(ControlsParam), intent(in)    :: controls
         real(real64),                    intent(in)    :: npp_rate
         real(real64),                    intent(inout) :: carbon_storage
         type(AllocationOutput),          intent(inout) :: result

         real(real64) :: storage_after_npp
         real(real64) :: storage_after_allocation
         real(real64) :: living_carbon
         real(real64) :: max_daily_allocation

         real(real64) :: leaf_required
         real(real64) :: delta_leaf_min_pipe
         real(real64) :: delta_leaf_min_root_nonnegative
         real(real64) :: delta_leaf_min
         real(real64) :: leaf_target

         real(real64) :: root_required
         real(real64) :: delta_root_min
         real(real64) :: root_target

         real(real64) :: sapwood_required_for_leaf_target
         real(real64) :: sapwood_target

         real(real64) :: leaf_deficit
         real(real64) :: root_deficit
         real(real64) :: sapwood_deficit

         real(real64) :: leaf_background_demand
         real(real64) :: root_background_demand
         real(real64) :: sapwood_background_demand

         real(real64) :: frac_leaf
         real(real64) :: frac_root
         real(real64) :: frac_sapwood

         real(real64) :: leaf_area_new
         real(real64) :: sapwood_area_from_mass
         real(real64) :: stem_carbon_total_new
         real(real64) :: height_power_total_stem
         real(real64) :: height_power_exponent
         real(real64) :: pi_over_four
         real(real64) :: starvation_share

         real(real64) :: structural_carbon_before
         real(real64) :: structural_carbon_after
         real(real64) :: whole_carbon_before
         real(real64) :: whole_carbon_after

         ! Reset the output object to a known state.
         result = AllocationOutput()

         result%carbon_storage_before = carbon_storage

         ! Convert the annualized NPP rate into carbon input over this time step.
         ! If npp_rate is in kgC per area per year and dt_years is 1/365,
         ! npp_daily has units of kgC per area per day.
         result%npp_daily = npp_rate * controls%dt_years

         ! Update the labile carbon storage. Negative NPP consumes storage.
         ! The unclamped value is kept for carbon-accounting diagnostics.
         result%storage_after_npp_unclamped = carbon_storage + result%npp_daily
         storage_after_npp = result%storage_after_npp_unclamped

         ! Storage cannot become negative. If NPP is strongly negative, the
         ! remaining deficit is first reported as unmet_storage_deficit. A
         ! starvation rule below then tries to pay part of this deficit by
         ! reducing living tissues.
         if (storage_after_npp < 0.0_real64) then
            result%unmet_storage_deficit = -storage_after_npp
            storage_after_npp = 0.0_real64
         else
            result%unmet_storage_deficit = 0.0_real64
         end if

         !--------------------------------------------------------------------
         ! Allometric correction direction following the original allocation
         ! logic.
         !--------------------------------------------------------------------
         ! Important: there is only one primary leaf requirement here.
         ! It comes from the pipe model and is exactly the same quantity used
         ! in the rigid normal-allocation solver:
         !
         !     leaf_required = leaf_requirement(state, params)
         !
         ! This is the leaf mass needed to support the current sapwood mass at
         ! the current height.
         leaf_required = leaf_requirement(state, params)

         ! Minimum leaf increment required by the pipe model.
         ! If the current leaf mass is already above the pipe-model requirement,
         ! this term is negative and will not create demand.
         delta_leaf_min_pipe = leaf_required - state%leaf_mass

         ! Minimum leaf increment required to avoid a negative root increment
         ! when the leaf-root relationship is imposed as:
         !
         !     root_new = leaf_new / leaf_to_root_ratio
         !
         ! In the rigid solver, root increment is computed as:
         !
         !     delta_root = (leaf_old + delta_leaf) / leaf_to_root_ratio
         !                  - root_old
         !
         ! To keep delta_root >= 0, delta_leaf must satisfy:
         !
         !     delta_leaf >= root_old * leaf_to_root_ratio - leaf_old
         !
         ! This is not a second independent leaf requirement. It is a numerical
         ! and biological constraint that prevents negative root allocation in
         ! the normal-growth pathway.
         delta_leaf_min_root_nonnegative = &
            state%root_mass * params%leaf_to_root_ratio - state%leaf_mass

         ! Effective minimum leaf increment implied by the same lower-bound
         ! logic used in the original normal allocation routine.
         ! This is a deficit used as a direction of gradual adjustment, not an
         ! increment that must be applied in a single daily time step.
         delta_leaf_min = max(0.0_real64, &
                              delta_leaf_min_pipe, &
                              delta_leaf_min_root_nonnegative)

         ! Leaf target associated with the minimum admissible leaf increment.
         leaf_target = state%leaf_mass + delta_leaf_min

         ! Root target is derived after the leaf target, following the original
         ! logic:
         !
         !     root_required = leaf_target / leaf_to_root_ratio
         !
         ! This means the pipe-model leaf requirement comes first, and the root
         ! requirement follows from functional balance.
         if (params%leaf_to_root_ratio > 0.0_real64) then
            root_required = leaf_target / params%leaf_to_root_ratio
         else
            root_required = state%root_mass
         end if

         delta_root_min = max(0.0_real64, root_required - state%root_mass)
         root_target = state%root_mass + delta_root_min

         ! Sapwood is different from leaf and root in the original rigid solver:
         ! it is not prescribed by an independent target. In normal allocation,
         ! sapwood increment is the remaining carbon after leaf and root:
         !
         !     delta_sapwood = c_available - delta_leaf - delta_root
         !
         ! For the gradual daily routine, however, we still need a sapwood demand
         ! direction. We therefore compute the sapwood mass that would support
         ! the current leaf target under the pipe model at the current height.
         ! This is a gradual target only; it is not forced instantly.
         sapwood_required_for_leaf_target = leaf_target * params%sla / params%latosa * &
                                            params%wood_density * state%height

         sapwood_target = max(state%sapwood_mass, sapwood_required_for_leaf_target)

         ! Positive deficits define the allometric correction direction.
         leaf_deficit = max(0.0_real64, leaf_target - state%leaf_mass)
         root_deficit = max(0.0_real64, root_target - state%root_mass)
         sapwood_deficit = max(0.0_real64, sapwood_target - state%sapwood_mass)

         ! Convert full allometric deficits into gradual demands. This is the key
         ! step that prevents instant correction of the whole plant structure.
         result%leaf_demand_daily = leaf_deficit / controls%allometric_adjustment_days
         result%root_demand_daily = root_deficit / controls%allometric_adjustment_days
         result%sapwood_demand_daily = sapwood_deficit / controls%allometric_adjustment_days

         !--------------------------------------------------------------------
         ! Small background structural demand.
         !--------------------------------------------------------------------
         ! These terms allow growth to continue when the plant is already close
         ! to the allometric correction targets. They are deliberately separated
         ! from the allometric correction terms above.
         !
         ! The background terms are optional and should be tested carefully. If
         ! they are set too high, they can dominate the allometric correction
         ! signal. If they are set to zero, allocation occurs only when there is
         ! an explicit allometric deficit.
         if (controls%leaf_background_timescale_years > 0.0_real64) then
            leaf_background_demand = leaf_target * controls%dt_years / &
                                     controls%leaf_background_timescale_years
         else
            leaf_background_demand = 0.0_real64
         end if

         if (controls%root_background_timescale_years > 0.0_real64) then
            root_background_demand = root_target * controls%dt_years / &
                                     controls%root_background_timescale_years
         else
            root_background_demand = 0.0_real64
         end if

         if (controls%sapwood_background_timescale_years > 0.0_real64) then
            sapwood_background_demand = sapwood_target * controls%dt_years / &
                                        controls%sapwood_background_timescale_years
         else
            sapwood_background_demand = 0.0_real64
         end if

         result%leaf_demand_daily = result%leaf_demand_daily + leaf_background_demand
         result%root_demand_daily = result%root_demand_daily + root_background_demand
         result%sapwood_demand_daily = result%sapwood_demand_daily + sapwood_background_demand

         result%total_demand_daily = result%leaf_demand_daily + &
                                      result%root_demand_daily + &
                                      result%sapwood_demand_daily

         ! The daily structural allocation is constrained by available storage,
         ! structural demand, and a maximum allowed growth fraction.
         living_carbon = state%leaf_mass + state%root_mass + state%sapwood_mass
         max_daily_allocation = controls%max_allocation_fraction * living_carbon
         result%max_daily_allocation = max_daily_allocation

         if (result%total_demand_daily > 0.0_real64 .and. &
             storage_after_npp > 0.0_real64 .and. &
             max_daily_allocation > 0.0_real64) then

            result%carbon_to_allocate = min(storage_after_npp, &
                                            result%total_demand_daily, &
                                            max_daily_allocation)

            frac_leaf = result%leaf_demand_daily / result%total_demand_daily
            frac_root = result%root_demand_daily / result%total_demand_daily
            frac_sapwood = result%sapwood_demand_daily / result%total_demand_daily

            result%delta_leaf = frac_leaf * result%carbon_to_allocate
            result%delta_root = frac_root * result%carbon_to_allocate
            result%delta_sapwood = frac_sapwood * result%carbon_to_allocate

         else

            result%carbon_to_allocate = 0.0_real64
            result%delta_leaf = 0.0_real64
            result%delta_root = 0.0_real64
            result%delta_sapwood = 0.0_real64

         end if

         ! Update structural carbon pools after positive growth allocation.
         result%leaf_mass_new = state%leaf_mass + result%delta_leaf
         result%root_mass_new = state%root_mass + result%delta_root
         result%sapwood_mass_new = state%sapwood_mass + result%delta_sapwood
         result%heartwood_mass_new = state%heartwood_mass

         !--------------------------------------------------------------------
         ! Starvation rule for negative NPP not covered by storage.
         !--------------------------------------------------------------------
         ! If storage is exhausted by negative NPP, the remaining deficit is split
         ! equally among the three living tissues. Leaf and fine-root losses are
         ! treated as carbon leaving the plant. Sapwood loss is treated as
         ! sapwood-to-heartwood conversion: living sapwood decreases, heartwood
         ! increases by the same amount, and total stem carbon is conserved.
         !
         ! The min() calls prevent any living pool from becoming negative when
         ! the requested loss is larger than the pool available in that tissue.
         if (result%unmet_storage_deficit > 0.0_real64) then
            starvation_share = result%unmet_storage_deficit / 3.0_real64

            result%leaf_starvation_loss = min(starvation_share, result%leaf_mass_new)
            result%root_starvation_loss = min(starvation_share, result%root_mass_new)
            result%sapwood_starvation_loss = min(starvation_share, result%sapwood_mass_new)
            result%sapwood_to_heartwood = result%sapwood_starvation_loss

            result%leaf_mass_new = result%leaf_mass_new - result%leaf_starvation_loss
            result%root_mass_new = result%root_mass_new - result%root_starvation_loss
            result%sapwood_mass_new = result%sapwood_mass_new - result%sapwood_starvation_loss
            result%heartwood_mass_new = result%heartwood_mass_new + result%sapwood_to_heartwood

            result%starvation_carbon_loss = result%leaf_starvation_loss + &
                                            result%root_starvation_loss
            result%unpaid_carbon_deficit = result%unmet_storage_deficit - &
                                           result%starvation_carbon_loss
            result%unpaid_carbon_deficit = max(0.0_real64, result%unpaid_carbon_deficit)
         end if

         ! Update storage after structural allocation.
         storage_after_allocation = storage_after_npp - result%carbon_to_allocate

         !--------------------------------------------------------------------
         ! Continuous compartment turnover.
         !--------------------------------------------------------------------
         ! Turnover is applied after NPP, starvation, and structural allocation.
         ! The rates are annual rates and are converted to the current timestep
         ! by multiplying by controls%dt_years. The min() guards prevent small
         ! numerical or extreme-timestep problems from making pools negative.
         result%leaf_turnover_loss = min(result%leaf_mass_new, &
            result%leaf_mass_new * l_turnover * controls%dt_years)
         result%root_turnover_loss = min(result%root_mass_new, &
            result%root_mass_new * r_turnover * controls%dt_years)
         result%sapwood_turnover_loss = min(result%sapwood_mass_new, &
            result%sapwood_mass_new * s_turnover * controls%dt_years)
         result%storage_turnover_loss = min(storage_after_allocation, &
            storage_after_allocation * sto_turnover * controls%dt_years)
         result%heartwood_turnover_loss = min(result%heartwood_mass_new, &
            result%heartwood_mass_new * h_turnover * controls%dt_years)

         result%sapwood_to_heartwood_turnover = result%sapwood_turnover_loss

         result%leaf_mass_new = result%leaf_mass_new - result%leaf_turnover_loss
         result%root_mass_new = result%root_mass_new - result%root_turnover_loss
         result%sapwood_mass_new = result%sapwood_mass_new - result%sapwood_turnover_loss
         result%heartwood_mass_new = result%heartwood_mass_new + &
            result%sapwood_to_heartwood_turnover - result%heartwood_turnover_loss

         carbon_storage = storage_after_allocation - result%storage_turnover_loss
         result%carbon_storage_after = carbon_storage

         result%total_sapwood_to_heartwood = result%sapwood_to_heartwood + &
            result%sapwood_to_heartwood_turnover

         result%turnover_carbon_loss = result%leaf_turnover_loss + &
            result%root_turnover_loss + result%storage_turnover_loss + &
            result%heartwood_turnover_loss

         ! For gradual allocation, height is computed from total stem carbon and
         ! height-diameter allometry. This avoids forcing the pipe model to be
         ! exactly satisfied in one daily step.
         stem_carbon_total_new = result%sapwood_mass_new + result%heartwood_mass_new
         height_power_exponent = 1.0_real64 + 2.0_real64 / params%allom3
         pi_over_four = pi / 4.0_real64

         if (stem_carbon_total_new > 0.0_real64 .and. params%wood_density > 0.0_real64) then
            height_power_total_stem = params%allom2**(2.0_real64 / params%allom3) * &
                                      (stem_carbon_total_new / params%wood_density) / &
                                      pi_over_four
            result%height_new = height_power_total_stem**(1.0_real64 / height_power_exponent)
         else
            result%height_new = state%height
         end if

         if (result%height_new > 0.0_real64) then
            result%stem_diameter_new = (result%height_new / params%allom2)**(1.0_real64 / params%allom3)
         else
            result%stem_diameter_new = 0.0_real64
         end if

         result%sapwood_area_new = result%leaf_mass_new * params%sla / params%latosa

         ! Diagnostics.
         ! Structural carbon accounting: carbon sent to growth must equal the
         ! sum of the structural increments.
         result%structural_increment_sum = result%delta_leaf + &
            result%delta_root + result%delta_sapwood

         result%structural_balance_error = result%carbon_to_allocate - &
            result%structural_increment_sum

         ! Keep the older diagnostic name for backward compatibility.
         result%carbon_balance_error = result%structural_balance_error

         ! Storage carbon accounting: final storage must equal initial storage
         ! plus NPP input, plus any reported unmet deficit caused by the zero
         ! lower bound, minus the carbon allocated to structure.
         result%storage_balance_error = result%carbon_storage_after - &
            (result%carbon_storage_before + result%npp_daily + &
             result%unmet_storage_deficit - result%carbon_to_allocate - &
             result%storage_turnover_loss)

         ! Whole-plant carbon accounting: structural + storage carbon should
         ! change by NPP, except for the part of the negative-NPP deficit that
         ! remains unpaid after the starvation rule.
         structural_carbon_before = state%leaf_mass + state%root_mass + &
            state%sapwood_mass + state%heartwood_mass

         structural_carbon_after = result%leaf_mass_new + result%root_mass_new + &
            result%sapwood_mass_new + result%heartwood_mass_new

         whole_carbon_before = structural_carbon_before + result%carbon_storage_before
         whole_carbon_after = structural_carbon_after + result%carbon_storage_after

         result%whole_plant_balance_error = (whole_carbon_after - whole_carbon_before) - &
            (result%npp_daily + result%unpaid_carbon_deficit - &
             result%turnover_carbon_loss)

         result%carbon_accounting_ok = &
            abs(result%structural_balance_error) <= carbon_accounting_tolerance .and. &
            abs(result%storage_balance_error) <= carbon_accounting_tolerance .and. &
            abs(result%whole_plant_balance_error) <= carbon_accounting_tolerance

         result%leaf_root_residual = result%leaf_mass_new - &
            params%leaf_to_root_ratio * result%root_mass_new

         leaf_area_new = result%leaf_mass_new * params%sla
         sapwood_area_from_mass = 0.0_real64
         if (result%height_new > 0.0_real64) then
            sapwood_area_from_mass = result%sapwood_mass_new / &
                                     (params%wood_density * result%height_new)
         end if

         result%pipe_model_residual = leaf_area_new - params%latosa * sapwood_area_from_mass
         result%allocation_residual_final = 0.0_real64

         result%message = "Gradual storage allocation used with starvation and turnover."

      end subroutine allocate_gradual_with_storage

end module carbon_allocation_offline_kernel
