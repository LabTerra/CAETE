   ! Copyright 2017- LabTerra

!     This program is free software: you can redistribute it and/or modify
!     it under the terms of the GNU General Public License as published by
!     the Free Software Foundation, either version 3 of the License, or
!     (at your option) any later version.)

!     This program is distributed in the hope that it will be useful,
!     but WITHOUT ANY WARRANTY; without even the implied warranty of
!     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!     GNU General Public License for more details.

!     You should have received a copy of the GNU General Public License
!     along with this program.  If not, see <http://www.gnu.org/licenses/>.

! AUTHORS: Bianca Rius, Bárbara Cardeli, Carolina Blanco, JP Darela, David Lapola
! contact: <biancafaziorius ( at ) gmail.com>
!
! Allocation of the allometric version of CAETE.
!
! Woody PLSs (allocation3): gradual, daily allocation with a labile carbon
! storage. The pipe model and the leaf:root balance (as in LPJ) define targets
! for leaf, fine root and sapwood; the PLS moves towards them over
! allometric_adjustment_days, and the daily growth is capped at
! max_allocation_fraction of the living carbon.
!
! Grasses (grass_allocation3): NPP split between leaf and fine root by the
! aleaf/aroot traits, without allometric constraints.
!
! Nutrient cycle (N, P): both use the same subroutines - nitrogen_fixation,
! nutrient_limited_growth, nutrient_turnover and cap_nutrient_reserve -
! following the processes of allocation.f90.
!
! Height is a function of the stem carbon (sapwood + heartwood). It is computed
! once per PLS per day by the caller (budget_allom.f90) and reused here, so that
! light competition and allocation see the same height.

module alloc3

   use types
   use global_par
   use photo, only: spec_leaf_area, realized_npp
   use carbon_costs, only: passive_uptake, retran_nutri_cost, fixed_n,&
                         & active_costn, active_costp,&
                         & select_active_strategy, prep_out_n, prep_out_p

   implicit none
   private

   public :: allocation3                ! woody gradual allocation with labile storage
   public :: grass_allocation3          ! grass allocation (NPP-proportional, no allometry)
   public :: nitrogen_fixation          ! carbon sent to the N fixers and fixed N (woody and grass)
   public :: nutrient_limited_growth    ! N/P limitation of growth, uptake and reserve refill (woody and grass)
   public :: nutrient_turnover          ! litter N/P content and resorption (woody and grass)
   public :: cap_nutrient_reserve       ! ceiling of the N and P reserves (woody and grass)
   public :: height_from_stem_carbon    ! (f) height from total stem carbon (sapwood+heartwood)
   public :: leaf_req_calc3             ! (f) leaf mass requirement to satisfy the pipe model
   public :: diameter_from_height       ! (f) diameter consistent with height_from_stem_carbon's own H-D relation
   public :: crown_area_from_diameter   ! (f) crown area (m2), capped at crown_area_max (funcs.f90 pls_allometry)

   contains

   !==========================================================================
   ! Height (m) from the stem carbon (sapwood + heartwood), the wood density
   ! and the allometric constants k_allom2 and k_allom3.
   !==========================================================================

   function height_from_stem_carbon(stem_carbon_ind, wd_allom) result(height)

      real(r_8), intent(in) :: stem_carbon_ind !gC/ind - sapwood + heartwood
      real(r_8), intent(in) :: wd_allom        !g/cm3 - wood density trait

      real(r_8) :: height !m - output

      real(r_8) :: wd_gm3
      real(r_8) :: stem_volume
      real(r_8) :: height_exponent

      height = 0.0D0
      wd_gm3 = wd_allom * 1.0D6 !g/cm3 -> g/m3

      ! Height-diameter allometry: H = k_allom2*D**k_allom3. Stem as a
      ! cylinder: V = (pi/4)*D**2*H. Replacing D and solving for H:
      !    H**(1 + 2/k_allom3) = V * k_allom2**(2/k_allom3) / (pi/4)

      if (stem_carbon_ind .gt. 0.0D0 .and. wd_gm3 .gt. 0.0D0) then
         stem_volume = stem_carbon_ind / wd_gm3 !m3
         height_exponent = 1.0D0 + 2.0D0/k_allom3
         height = ((k_allom2**(2.0D0/k_allom3)) * stem_volume / (pi/4.0D0))**(1.0D0/height_exponent)
      endif

   end function height_from_stem_carbon


   !==========================================================================
   ! Leaf mass that the current sapwood mass supports under the pipe model,
   ! given the height.
   !==========================================================================

   function leaf_req_calc3(sap_in_ind, height, sla_allom, wd_allom) result(leaf_req)

      real(r_8), intent(in) :: sap_in_ind !gC/ind - sapwood
      real(r_8), intent(in) :: height     !m
      real(r_8), intent(in) :: sla_allom  !m2/gC
      real(r_8), intent(in) :: wd_allom   !g/cm3

      real(r_8) :: leaf_req !gC/ind - output

      leaf_req = 0.0D0

      if (height .gt. 0.0D0 .and. wd_allom .gt. 0.0D0 .and. sla_allom .gt. 0.0D0) then
         leaf_req = klatosa * sap_in_ind / ((wd_allom*1.0D6) * height * sla_allom)
      endif

   end function leaf_req_calc3


   !==========================================================================
   ! Diameter (m) from the height, by the inverse of the height-diameter
   ! allometry used in height_from_stem_carbon (H = k_allom2*D**k_allom3).
   !==========================================================================

   function diameter_from_height(height) result(diameter)

      real(r_8), intent(in) :: height !m - from height_from_stem_carbon

      real(r_8) :: diameter !m - output

      diameter = 0.0D0

      if (height .gt. 0.0D0) then
         diameter = (height/k_allom2)**(1.0D0/k_allom3)
      endif

   end function diameter_from_height

   !==========================================================================
   ! Crown area (m2) from the diameter, limited to crown_area_max (same
   ! relation and ceiling of pls_allometry in funcs.f90).
   !==========================================================================

   function crown_area_from_diameter(diameter) result(crown_area)

      real(r_8), intent(in) :: diameter !m - from diameter_from_height

      real(r_8) :: crown_area !m2 - output

      crown_area = 0.0D0

      if (diameter .gt. 0.0D0) then
         crown_area = min(crown_area_max, k_allom1*(diameter**krp))
      endif

   end function crown_area_from_diameter

   !==========================================================================
   ! POTENTIAL N FIXATION, as in allocation.f90: the pdia fraction (trait) of
   ! today's positive NPP can be sent to the N fixers (ctonfix) and fixes
   ! n_fixed (cc.f90's fixed_n), available for growth in the same day.
   ! Different from allocation.f90, only the N that today's growth needs is
   ! fixed (nutrient_limited_growth); the callers return the unused carbon to
   ! the storage. This keeps the N reserve from growing without limit when
   ! growth is limited by P.
   ! Shared by the woody and grass paths. npp_daily and ctonfix in g(C) m-2.
   !==========================================================================

   subroutine nitrogen_fixation(dt, npp_daily, ts, ctonfix, n_fixed)

      real(r_8), dimension(ntraits), intent(in) :: dt !PLS attributes
      real(r_8), intent(in) :: npp_daily !g(C) m-2 - today's NPP
      real(r_8), intent(in) :: ts        !soil temp oC

      real(r_8), intent(out) :: ctonfix  !g(C) m-2 - maximum carbon sent to the N fixers
      real(r_8), intent(out) :: n_fixed  !g(N) m-2 - N fixed with ctonfix

      real(r_8) :: pdia

      pdia = dt(17)

      ctonfix = pdia * max(0.0D0, npp_daily)
      n_fixed = fixed_n(ctonfix, ts)
      if (n_fixed .lt. 0.0D0) n_fixed = 0.0D0

   end subroutine nitrogen_fixation


   !==========================================================================
   ! NUTRIENT (N,P) limitation of structural growth, uptake and reserve
   ! refill - shared by the woody (allocation3) and grass (grass_allocation3)
   ! paths. All carbon arguments are g(C) m-2, all nutrient arguments g m-2.
   !
   ! Called after the caller has defined today's growth (carbon_to_allocate).
   ! It can only reduce that growth, never increase it.
   !
   ! Sequence (same processes of allocation.f90):
   !  - Demand from the tissue N:C and P:C traits.
   !  - Availability: daily accessible fraction of the solution pools
   !    (mult_factor_n/mult_factor_p) + PLS reserve + N fixed today.
   !  - Limitation of each organ (realized_npp), with the available N and P
   !    offered by priority: wood (sapwood), fine root, leaf.
   !  - Uptake: reserve first, then N fixation (only the N still needed),
   !    remainder from the soil.
   !  - Passive uptake (passive_uptake, cc.f90): what it brings beyond
   !    today's demand refills the reserve.
   !  - Active uptake of the demand not covered by the passive uptake
   !    (to_pay): cost and strategy from cc.f90.
   !==========================================================================

   subroutine nutrient_limited_growth(dt, wsoil, te, croot&
      &, mineral_n, labile_p, on, sop, op, sto_n_in, sto_p_in, n_fixed&
      &, delta_leaf, delta_root, delta_sapwood, carbon_to_allocate&
      &, from_storage_n, from_storage_p, n_to_storage, p_to_storage&
      &, nitrogen_uptake, phosphorus_uptake, limiting_nutrient&
      &, c_costs_of_uptake, uptk_strategy)

      real(r_8), dimension(ntraits), intent(in) :: dt !PLS attributes
      real(r_8), intent(in) :: wsoil     !soil water depth (mm)
      real(r_8), intent(in) :: te        !plant transpiration (mm/s)
      real(r_8), intent(in) :: croot     !g(C) m-2 - fine root carbon (active uptake costs)
      real(r_8), intent(in) :: mineral_n, labile_p, on, sop, op !g m-2
      real(r_8), intent(in) :: sto_n_in, sto_p_in !g m-2 - PLS N/P reserve (previous day)
      real(r_8), intent(inout) :: n_fixed !g m-2 - N fixation: in = potential (nitrogen_fixation), out = used by growth

      !structural growth, g(C) m-2: in = carbon-capped, out = nutrient-throttled
      real(r_8), intent(inout) :: delta_leaf, delta_root, delta_sapwood
      real(r_8), intent(inout) :: carbon_to_allocate

      real(r_8), intent(out) :: from_storage_n, from_storage_p !drawn from the PLS reserve for growth
      real(r_8), intent(out) :: n_to_storage, p_to_storage     !passive uptake beyond demand -> reserve
      real(r_8), dimension(2), intent(out) :: nitrogen_uptake   !(1) from mineral_n, (2) from on
      real(r_8), dimension(3), intent(out) :: phosphorus_uptake !(1) from labile_p, (2) from sop, (3) from op
      !Limiting nutrient: dim1 = leaf wood root, code: 1=N 2=P 4=N,COLIM 5=P,COLIM 6=COLIM 0=NOLIM
      integer(i_2), dimension(3), intent(out) :: limiting_nutrient
      real(r_8), intent(out) :: c_costs_of_uptake    !g(C) m-2 - cost of the active uptake
      integer(i_4), dimension(2), intent(out) :: uptk_strategy  !(1) N, (2) P - 0 = passive

      real(r_8) :: leaf_n2c, wood_n2c, root_n2c, leaf_p2c, wood_p2c, root_p2c
      real(r_8) :: avail_n_soil, avail_p_soil, avail_n_total, avail_p_total
      real(r_8) :: rnpp_n, rnpp_p
      logical(l_1) :: n_limited, p_limited

      !per-organ limitation (same parameters and codes as allocation.f90)
      integer(i_2), parameter :: nitrog = 1_i_2
      integer(i_2), parameter :: phosph = 2_i_2
      integer(i_2), parameter :: colimi = 3_i_2
      integer(i_4), parameter :: leaf = 1
      integer(i_4), parameter :: wood = 2
      integer(i_4), parameter :: root = 3
      integer(i_4) :: i
      real(r_8), dimension(3) :: growth   !g(C) m-2 - today's growth of leaf, wood (sapwood), root
      real(r_8), dimension(3) :: n2c, p2c !N:C and P:C of leaf, wood, root
      real(r_8) :: rem_n, rem_p           !g m-2 - N and P still available for the next organ
      integer(i_4) :: k
      integer(i_4), dimension(3), parameter :: order = (/wood, root, leaf/) !priority of the organs
      real(r_8) :: realized_n_demand, realized_p_demand
      real(r_8) :: from_soil_n, from_soil_p
      real(r_8), dimension(2) :: to_pay, to_sto, plant_passive_uptake

      !active uptake (same names as allocation.f90)
      real(r_8) :: amp                        !trait: AM fraction of the mycorrhizal association
      real(r_8) :: aux_on, aux_sop, aux_op    !g m-2 - accessible fraction of the on, sop and op pools
      real(r_8), dimension(6) :: ccn          !g(C) g(N)-1 - cost of N uptake by strategy
      real(r_8), dimension(8) :: ccp          !g(C) g(P)-1 - cost of P uptake by strategy
      real(r_8) :: unit_cost_n, unit_cost_p   !g(C) g(Nutrient)-1 - cost of the selected strategy
      real(r_8) :: active_nupt_cost, active_pupt_cost !g(C) m-2
      integer(i_4) :: naquis_strat, paquis_strat

      ! initialize ALL outputs
      from_storage_n         = 0.0D0
      from_storage_p         = 0.0D0
      n_to_storage           = 0.0D0
      p_to_storage           = 0.0D0
      nitrogen_uptake(:)     = 0.0D0
      phosphorus_uptake(:)   = 0.0D0
      limiting_nutrient(:)   = 0_i_2
      c_costs_of_uptake      = 0.0D0
      uptk_strategy(:)       = 0

      !nutrient stoichiometry traits (same indices as allocation.f90)
      leaf_n2c = dt(10) ! Nutrient:Carbon Ratios gg-1
      wood_n2c = dt(11)
      root_n2c = dt(12)
      leaf_p2c = dt(13)
      wood_p2c = dt(14)
      root_p2c = dt(15)
      amp = dt(16)

      avail_n_soil = max(0.0D0, mineral_n) * mult_factor_n
      avail_p_soil = max(0.0D0, labile_p)  * mult_factor_p

      !accessible fraction of the organic N, sorbed P and organic P pools
      aux_on  = max(0.0D0, on)  * mult_factor_n
      aux_sop = max(0.0D0, sop) * mult_factor_p
      aux_op  = max(0.0D0, op)  * mult_factor_p

      if (carbon_to_allocate .gt. 0.0D0) then

         avail_n_total = avail_n_soil + max(0.0D0, sto_n_in) + n_fixed
         avail_p_total = avail_p_soil + max(0.0D0, sto_p_in)

         !PER-ORGAN LIMITATION. realized_npp is applied to each organ (as in
         !allocation.f90), but the available N and P are offered by priority:
         !wood (sapwood) first, then fine root, then leaf. Each organ takes
         !what its growth needs and the remainder goes to the next one. Wood
         !has low N:C and P:C, so it takes little from root and leaf.
         growth(leaf) = delta_leaf
         growth(wood) = delta_sapwood
         growth(root) = delta_root
         n2c = (/leaf_n2c, wood_n2c, root_n2c/)
         p2c = (/leaf_p2c, wood_p2c, root_p2c/)

         rem_n = avail_n_total
         rem_p = avail_p_total
         do k = 1, 3
            i = order(k)
            if (growth(i) .le. 0.0D0) cycle

            call realized_npp(growth(i), growth(i)*n2c(i), rem_n, rnpp_n, n_limited)
            call realized_npp(growth(i), growth(i)*p2c(i), rem_p, rnpp_p, p_limited)

            !limiting nutrient code, same as allocation.f90
            if (n_limited .and. p_limited) then
               if (abs(rnpp_n - rnpp_p) .lt. 1.0D-6) then
                  limiting_nutrient(i) = colimi + colimi
               else if (rnpp_n .gt. rnpp_p) then
                  limiting_nutrient(i) = colimi + phosph
               else
                  limiting_nutrient(i) = colimi + nitrog
               endif
            else if (n_limited) then
               limiting_nutrient(i) = nitrog
            else if (p_limited) then
               limiting_nutrient(i) = phosph
            endif

            growth(i) = min(growth(i), rnpp_n, rnpp_p)
            rem_n = max(0.0D0, rem_n - growth(i)*n2c(i))
            rem_p = max(0.0D0, rem_p - growth(i)*p2c(i))
         enddo

         delta_leaf    = growth(leaf)
         delta_sapwood = growth(wood)
         delta_root    = growth(root)

         !carbon actually consumed by growth after the nutrient throttle - the
         !caller's storage balance must use this, not the pre-throttle value
         carbon_to_allocate = delta_leaf + delta_root + delta_sapwood

         realized_n_demand = delta_leaf*leaf_n2c + delta_root*root_n2c + delta_sapwood*wood_n2c
         realized_p_demand = delta_leaf*leaf_p2c + delta_root*root_p2c + delta_sapwood*wood_p2c
      else
         realized_n_demand = 0.0D0
         realized_p_demand = 0.0D0
      endif

      !uptake: draw from the PLS's own reserve first, then fix N (only what
      !the demand still needs), remainder from soil
      from_storage_n = min(max(0.0D0, sto_n_in), realized_n_demand)
      n_fixed        = max(0.0D0, min(n_fixed, realized_n_demand - from_storage_n))
      from_soil_n    = max(0.0D0, min(avail_n_soil, realized_n_demand - from_storage_n - n_fixed))
      from_storage_p = min(max(0.0D0, sto_p_in), realized_p_demand)
      from_soil_p    = max(0.0D0, min(avail_p_soil, realized_p_demand - from_storage_p))

      !passive uptake (transpiration stream). to_sto = passive uptake beyond
      !what today's growth takes from the soil -> refills the reserve.
      !to_pay = soil demand NOT covered by passive uptake -> active uptake.
      call passive_uptake(wsoil, avail_n_soil, avail_p_soil, from_soil_n, from_soil_p, te&
         &, to_pay, to_sto, plant_passive_uptake)

      n_to_storage = max(0.0D0, to_sto(1))
      p_to_storage = max(0.0D0, to_sto(2))

      !ACTIVE UPTAKE (as in allocation.f90): to_pay has a carbon cost. The
      !cheapest strategy (cc.f90) gives the cost per unit of nutrient and the
      !pool that provides to_pay: solution pool (strategies 1-4), organic
      !pools (N: 5-6, P: 5-7) or sorbed P (8). The on/sop/op pools do not
      !increase the growth ceiling (avail_*_total), as in allocation.f90.
      !The cost is the unit cost times to_pay, g(C) m-2.

      !N
      active_nupt_cost = 0.0D0
      naquis_strat = 0 !passive uptake
      if (to_pay(1) .gt. 0.0D0) then
         call active_costn(amp, avail_n_soil - plant_passive_uptake(1), aux_on, croot, ccn)
         call select_active_strategy(ccn, unit_cost_n, naquis_strat)
         call prep_out_n(naquis_strat, from_soil_n, to_pay(1), nitrogen_uptake)
         !the organic pool gives at most its accessible fraction; the rest
         !comes from the solution pool
         if (nitrogen_uptake(2) .gt. aux_on) then
            nitrogen_uptake(1) = nitrogen_uptake(1) + (nitrogen_uptake(2) - aux_on)
            nitrogen_uptake(2) = aux_on
         endif
         active_nupt_cost = unit_cost_n * to_pay(1)
      else
         nitrogen_uptake(1) = from_soil_n
         nitrogen_uptake(2) = 0.0D0
      endif
      uptk_strategy(1) = naquis_strat

      !P
      active_pupt_cost = 0.0D0
      paquis_strat = 0 !passive uptake
      if (to_pay(2) .gt. 0.0D0) then
         call active_costp(amp, avail_p_soil - plant_passive_uptake(2), aux_sop, aux_op, croot, ccp)
         call select_active_strategy(ccp, unit_cost_p, paquis_strat)
         call prep_out_p(paquis_strat, from_soil_p, to_pay(2), phosphorus_uptake)
         if (phosphorus_uptake(2) .gt. aux_sop) then
            phosphorus_uptake(1) = phosphorus_uptake(1) + (phosphorus_uptake(2) - aux_sop)
            phosphorus_uptake(2) = aux_sop
         endif
         if (phosphorus_uptake(3) .gt. aux_op) then
            phosphorus_uptake(1) = phosphorus_uptake(1) + (phosphorus_uptake(3) - aux_op)
            phosphorus_uptake(3) = aux_op
         endif
         active_pupt_cost = unit_cost_p * to_pay(2)
      else
         phosphorus_uptake(1) = from_soil_p
         phosphorus_uptake(2) = 0.0D0
         phosphorus_uptake(3) = 0.0D0
      endif
      uptk_strategy(2) = paquis_strat

      !the passive uptake beyond today's demand also leaves the solution
      !pools (never exceeds avail_*_soil: passive_uptake caps the passive flux
      !at the accessible pool)
      nitrogen_uptake(1)   = nitrogen_uptake(1)   + n_to_storage
      phosphorus_uptake(1) = phosphorus_uptake(1) + p_to_storage

      !carbon cost of the active uptake, paid in the next day
      c_costs_of_uptake = active_nupt_cost + active_pupt_cost

   end subroutine nutrient_limited_growth

   !=================================================================
   ! Ceiling of the N and P reserves of the PLS: the N (P) content of its
   ! leaves and fine roots, i.e. enough to rebuild them once. The excess is
   ! first taken from the passive uptake beyond demand (it stays in the soil)
   ! and the rest leaves the plant with the leaf litter.
   !=================================================================
   subroutine cap_nutrient_reserve(sto_max_n, sto_max_p, n_to_storage, p_to_storage&
      &, nitrogen_uptake, phosphorus_uptake, litter_nutrient_content, sto_n_out, sto_p_out)

      real(r_8), intent(in) :: sto_max_n, sto_max_p            !g m-2
      real(r_8), intent(inout) :: n_to_storage, p_to_storage   !g m-2
      real(r_8), dimension(2), intent(inout) :: nitrogen_uptake
      real(r_8), dimension(3), intent(inout) :: phosphorus_uptake
      real(r_8), dimension(6), intent(inout) :: litter_nutrient_content
      real(r_8), intent(inout) :: sto_n_out, sto_p_out         !g m-2

      real(r_8) :: excess, not_taken

      excess = max(0.0D0, sto_n_out - max(0.0D0, sto_max_n))
      not_taken = min(excess, n_to_storage)
      n_to_storage = n_to_storage - not_taken
      nitrogen_uptake(1) = max(0.0D0, nitrogen_uptake(1) - not_taken)
      litter_nutrient_content(1) = litter_nutrient_content(1) + (excess - not_taken)
      sto_n_out = sto_n_out - excess

      excess = max(0.0D0, sto_p_out - max(0.0D0, sto_max_p))
      not_taken = min(excess, p_to_storage)
      p_to_storage = p_to_storage - not_taken
      phosphorus_uptake(1) = max(0.0D0, phosphorus_uptake(1) - not_taken)
      litter_nutrient_content(4) = litter_nutrient_content(4) + (excess - not_taken)
      sto_p_out = sto_p_out - excess

   end subroutine cap_nutrient_reserve


   !==========================================================================
   ! LITTER NUTRIENT CONTENT AND RESORPTION - shared by the woody and grass
   ! paths. All carbon arguments are g(C) m-2, nutrient outputs g m-2.
   !
   ! Litter/turnover N,P content is computed stoichiometrically from the
   ! carbon flux times the current trait N:C/P:C ratio (same convention
   ! allocation.f90 uses - there is no persisted per-tissue N/P mass pool,
   ! N/P "in" a compartment is always current_C x trait ratio).
   !
   ! Turnover: leaf litter and the sapwood->heartwood conversion both undergo
   ! resorption (resorpt_frac) before the nutrient leaves the living pool;
   ! heartwood is therefore assumed to already carry only the
   ! post-resorption ratio by the time it turns over as CWD (heart_turn).
   ! Fine-root litter has no resorption modeled, matching allocation.f90.
   !
   ! Starvation: leaf/root carbon lost to pay an unmet carbon deficit is
   ! respired, not shed - it never becomes litter carbon. Its N/P is
   ! therefore fully remobilized into the PLS reserve (sending it to litter
   ! would create litter nutrient with no litter carbon). Sapwood converted
   ! to heartwood by starvation is treated exactly like the turnover
   ! conversion (resorpt_frac goes to the reserve, the rest stays in
   ! heartwood at the post-resorption ratio).
   !
   ! Carbon cost of resorption: cc.f90's retran_nutri_cost applied to the
   ! leaf litter resorption only, as in allocation.f90. The resorption of the
   ! sapwood->heartwood conversion and the starvation remobilization have no
   ! cost.
   !==========================================================================

   subroutine nutrient_turnover(dt, leaf_turn, root_turn, sap_to_heart_turn, heart_turn&
      &, leaf_starv, root_starv, sap_to_heart_starv&
      &, litter_nutrient_content, resorbed_n, resorbed_p, c_cost_resorpt)

      real(r_8), dimension(ntraits), intent(in) :: dt !PLS attributes
      real(r_8), intent(in) :: leaf_turn, root_turn, sap_to_heart_turn, heart_turn !g(C) m-2 - turnover
      real(r_8), intent(in) :: leaf_starv, root_starv, sap_to_heart_starv          !g(C) m-2 - starvation

      real(r_8), dimension(6), intent(out) :: litter_nutrient_content !leaf-N,root-N,cwd-N,leaf-P,root-P,cwd-P
      real(r_8), intent(out) :: resorbed_n, resorbed_p     !g m-2 - returned to the PLS reserve
      real(r_8), intent(out) :: c_cost_resorpt             !g(C) m-2 - carbon cost of leaf resorption

      real(r_8) :: leaf_n2c, wood_n2c, root_n2c, leaf_p2c, wood_p2c, root_p2c, resorpt_frac
      real(r_8) :: sap_to_heart
      real(r_8) :: n_leaf, p_leaf !g m-2 - N, P in leaf litter before resorption

      ! initialize ALL outputs
      litter_nutrient_content(:) = 0.0D0
      resorbed_n                 = 0.0D0
      resorbed_p                 = 0.0D0
      c_cost_resorpt             = 0.0D0

      resorpt_frac = dt(2)
      leaf_n2c = dt(10) ! Nutrient:Carbon Ratios gg-1
      wood_n2c = dt(11)
      root_n2c = dt(12)
      leaf_p2c = dt(13)
      wood_p2c = dt(14)
      root_p2c = dt(15)

      sap_to_heart = sap_to_heart_turn + sap_to_heart_starv

      litter_nutrient_content(1) = leaf_turn*leaf_n2c*(1.0D0 - resorpt_frac)
      litter_nutrient_content(2) = root_turn*root_n2c
      litter_nutrient_content(3) = heart_turn*wood_n2c*(1.0D0 - resorpt_frac)
      litter_nutrient_content(4) = leaf_turn*leaf_p2c*(1.0D0 - resorpt_frac)
      litter_nutrient_content(5) = root_turn*root_p2c
      litter_nutrient_content(6) = heart_turn*wood_p2c*(1.0D0 - resorpt_frac)

      resorbed_n = leaf_turn*leaf_n2c*resorpt_frac + sap_to_heart*wood_n2c*resorpt_frac&
         & + leaf_starv*leaf_n2c + root_starv*root_n2c
      resorbed_p = leaf_turn*leaf_p2c*resorpt_frac + sap_to_heart*wood_p2c*resorpt_frac&
         & + leaf_starv*leaf_p2c + root_starv*root_p2c

      !carbon cost of the leaf resorption (N + P), paid in the next day
      n_leaf = leaf_turn*leaf_n2c
      p_leaf = leaf_turn*leaf_p2c
      c_cost_resorpt = retran_nutri_cost(n_leaf, n_leaf*resorpt_frac, 1)&
         & + retran_nutri_cost(p_leaf, p_leaf*resorpt_frac, 2)

   end subroutine nutrient_turnover

   !==========================================================================
   ! Gradual daily allocation with labile carbon storage for woody PLSs.
   ! height_in is the height computed by the caller in this timestep (the same
   ! value used by the light competition).
   !==========================================================================

   subroutine allocation3(step, ri, p, dt, npp, npp_costs, ts, wsoil, te&
      &, leaf_in, wood_in, root_in, sap_in, heart_in, sto_in, height_in&
      &, mineral_n, labile_p, on, sop, op, sto_n_in, sto_p_in&
      &, leaf_out, wood_out, root_out, sap_out, heart_out, sto_out&
      &, sto_n_out, sto_p_out, leaf_litter, root_litter, cwd&
      &, nitrogen_uptake, phosphorus_uptake&
      &, litter_nutrient_content, limiting_nutrient&
      &, c_costs_of_uptake, uptk_strategy, ctonfix&
      &, leaf_req, leaf_inc_min, root_inc_min)

      !VARIABLE INPUTS
      real(r_8), dimension(ntraits), intent(in) :: dt !PLS attributes
      integer(i_4), intent(in) :: p, ri, step

      real(r_8), intent(in) :: leaf_in, wood_in, root_in, sap_in, heart_in, sto_in !kgC/m2
      real(r_8), intent(in) :: npp        !kgC/m2/yr - annualized NPP rate
      real(r_8), intent(in) :: height_in  !m - precomputed this timestep by the caller

      !NUTRIENT INPUTS (g/m2). mineral_n/labile_p/on/sop/op follow the same
      !names/units/convention as allocation.f90: "on" (organic N), "sop"
      !(sorbed P), "op" (organic P).

      real(r_8), intent(in) :: mineral_n, labile_p, on, sop, op
      real(r_8), intent(in) :: sto_n_in, sto_p_in !g/m2 - PLS N/P reserve (previous day)
      real(r_8), intent(in) :: npp_costs !g(C)/m2 - previous-day C cost of nutrient uptake
      real(r_8), intent(in) :: ts        !soil temp oC (N fixation)
      real(r_8), intent(in) :: wsoil     !soil water depth (mm)
      real(r_8), intent(in) :: te        !plant transpiration (mm/s)

      !VARIABLE OUTPUTS
      real(r_8), intent(out) :: leaf_out, wood_out, root_out, sap_out, heart_out, sto_out !kgC/m2

      !NUTRIENT OUTPUTS (g/m2)
      real(r_8), intent(out) :: sto_n_out, sto_p_out !updated PLS N/P reserve
      real(r_8), dimension(2), intent(out) :: nitrogen_uptake   !(1) from mineral_n, (2) from on
      real(r_8), dimension(3), intent(out) :: phosphorus_uptake !(1) from labile_p, (2) from sop, (3) from op
      real(r_8), dimension(6), intent(out) :: litter_nutrient_content !leaf-N,root-N,cwd-N,leaf-P,root-P,cwd-P
      !Limiting nutrient: dim1 = leaf wood root, code: 1=N 2=P 4=N,COLIM 5=P,COLIM 6=COLIM 0=NOLIM
      integer(i_2), dimension(3), intent(out) :: limiting_nutrient
      real(r_8), intent(out) :: c_costs_of_uptake    !g(C)/m2 - C cost of nutrient uptake and resorption (paid in the next day)
      integer(i_4), dimension(2), intent(out) :: uptk_strategy  !(1) N, (2) P uptake strategy (0 = passive)
      real(r_8), intent(out) :: ctonfix              !g(C)/m2 - C sent to N fixers

      !LITTER CARBON OUTPUTS (gC/m2/day) - turnover fluxes leaving the plant
      real(r_8), intent(out) :: leaf_litter, root_litter, cwd

      real(r_8), intent(out) :: leaf_req      !gC/ind - pipe-model leaf requirement (diagnostic)
      real(r_8), intent(out) :: leaf_inc_min  !gC/ind/day - leaf demand (diagnostic)
      real(r_8), intent(out) :: root_inc_min  !gC/ind/day - root demand (diagnostic)

      !INTERNAL VARIABLES
      real(r_8) :: dens_in
      real(r_8) :: leaf_in_ind, root_in_ind, sap_in_ind, heart_in_ind, sto_in_ind
      real(r_8) :: sla_allom, wd_allom
      real(r_8) :: dt_years, npp_daily

      real(r_8) :: storage_after_npp, unmet_storage_deficit

      real(r_8) :: leaf_required
      real(r_8) :: delta_leaf_min_pipe, delta_leaf_min_root_nonneg, delta_leaf_min
      real(r_8) :: leaf_target, root_required, delta_root_min, root_target
      real(r_8) :: sapwood_required_for_leaf_target, sapwood_target
      real(r_8) :: leaf_deficit, root_deficit, sapwood_deficit
      real(r_8) :: leaf_demand_daily, root_demand_daily, sapwood_demand_daily, total_demand_daily
      real(r_8) :: leaf_background_demand, root_background_demand, sapwood_background_demand
      real(r_8) :: living_carbon, max_daily_alloc, carbon_to_allocate
      real(r_8) :: frac_leaf, frac_root, frac_sapwood
      real(r_8) :: delta_leaf, delta_root, delta_sapwood
      real(r_8) :: leaf_mass_new, root_mass_new, sapwood_mass_new, heartwood_mass_new

      real(r_8) :: starvation_share, leaf_starv, root_starv, sap_starv, sap_to_heart

      real(r_8) :: storage_after_alloc
      real(r_8) :: leaf_turn, root_turn, sap_turn, sto_turn, heart_turn, sap_to_heart_turn
      real(r_8) :: sto_final_ind

      real(r_8) :: structural_sum, structural_balance_error, storage_balance_error
      logical(l_1) :: carbon_accounting_ok

      !NUTRIENT INTERNAL VARIABLES
      real(r_8) :: from_storage_n, from_storage_p, n_to_storage, p_to_storage
      real(r_8) :: resorbed_n, resorbed_p, c_cost_resorpt
      real(r_8) :: n_fixed       !g(N) m-2 - N fixation (potential, then used)
      real(r_8) :: n_fixed_pot   !g(N) m-2 - potential N fixation
      real(r_8) :: ctonfix_refund !g(C) m-2 - carbon not used by the N fixers, back to storage

      ! initialize ALL outputs
      leaf_out                   = 0.0D0
      wood_out                   = 0.0D0
      root_out                   = 0.0D0
      sap_out                    = 0.0D0
      heart_out                  = 0.0D0
      sto_out                    = 0.0D0
      sto_n_out                  = 0.0D0
      sto_p_out                  = 0.0D0
      leaf_litter                = 0.0D0
      root_litter                = 0.0D0
      cwd                        = 0.0D0
      nitrogen_uptake(:)         = 0.0D0
      phosphorus_uptake(:)       = 0.0D0
      litter_nutrient_content(:) = 0.0D0
      limiting_nutrient(:)       = 0_i_2
      c_costs_of_uptake          = 0.0D0
      uptk_strategy(:)           = 0
      ctonfix                    = 0.0D0
      leaf_req                   = 0.0D0
      leaf_inc_min               = 0.0D0
      root_inc_min               = 0.0D0

      !take PLS traits
      ! SLA via Reich et al. (1997), derived from leaf longevity (tau_leaf, dt(3)).
      sla_allom  = spec_leaf_area(dt(3))  !m2/gC
      wd_allom   = dt(19)              !g/cm3

      dt_years = 1.0D0/year_days

      !provisory !until there are individuals and FPC/establishment is implemented
      dens_in = 1.0D0

      !convert kgC/m2 -> gC/ind
      leaf_in_ind  = (leaf_in/dens_in)*1.0D3
      root_in_ind  = (root_in/dens_in)*1.0D3
      sap_in_ind   = (sap_in/dens_in)*1.0D3
      heart_in_ind = (heart_in/dens_in)*1.0D3
      sto_in_ind   = (sto_in/dens_in)*1.0D3

      !annualized NPP rate -> carbon available today (gC/ind)
      npp_daily = (npp/dens_in)*1.0D3*dt_years

      !--------------------------------------------------------------------
      ! Potential N fixation: maximum carbon sent to the fixers (ctonfix)
      ! and N fixed with it (n_fixed)
      !--------------------------------------------------------------------
      call nitrogen_fixation(dt, npp_daily, ts, ctonfix, n_fixed)
      n_fixed_pot = n_fixed

      !--------------------------------------------------------------------
      ! Update labile storage with today's NPP. Negative NPP consumes storage;
      ! if storage would go negative, the remainder is an unmet deficit that
      ! triggers the starvation rule below.
      ! The carbon costs of nutrient uptake of the previous day (npp_costs)
      ! are paid here, before growth, as in allocation.f90
      ! (npp_pot = npp_pot - npp_to_fixer - npp_costs), together with the
      ! carbon sent to the N fixers. What NPP + storage cannot pay joins the
      ! unmet deficit instead of becoming a debt for the next day.
      !--------------------------------------------------------------------

      storage_after_npp = sto_in_ind + npp_daily - npp_costs - ctonfix
      if (storage_after_npp .lt. 0.0D0) then
         unmet_storage_deficit = abs(storage_after_npp)
         storage_after_npp = 0.0D0
      else
         unmet_storage_deficit = 0.0D0
      endif

      !--------------------------------------------------------------------
      ! Allometric targets (pipe model + leaf:root balance). They give the
      ! direction of the daily demand; they are not imposed in a single day.
      !--------------------------------------------------------------------
      
      leaf_required = leaf_req_calc3(sap_in_ind, height_in, sla_allom, wd_allom)

      delta_leaf_min_pipe = leaf_required - leaf_in_ind
      delta_leaf_min_root_nonneg = root_in_ind*ltor - leaf_in_ind
      delta_leaf_min = max(0.0D0, delta_leaf_min_pipe, delta_leaf_min_root_nonneg)

      leaf_target = leaf_in_ind + delta_leaf_min

      if (ltor .gt. 0.0D0) then
         root_required = leaf_target/ltor
      else
         root_required = root_in_ind
      endif
      delta_root_min = max(0.0D0, root_required - root_in_ind)
      root_target = root_in_ind + delta_root_min

      !sapwood mass that would support the leaf target under the pipe model
      !(gradual target only - not forced instantly)
      sapwood_required_for_leaf_target = leaf_target*sla_allom/klatosa*(wd_allom*1.0D6)*height_in
      sapwood_target = max(sap_in_ind, sapwood_required_for_leaf_target)

      leaf_deficit    = max(0.0D0, leaf_target    - leaf_in_ind)
      root_deficit    = max(0.0D0, root_target    - root_in_ind)
      sapwood_deficit = max(0.0D0, sapwood_target - sap_in_ind)

      !convert full allometric deficits into gradual daily demand
      leaf_demand_daily    = leaf_deficit/allometric_adjustment_days
      root_demand_daily    = root_deficit/allometric_adjustment_days
      sapwood_demand_daily = sapwood_deficit/allometric_adjustment_days

      !small background structural demand (keeps growth going near allometric balance)
      if (leaf_background_timescale_years .gt. 0.0D0) then
         leaf_background_demand = leaf_target*dt_years/leaf_background_timescale_years
      else
         leaf_background_demand = 0.0D0
      endif

      if (root_background_timescale_years .gt. 0.0D0) then
         root_background_demand = root_target*dt_years/root_background_timescale_years
      else
         root_background_demand = 0.0D0
      endif

      if (sapwood_background_timescale_years .gt. 0.0D0) then
         sapwood_background_demand = sapwood_target*dt_years/sapwood_background_timescale_years
      else
         sapwood_background_demand = 0.0D0
      endif

      leaf_demand_daily    = leaf_demand_daily    + leaf_background_demand
      root_demand_daily    = root_demand_daily    + root_background_demand
      sapwood_demand_daily = sapwood_demand_daily + sapwood_background_demand

      total_demand_daily = leaf_demand_daily + root_demand_daily + sapwood_demand_daily

      !daily structural allocation limited by storage, demand and max growth fraction
      living_carbon = leaf_in_ind + root_in_ind + sap_in_ind
      max_daily_alloc = max_allocation_fraction*living_carbon

      if (total_demand_daily .gt. 0.0D0 .and. storage_after_npp .gt. 0.0D0 &
         &.and. max_daily_alloc .gt. 0.0D0) then

         carbon_to_allocate = min(storage_after_npp, total_demand_daily, max_daily_alloc)

         frac_leaf    = leaf_demand_daily/total_demand_daily
         frac_root    = root_demand_daily/total_demand_daily
         frac_sapwood = sapwood_demand_daily/total_demand_daily

         delta_leaf    = frac_leaf*carbon_to_allocate
         delta_root    = frac_root*carbon_to_allocate
         delta_sapwood = frac_sapwood*carbon_to_allocate
      else
         carbon_to_allocate = 0.0D0
         delta_leaf    = 0.0D0
         delta_root    = 0.0D0
         delta_sapwood = 0.0D0
      endif

      !--------------------------------------------------------------------
      ! NUTRIENT (N,P): throttle delta_leaf/root/sapwood and carbon_to_allocate
      ! by N/P availability, and get today's uptake and reserve fluxes.
      ! gC/ind == gC/m2 here (dens_in = 1), same units the helper expects.
      !--------------------------------------------------------------------

      call nutrient_limited_growth(dt, wsoil, te, root_in_ind&
         &, mineral_n, labile_p, on, sop, op, sto_n_in, sto_p_in, n_fixed&
         &, delta_leaf, delta_root, delta_sapwood, carbon_to_allocate&
         &, from_storage_n, from_storage_p, n_to_storage, p_to_storage&
         &, nitrogen_uptake, phosphorus_uptake, limiting_nutrient&
         &, c_costs_of_uptake, uptk_strategy)

      !carbon of the N fixation not used goes back to the storage
      ctonfix_refund = 0.0D0
      if (n_fixed_pot .gt. 0.0D0) then
         ctonfix_refund = ctonfix*(1.0D0 - n_fixed/n_fixed_pot)
      else
         ctonfix_refund = ctonfix
      endif
      ctonfix = ctonfix - ctonfix_refund

      leaf_mass_new      = leaf_in_ind  + delta_leaf
      root_mass_new      = root_in_ind  + delta_root
      sapwood_mass_new   = sap_in_ind   + delta_sapwood
      heartwood_mass_new = heart_in_ind

      !--------------------------------------------------------------------
      ! Starvation rule: negative NPP not covered by storage is split equally
      ! among leaf, fine-root and sapwood. Leaf/root losses leave the plant;
      ! sapwood loss becomes heartwood (sapwood-to-heartwood conversion).
      !--------------------------------------------------------------------
      leaf_starv   = 0.0D0
      root_starv   = 0.0D0
      sap_starv    = 0.0D0
      sap_to_heart = 0.0D0

      if (unmet_storage_deficit .gt. 0.0D0) then
         starvation_share = unmet_storage_deficit/3.0D0

         leaf_starv = min(starvation_share, leaf_mass_new)
         root_starv = min(starvation_share, root_mass_new)
         sap_starv  = min(starvation_share, sapwood_mass_new)
         sap_to_heart = sap_starv

         leaf_mass_new      = leaf_mass_new      - leaf_starv
         root_mass_new      = root_mass_new      - root_starv
         sapwood_mass_new   = sapwood_mass_new   - sap_starv
         heartwood_mass_new = heartwood_mass_new + sap_to_heart
      endif

      storage_after_alloc = storage_after_npp - carbon_to_allocate + ctonfix_refund

      !--------------------------------------------------------------------
      ! Continuous compartment turnover (annual rates converted to a daily
      ! loss via dt_years = 1/year_days). Sapwood turnover becomes heartwood.
      !--------------------------------------------------------------------
      leaf_turn  = min(leaf_mass_new,      leaf_mass_new      * l_turnover   * dt_years)
      root_turn  = min(root_mass_new,      root_mass_new      * r_turnover   * dt_years)
      sap_turn   = min(sapwood_mass_new,   sapwood_mass_new   * s_turnover   * dt_years)
      sto_turn   = min(storage_after_alloc, storage_after_alloc * sto_turnover * dt_years)
      heart_turn = min(heartwood_mass_new, heartwood_mass_new * h_turnover   * dt_years)

      sap_to_heart_turn = sap_turn

      leaf_mass_new      = leaf_mass_new      - leaf_turn
      root_mass_new      = root_mass_new      - root_turn
      sapwood_mass_new   = sapwood_mass_new   - sap_turn
      heartwood_mass_new = heartwood_mass_new + sap_to_heart_turn - heart_turn

      sto_final_ind = storage_after_alloc - sto_turn

      !--------------------------------------------------------------------
      ! LITTER CARBON, LITTER NUTRIENT CONTENT AND RESORPTION.
      ! Litter carbon is the turnover flux only: starvation losses are
      ! respired (leaf/root) or kept as heartwood (sapwood), and sto_turn is
      ! not routed to litter.
      !--------------------------------------------------------------------

      leaf_litter = leaf_turn*dens_in
      root_litter = root_turn*dens_in
      cwd         = heart_turn*dens_in

      call nutrient_turnover(dt, leaf_turn, root_turn, sap_to_heart_turn, heart_turn&
         &, leaf_starv, root_starv, sap_to_heart&
         &, litter_nutrient_content, resorbed_n, resorbed_p, c_cost_resorpt)

      !total carbon cost of today (uptake + resorption), paid in the next day
      c_costs_of_uptake = c_costs_of_uptake + c_cost_resorpt

      !reserve balance: previous reserve - drawn for growth + passive uptake
      !beyond today's demand + resorbed/remobilized N,P (soil uptake and
      !fixed N that meet today's demand go straight into structural tissue
      !and never pass through the reserve)
      sto_n_out = max(0.0D0, sto_n_in) - from_storage_n + n_to_storage + resorbed_n
      sto_p_out = max(0.0D0, sto_p_in) - from_storage_p + p_to_storage + resorbed_p

      !ceiling of the reserves: N and P content of leaves and fine roots
      call cap_nutrient_reserve((leaf_mass_new*dt(10) + root_mass_new*dt(12))*dens_in&
         &, (leaf_mass_new*dt(13) + root_mass_new*dt(15))*dens_in, n_to_storage, p_to_storage&
         &, nitrogen_uptake, phosphorus_uptake, litter_nutrient_content, sto_n_out, sto_p_out)

      !--------------------------------------------------------------------
      ! Check of the internal carbon balance (prints a warning if it fails)
      !--------------------------------------------------------------------
      structural_sum = delta_leaf + delta_root + delta_sapwood
      structural_balance_error = carbon_to_allocate - structural_sum

      storage_balance_error = sto_final_ind - (sto_in_ind + npp_daily - npp_costs - ctonfix &
         &+ unmet_storage_deficit - carbon_to_allocate - sto_turn)

      carbon_accounting_ok = (abs(structural_balance_error) .le. tol) &
         &.and. (abs(storage_balance_error) .le. tol)

      if (.not. carbon_accounting_ok) then
         print*, 'WARNING alloc3/allocation3: carbon accounting mismatch for PLS', p, &
            &'structural_balance_error=', structural_balance_error, &
            &'storage_balance_error=', storage_balance_error
      endif

      !--------------------------------------------------------------------
      ! Diagnostics returned to the caller (not used for control flow downstream)
      !--------------------------------------------------------------------
      leaf_req     = leaf_required
      leaf_inc_min = leaf_demand_daily
      root_inc_min = root_demand_daily

      !convert back to kgC/m2
      leaf_out  = (leaf_mass_new*dens_in)/1.0D3
      root_out  = (root_mass_new*dens_in)/1.0D3
      sap_out   = (sapwood_mass_new*dens_in)/1.0D3
      heart_out = (heartwood_mass_new*dens_in)/1.0D3
      sto_out   = (sto_final_ind*dens_in)/1.0D3
      wood_out  = sap_out + heart_out

   end subroutine allocation3

   !==========================================================================
   ! Grass allocation (awood <= 0), without allometric constraints.
   ! Positive NPP is split between leaf and fine root by the aleaf/aroot
   ! traits. The daily growth is capped at max_allocation_fraction of the
   ! living carbon and then limited by N and P (nutrient_limited_growth). NPP
   ! above the cap, or not used because of nutrient limitation, goes to the
   ! storage. Negative NPP is paid by the storage; what is left is split
   ! between leaf and root. Turnover, litter N/P and resorption as in the
   ! woody path. Sapwood and heartwood are always zero.
   !==========================================================================

   subroutine grass_allocation3(p, dt, npp, npp_costs, ts, wsoil, te&
      &, leaf_in, root_in, sto_in&
      &, mineral_n, labile_p, on, sop, op, sto_n_in, sto_p_in&
      &, leaf_out, wood_out, root_out, sap_out, heart_out, sto_out&
      &, sto_n_out, sto_p_out, leaf_litter, root_litter, cwd&
      &, nitrogen_uptake, phosphorus_uptake&
      &, litter_nutrient_content, limiting_nutrient&
      &, c_costs_of_uptake, uptk_strategy, ctonfix)

      integer(i_4), intent(in) :: p
      real(r_8), dimension(ntraits), intent(in) :: dt
      real(r_8), intent(in) :: npp !kgC/m2/yr
      real(r_8), intent(in) :: leaf_in, root_in, sto_in !kgC/m2

      !NUTRIENT INPUTS (g/m2) - same meaning as in allocation3
      real(r_8), intent(in) :: mineral_n, labile_p, on, sop, op
      real(r_8), intent(in) :: sto_n_in, sto_p_in
      real(r_8), intent(in) :: npp_costs !g(C)/m2 - previous-day C cost of nutrient uptake
      real(r_8), intent(in) :: ts        !soil temp oC (N fixation)
      real(r_8), intent(in) :: wsoil     !soil water depth (mm)
      real(r_8), intent(in) :: te        !plant transpiration (mm/s)

      real(r_8), intent(out) :: leaf_out, wood_out, root_out, sap_out, heart_out, sto_out

      !NUTRIENT OUTPUTS (g/m2) - same meaning as in allocation3
      real(r_8), intent(out) :: sto_n_out, sto_p_out
      real(r_8), dimension(2), intent(out) :: nitrogen_uptake
      real(r_8), dimension(3), intent(out) :: phosphorus_uptake
      real(r_8), dimension(6), intent(out) :: litter_nutrient_content
      integer(i_2), dimension(3), intent(out) :: limiting_nutrient
      real(r_8), intent(out) :: c_costs_of_uptake    !g(C)/m2 - C cost of nutrient uptake and resorption
      integer(i_4), dimension(2), intent(out) :: uptk_strategy  !(1) N, (2) P uptake strategy
      real(r_8), intent(out) :: ctonfix              !g(C)/m2 - C sent to N fixers

      !LITTER CARBON OUTPUTS (gC/m2/day)
      real(r_8), intent(out) :: leaf_litter, root_litter, cwd

      real(r_8) :: aleaf, aroot, asto
      real(r_8) :: dt_years, npp_daily, npp_net
      real(r_8) :: leaf_mass_new, root_mass_new, sto_mass_new
      real(r_8) :: leaf_turn, root_turn, sto_turn
      real(r_8) :: deficit
      real(r_8) :: living_carbon, max_daily_alloc, structural_demand, carbon_to_allocate, excess
      real(r_8) :: frac_leaf, frac_root

      !nutrient internal variables (carbon in gC/m2 for the shared helpers)
      real(r_8) :: delta_leaf_g, delta_root_g, delta_sap_g, carbon_to_allocate_g
      real(r_8) :: leaf_starv, root_starv
      real(r_8) :: from_storage_n, from_storage_p, n_to_storage, p_to_storage
      real(r_8) :: resorbed_n, resorbed_p, c_cost_resorpt
      real(r_8) :: n_fixed       !g(N) m-2 - N fixation (potential, then used)
      real(r_8) :: n_fixed_pot   !g(N) m-2 - potential N fixation
      real(r_8) :: ctonfix_refund !g(C) m-2 - carbon not used by the N fixers, back to storage

      ! initialize ALL outputs
      leaf_out                   = 0.0D0
      wood_out                   = 0.0D0
      root_out                   = 0.0D0
      sap_out                    = 0.0D0
      heart_out                  = 0.0D0
      sto_out                    = 0.0D0
      sto_n_out                  = 0.0D0
      sto_p_out                  = 0.0D0
      leaf_litter                = 0.0D0
      root_litter                = 0.0D0
      cwd                        = 0.0D0
      nitrogen_uptake(:)         = 0.0D0
      phosphorus_uptake(:)       = 0.0D0
      litter_nutrient_content(:) = 0.0D0
      limiting_nutrient(:)       = 0_i_2
      c_costs_of_uptake          = 0.0D0
      uptk_strategy(:)           = 0
      ctonfix                    = 0.0D0

      aleaf = dt(6) ! ALLOCATION  (proportion  %/100)
      aroot = dt(8)
      !remaining NPP fraction (not sent to leaf/root) accumulates in storage
      asto  = max(0.0D0, 1.0D0 - aleaf - aroot)

      dt_years  = 1.0D0/year_days
      npp_daily = npp*dt_years !kgC/m2/day

      !potential N fixation: maximum carbon sent to the fixers (ctonfix) and
      !N fixed with it (n_fixed)
      call nitrogen_fixation(dt, npp_daily*1.0D3, ts, ctonfix, n_fixed)
      n_fixed_pot = n_fixed

      !carbon costs of nutrient uptake of the previous day and the carbon
      !sent to the N fixers are paid before growth, as in allocation.f90
      !(npp_pot = npp_pot - npp_to_fixer - npp_costs)
      npp_net = npp_daily - (npp_costs + ctonfix)/1.0D3 !kgC/m2/day

      leaf_mass_new = leaf_in
      root_mass_new = root_in
      sto_mass_new  = sto_in

      leaf_starv = 0.0D0
      root_starv = 0.0D0

      delta_leaf_g = 0.0D0
      delta_root_g = 0.0D0
      delta_sap_g  = 0.0D0 !no sapwood in grasses
      carbon_to_allocate_g = 0.0D0

      structural_demand = 0.0D0

      if (npp_net .ge. 0.0D0) then

         !structural growth capped at max_allocation_fraction of living carbon/day

         living_carbon = leaf_in + root_in
         max_daily_alloc = max_allocation_fraction*living_carbon

         structural_demand = (aleaf + aroot)*npp_net

         if (structural_demand .gt. 0.0D0 .and. max_daily_alloc .gt. 0.0D0) then
            carbon_to_allocate = min(structural_demand, max_daily_alloc)
         else
            carbon_to_allocate = 0.0D0
         endif

         if (structural_demand .gt. 0.0D0) then
            frac_leaf = aleaf/(aleaf + aroot)
            frac_root = aroot/(aleaf + aroot)
         else
            frac_leaf = 0.0D0
            frac_root = 0.0D0
         endif

         delta_leaf_g = frac_leaf*carbon_to_allocate*1.0D3
         delta_root_g = frac_root*carbon_to_allocate*1.0D3
         carbon_to_allocate_g = carbon_to_allocate*1.0D3
      endif

      !NUTRIENT (N,P): throttle today's structural growth by N/P availability
      !and get uptake and reserve fluxes. Called on every day (also with zero
      !growth) so passive uptake keeps refilling the reserve.
      
      call nutrient_limited_growth(dt, wsoil, te, root_in*1.0D3&
         &, mineral_n, labile_p, on, sop, op, sto_n_in, sto_p_in, n_fixed&
         &, delta_leaf_g, delta_root_g, delta_sap_g, carbon_to_allocate_g&
         &, from_storage_n, from_storage_p, n_to_storage, p_to_storage&
         &, nitrogen_uptake, phosphorus_uptake, limiting_nutrient&
         &, c_costs_of_uptake, uptk_strategy)

      if (npp_net .ge. 0.0D0) then

         leaf_mass_new = leaf_mass_new + delta_leaf_g/1.0D3
         root_mass_new = root_mass_new + delta_root_g/1.0D3

         !structural demand not realized (above the daily cap, or throttled
         !by N/P) joins storage, same as the asto-directed NPP fraction
         excess = structural_demand - carbon_to_allocate_g/1.0D3

         sto_mass_new  = sto_mass_new  + asto*npp_net + excess
      else
         if ((sto_mass_new + npp_net) .ge. 0.0D0) then
            sto_mass_new = sto_mass_new + npp_net
         else
            deficit = -(sto_mass_new + npp_net)
            sto_mass_new = 0.0D0
            leaf_starv = min(leaf_mass_new, 0.5D0*deficit)
            root_starv = min(root_mass_new, 0.5D0*deficit)
            leaf_mass_new = leaf_mass_new - leaf_starv
            root_mass_new = root_mass_new - root_starv
         endif
      endif

      !carbon of the N fixation not used goes back to the storage
      ctonfix_refund = 0.0D0
      if (n_fixed_pot .gt. 0.0D0) then
         ctonfix_refund = ctonfix*(1.0D0 - n_fixed/n_fixed_pot)
      else
         ctonfix_refund = ctonfix
      endif
      ctonfix = ctonfix - ctonfix_refund
      sto_mass_new = sto_mass_new + ctonfix_refund/1.0D3

      leaf_turn = min(leaf_mass_new, leaf_mass_new*l_turnover*dt_years)
      root_turn = min(root_mass_new, root_mass_new*r_turnover*dt_years)
      sto_turn  = sto_mass_new*sto_turnover*dt_years

      leaf_out = max(0.0D0, leaf_mass_new - leaf_turn)
      root_out = max(0.0D0, root_mass_new - root_turn)
      sto_out  = max(0.0D0, sto_mass_new  - sto_turn)

      sap_out   = 0.0D0
      heart_out = 0.0D0
      wood_out  = 0.0D0

      !litter carbon (gC/m2/day): turnover only - starvation losses are respired
      leaf_litter = leaf_turn*1.0D3
      root_litter = root_turn*1.0D3
      cwd         = 0.0D0

      call nutrient_turnover(dt, leaf_turn*1.0D3, root_turn*1.0D3, 0.0D0, 0.0D0&
         &, leaf_starv*1.0D3, root_starv*1.0D3, 0.0D0&
         &, litter_nutrient_content, resorbed_n, resorbed_p, c_cost_resorpt)

      !total carbon cost of today (uptake + resorption), paid in the next day
      c_costs_of_uptake = c_costs_of_uptake + c_cost_resorpt

      sto_n_out = max(0.0D0, sto_n_in) - from_storage_n + n_to_storage + resorbed_n
      sto_p_out = max(0.0D0, sto_p_in) - from_storage_p + p_to_storage + resorbed_p

      !ceiling of the reserves: N and P content of leaves and fine roots
      call cap_nutrient_reserve((leaf_out*dt(10) + root_out*dt(12))*1.0D3&
         &, (leaf_out*dt(13) + root_out*dt(15))*1.0D3, n_to_storage, p_to_storage&
         &, nitrogen_uptake, phosphorus_uptake, litter_nutrient_content, sto_n_out, sto_p_out)

   end subroutine grass_allocation3

end module alloc3
