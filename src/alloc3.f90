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
! This module replaces allocation2.f90 (alloc2). It implements a gradual, daily,
! storage-based allometric allocation scheme for woody PLSs (based on the LPJ pipe
! model / allometric relations, but relaxed toward the allometric targets over
! allometric_adjustment_days instead of forcing an instantaneous bisection solution),
! plus a simple NPP-proportional allocation scheme for grasses (aleaf/aroot split,
! no allometric constraints, following the same conceptual pattern used for
! herbaceous strategies in alloc/allocation.f90).
!
! Height is a pure function of current stem carbon (sapwood+heartwood) via
! height_from_stem_carbon. It carries no independent state: it is recomputed once
! per PLS per timestep by the caller (budget_allom.f90, canopy pre-loop) and that
! same value is reused here (never recalculated) so that the light-competition
! canopy structure and the allometric demand calculation always see an identical
! height within the same timestep.

module alloc3

   use types
   use global_par
   use photo, only: spec_leaf_area

   implicit none
   private

   public :: allocation3                ! woody gradual allocation with labile storage
   public :: grass_allocation3          ! grass allocation (NPP-proportional, no allometry)
   public :: height_from_stem_carbon    ! (f) height from total stem carbon (sapwood+heartwood)
   public :: leaf_req_calc3             ! (f) leaf mass requirement to satisfy the pipe model
   public :: diameter_from_height       ! (f) diameter consistent with height_from_stem_carbon's own H-D relation
   public :: crown_area_from_diameter   ! (f) crown area (m2), capped at crown_area_max (funcs.f90 pls_allometry)

   contains

   !==========================================================================
   !> Height from total stem carbon (sapwood + heartwood), wood density and the
   !> global allometric constants (k_allom2, k_allom3). Pure function of the
   !> current pools: no persisted/independent height state.
   !==========================================================================

   function height_from_stem_carbon(stem_carbon_ind, wd_allom) result(height)

      real(r_8), intent(in) :: stem_carbon_ind !gC/ind - sapwood + heartwood
      real(r_8), intent(in) :: wd_allom        !g/cm3 - wood density trait

      real(r_8) :: height !m - output

      real(r_8) :: wd_gm3
      real(r_8) :: stem_volume
      real(r_8) :: height_exponent

      height = 0.0D0
      wd_gm3 = wd_allom * 1.0D6 !g/cm3 -> g/m3 (same conversion used throughout alloc2)

      ! Derivation: height-diameter allometry H = k_allom2*D**k_allom3 (1), stem
      ! modeled as a cylinder V = (pi/4)*D**2*H (2). Substituting D = (H/k_allom2)
      ! **(1/k_allom3), from (1), into (2) and solving for H gives:
      !    H**(1 + 2/k_allom3) = V * k_allom2**(2/k_allom3) / (pi/4)
      ! height_exponent is that combined exponent (1 + 2/k_allom3); the height
      ! itself is obtained by taking the (1/height_exponent)-th root of the
      ! right-hand side below.

      if (stem_carbon_ind .gt. 0.0D0 .and. wd_gm3 .gt. 0.0D0) then
         stem_volume = stem_carbon_ind / wd_gm3 !m3
         height_exponent = 1.0D0 + 2.0D0/k_allom3
         height = ((k_allom2**(2.0D0/k_allom3)) * stem_volume / (pi/4.0D0))**(1.0D0/height_exponent)
      end if

   end function height_from_stem_carbon


   !==========================================================================
   !> Leaf mass required to maintain the existing sapwood mass under the pipe
   !> model, given the current height (algebraically equivalent to alloc2's
   !> leaf_req_calc, written without the redundant 1.0D3 juggling).
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
      end if

   end function leaf_req_calc3


   !==========================================================================
   !> Diameter consistent with a given height, via the inverse of the exact
   !> same height-diameter allometry (H = k_allom2*D**k_allom3) that
   !> height_from_stem_carbon solved for H. Deriving D from H (rather than
   !> recomputing it independently from stem volume, as funcs.f90's
   !> pls_allometry does) guarantees the two are always consistent with each
   !> other by construction - same underlying equation, no risk of numerical
   !> drift between two independently-solved formulas.
   !==========================================================================

   function diameter_from_height(height) result(diameter)

      real(r_8), intent(in) :: height !m - from height_from_stem_carbon

      real(r_8) :: diameter !m - output

      diameter = 0.0D0

      if (height .gt. 0.0D0) then
         diameter = (height/k_allom2)**(1.0D0/k_allom3)
      end if

   end function diameter_from_height

   !==========================================================================
   !> Crown area (m2) from diameter, capped at crown_area_max (global_par) -
   !> same allometric relation and ceiling used by funcs.f90's pls_allometry
   !> (LPJ-mFire establishment.f90 convention: no individual crown exceeds
   !> crown_area_max regardless of how large the stem carbon pool grows).
   !==========================================================================

   function crown_area_from_diameter(diameter) result(crown_area)

      real(r_8), intent(in) :: diameter !m - from diameter_from_height

      real(r_8) :: crown_area !m2 - output

      crown_area = 0.0D0

      if (diameter .gt. 0.0D0) then
         crown_area = min(crown_area_max, k_allom1*(diameter**krp))
      end if

   end function crown_area_from_diameter

   !==========================================================================
   !> Gradual daily allocation with labile carbon storage for woody PLSs.
   !> height_in must be the value already computed this timestep by the caller
   !> (height_from_stem_carbon applied to the pre-allocation pools) - it is
   !> reused here, never recalculated, to guarantee it matches exactly what the
   !> canopy/light-competition pre-loop used for the same PLS in this timestep.
   !==========================================================================

   subroutine allocation3(step, ri, p, dt, npp, leaf_in, wood_in, root_in, sap_in, heart_in, sto_in, height_in&
      &, leaf_out, wood_out, root_out, sap_out, heart_out, sto_out&
      &, leaf_req, leaf_inc_min, root_inc_min)

      !VARIABLE INPUTS
      real(r_8), dimension(ntraits), intent(in) :: dt !PLS attributes
      integer(i_4), intent(in) :: p, ri, step

      real(r_8), intent(in) :: leaf_in, wood_in, root_in, sap_in, heart_in, sto_in !kgC/m2
      real(r_8), intent(in) :: npp        !kgC/m2/yr - annualized NPP rate
      real(r_8), intent(in) :: height_in  !m - precomputed this timestep by the caller

      !VARIABLE OUTPUTS
      real(r_8), intent(out) :: leaf_out, wood_out, root_out, sap_out, heart_out, sto_out !kgC/m2

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
      ! Update labile storage with today's NPP. Negative NPP consumes storage;
      ! if storage would go negative, the remainder is an unmet deficit that
      ! triggers the starvation rule below.
      !--------------------------------------------------------------------

      storage_after_npp = sto_in_ind + npp_daily
      if (storage_after_npp .lt. 0.0D0) then
         unmet_storage_deficit = abs(storage_after_npp)
         storage_after_npp = 0.0D0
      else
         unmet_storage_deficit = 0.0D0
      end if

      !--------------------------------------------------------------------
      ! Allometric correction direction (pipe model + leaf:root functional
      ! balance), following the same logic as alloc2's leaf_req_calc/
      ! leaf_inc_min_calc/root_inc_min_calc, but used here as a gradual demand
      ! direction instead of a hard constraint solved by bisection.
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
      end if
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
      end if

      if (root_background_timescale_years .gt. 0.0D0) then
         root_background_demand = root_target*dt_years/root_background_timescale_years
      else
         root_background_demand = 0.0D0
      end if

      if (sapwood_background_timescale_years .gt. 0.0D0) then
         sapwood_background_demand = sapwood_target*dt_years/sapwood_background_timescale_years
      else
         sapwood_background_demand = 0.0D0
      end if

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
      end if

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
      end if

      storage_after_alloc = storage_after_npp - carbon_to_allocate

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
      ! Internal carbon-accounting safety net (diagnostic only - not exposed
      ! to the caller, matching the debug-print convention already used
      ! elsewhere in this codebase, e.g. alloc2's bisection loop warnings).
      !--------------------------------------------------------------------
      structural_sum = delta_leaf + delta_root + delta_sapwood
      structural_balance_error = carbon_to_allocate - structural_sum

      storage_balance_error = sto_final_ind - (sto_in_ind + npp_daily &
         &+ unmet_storage_deficit - carbon_to_allocate - sto_turn)

      carbon_accounting_ok = (abs(structural_balance_error) .le. tol) &
         &.and. (abs(storage_balance_error) .le. tol)

      if (.not. carbon_accounting_ok) then
         print*, 'WARNING alloc3/allocation3: carbon accounting mismatch for PLS', p, &
            &'structural_balance_error=', structural_balance_error, &
            &'storage_balance_error=', storage_balance_error
      end if

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
   !> Grass allocation (awood <= 0): no allometric/pipe-model constraints.
   !> Positive NPP is split between leaf and fine root via the aleaf/aroot
   !> trait fractions (same conceptual pattern as alloc/allocation.f90, but
   !> carbon-only - no nutrient cycling here). Structural growth (leaf+root)
   !> is capped at max_allocation_fraction of current living carbon per day -
   !> the same relative-rate cap used by the woody allocation3 path - so that
   !> a day of unusually high NPP cannot instantly compound into unbounded
   !> leaf/root mass. NPP in excess of that cap accumulates in storage
   !> (sto) instead of being forced into structure immediately. Sapwood/
   !> heartwood are always zero. Negative NPP is paid from storage first;
   !> any remaining deficit is split between leaf and root. Turnover uses
   !> the same global annual rates and daily conversion (dt_years) as the
   !> woody path.
   !==========================================================================

   subroutine grass_allocation3(p, dt, npp, leaf_in, root_in, sto_in&
      &, leaf_out, wood_out, root_out, sap_out, heart_out, sto_out)

      integer(i_4), intent(in) :: p
      real(r_8), dimension(ntraits), intent(in) :: dt
      real(r_8), intent(in) :: npp !kgC/m2/yr
      real(r_8), intent(in) :: leaf_in, root_in, sto_in !kgC/m2

      real(r_8), intent(out) :: leaf_out, wood_out, root_out, sap_out, heart_out, sto_out

      real(r_8) :: aleaf, aroot, asto
      real(r_8) :: dt_years, npp_daily
      real(r_8) :: leaf_mass_new, root_mass_new, sto_mass_new
      real(r_8) :: leaf_turn, root_turn, sto_turn
      real(r_8) :: deficit
      real(r_8) :: living_carbon, max_daily_alloc, structural_demand, carbon_to_allocate, excess
      real(r_8) :: frac_leaf, frac_root

      aleaf = dt(6)
      aroot = dt(8)
      !remaining NPP fraction (not sent to leaf/root) accumulates in storage
      asto  = max(0.0D0, 1.0D0 - aleaf - aroot)

      dt_years  = 1.0D0/year_days
      npp_daily = npp*dt_years !kgC/m2/day

      leaf_mass_new = leaf_in
      root_mass_new = root_in
      sto_mass_new  = sto_in

      if (npp_daily .ge. 0.0D0) then

         !structural growth capped at max_allocation_fraction of living carbon/day,
         !mirroring the woody path's max_daily_alloc (see allocation3 above)

         living_carbon = leaf_in + root_in
         max_daily_alloc = max_allocation_fraction*living_carbon

         structural_demand = (aleaf + aroot)*npp_daily

         if (structural_demand .gt. 0.0D0 .and. max_daily_alloc .gt. 0.0D0) then
            carbon_to_allocate = min(structural_demand, max_daily_alloc)
         else
            carbon_to_allocate = 0.0D0
         end if
         excess = structural_demand - carbon_to_allocate

         if (structural_demand .gt. 0.0D0) then
            frac_leaf = aleaf/(aleaf + aroot)
            frac_root = aroot/(aleaf + aroot)
         else
            frac_leaf = 0.0D0
            frac_root = 0.0D0
         end if

         leaf_mass_new = leaf_mass_new + frac_leaf*carbon_to_allocate
         root_mass_new = root_mass_new + frac_root*carbon_to_allocate

         !unallocated structural demand (above the daily cap) joins storage,
         !same as the asto-directed NPP fraction

         sto_mass_new  = sto_mass_new  + asto*npp_daily + excess
      else
         if ((sto_mass_new + npp_daily) .ge. 0.0D0) then
            sto_mass_new = sto_mass_new + npp_daily
         else
            deficit = -(sto_mass_new + npp_daily)
            sto_mass_new = 0.0D0
            leaf_mass_new = max(0.0D0, leaf_mass_new - 0.5D0*deficit)
            root_mass_new = max(0.0D0, root_mass_new - 0.5D0*deficit)
         end if
      end if

      leaf_turn = leaf_mass_new*l_turnover*dt_years
      root_turn = root_mass_new*r_turnover*dt_years
      sto_turn  = sto_mass_new*sto_turnover*dt_years

      leaf_out = max(0.0D0, leaf_mass_new - leaf_turn)
      root_out = max(0.0D0, root_mass_new - root_turn)
      sto_out  = max(0.0D0, sto_mass_new  - sto_turn)

      sap_out   = 0.0D0
      heart_out = 0.0D0
      wood_out  = 0.0D0

   end subroutine grass_allocation3

end module alloc3
