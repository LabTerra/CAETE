! Copyright 2023- LabTerra

!     This program is free software: you can redistribute it and/or modify
!     it under the terms of the GNU General Public License as published by
!     the Free Software Foundation, either version 3 of the License, or
!     (at your option) any later version.

!     This program is distributed in the hope that it will be useful,
!     but WITHOUT ANY WARRANTY; without even the implied warranty of
!     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!     GNU General Public License for more details.

!     You should have received a copy of the GNU General Public License
!     along with this program.  If not, see <http://www.gnu.org/licenses/>.

! contacts :: David Montenegro Lapola <lapoladm ( at ) gmail.com>
! Author: Bianca Rius and JP Darela
! This program is based on the work of those that gave us the INPE-CPTEC-PVM2 model

! This program is a modified version of budget1. It is being developed to desconsider
! nutri cycle for allocation and to implement allocation following allometric restrictions

module budget_allom
   implicit none
   private
 
   public :: daily_budget_allom
 
contains
 
   subroutine daily_budget_allom(step, dt, w1, w2, wmax_in, ts, temp, p0, ipar, rh, catm&
      &, cleaf_in, cwood_in, croot_in, csap_in, cheart_in, csto_in&
      &, dleaf_in, dwood_in, droot_in, dsap_in, dheart_in, dsto_in&
      &, cleaf_out, cwood_out, croot_out, csap_out, cheart_out, csto_out& !outputs
      &, dleaf_out, dwood_out, droot_out, dsap_out, dheart_out, dsto_out&
      &, cleaf_grd, cwood_grd, croot_grd, csap_grd, cheart_grd, csto_grd&
      &, evavg, epavg, phavg, aravg, nppavg, laiavg, rcavg&
      &, f5avg, rmavg, rgavg, wueavg, cueavg, vcmax_1&
      &, specific_la_1, ocpavg)

      use types
      use global_par, only: ntraits, npls
      use alloc
      use productivity
      use omp_lib

      use photo, only: pft_area_frac, sto_resp
      use water, only: evpot2, penman, available_energy, runoff
      use alloc2

      !==========================================================================
      !     ----------------------------INPUTS-------------------------------

      !PLS traits (traits)
      real(r_8),dimension(ntraits,npls),intent(in) :: dt
            !Water
      real(r_8),intent(in) :: w1  !Initial (previous month last day) soil moisture storage (mm) - upper layer
      real(r_8),intent(in) :: w2  !Initial (previous month last day) soil moisture storage (mm) - lower layer

      !Climatic data
      real(r_4),intent(in) :: temp    ! Surface air temperature (oC)
      real(r_4),intent(in) :: p0      ! Surface pressure (mb)
      real(r_4),intent(in) :: ipar    ! Incident photosynthetic active radiation mol Photons m-2 s-1
      real(r_4),intent(in) :: rh      ! Relative humidity 
      real(r_8),intent(in) :: catm    ! ATM CO2 concentration ppm
      real(r_4),intent(in) :: ts      ! Soil temperature (oC)

      !Soil
      real(r_8),intent(in) :: wmax_in ! Saturation point

      !Vegetation pools
      real(r_8),dimension(npls),intent(in) :: cleaf_in
      real(r_8),dimension(npls),intent(in) :: cwood_in
      real(r_8),dimension(npls),intent(in) :: croot_in
      real(r_8),dimension(npls),intent(in) :: csap_in
      real(r_8),dimension(npls),intent(in) :: cheart_in
      real(r_8),dimension(npls),intent(in) :: csto_in

      !Delta vegetation pools
      real(r_8),dimension(npls),intent(in) :: dleaf_in
      real(r_8),dimension(npls),intent(in) :: dwood_in
      real(r_8),dimension(npls),intent(in) :: droot_in
      real(r_8),dimension(npls),intent(in) :: dsap_in
      real(r_8),dimension(npls),intent(in) :: dheart_in
      real(r_8),dimension(npls),intent(in) :: dsto_in



      

      


      !==========================================================================

      !==========================================================================
      !     ----------------------------OUTPUTS-------------------------------

      !DAILY OUTPUTS
      !Vegetation pools
      real(r_8),dimension(npls),intent(out) :: cleaf_out
      real(r_8),dimension(npls),intent(out) :: cwood_out
      real(r_8),dimension(npls),intent(out) :: croot_out
      real(r_8),dimension(npls),intent(out) :: csap_out
      real(r_8),dimension(npls),intent(out) :: cheart_out
      real(r_8),dimension(npls),intent(out) :: csto_out


      !Delta vegetation pools
      real(r_8),dimension(npls),intent(out) :: dleaf_out
      real(r_8),dimension(npls),intent(out) :: dwood_out
      real(r_8),dimension(npls),intent(out) :: droot_out
      real(r_8),dimension(npls),intent(out) :: dsap_out
      real(r_8),dimension(npls),intent(out) :: dheart_out
      real(r_8),dimension(npls),intent(out) :: dsto_out




      real(r_8),dimension(npls),intent(out) :: ocpavg    ! [0-1] Gridcell occupation

      !CWM OUTPUTS
      real(r_4),intent(out) :: epavg          !Maximum evapotranspiration (mm/day)
      real(r_8),intent(out) :: evavg          !Actual evapotranspiration Daily average (mm/day)
      real(r_8),intent(out) :: phavg          !Daily photosynthesis (Kg m-2 y-1)
      real(r_8),intent(out) :: aravg          !Daily autotrophic respiration (Kg m-2 y-1)
      real(r_8),intent(out) :: nppavg         !Daily NPP (average between PFTs)(Kg m-2 y-1)
      real(r_8),intent(out) :: laiavg         !Daily leaf19010101', '19551231 area Index m2m-2
      real(r_8),intent(out) :: rcavg          !Daily canopy resistence s/m
      real(r_8),intent(out) :: f5avg          !Daily canopy resistence s/m
      real(r_8),intent(out) :: rmavg          !maintenance/growth respiration (Kg m-2 y-1)
      real(r_8),intent(out) :: rgavg          !maintenance/growth respiration (Kg m-2 y-1)
      real(r_8),intent(out) :: wueavg         ! Water use efficiency
      real(r_8),intent(out) :: cueavg         ! [0-1]
      real(r_8),intent(out) :: vcmax_1          ! µmol m-2 s-1
      real(r_8),intent(out) :: specific_la_1    ! m2 g(C)-1

      !carbon veg. pools for the gridcell (grd)
      !Values considering all the PLSs weighted by their relative C contribution
      real(r_8), intent(out) :: cleaf_grd
      real(r_8), intent(out) :: cwood_grd
      real(r_8), intent(out) :: croot_grd
      real(r_8), intent(out) :: csap_grd
      real(r_8), intent(out) :: cheart_grd
      real(r_8), intent(out) :: csto_grd





      !==========================================================================

      !==========================================================================
      !     ---------------------INTERNAL VARIABLES-------------------------------

      !index for PLS loops
      integer(i_4) :: i
      integer(i_4) :: p
      integer(i_4) :: counter
      integer(i_4) :: ri
      integer(i_4) :: step

      !PLSs
      real(r_8),dimension(ntraits) :: dt1 ! Store one PLS attributes array (1D)
      integer(i_4) :: nlen !number of alive PLSs
      integer(i_4), dimension(:), allocatable :: lp ! index of living PLSs/living grasses
      real(r_8), dimension(:), allocatable :: idx_grasses !grass identifier

      !Carbon vegetation pools
      real(r_8),dimension(npls) :: cleaf_pls
      real(r_8),dimension(npls) :: cwood_pls
      real(r_8),dimension(npls) :: croot_pls
      real(r_8),dimension(npls) :: csap_pls
      real(r_8),dimension(npls) :: cheart_pls
      real(r_8),dimension(npls) :: csto_pls


      !Carbon vegetation pools after allocation routine
      real(r_8),dimension(:), allocatable :: cleaf_pls2
      real(r_8),dimension(:), allocatable :: cwood_pls2
      real(r_8),dimension(:), allocatable :: croot_pls2
      real(r_8),dimension(:), allocatable :: csap_pls2
      real(r_8),dimension(:), allocatable :: cheart_pls2
      real(r_8),dimension(:), allocatable :: csto_pls2

      !carbon vegetation pools discounting c deficit
      real(r_8),dimension(:), allocatable :: cleaf_int
      real(r_8),dimension(:), allocatable :: cwood_int
      real(r_8),dimension(:), allocatable :: croot_int
      real(r_8),dimension(:), allocatable :: csap_int
      real(r_8),dimension(:), allocatable :: cheart_int
      real(r_8),dimension(:), allocatable :: csto_int

      real(r_8),dimension(:), allocatable :: leaf_req
      real(r_8),dimension(:), allocatable :: leaf_inc_min
      real(r_8),dimension(:), allocatable :: root_inc_min



      !Carbon vegetation pools (auxiliar for internal convertions)
      real(r_8),dimension(:), allocatable :: cleaf_pls_aux
      real(r_8),dimension(:), allocatable :: cwood_pls_aux
      real(r_8),dimension(:), allocatable :: croot_pls_aux
      real(r_8),dimension(:), allocatable :: csap_pls_aux
      real(r_8),dimension(:), allocatable :: cheart_pls_aux
      real(r_8),dimension(:), allocatable :: csto_pls_aux


      !Delta veg pools
      real(r_8),dimension(npls) :: dleaf
      real(r_8),dimension(npls) :: dwood
      real(r_8),dimension(npls) :: droot
      real(r_8),dimension(npls) :: dsap
      real(r_8),dimension(npls) :: dheart
      real(r_8),dimension(npls) :: dsto

      !Carbon vegetation pools (auxiliar for internal convertions)
      real(r_8),dimension(:), allocatable :: dleaf_pls_aux
      real(r_8),dimension(:), allocatable :: dwood_pls_aux
      real(r_8),dimension(:), allocatable :: droot_pls_aux
      real(r_8),dimension(:), allocatable :: dsap_pls_aux
      real(r_8),dimension(:), allocatable :: dheart_pls_aux
      real(r_8),dimension(:), allocatable :: dsto_pls_aux



      !Carbon Cycle
      real(r_4),dimension(:),allocatable :: ph           !Canopy gross photosynthesis (kgC/m2/yr)
      real(r_4),dimension(:),allocatable :: ar           !Autotrophic respiration (kgC/m2/yr)
      real(r_4),dimension(:),allocatable :: nppa         !Net primary productivity / auxiliar
      real(r_8),dimension(:),allocatable :: laia         !Leaf area index (m2 leaf/m2 area)
      real(r_8),dimension(:),allocatable :: f5           !Foliar Photosynthesis (mol/m2/s)
      real(r_4),dimension(:),allocatable :: vpd          !Vapor Pressure deficit
      real(r_4),dimension(:),allocatable :: rm           !Maintenance respiration (kgC/m2/yr)
      real(r_4),dimension(:),allocatable :: rg           !Growth respiration (kgC/m2/yr)
      real(r_4),dimension(:),allocatable :: cue          !Carbon use efficiency
      real(r_4),dimension(:),allocatable :: c_def        !Carbon deficit due to negative NPP (kgC/m2/yr) - i.e. ph < ar
      real(r_8),dimension(:),allocatable :: vcmax        !Rubisco maximum velocity of carboxilation (µmol/m2/s)
      real(r_8),dimension(:),allocatable :: specific_la  !Specific leaf area (m2/kg)
      real(r_8),dimension(:),allocatable :: litter_l     ! leaf litter
      real(r_8),dimension(:),allocatable :: cwd          ! coarse wood debris (to litter)
      real(r_8),dimension(:),allocatable :: litter_fr    ! fine roots litter

      !water pools
      real(r_8) :: w  !Daily soil moisture storage (mm)
      !Water cycle
      real(r_4),dimension(:),allocatable :: evap   !Actual evapotranspiration (mm/day)
      real(r_4),dimension(:),allocatable :: wue    !Water use efficiency
      real(r_4),dimension(:),allocatable :: rc2    !Canopy resistence (s/m)
      real(r_8),dimension(:),allocatable :: tra    !Transpiration (mm/s)
      real(r_4) :: emax                            !Maximum evapotranspiration (mm/day)

      !soil
      real(r_4) :: soil_temp !soil temperature
      real(r_8) :: soil_sat  !soil water saturation point
      
      !area frac occupation
      real(r_8),    dimension(npls) :: awood_aux !define if tree or grass
      logical(l_1), dimension(npls) :: ocp_wood  !wood occupation
      integer(i_4), dimension(npls) :: run       !verifies if the PLS is alive
      real(r_8),    dimension(npls) :: ocp_mm    ! TODO include cabon of dead plssss in the cicle? (not implemented)
      real(r_8),dimension(:),allocatable :: ocp_coeffs !occupancy coefficients for each PLS


      !===========================================================================

      !Initializing

      !ATENÇÃO, VERIFICAR QUESTÃO DO TOTAL WOOD
      
      do i = 1, npls
         awood_aux(i) = dt(7,i)

         cleaf_pls(i)  = cleaf_in(i)
         cwood_pls(i)  = cwood_in(i)
         croot_pls(i)  = croot_in(i)
         csap_pls(i)   = csap_in(i)
         cheart_pls(i) = cheart_in(i)
         csto_pls(i)   = csto_in(i)


         dleaf(i)  = dleaf_in(i)
         dwood(i)  = dwood_in(i)
         droot(i)  = droot_in(i)
         dsap(i)   = dsap_in(i)
         dheart(i) = dheart_in(i)
         dsto(i)   = dsto_in(i)
         

         
         ! cleaf_out(i) = cleaf_pls(i) + 1.
         ! print*,'cleaf_in',cleaf_in(i), i
      
      enddo

      w = w1 + w2        
      soil_temp = ts     
      soil_sat = wmax_in

      call pft_area_frac(cleaf_pls, croot_pls, cwood_pls, awood_aux,&
      &                 ocpavg, ocp_wood, run, ocp_mm)

      nlen = sum(run)    ! New length for the arrays in the main loop
                         ! get the total number of alives

      ! print*, 'nlen ====>', nlen

      allocate(lp(nlen))
      allocate(ocp_coeffs(nlen))
      allocate(idx_grasses(nlen))

      ! Get only living PLSs
      counter = 1
      do p = 1,npls
         if(run(p).eq. 1) then
            lp(counter) = p
            ocp_coeffs(counter) = ocpavg(p)
            counter = counter + 1
         endif
      enddo
      
      ! Identify grasses
      idx_grasses(:) = 1.0D0
      
      do p = 1, nlen
         if (awood_aux(lp(p)) .le. 0.0D0) idx_grasses(p) = 0.0D0
         !This variable multipies ocp_coeffs for the wood tissues. If it is
         !a grass it turns the ocp_coeffs for wood_tissues = 0
      enddo

      !dimensioning according to alive PLSs
      allocate(evap(nlen))
      allocate(nppa(nlen))
      allocate(ph(nlen))
      allocate(ar(nlen))
      allocate(laia(nlen))
      allocate(f5(nlen))
      allocate(vpd(nlen))
      allocate(rc2(nlen))
      allocate(rm(nlen))
      allocate(rg(nlen))
      allocate(wue(nlen))
      allocate(cue(nlen))
      allocate(c_def(nlen))
      allocate(vcmax(nlen))
      allocate(specific_la(nlen))
      allocate(litter_l(nlen))
      allocate(cwd(nlen))
      allocate(litter_fr(nlen))
      allocate(tra(nlen))

      allocate(cleaf_pls2(nlen))
      allocate(cwood_pls2(nlen))
      allocate(croot_pls2(nlen))
      allocate(csap_pls2(nlen))
      allocate(cheart_pls2(nlen))
      allocate(csto_pls2(nlen))

      allocate(leaf_req(nlen))
      allocate(leaf_inc_min(nlen))
      allocate(root_inc_min(nlen))

      allocate(cleaf_int(nlen))
      allocate(cwood_int(nlen))
      allocate(croot_int(nlen))
      allocate(csap_int(nlen))
      allocate(cheart_int(nlen))
      allocate(csto_int(nlen))

      allocate(dleaf_pls_aux(nlen))
      allocate(dwood_pls_aux(nlen))
      allocate(droot_pls_aux(nlen))
      allocate(dsap_pls_aux(nlen))
      allocate(dheart_pls_aux(nlen))
      allocate(dsto_pls_aux(nlen))

      !     Maximum evapotranspiration (emax)
      !     =================================
      emax = evpot2(p0,temp,rh,available_energy(temp))
      soil_temp = ts

      ! FAZER NUmthreads função de nlen pra otimizar a criação de trheads
      if (nlen .le. 20) then
         call OMP_SET_NUM_THREADS(1)
      else if (nlen .le. 100) then
         call OMP_SET_NUM_THREADS(1)
      else if (nlen .le. 300) then
         call OMP_SET_NUM_THREADS(2)
      else if (nlen .le. 600) then
         call OMP_SET_NUM_THREADS(3)
      else
         call OMP_SET_NUM_THREADS(3)
      endif
      !$OMP PARALLEL DO &
      !$OMP SCHEDULE(AUTO) &
      !$OMP DEFAULT(SHARED)



      do p = 1,nlen
         
         ri = lp(p) !get the correspondentt value of for that specific pls (don't lose the original PLS index)
         dt1 = dt(:,ri) ! Pick up the pls functional attributes list
        
         !here ri is real index. The outputs use p to save memory, but at the end of the doc
         !it is transformed in p

         
         
         call prod(dt1, ocp_wood(ri), catm, temp, soil_temp, p0, w, ipar&
            &, rh, emax, cleaf_pls(ri), csap_pls(ri), croot_pls(ri), dleaf(ri), dsap(ri), droot(ri)&
            &, soil_sat, ph(p), ar(p), nppa(p), laia(p), f5(p), vpd(p)&
            &, rm(p), rg(p), rc2(p), wue(p), c_def(p), vcmax(p), specific_la(p), tra(p))


         ! if (p.eq.1259)then
            ! print*,'_____________'
            ! print*, 'cleaf_pls',cleaf_pls(p), p
            ! print*, 'cleaf_pls2',cleaf_pls2(p), p
            ! print*, 'dleaf aux', dleaf_pls_aux(p), p
            ! print*, 'ph', ph(p), p
            ! print*, 'nppa', nppa(p), p
            ! print*, 'rm', rm(p), p
            ! print*, 'ar', ar(p), p
            ! print*, 'rg', rg(p), p
   
            ! print*,'_____________'
         ! endif
         
         evap(p) = penman(p0, temp, rh, available_energy(temp), rc2(p)) !actual evapotranspiration (evap, mm/day)
         ! if (p.eq.1460)then
            ! print*, ''
            ! print*, 'csap in', csap_pls(ri), p, step
         ! endif
         
         call allocation2(step, ri, p, dt1,nppa(p)&
            &,cleaf_pls(ri), cwood_pls(ri), croot_pls(ri), csap_pls(ri), cheart_pls(ri), csto_pls(ri)&
            &,cleaf_pls2(p), cwood_pls2(p), croot_pls2(p), csap_pls2(p), cheart_pls2(p), csto_pls2(p)&
            &,leaf_req(p), leaf_inc_min(p), root_inc_min(p))
         ! if (p.eq.1460) then
         ! if(csap_pls2(p).eq.0.0D0)then
         !    print*, 'leaf out alloc', cleaf_pls2(p), step,p
         !    print*, 'sap out alloc', csap_pls2(p), step,p
         ! endif
         ! endif
         ! print*, 'wood', cwood_pls2(p), p
         ! print*, 'sap', csap_pls2(p), p
         ! print*, 'heart', cheart_pls2(p), p
         

            !Carbon use efficiency & Delta C
         if(ph(p) .eq. 0.0 .or. nppa(p) .eq. 0.0) then
            cue(p) = 0.0
         else
            cue(p) = nppa(p)/ph(p)
         endif


         !Mass balance (c deficit)
         c_def(p) = c_def(p)/2.73791 !/2.73791 transforms to year

         if (c_def(p).gt.0.0D0) then
            if (dt1(7) .gt. 0.0D0) then
               cleaf_int(p) = cleaf_pls2(p) - (c_def(p) * 0.15)
               croot_int(p) = croot_pls2(p) - (c_def(p) * 0.15)
               csap_int(p)  = csap_pls2(p) - (c_def(p) * 0.15)
               cheart_int(p) = cheart_pls2(p) + (csap_pls2(p) - (c_def(p) * 0.15))
               csto_int(p) = csto_pls2(p) - (c_def(p)*0.55)
               cwood_int(p) = csap_int(p) + cheart_int(p)
               
            else
               cleaf_int(p) = cleaf_pls2(p) - (c_def(p) * 0.5)
               croot_int(p) = croot_pls2(p) - (c_def(p) * 0.5)
               csap_int(p)  = 0.0D0
               cheart_int(p) = 0.0D0
               csto_int(p) = 0.0D0
               cwood_int(p) = 0.0D0
            endif
         else 
            if (dt1(7) .gt. 0.0D0)then
               cleaf_int(p) = cleaf_pls2(p) 
               croot_int(p) = croot_pls2(p) 
               csap_int(p)  = csap_pls2(p)
               cheart_int(p) = cheart_pls2(p) 
               csto_int(p) = csto_pls2(p) 
               cwood_int(p) = csap_int(p) + cheart_int(p)
            else
               cleaf_int(p) = cleaf_pls2(p) 
               croot_int(p) = croot_pls2(p) 
               csap_int(p)  = 0.0D0
               cheart_int(p) = 0.0D0
               csto_int(p) = 0.0D0
               cwood_int(p) = 0.0D0
            endif
         endif

         if (cleaf_int(p).lt.0.0D0)  cleaf_int(p) = 0.0D0
         if (croot_int(p).lt.0.0D0)  croot_int(p) = 0.0D0
         if (csap_int(p).lt.0.0D0)   csap_int(p) = 0.0D0
         if (cheart_int(p).lt.0.0D0) cheart_int(p) = 0.0D0
         if (csto_int(p).lt.0.0D0)   csto_int(p) = 0.0D0
         if (cwood_int(p).lt.0.0D0)  cwood_int(p) = 0.0D0

         
               ! print*, 'after allocation and c def inside bdgt',p
               ! print*, 'l', cleaf_int(p)
               ! print*, 'w', cwood_int(p)
               ! print*, 'r', croot_int(p)
               ! print*, 'sap', csap_int(p)
               ! print*, 'h', cheart_int(p)
               ! print*, 'sto', csto_int(p)
               ! print*, 'ph', ph(p)
               ! print*, 'c_def', c_def(p)

      
         !estimate growth of storage pool (acho que isso vai ser dentro da alloc)
         !calculate storage growth respi(onde isso?)

         ! growth_stoc = 0.0D0
         ! mr_sto = 0.0D0
         ! sr = 0.0D0

         
         
         !calculating deltas
         if (dt1(7) .gt. 0.0D0)then
            dleaf_pls_aux(p)  = cleaf_pls2(p)  - cleaf_pls(ri)
            droot_pls_aux(p)  = croot_pls2(p)  - croot_pls(ri)
            dsap_pls_aux(p)   = csap_pls2(p)   - csap_pls(ri)
            dheart_pls_aux(p) = cheart_pls2(p) - cheart_pls(ri)
            dsto_pls_aux(p)   = csto_pls2(p)   - csto_pls(ri)
            dwood_pls_aux(p)  = cwood_pls2(p)  - cwood_pls2(ri)
         else
            dleaf_pls_aux(p)  = cleaf_pls2(p)  - cleaf_pls(ri)
            droot_pls_aux(p)  = croot_pls2(p)  - croot_pls(ri)
            dsap_pls_aux(p)   = 0.0D0
            dheart_pls_aux(p) = 0.0D0
            dsto_pls_aux(p)   = csto_pls2(p)   - csto_pls(ri)
            dwood_pls_aux(p)  = 0.0D0
         endif


         if(dleaf_pls_aux(p).lt.0.0D0)  dleaf_pls_aux(p) = 0.0D0
         if(droot_pls_aux(p).lt.0.0D0)  droot_pls_aux(p) = 0.0D0
         if(dsap_pls_aux(p).lt.0.0D0)   dsap_pls_aux(p) = 0.0D0
         if(dheart_pls_aux(p).lt.0.0D0) dheart_pls_aux(p) = 0.0D0
         if(dsto_pls_aux(p).lt.0.0D0)   dsto_pls_aux(p) = 0.0D0
         if(dwood_pls_aux(p).lt.0.0D0)  dwood_pls_aux(p) = 0.0D0


      enddo
      !$OMP END PARALLEL DO

      epavg = emax

      !Fill output data
      evavg  = 0.0D0
      phavg  = 0.0D0
      aravg  = 0.0D0
      nppavg = 0.0D0
      laiavg = 0.0D0        
      rcavg  = 0.0D0        
      f5avg  = 0.0D0        
      rmavg  = 0.0D0       
      rgavg  = 0.0D0       
      wueavg = 0.0D0       
      cueavg = 0.0D0       
      vcmax_1 = 0.0D0       
      specific_la_1 = 0.0D0
      
      cleaf_out(:)  = 0.0D0
      cwood_out(:)  = 0.0D0
      croot_out(:)  = 0.0D0
      csap_out(:)   = 0.0D0
      cheart_out(:) = 0.0D0
      csto_out(:)   = 0.0D0

      dleaf_out(:)  = 0.0D0
      dwood_out(:)  = 0.0D0
      droot_out(:)  = 0.0D0
      dsap_out(:)   = 0.0D0
      dheart_out(:) = 0.0D0
      dsto_out(:)   = 0.0D0


      cleaf_grd  = 0.0D0
      cwood_grd  = 0.0D0
      croot_grd  = 0.0D0
      csap_grd   = 0.0D0
      cheart_grd = 0.0D0
      csto_grd   = 0.0D0


      ! Calculate CWM for ecosystem processes
 
      ! Filter NaN in ocupation (abundance) coefficients
      do p = 1, nlen
         if(isnan(ocp_coeffs(p))) ocp_coeffs(p) = 0.0D0
      enddo

      evavg         = sum(real(evap, kind=r_8) * ocp_coeffs, mask= .not. isnan(evap))
      phavg         = sum(real(ph, kind=r_8) * ocp_coeffs, mask= .not. isnan(ph))
      aravg         = sum(real(ar, kind=r_8) * ocp_coeffs, mask= .not. isnan(ar))
      nppavg        = sum(real(nppa, kind=r_8) * ocp_coeffs, mask= .not. isnan(nppa))
      laiavg        = sum(laia * ocp_coeffs, mask= .not. isnan(laia))
      rcavg         = sum(real(rc2, kind=r_8) * ocp_coeffs, mask= .not. isnan(rc2))
      f5avg         = sum(f5 * ocp_coeffs, mask= .not. isnan(f5))
      rmavg         = sum(real(rm, kind=r_8) * ocp_coeffs, mask= .not. isnan(rm))
      rgavg         = sum(real(rg, kind=r_8) * ocp_coeffs, mask= .not. isnan(rg))
      wueavg        = sum(real(wue, kind=r_8) * ocp_coeffs, mask= .not. isnan(wue))
      cueavg        = sum(real(cue, kind=r_8) * ocp_coeffs, mask= .not. isnan(cue))
      vcmax_1       = sum(vcmax * ocp_coeffs, mask= .not. isnan(vcmax))
      specific_la_1 = sum(specific_la * ocp_coeffs, mask= .not. isnan(specific_la))

      cleaf_grd  = sum(cleaf_int  * ocp_coeffs, mask = .not. isnan(cleaf_int))
      cwood_grd  = sum(cwood_int * ocp_coeffs, mask = .not. isnan(cwood_int ))
      croot_grd  = sum(croot_int  * ocp_coeffs, mask = .not. isnan(croot_int ))
      csap_grd   = sum(csap_int   * ocp_coeffs, mask = .not. isnan(csap_int  ))
      cheart_grd = sum(cheart_int * ocp_coeffs, mask = .not. isnan(cheart_int))
      csto_grd   = sum(csto_int   * ocp_coeffs, mask = .not. isnan(csto_int  ))

      !daily output to carbon pools (not CWM)
      do p = 1, nlen
         ri = lp(p)
         

         cleaf_out(ri)  =  cleaf_int(p) 
         croot_out(ri)  =  croot_int(p)
         cheart_out(ri) =  cheart_int(p)
         csap_out(ri)   =  csap_int(p)
         csto_out(ri)   =  csto_int(p)
         cwood_out(ri)  =  cheart_out(ri) + csap_out(ri)

      
         !deltas
         dleaf_out(ri)  =  dleaf_pls_aux(p)
         droot_out(ri)  =  droot_pls_aux(p)
         dheart_out(ri) =  dheart_pls_aux(p)
         dsap_out(ri)   =  dsap_pls_aux(p)
         dsto_out(ri)   =  dsto_pls_aux(p)
         dwood_out(ri)  =  dheart_out(ri) + dsap_out(ri)

         
      enddo


      deallocate(lp)
      deallocate(evap)
      deallocate(nppa)
      deallocate(ph)
      deallocate(ar)
      deallocate(laia)
      deallocate(f5)
      deallocate(vpd)
      deallocate(rc2)
      deallocate(rm)
      deallocate(rg)
      deallocate(wue)
      deallocate(cue)
      deallocate(c_def)
      deallocate(vcmax)
      deallocate(specific_la)
      deallocate(litter_l)
      deallocate(cwd)
      deallocate(litter_fr)
      deallocate(tra)
      deallocate(idx_grasses)

      deallocate(cleaf_pls2)
      deallocate(cwood_pls2)
      deallocate(croot_pls2)
      deallocate(csap_pls2)
      deallocate(cheart_pls2)
      deallocate(csto_pls2)

      deallocate(leaf_req)
      deallocate(leaf_inc_min)
      deallocate(root_inc_min)

      deallocate(cleaf_int)
      deallocate(cwood_int)
      deallocate(croot_int)
      deallocate(csap_int)
      deallocate(cheart_int)
      deallocate(csto_int)

      deallocate(dleaf_pls_aux)
      deallocate(dwood_pls_aux)
      deallocate(droot_pls_aux)
      deallocate(dsap_pls_aux)
      deallocate(dheart_pls_aux)
      deallocate(dsto_pls_aux)




   end subroutine daily_budget_allom
 
 end module budget_allom
 