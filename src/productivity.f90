! Copyright 2017- LabTerra

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
! Author: JP Darela
! This program is based on the work of those that gave us the INPE-CPTEC-PVM2 model

module productivity
  implicit none
  private

  public :: prod, prod_allom


contains

  ! [LIGHT COMP] Prod recebe linc_layer, nl_shared e lsize_shared
  ! vindos do pre-loop de budget.f90, onde o dossel compartilhado foi calculado.
    subroutine prod(dt,catm,temp,ts,p0,w,ipar,rh,emax,cl1_prod,&
        & ca1_prod,cf1_prod,beta_leaf,beta_awood,beta_froot,height1,&
        & linc_layer,nl_shared,lsize_shared,wmax,ph,ar,&
        & nppa,laia,f5,vpd,rm,rg,rc,wue,c_defcit,vm_out,sla,e)

        use types
        use global_par
        use photo_par
        use photo
        use water

    !Input
    !-----
        real(r_8),dimension(ntraits),intent(in) :: dt ! PLS data
        real(r_8), intent(in) :: temp, ts                 !Mean monthly temperature (oC)
        real(r_8), intent(in) :: p0                   !Mean surface pressure (hPa)
        real(r_8), intent(in) :: w                    !Soil moisture kg m-2
        real(r_8), intent(in) :: ipar                 !Incident photosynthetic active radiation (w/m2)
        real(r_8), intent(in) :: rh,emax !Relative humidity/MAXIMUM EVAPOTRANSPIRATION
        real(r_8), intent(in) :: catm, cl1_prod, cf1_prod, ca1_prod        !Carbon in plant tissues (kg/m2)
        real(r_8), intent(in) :: beta_leaf            !npp allocation to carbon pools (kg/m2/day)
        real(r_8), intent(in) :: beta_awood
        real(r_8), intent(in) :: beta_froot, wmax
        !real(r_8), intent(in) :: sla1
        real(r_8), intent(in) :: height1
        ! [LIGHT COMP] Novos argumentos: dossel compartilhado calculado em budget.f90
        integer(i_4), intent(in) :: nl_shared       ! numero de camadas do dossel
        real(r_8),    intent(in) :: lsize_shared     ! tamanho de cada camada (m)
        real(r_8), dimension(nl_shared), intent(in) :: linc_layer ! luz incidente por camada
        ! logical(l_1), intent(in) :: light_limit                !True for no ligth limitation

    !     Output
    !     ------
        real(r_8), intent(out) :: ph                   !Canopy gross photosynthesis (kgC/m2/yr)
        real(r_8), intent(out) :: rc                   !Stomatal resistence (not scaled to canopy!) (s/m)
        real(r_8), intent(out) :: laia                 !Leaf area index (m2 leaf/m2 area) 
        real(r_8), intent(out) :: ar                   !Autotrophic respiration (kgC/m2/yr)
        real(r_8), intent(out) :: nppa                 !Net primary productivity (kgC/m2/yr)
        real(r_8), intent(out) :: vpd
        real(r_8), intent(out) :: f5                   !Water stress response modifier (unitless)
        real(r_8), intent(out) :: rm                   !Maintenance respiration (kgC/m2/yr) 
        real(r_8), intent(out) :: rg
        real(r_8), intent(out) :: wue
        real(r_8), intent(out) :: c_defcit     ! Diagnostic only (gC/m2/day): magnitude of ar-ph when ar > ph.
                                                ! nppa itself carries the (possibly negative) sign and is what
                                                ! alloc3.f90 actually uses to pay the deficit from storage.
        real(r_8), intent(out) :: e,sla     !sla   !specific leaf area (m2/kg)
        real(r_8), intent(out) :: vm_out
    !     Internal
    !     --------

        real(r_8) :: awood                  !wood allocation fraction (to identify grasses)
        real(r_8) :: g1
        real(r_8) :: c4

        real(r_8) :: n2cl
        real(r_8) :: n2cl_resp
        real(r_8) :: n2cw_resp
        real(r_8) :: n2cf_resp
        real(r_8) :: p2cl
        integer(i_4) :: c4_int
        real(r_8) :: jl_out

        real(r_8) :: f1       !Leaf level gross photosynthesis (molCO2/m2/s)
        real(r_8) :: f1a      !auxiliar_f1
        ! [SUN/SHADE FIX] Sun and shade assimilation rates (before and after water stress)
        real(r_8) :: f1a_sun, f1a_shade  ! raw rates from photosynthesis_rate
        real(r_8) :: f1_sun,  f1_shade   ! water-stress-adjusted rates for gross_ph
        real(r_8) :: rc_pot, rc_aux

    !getting pls parameters

        g1  = dt(1)
        awood = dt(7)
        c4  = dt(9)
        n2cl = dt(10)
        n2cl_resp = n2cl
        n2cw_resp = dt(11)
        n2cf_resp = dt(12)
        p2cl = dt(13)

        n2cl = n2cl * 1.0D3 ! N in leaf mg g-1
        p2cl = p2cl * 1.0D3 ! P in leaf mg g-1

        c4_int = idnint(c4)

    !     ==============
    !     Photosynthesis
    !     ==============
    ! rate (molCO2/m2/s)
    
        ! SLA from the SLA trait of the PLS (sla_random, dt(18)), converted to m2 gC-1.
        sla = sla_trait(dt(18))  ! m2/gC
    
        ! [LIGHT COMP] Repassa linc_layer, nl_shared e lsize_shared para
        ! photosynthesis_rate, que usara esses valores em vez de recalcular
        ! o dossel individualmente para cada PLS.
        ! [SUN/SHADE FIX] Also captures f1a_sun and f1a_shade (new outputs).
        
        call photosynthesis_rate(catm,temp,p0,ipar,sla,c4_int,n2cl,&
            & p2cl,cl1_prod,ca1_prod,height1,&
            & linc_layer,nl_shared,lsize_shared,f1a,vm_out,jl_out,&
            & f1a_sun,f1a_shade,rh)
    
        ! VPD
        !========
        vpd = vapor_p_defcit(temp,rh)

        !Stomatal resistence
        !===================
        rc_pot = canopy_resistence(vpd, f1a, g1, catm,temp) ! Potential RCM leaf level - s m-1

        !Water stress response modifier (dimensionless)
        !----------------------------------------------
        f5 =  water_stress_modifier(w, cf1_prod, rc_pot, emax, wmax)
    
    !     Photosysthesis minimum and maximum temperature
    !     ----------------------------------------------
    
        if ((temp.ge.-10.0).and.(temp.le.50.0)) then

            f1 = f1a * f5 ! :water stress factor ! Ancient floating-point underflow spring (from CPTEC-PVM2)

            ! [SUN/SHADE FIX] Apply the same water-stress factor to sun and shade rates.
            ! f5 is derived from the canopy-mean f1a, so it applies uniformly to both
            ! fractions (both are within the same atmospheric/soil water environment).
            
            f1_sun   = f1a_sun   * f5
            f1_shade = f1a_shade * f5

        else
            f1 = 0.0D0      !Temperature above/below photosynthesis windown

            ! [SUN/SHADE FIX] Zero both fractions outside the temperature window
            f1_sun   = 0.0D0
            f1_shade = 0.0D0
        endif
    
        rc_aux = canopy_resistence(vpd, f1, g1, catm,temp)  ! RCM leaf level -!s m-1
    
        wue = water_ue(f1, rc_aux, p0, vpd,temp)
    
        !     calcula a transpiração em mm/s
        e = transpiration(rc_aux, p0, vpd, 2,temp)
    
        ! Leaf area index (m2/m2)
        ! recalcula rc e escalona para dossel
        ! laia = 0.2D0 * dexp((2.5D0 * f1)/p25)
        laia = leaf_area_index(cl1_prod, sla)
        ! laia = f_four(0, cl1_prod, sla) + f_four(1, cl1_prod, sla)
        rc = rc_aux * real(laia,kind=r_8) ! RCM -!s m-1 ! CANOPY SCALING --
    
    !     Canopy gross photosynthesis (kgC/m2/yr)
    !     =======================================x
    
        ! [SUN/SHADE FIX] Pass f1_sun and f1_shade separately so gross_ph can
        ! compute A_sun*f4sun + A_shade*f4shade (De Pury & Farquhar 1997, Eq. 24).
        ! Previously: gross_ph(f1, cl1_prod, sla) used a single rate for both fractions.
        ph =  gross_ph(f1_sun, f1_shade, cl1_prod, sla)  ! kg m-2 year-1
    
    !     Autothrophic respiration
    !     ========================
    !     Maintenance respiration (kgC/m2/yr) (based in Ryan 1991)
        rm = m_resp(temp,ts,cl1_prod,cf1_prod,ca1_prod &
            &,n2cl_resp,n2cw_resp,n2cf_resp,awood)
    
    ! c     Growth respiration (KgC/m2/yr)(based in Ryan 1991; Sitch et al.
    ! c     2003; Levis et al. 2004)
        rg = g_resp(beta_leaf,beta_awood, beta_froot,awood)
    
        if (rg.lt.0) then
            rg = 0.0
        endif
    
    !     c Autotrophic (plant) respiration -ar- (kgC/m2/yr)
    !     Respiration minimum and maximum temperature
    !     -------------------------------------------
        if ((temp.ge.-10.0).and.(temp.le.50.0)) then
            ar = rm + rg
        else
            ar = 0.0               !Temperature above/below respiration windown
        endif
    !     Net primary productivity(kgC/m2/yr)
    !     ====================================
        nppa = ph - ar

        ! c_defcit is a diagnostic only (magnitude of the shortfall when ar > ph).
        ! nppa is NOT clamped to zero here: alloc3.f90 (allocation3/grass_allocation3)
        ! consumes negative nppa directly, paying it from labile storage first and
        ! only then from structural pools (starvation rule). Zeroing nppa here would
        ! hide the deficit from that mechanism and force it to be re-applied
        ! downstream, double counting it (see budget_allom.f90).
        if(ar .gt. ph) then
            c_defcit = ((ar - ph) * 2.73791D0) ! tranform kg m-2 year-1 in g m-2 day-1
        else
            c_defcit = 0.0D0
        endif
 
    end subroutine prod

  ! Version of prod used only by the allometric version (budget_allom.f90).
  ! It differs from prod in the carbon-water coupling: canopy resistance (note [N1]
  ! at the end of this file) and ci from the Medlyn conductance (photosynthesis_rate_allom,
  ! note [N4] in funcs.f90); prod is kept unchanged for the classic version (budget.f90).
    subroutine prod_allom(dt,catm,temp,ts,p0,w,ipar,rh,emax,cl1_prod,&
        & ca1_prod,cf1_prod,beta_leaf,beta_awood,beta_froot,height1,&
        & linc_layer,nl_shared,lsize_shared,wmax,ph,ar,&
        & nppa,laia,f5,vpd,rm,rg,rc,wue,c_defcit,vm_out,sla,e)

        use types
        use global_par
        use photo_par
        use photo
        use water

    !Input
    !-----
        real(r_8),dimension(ntraits),intent(in) :: dt ! PLS data
        real(r_8), intent(in) :: temp, ts                 !Mean monthly temperature (oC)
        real(r_8), intent(in) :: p0                   !Mean surface pressure (hPa)
        real(r_8), intent(in) :: w                    !Soil moisture kg m-2
        real(r_8), intent(in) :: ipar                 !Incident photosynthetic active radiation (w/m2)
        real(r_8), intent(in) :: rh,emax !Relative humidity/MAXIMUM EVAPOTRANSPIRATION
        real(r_8), intent(in) :: catm, cl1_prod, cf1_prod, ca1_prod        !Carbon in plant tissues (kg/m2)
        real(r_8), intent(in) :: beta_leaf            !npp allocation to carbon pools (kg/m2/day)
        real(r_8), intent(in) :: beta_awood
        real(r_8), intent(in) :: beta_froot, wmax
        !real(r_8), intent(in) :: sla1
        real(r_8), intent(in) :: height1
        ! [LIGHT COMP] Novos argumentos: dossel compartilhado calculado em budget.f90
        integer(i_4), intent(in) :: nl_shared       ! numero de camadas do dossel
        real(r_8),    intent(in) :: lsize_shared     ! tamanho de cada camada (m)
        real(r_8), dimension(nl_shared), intent(in) :: linc_layer ! luz incidente por camada
        ! logical(l_1), intent(in) :: light_limit                !True for no ligth limitation

    !     Output
    !     ------
        real(r_8), intent(out) :: ph                   !Canopy gross photosynthesis (kgC/m2/yr)
        real(r_8), intent(out) :: rc                   !Stomatal resistence (not scaled to canopy!) (s/m)
        real(r_8), intent(out) :: laia                 !Leaf area index (m2 leaf/m2 area) 
        real(r_8), intent(out) :: ar                   !Autotrophic respiration (kgC/m2/yr)
        real(r_8), intent(out) :: nppa                 !Net primary productivity (kgC/m2/yr)
        real(r_8), intent(out) :: vpd
        real(r_8), intent(out) :: f5                   !Water stress response modifier (unitless)
        real(r_8), intent(out) :: rm                   !Maintenance respiration (kgC/m2/yr) 
        real(r_8), intent(out) :: rg
        real(r_8), intent(out) :: wue
        real(r_8), intent(out) :: c_defcit     ! Diagnostic only (gC/m2/day): magnitude of ar-ph when ar > ph.
                                                ! nppa itself carries the (possibly negative) sign and is what
                                                ! alloc3.f90 actually uses to pay the deficit from storage.
        real(r_8), intent(out) :: e,sla     !sla   !specific leaf area (m2/kg)
        real(r_8), intent(out) :: vm_out
    !     Internal
    !     --------

        real(r_8) :: awood                  !wood allocation fraction (to identify grasses)
        real(r_8) :: g1
        real(r_8) :: c4

        real(r_8) :: n2cl
        real(r_8) :: n2cl_resp
        real(r_8) :: n2cw_resp
        real(r_8) :: n2cf_resp
        real(r_8) :: p2cl
        integer(i_4) :: c4_int
        real(r_8) :: jl_out

        real(r_8) :: f1       !Leaf level gross photosynthesis (molCO2/m2/s)
        real(r_8) :: f1a      !auxiliar_f1
        ! [SUN/SHADE FIX] Sun and shade assimilation rates (before and after water stress)
        real(r_8) :: f1a_sun, f1a_shade  ! raw rates from photosynthesis_rate
        real(r_8) :: f1_sun,  f1_shade   ! water-stress-adjusted rates for gross_ph
        real(r_8) :: rc_pot, rc_aux
        real(r_8) :: rc_sun, rc_shade, gc_canopy  ! canopy resistance (two-big-leaf)
        real(r_8) :: rd_leaf  ! leaf respiration per unit leaf area (molCO2/m2/s), note [N7]

    !getting pls parameters

        g1  = dt(1)
        awood = dt(7)
        c4  = dt(9)
        n2cl = dt(10)
        n2cl_resp = n2cl
        n2cw_resp = dt(11)
        n2cf_resp = dt(12)
        p2cl = dt(13)

        n2cl = n2cl * 1.0D3 ! N in leaf mg g-1
        p2cl = p2cl * 1.0D3 ! P in leaf mg g-1

        c4_int = idnint(c4)

    !     ==============
    !     Photosynthesis
    !     ==============
    ! rate (molCO2/m2/s)

        ! SLA from the SLA trait of the PLS (sla_random, dt(18)), converted to m2 gC-1.
        sla = sla_trait(dt(18))  ! m2/gC

        ! [LIGHT COMP] Repassa linc_layer, nl_shared e lsize_shared para
        ! photosynthesis_rate, que usara esses valores em vez de recalcular
        ! o dossel individualmente para cada PLS.
        ! [SUN/SHADE FIX] Also captures f1a_sun and f1a_shade (new outputs).

        ! VPD (kPa): also sets ci in photosynthesis_rate_allom
        !========
        vpd = vapor_p_defcit(temp,rh)

        ! ci from the Medlyn conductance (Medlyn et al. 2011; note [N4] in funcs.f90)
        call photosynthesis_rate_allom(catm,temp,p0,ipar,sla,c4_int,n2cl,&
            & p2cl,cl1_prod,ca1_prod,height1,&
            & linc_layer,nl_shared,lsize_shared,f1a,vm_out,jl_out,&
            & f1a_sun,f1a_shade,rh,g1,vpd)

        ! Leaf area index (m2/m2)
        laia = leaf_area_index(cl1_prod, sla)

        ! Leaf respiration Rd per unit leaf area (molCO2/m2/s): leaf term of the
        ! maintenance respiration (Ryan 1991, as in m_resp) divided by the LAI.
        ! The Medlyn conductance uses the net assimilation max(A - Rd, 0); GPP and
        ! the respiration of the PLS are unchanged. See note [N7] at the end.
        if (laia .gt. 0.0D0) then
            rd_leaf = leaf_maint_resp(temp, cl1_prod, n2cl_resp) / (0.012D0 * 31557600.0D0) / laia
        else
            rd_leaf = 0.0D0
        endif

        !Potential canopy resistance (no water stress)
        !============================================
        ! Same two-big-leaf scaling as the canopy resistance rc below ([N1]); the
        ! demand of the water stress modifier uses the canopy conductance (Gerten
        ! et al. 2004). See note [N6] in budget_allom.f90.
        gc_canopy = f_four(1, cl1_prod, sla) / canopy_resistence(vpd, max(f1a_sun - rd_leaf, 0.0D0), g1, catm, temp) &
                & + f_four(2, cl1_prod, sla) / canopy_resistence(vpd, max(f1a_shade - rd_leaf, 0.0D0), g1, catm, temp)
        if (gc_canopy .gt. 0.0D0) then
            rc_pot = 1.0D0 / gc_canopy  ! s m-1, canopy
        else
            rc_pot = rcmax
        endif

        !Water stress response modifier (dimensionless)
        !----------------------------------------------
        ! demand: equilibrium evapotranspiration (emax) and canopy conductance
        f5 =  water_stress_modifier_allom(w, cf1_prod, rc_pot, emax, wmax)
    
    !     Photosysthesis minimum and maximum temperature
    !     ----------------------------------------------
    
        if ((temp.ge.-10.0).and.(temp.le.50.0)) then

            f1 = f1a * f5 ! :water stress factor ! Ancient floating-point underflow spring (from CPTEC-PVM2)

            ! [SUN/SHADE FIX] Apply the same water-stress factor to sun and shade rates.
            ! f5 is derived from the canopy-mean f1a, so it applies uniformly to both
            ! fractions (both are within the same atmospheric/soil water environment).
            
            f1_sun   = f1a_sun   * f5
            f1_shade = f1a_shade * f5

        else
            f1 = 0.0D0      !Temperature above/below photosynthesis windown

            ! [SUN/SHADE FIX] Zero both fractions outside the temperature window
            f1_sun   = 0.0D0
            f1_shade = 0.0D0
        endif
    
        ! sunlit leaf, net assimilation (note [N7]): leaf transpiration e and wue (A_net/E)
        rc_aux = canopy_resistence(vpd, max(f1 - rd_leaf, 0.0D0), g1, catm,temp)  ! RCM leaf level -!s m-1

        wue = water_ue(max(f1 - rd_leaf, 0.0D0), rc_aux, p0, vpd,temp)

        !     calcula a transpiração em mm/s
        e = transpiration(rc_aux, p0, vpd, 2,temp)

        ! Canopy resistance (two-big-leaf; De Pury & Farquhar 1997; Leuning et al. 1995):
        ! leaves are parallel conductances, gc = gs_sun*f4sun + gs_shade*f4shade, with the
        ! same sun/shade weights used for GPP in gross_ph. See note [N1] at the end.
        ! gs from the net assimilation of each leaf class (note [N7]).
        rc_sun   = canopy_resistence(vpd, max(f1_sun   - rd_leaf, 0.0D0), g1, catm, temp)  ! s m-1, leaf
        rc_shade = canopy_resistence(vpd, max(f1_shade - rd_leaf, 0.0D0), g1, catm, temp)  ! s m-1, leaf
        gc_canopy = f_four(1, cl1_prod, sla) / rc_sun + f_four(2, cl1_prod, sla) / rc_shade  ! m s-1
        if (gc_canopy .gt. 0.0D0) then
            rc = 1.0D0 / gc_canopy  ! s m-1, canopy
        else
            rc = rcmax              ! no leaves: closed canopy (no transpiration in penman)
        endif
    
    !     Canopy gross photosynthesis (kgC/m2/yr)
    !     =======================================x
    
        ! [SUN/SHADE FIX] Pass f1_sun and f1_shade separately so gross_ph can
        ! compute A_sun*f4sun + A_shade*f4shade (De Pury & Farquhar 1997, Eq. 24).
        ! Previously: gross_ph(f1, cl1_prod, sla) used a single rate for both fractions.
        ph =  gross_ph(f1_sun, f1_shade, cl1_prod, sla)  ! kg m-2 year-1
    
    !     Autothrophic respiration
    !     ========================
    !     Maintenance respiration (kgC/m2/yr) (based in Ryan 1991)
        rm = m_resp(temp,ts,cl1_prod,cf1_prod,ca1_prod &
            &,n2cl_resp,n2cw_resp,n2cf_resp,awood)
    
    ! c     Growth respiration (KgC/m2/yr)(based in Ryan 1991; Sitch et al.
    ! c     2003; Levis et al. 2004)
        rg = g_resp(beta_leaf,beta_awood, beta_froot,awood)
    
        if (rg.lt.0) then
            rg = 0.0
        endif
    
    !     c Autotrophic (plant) respiration -ar- (kgC/m2/yr)
    !     Respiration minimum and maximum temperature
    !     -------------------------------------------
        if ((temp.ge.-10.0).and.(temp.le.50.0)) then
            ar = rm + rg
        else
            ar = 0.0               !Temperature above/below respiration windown
        endif
    !     Net primary productivity(kgC/m2/yr)
    !     ====================================
        nppa = ph - ar

        ! c_defcit is a diagnostic only (magnitude of the shortfall when ar > ph).
        ! nppa is NOT clamped to zero here: alloc3.f90 (allocation3/grass_allocation3)
        ! consumes negative nppa directly, paying it from labile storage first and
        ! only then from structural pools (starvation rule). Zeroing nppa here would
        ! hide the deficit from that mechanism and force it to be re-applied
        ! downstream, double counting it (see budget_allom.f90).
        if(ar .gt. ph) then
            c_defcit = ((ar - ph) * 2.73791D0) ! tranform kg m-2 year-1 in g m-2 day-1
        else
            c_defcit = 0.0D0
        endif
 
    end subroutine prod_allom

end module productivity

! ==============================================================================
! NOTES
! ==============================================================================
!
! [N1] Canopy resistance in prod_allom (two-big-leaf scaling)
!
!   Penman-Monteith treats the canopy as one surface with a bulk (canopy)
!   conductance gc (Monteith & Unsworth 2013). Leaves exchange water vapour in
!   parallel, so their conductances add up; resistances add up only in series.
!   The leaf-to-canopy scaling therefore is
!
!       gc = sum over leaves of gs_leaf        (per unit ground area)
!
!   and NOT rc_canopy = rc_leaf * LAI (series), as in prod. Dividing a single
!   leaf resistance by LAI would also be wrong here: the leaf rate f1 is the rate
!   of the SUNLIT leaves (photosynthesis_rate returns f1ab = f1ab_sun), so it
!   would treat every leaf as sunlit.
!
!   The sun/shade (two-big-leaf) model of De Pury & Farquhar (1997), already used
!   for GPP in gross_ph, gives the conductance in the same way (Leuning et al. 1995;
!   Wang & Leuning 1998): the sunlit and shaded conductances are weighted by their
!   leaf areas; adding conductances without this weighting biases the big-leaf
!   Penman-Monteith transpiration (Leuning et al. 1995).
!
!       gc = gs_sun * f4sun + gs_shade * f4shade
!       gs_i = Medlyn et al. (2011) conductance of leaf class i, from A_i
!       f4sun   = (1 - exp(-kb*LAI)) / kb      (sunlit leaf area, f_four(1))
!       f4shade = (1 - exp(-kd*LAI)) / kd      (shaded leaf area as in gross_ph, f_four(2))
!       rc = 1 / gc
!
!   Since Medlyn's gs is linear in A, this is the same as applying Medlyn to the
!   canopy assimilation A_sun*f4sun + A_shade*f4shade: carbon (gross_ph) and water
!   (rc) are scaled with the same weights.
!
!   Check: Kelliher et al. (1995) found that the maximum canopy conductance is
!   about 3 times the maximum leaf conductance across vegetation types. With the
!   mean conditions of gridcell 186-239 (LAI 6.8), gc / gs_sun = 2.8.
!
!   rc_aux (sunlit leaf) is kept for the leaf transpiration e (output tra) and
!   for wue. The passive nutrient uptake no longer uses e: it uses the canopy
!   transpiration evap(p) (note [N5] in budget_allom.f90).
!
!   The potential canopy resistance rc_pot (no water stress), used by the water
!   stress modifier, is scaled in the same way (note [N6] in budget_allom.f90).
!
! [N7] Net assimilation in the Medlyn conductance
!
!   Medlyn et al. (2011) and the g1 values of Lin et al. (2015) refer to the net
!   leaf assimilation, An = Ag - Rd. photosynthesis_rate_allom returns the gross
!   rate Ag (24 h mean). The leaf respiration of CAETE is inside the maintenance
!   respiration (m_resp, Ryan 1991): its leaf term, per unit leaf area,
!
!       Rd = rml / (0.012 * 31557600) / LAI        (molCO2 m-2 s-1)
!       rml = n2cl * Cleaf * 25 * exp(0.04 T)      (leaf_maint_resp, kgC m-2 yr-1)
!
!   is used, so that Rd is the same respiration already paid in the carbon
!   balance. The conductance of each leaf class uses max(A_i - Rd, 0); when
!   An = 0, gs = g0 (canopy_resistence). Only the conductance (and rc, rc_pot,
!   e and wue = An/E) changes: GPP, ar and nppa are unchanged (no double
!   counting). ci is not affected, since ci/ca = g1/(g1 + sqrt(D)) does not
!   depend on A ([N4] in funcs.f90).
!
!   Bench (gridcell 186-239 mean conditions, g1 = 4): Rd = 0.28 umol m-2 s-1,
!   ~7% of the 24 h mean gross rate of the sunlit leaves; canopy resistance
!   137 -> 149 s m-1, evapotranspiration 3.79 -> 3.72 mm day-1.
!   Limitation: Rd is the same for sunlit and shaded leaves (in De Pury &
!   Farquhar 1997 it follows the Vcmax profile) and is a 24 h mean.
!
! References
!   De Pury, D.G.G. & Farquhar, G.D. (1997). Simple scaling of photosynthesis from
!     leaves to canopies without the errors of big-leaf models. Plant, Cell &
!     Environment, 20, 537-557.
!   Lin, Y.-S. et al. (2015). Optimal stomatal behaviour around the world. Nature
!     Climate Change, 5, 459-464.
!   Ryan, M.G. (1991). Effects of climate change on plant respiration. Ecological
!     Applications, 1, 157-167.
!   Kelliher, F.M., Leuning, R., Raupach, M.R. & Schulze, E.-D. (1995). Maximum
!     conductances for evaporation from global vegetation types. Agricultural and
!     Forest Meteorology, 73, 1-16.
!   Leuning, R., Kelliher, F.M., De Pury, D.G.G. & Schulze, E.-D. (1995). Leaf
!     nitrogen, photosynthesis, conductance and transpiration: scaling from leaves
!     to canopies. Plant, Cell & Environment, 18, 1183-1200.
!   Medlyn, B.E. et al. (2011). Reconciling the optimal and empirical approaches to
!     modelling stomatal conductance. Global Change Biology, 17, 2134-2144.
!   Monteith, J.L. & Unsworth, M.H. (2013). Principles of Environmental Physics,
!     4th ed. Academic Press.
!   Wang, Y.-P. & Leuning, R. (1998). A two-leaf model for canopy conductance,
!     photosynthesis and partitioning of available energy I. Agricultural and
!     Forest Meteorology, 91, 89-111.
! ==============================================================================
