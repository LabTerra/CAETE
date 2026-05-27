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

  public :: prod


contains

  ! [LIGHT COMP] Prod recebe linc_layer, nl_shared e lsize_shared
  ! vindos do pre-loop de budget.f90, onde o dossel compartilhado foi calculado.
    subroutine prod(dt,catm,temp,ts,p0,w,ipar,rh,emax,cl1_prod,&
        & ca1_prod,cf1_prod,beta_leaf,beta_awood,beta_froot,height1,&
        & linc_layer,nl_shared,lsize_shared,wmax,&
        & psisoil,psi50,klmax,krcmax,psixylem,kxylem,knorm,&
        & ph,ar,nppa,laia,f5,vpd,rm,rg,rc,wue,c_defcit,vm_out,sla,e)

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
        real(r_8), intent(in) :: beta_froot
        real(r_8), intent(in) :: wmax
        real(r_8), intent(in) :: psisoil
        !real(r_8), intent(in) :: sla1
        real(r_8), intent(in) :: height1
        ! [LIGHT COMP] Novos argumentos: dossel compartilhado calculado em budget.f90
        integer(i_4), intent(in) :: nl_shared       ! numero de camadas do dossel
        real(r_8),    intent(in) :: lsize_shared     ! tamanho de cada camada (m)
        real(r_8), dimension(nl_shared), intent(in) :: linc_layer ! luz incidente por camada
        ! logical(l_1), intent(in) :: light_limit                !True for no ligth limitation

    !     Output
    !     ------
        real(r_8), intent(out) :: psi50                !xylem water potential when the plant loses 50% of their maximum xylem conductance (MPa)
        real(r_8), intent(out) :: klmax                !Maximum xylem conductivity per unit leaf area (kgm-1s-1MPa-1)
        real(r_8), intent(out) :: krcmax               !Maximum xylem conductance per unit leaf area (molm-2s-1Mpa-1)
        real(r_8), intent(out) :: psixylem             !Xylem water potential (MPa)
        real(r_8), intent(out) :: kxylem                  !Xylem conductance (molm-2s-1MPa-1)
        real(r_8), intent(out) :: knorm                   !Returns normalized xylem conductance (dimensionless)
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
        real(r_8), intent(out) :: c_defcit     ! Carbon deficit gm-2 if it is positive, aresp was greater than npp + sto2(1)
        real(r_8), intent(out) :: e,sla     !sla   !specific leaf area (m2/kg)
        real(r_8), intent(out) :: vm_out


    !     Internal
    !     --------

        real(r_8) :: tleaf,awood            !leaf/wood turnover time (yr)
        real(r_8) :: g1
        real(r_8) :: c4

        real(r_8) :: n2cl
        real(r_8) :: n2cl_resp
        real(r_8) :: n2cw_resp
        real(r_8) :: n2cf_resp
        real(r_8) :: p2cl
        real(r_8) :: wd_allom
        integer(i_4) :: c4_int
        real(r_8) :: jl_out

        real(r_8) :: f1       !Leaf level gross photosynthesis (molCO2/m2/s)
        real(r_8) :: f1a      !auxiliar_f1
        ! [SUN/SHADE FIX] Sun and shade assimilation rates (before and after water stress)
        real(r_8) :: f1a_sun, f1a_shade  ! raw rates from photosynthesis_rate
        real(r_8) :: f1_sun,  f1_shade   ! water-stress-adjusted rates for gross_ph
        real(r_8) :: rc_pot, rc_aux
        real(r_8) :: e_pot

        !Hydraulic parameters
        !real(r_8) :: psi50

    !getting pls parameters


        g1  = dt(1)
        tleaf = dt(3)
        awood = dt(7)
        c4  = dt(9)
        n2cl = dt(10)
        n2cl_resp = n2cl
        n2cw_resp = dt(11)
        n2cf_resp = dt(12)
        p2cl = dt(13)
        wd_allom = dt(19)


        n2cl = n2cl * 1.0D3 ! N in leaf mg g-1
        p2cl = p2cl * 1.0D3 ! P in leaf mg g-1

        c4_int = idnint(c4)

    !     ==============
    !     Photosynthesis
    !     ==============
    ! rate (molCO2/m2/s)
    
        ! [SLA CALCULADO] Calcula sla ANTES de photosynthesis_rate
        ! spec_leaf_area(tleaf) deve ser chamada aqui para que sla
        ! esteja disponivel na chamada de photosynthesis_rate abaixo
        sla = spec_leaf_area(tleaf)
    
        ! [LIGHT COMP] Repassa linc_layer, nl_shared e lsize_shared para
        ! photosynthesis_rate, que usara esses valores em vez de recalcular
        ! o dossel individualmente para cada PLS.
        ! [SUN/SHADE FIX] Also captures f1a_sun and f1a_shade (new outputs).
        
        call photosynthesis_rate(catm,temp,p0,ipar,sla,c4_int,n2cl,&
            & p2cl,cl1_prod,ca1_prod,height1,&
            & linc_layer,nl_shared,lsize_shared,f1a,vm_out,jl_out,&
            & f1a_sun,f1a_shade)
    
        ! VPD
        !========
        vpd = vapor_p_defcit(temp,rh)
    
        !Stomatal resistence
        !===================
        rc_pot = canopy_resistence(vpd, f1a, g1, catm,temp) ! Potential RCM leaf level - s m-1

        !Stomatal resistence
        !===================
        e_pot = transpiration(rc_aux, p0, vpd, 2,temp)
        !print*, 'e_pot:',e_pot

        !==========================
        !  Hydraulic without stress --
        !==========================

        !   P50
        !=========
        psi50 = psi_fifty(wd_allom,ca1_prod)
        !print*,'P50',psi50, 'wd',wd_allom

        ! Klmax
        !=========
        klmax = conductivity_xylemleaf(wd_allom,jl_out* 1e6,ca1_prod)
        !print*,'klmax',klmax

        ! Krcmax
        !=========
        krcmax = conductance_xylemax(klmax,height1,ca1_prod)   
        !print*,'krcmax',krcmax,'klmax',klmax

        ! Psixylem
        !=========
        psixylem = xylem_waterpotential(psisoil,height1,e_pot,krcmax,ca1_prod)
        print*,'psixylem',psixylem, 'psisoil',psisoil

        ! k xylem
        !=========
        kxylem = xylem_conductance(krcmax,psixylem,psi50,ca1_prod)
        !print*,'KXYLEM:',kxylem, 'P50:',psi50, 'KRCMAX:',krcmax

        ! k xylem
        !=========
        knorm = conductance_normalized(krcmax,kxylem,ca1_prod)
        !print*,'knorm:',knorm,'krcmax:',krcmax,'kxylem:',kxylem
    
        !Water stress response modifier (dimensionless)
        !----------------------------------------------
        f5 =  water_stress_modifier(w, cf1_prod, rc_pot, emax, wmax, knorm, ca1_prod)
        print*,'f5',f5
    
    
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
        !print*, 'e_real:',e
    
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
    ! this operation affects the model mass balance
    ! If ar is bigger than ph, what is the source or respired C?
    
        if(ar .gt. ph) then
        c_defcit = ((ar - ph) * 2.73791D0) ! tranform kg m-2 year-1 in  g m-2 day-1
        nppa = 0.0D0
        else
        c_defcit = 0.0D0
        endif
 
    end subroutine prod
end module productivity
