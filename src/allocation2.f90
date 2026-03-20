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
!
! This module was developed to include allometric constraints on carbon distribution
!for plant tissues. It is mainly based on LPJ scheme (Sitch et al. 2003; Smith et al. 2001),
!although some implementations were developed by CAETÊ group.
!We also used LPJ code as reference: LPJ-LMFire.
!You can access a conceptual diagram for this allocation scheme at Bianca Rius' github:
!https://github.com/BiancaRius/CAETE-DVM-alloc-allom

module alloc2

    use types
    use global_par
    
    implicit none
    private

    public:: allocation2,& !calculates carbon pools output from NPP and carbon from preivous step
             height_calc,& !(f)calculates height
             leaf_req_calc,& !(f)leaf mass requeriment to satisfy allometry
             leaf_inc_min_calc,& !(f) minimum leaf increment to satisfy allocation equations
             root_inc_min_calc,& !(f) minimum root increment to satisfy allocation equations
             normal_alloc, & !(f)regular allocation process
             root_bisec_calc,& !solves bisection method for normal allocation
             positive_leaf_inc_min,&
             mortality_turnover,& !(s) accounts for mortality through turnover
             storage_accumulation,& !allocation for storage compartment when normal allocation is not possible
             reallocation,& !use of storage when normal allocation is not possible
             mortality_greff

    contains

    subroutine allocation2(step,ri, p, dt, npp, leaf_in, wood_in, root_in, sap_in, heart_in, sto_in&
        &, leaf_out, wood_out, root_out, sap_out, heart_out, sto_out&
        &, leaf_req, leaf_inc_min, root_inc_min)
    
        
        !VARIABLE INPUTS
        real(r_8), dimension(ntraits),intent(in) :: dt  ! PLS attributes
        integer(i_4), intent(in) :: p,ri,step  ! PLS 


        !carbon inputs (kgC/m2)
        real(r_8), intent(in) :: leaf_in
        real(r_8), intent(in) :: root_in
        real(r_8), intent(in) :: sap_in
        real(r_8), intent(in) :: heart_in
        real(r_8), intent(in) :: sto_in
        real(r_8), intent(in) :: wood_in
        real(r_4), intent(in) :: npp




        !provisory
        real(r_8) :: dens_in

        !input of individuals density (ind/m2)
        ! real(r_8), intent(in) :: dens_in !ind/m2 initial density of individuals

        !input of carbon available gc/m2/time_step
        real(r_4):: bminc_in ! carbon (NPP) available to be allocated
        

        !VARIABLES OUTPUTS 
        !carbon outputs (kgC/m2)
        real(r_8), intent(out) :: leaf_out
        real(r_8), intent(out) :: wood_out
        real(r_8), intent(out) :: root_out
        real(r_8), intent(out) :: sap_out
        real(r_8), intent(out) :: heart_out
        real(r_8), intent(out) :: sto_out

        real(r_8), intent(out) :: leaf_req
        real(r_8), intent(out) :: leaf_inc_min
        real(r_8), intent(out) :: root_inc_min

        !INTERNAL VARIABLES

        !carbon (gC) in compartments considering the density (ind/m2)
        real(r_8) :: leaf_in_ind
        real(r_8) :: root_in_ind
        real(r_8) :: sap_in_ind
        real(r_8) :: heart_in_ind
        real(r_8) :: sto_in_ind
        real(r_8) :: wood_in_ind 

        !carbon available for allocation (gC) considering the density of individuals
        real(r_8) :: bminc_in_ind
        
        !Height
        !dwood !!####****!! ATTENTION: dwood is already transformed to gC/m3 at constants.f90
        real(r_8) :: height
        
        !variables allocation (increase for each compartment)
        real(r_8) :: leaf_inc_alloc
        real(r_8) :: root_inc_alloc
        real(r_8) :: sap_inc_alloc
        real(r_8) :: heart_inc_alloc
        real(r_8) :: sto_inc_alloc
        real(r_8) :: total_inc !sum all the increments (it is the deltaC)

        !C deficit (when NPP < 0)
        real(r_8) :: c_deficit

        !variable update that goes to turnover mortality (compartment_in_ind + compartiment_inc_alloc)
        real(r_8) :: leaf_updt
        real(r_8) :: wood_updt
        real(r_8) :: root_updt
        real(r_8) :: sap_updt
        real(r_8) :: heart_updt
        real(r_8) :: sto_updt

        !amount of C lost with turnover processes
        real(r_8) :: leaf_turn
        real(r_8) :: root_turn
        real(r_8) :: sap_turn
        real(r_8) :: heart_turn
        real(r_8) :: sto_turn

        !used to identify wood/non wood strategies
        real(r_8) :: awood

        !variant trait
        !(SLA - m2/g)
        real(r_8) :: sla_allom
         !(WD - g/cm3)
        real(r_8) :: wd_allom

        !internal
        real(r_8) :: bminc_internal

        !array to store inc vars
        real(r_8), dimension(5) :: incs
        integer(i_4) :: i
        
        !take the allocation proportion to wood (to identify) the grasses
        awood = dt(7)
        sla_allom = dt(18)
        wd_allom = dt(19) !*1.D6

        !initializing variables
        leaf_in_ind  = 0.0D0
        root_in_ind  = 0.0D0
        sap_in_ind   = 0.0D0
        heart_in_ind = 0.0D0
        wood_in_ind  = 0.0D0
        sto_in_ind   = 0.0D0

        leaf_inc_min = 0.0D0
        root_inc_min = 0.0D0

        leaf_out  = 0.0D0
        wood_out  = 0.0D0
        root_out  = 0.0D0
        sap_out   = 0.0D0
        heart_out = 0.0D0
        sto_out   = 0.0D0
 
        height   = 0.0D0
        leaf_req = 0.0D0

        leaf_inc_alloc  = 0.0D0
        root_inc_alloc  = 0.0D0
        sap_inc_alloc   = 0.0D0
        heart_inc_alloc = 0.0D0
        sto_inc_alloc   = 0.0D0

        c_deficit = 0.0D0

        leaf_updt    = 0.0D0
        wood_updt    = 0.0D0
        root_updt    = 0.0D0
        sap_updt     = 0.0D0
        heart_updt   = 0.0D0
        sto_updt     = 0.0D0

        leaf_turn = 0.0D0
        root_turn = 0.0D0
        sap_turn = 0.0D0
        heart_turn = 0.0D0
        sto_turn = 0.0D0

        bminc_internal = 0.0D0

        !wood/non wood strategies !provisory
        if (awood .le. 0.0D0) then
            ! print*, 'sou grama', p
            leaf_out = 0.3
            root_out = 0.2
            sap_out = 0.0
            heart_out = 0.0
            sto_out = 1.
            wood_out = 0.0
        endif     

        !provisory !until there no individuals and the FPC/establishment is not implemented
        dens_in = 1.
        
        !carbon (gC) in compartments considering the density (ind/m2) -- NOT APPLIED IN THIS VERSION
        leaf_in_ind = (leaf_in/dens_in)*1.D3
        root_in_ind = (root_in/dens_in)*1.D3
        sap_in_ind = (sap_in/dens_in)*1.D3 
        heart_in_ind = (heart_in/dens_in)*1.D3
        sto_in_ind = (sto_in/dens_in)*1.D3
        wood_in_ind = sap_in_ind + heart_in_ind


        bminc_in_ind = (npp/dens_in)*1.D3
        ! print*, 'initial bminc', bminc_in_ind
      
        ! call functions to allocation logic
        height = height_calc(wood_in_ind, sap_in_ind, leaf_in_ind, wd_allom)
        ! print*, 'height', height
      
        ! if (height.le.0.0D0) then
        !     print*, 'HEIGHT LE 0', wood_in_ind
        ! endif

        ! !leaf requirement
        leaf_req = leaf_req_calc(sap_in_ind, height, p, sla_allom, wd_allom)
        ! if (p.eq.1460) then
            ! print*, 'leaf_req função  ','s in:', sap_in_ind, leaf_req, p, 'l in:',leaf_in_ind, step
        ! endif
        ! print*,'leaf req id nan', leaf_req, p
            ! !minimum increment to leaf
        leaf_inc_min = leaf_inc_min_calc(leaf_req, leaf_in_ind)    
        
        !minimum increment to root
        root_inc_min = root_inc_min_calc(leaf_req, root_in_ind)
      

    !!conditions for allocation!!! see fluxogram in https://lucid.app/lucidchart/74db0739-29ee-4894-9ecc-42b2cf3d0ae5/edit?invitationId=inv_d3a94efe-b397-45df-9af2-9467d19bee97&page=0_0#
        
       
        if (leaf_inc_min.gt.0.0D0.and.root_inc_min.gt.0.0D0) then
            ! PRINT*, ''
            
            ! print*,'leaf and root inc minimum are > 0' !ok
                           
            if((bminc_in_ind.gt.0)) then
             
                ! print*, 'NPP > 0.0'
                
                if (bminc_in_ind.ge.(root_inc_min + leaf_inc_min)) then
                    ! print*, "ALOCAÇÃO NORMAL1"
                                        
                    ! print*, 'NPP > sum of root and leaf inc min' 
                    ! print*, 'call normal alloc' 

                    call normal_alloc(leaf_inc_min, leaf_in_ind, root_in_ind, bminc_in_ind, sla_allom, p,&
                    wd_allom, sap_in_ind, heart_in_ind, leaf_inc_alloc, root_inc_alloc, sap_inc_alloc, heart_inc_alloc)

                else
                    ! print*, 'NPP < sum of root and leaf inc min' !ok

                    if ((sto_in_ind + bminc_in_ind).ge.(root_inc_min + leaf_inc_min) ) then !!AND NUTRIENTS
                        ! print*, 'NORMAL ALLOC2'        
                        ! print*, 'storage + NPP is enough to normal allocation '                       
                        
                        bminc_internal = sto_in_ind + bminc_in_ind

                        call normal_alloc(leaf_inc_min, leaf_in_ind, root_in_ind, bminc_internal, sla_allom, p,&
                        wd_allom, sap_in_ind, heart_in_ind, leaf_inc_alloc, root_inc_alloc, sap_inc_alloc, heart_inc_alloc)
                        
                        !all the C available in storage is used (it is a type of reallocation)
                        sto_inc_alloc = -leaf_inc_alloc - root_inc_alloc - sap_inc_alloc +  bminc_in_ind
                        ! print*,'sto inc negativo', sto_inc_alloc
                        
                    else
                        ! print*, 'storage + npp < inc min, non used npp goes to storage'
                        
                        sto_inc_alloc = bminc_in_ind 

                    end if 
                end if

            else
                ! print*, 'NPP < 0'         

                if ((sto_in_ind).ge.(root_inc_min + leaf_inc_min) ) then
                    ! print*, 'NORMAL ALLOC3'
                    ! print*, 'NPP < 0 but storage > minimum requirement'

                    call normal_alloc(leaf_inc_min, leaf_in_ind, root_in_ind, sto_in_ind, sla_allom, p,&
                        wd_allom, sap_in_ind, heart_in_ind, leaf_inc_alloc, root_inc_alloc, sap_inc_alloc, heart_inc_alloc)

                    !all the C available in storage is used (it is a type of reallocation)
                    sto_inc_alloc = -leaf_inc_alloc - root_inc_alloc - sap_inc_alloc +  bminc_in_ind
 
                    ! print*,'sto inc negativo', sto_inc_alloc
                else
                    ! print*, 'NPP < 0 and storage < minimum requirement' !there is no allocation
                                        
                    leaf_inc_alloc = 0.0D0 
                    root_inc_alloc = 0.0D0 
                    sap_inc_alloc  = 0.0D0
                    !when sap dies it turns into heartwood
                    heart_inc_alloc = 0.0D0 
                    sto_inc_alloc = 0.0D0 !- (c_deficit*0.55)
                    
                end if
                    
            endif

        else
            ! PRINT*, '' 
            ! print*, "ABNORMAL",p
            ! print*, 'leaf and root inc minimum are < 0' ,p
          
            if (bminc_in_ind .gt. 0.0D0) then
                ! print*, 'NPP > 0',p
                ! print*, 'increment goes to storage',p
                

                leaf_inc_alloc = 0.0D0 
                root_inc_alloc = 0.0D0 
                sap_inc_alloc  = 0.0D0 
                heart_inc_alloc = 0.0D0 
                sto_inc_alloc = bminc_in_ind 
        
            else 
                ! print*, 'NPP < 0 and minimum requirement < 0, there is no allocation',p
                leaf_inc_alloc = 0.0D0 
                root_inc_alloc = 0.0D0 
                sap_inc_alloc  = 0.0D0 
                heart_inc_alloc = 0.0D0 
                sto_inc_alloc = 0.0D0 
            end if               
        endif
        ! print*, 'sap inc after alloc', sap_inc_alloc, p
        ! print*, ''
        ! if (p.eq.1460) then 
            ! print*, 'leaf inc alloc', leaf_inc_alloc, p
            ! print*, 'sap inc alloc', sap_inc_alloc, p
        ! endif
        
        !identifyt vars in array incs
        incs(1) = leaf_inc_alloc
        incs(2) = root_inc_alloc
        incs(3) = sap_inc_alloc
        incs(4) = heart_inc_alloc
        ! incs(5) = sto_inc_alloc

        
        !verify if inc is lt 0 for the compartments except for storage (it can be negative)
        do i = 1, 4
            if (incs(i).lt.0.0D0) then
                incs(i) = 0.0D0
            endif
        enddo

        total_inc = incs(1) + incs(2) + incs(3) + incs(4)

        ! print*, 'TOTAL INC ==== ', total_inc

        
    !     !Update variable and add Increment to C compartments 
        leaf_updt    = leaf_in_ind  + incs(1)
        root_updt    = root_in_ind  + incs(2)
        sap_updt     = sap_in_ind   + incs(3)
        heart_updt   = heart_in_ind + incs(4)
        wood_updt    = sap_updt     + heart_updt

        !here the sto inc alloc do not use the vars "incs" because there is situations where it needs to be negative
        sto_updt     = sto_in_ind   + sto_inc_alloc
        ! if(sto_inc_alloc.lt.0.0D0)then
            ! print*, 'sto updt inc lt 0','updt', sto_updt,'in_ind', sto_in_ind, 'inc',sto_inc_alloc
        ! endif

        ! 
        ! print*, '_______increment_allocation_________', p
        ! print*, 'leaf updt', leaf_updt
        ! print*, 'sap updt', sap_updt
        ! print*, 'heart updt', heart_updt
        ! print*, 'root updt', root_updt
        ! print*, 'sto updt', sto_updt
        ! print*, 'wood total updt', wood_updt
        ! print*, '____________________________________'


    !mortality through turnover
        call mortality_turnover(p, dt, leaf_updt, root_updt, sap_updt, heart_updt,sto_updt,&
            leaf_turn, root_turn, sap_turn, heart_turn, sto_turn)

        !discout C due to turnover and transform variable in kgC/m2 to ouput
        leaf_out = ((leaf_updt - leaf_turn)*dens_in)/1.D3
        ! ! if(p.eq.1460)then
        !     print*, 'leaf out', leaf_out, 'leaf updt', leaf_updt, 'leaf turn', leaf_turn
        ! ! endif 
        root_out = ((root_updt - root_turn)*dens_in)/1.D3

        sap_out  = ((sap_updt - sap_turn)*dens_in)/1.D3

        ! if(p.eq.1460)then
            ! print*, p,'sap out', sap_out, 'sap updt', sap_updt, 'sap turn', sap_turn
            ! print*, ''
        ! endif 

        heart_out = (((heart_updt - heart_turn) + sap_turn)*dens_in)/1.D3
        
        if (sap_out.ge.0.0) then
            heart_out = (((heart_updt - heart_turn) + sap_turn)*dens_in)/1.D3
        else 
            heart_out = (((heart_updt - heart_turn) + abs(sap_out))*dens_in)/1.D3   
        endif
        
        sto_out = ((sto_updt - sto_turn) * dens_in)/1.D3
        wood_out = sap_out + heart_out

    end subroutine allocation2

    function height_calc (wood_in_ind, sap_in_ind, leaf_in_ind, wd_allom) result (height)
        
        real(r_8), intent(in) :: wood_in_ind !gC/ind - total wood (sap + heart) carbon stock
        real(r_8), intent(in) :: sap_in_ind !gC/ind - sapwood
        real(r_8), intent(in) :: leaf_in_ind !gC/ind
        real(r_8), intent(in) :: wd_allom

        real(r_8) :: sap_xsa !parameter to calculate height (comes from the LPJmlfire code)
        real(r_8) :: height !m - output

        !internal variable 
        real(r_8) :: diameter 

        !initializing variables
        diameter = 0.0D0
        height = 0.0D0

        
        !Calculating diameter (necessary to 'nppheight)
        !WD *1D6 transforms from g/cm3 to g/m3
        diameter = ((sap_in_ind)/(wd_allom*1.D6)*pi*k_allom2)**(1/(2+k_allom3))
        !print*, 'diameter', diameter

        !Height 
        height = k_allom2*(diameter**k_allom3)
        ! print*, 'HEIGHT', 'HEIGHT','HEIGHT LPJ', height

    end function height_calc

    function leaf_req_calc (sap_in_ind, height, p, sla_allom, wd_allom)  result (leaf_req)
    
        real(r_8), intent(in) :: sap_in_ind !gC - sapwood input
        real(r_8), intent(in) :: height !me
        real(r_8), intent(in) :: sla_allom !variant trait m2/g
        real(r_8), intent(in) :: wd_allom !variant trait g/cm3

        integer(i_4), intent(in) :: p
       
        real(r_8) :: leaf_req !gC - output- leaf mass requeriment to satisfy allometry
        
        !initializing variables
        leaf_req = 0.0D0
        
        !DWOOD = AQUI TEM QUE SER EM KG/M³ ENTÃO PEGA O VALOR EM G/CM³ E MULTIPLICA POR 1.D3
        if (sap_in_ind.gt.0.0D0) then
            leaf_req = (klatosa * (sap_in_ind/1000.) / ((wd_allom*1.D3) * height * sla_allom*1000))*1000.
        else 
            leaf_req = 0.0D0
        endif
    end function leaf_req_calc

    function leaf_inc_min_calc (leaf_req, leaf_in_ind) result (leaf_inc_min)   

        real(r_8), intent(in) :: leaf_req !gC leaf mass requeriment to satisfy allometry
        real(r_8), intent(in) :: leaf_in_ind !gC leaf input
        
        real(r_8) :: leaf_inc_min !gC -output- minimum leaf increment to satisfy allocation equations

        !initializing variables
        leaf_inc_min = 0.0D0

        if(leaf_req.gt.0.0D0) then
            leaf_inc_min = leaf_req - leaf_in_ind
        else 
            leaf_inc_min = 0.0D0
        endif
        ! print*,'leaf inc min', leaf_inc_min, 'leaf in', leaf_in_ind, 'leaf req', leaf_req


    end function leaf_inc_min_calc

    function root_inc_min_calc (leaf_req, root_in_ind) result (root_inc_min) !ROOT MASS MINIMO
  
        !calculate minimum root production to support this leaf mass (i.e. lm_ind + lminc_ind_min)
        !May be negative following a reduction in soil water limitation (increase in lm2rm) relative to last year
        real(r_8), intent(in) :: root_in_ind !gC root input
        real(r_8), intent(in) :: leaf_req  !gC leaf mass requeriment to satisfy allometry
  
        real(r_8) :: root_inc_min !gC -output- minimum root increment to satisfy allocation equation
        
        root_inc_min = leaf_req / ltor - root_in_ind
        ! print*, 'root inc min', root_inc_min

    end function root_inc_min_calc

    subroutine normal_alloc (leaf_inc_min, leaf_in_ind, root_in_ind, bminc_in_ind, sla_allom, p,&
        wd_allom, sap_in_ind, heart_in_ind, leaf_inc_alloc, root_inc_alloc, sap_inc_alloc, heart_inc_alloc)

        real(r_8), intent(in) :: leaf_inc_min 
        real(r_8), intent(in) :: leaf_in_ind  
        real(r_8), intent(in) :: root_in_ind
        real(r_8), intent(in) :: sap_in_ind  
        real(r_8), intent(in) :: heart_in_ind
        real(r_8), intent(in) :: bminc_in_ind
        real(r_8), intent(in) :: sla_allom
        real(r_8), intent(in) :: wd_allom
        integer(i_4), intent(in) :: p

        real(r_8), intent(out) :: leaf_inc_alloc
        real(r_8), intent(out) :: root_inc_alloc
        real(r_8), intent(out) :: sap_inc_alloc
        real(r_8), intent(out) :: heart_inc_alloc

        real(r_8) :: x1
        real(r_8) :: x2
        real(r_8) :: dx
        real(r_8) :: fx1     


        !initializing variables
        x1 = 0.0D0
        x2 = 0.0D0
        dx = 0.0D0
        leaf_inc_alloc = 0.0D0
        root_inc_alloc = 0.0D0
        sap_inc_alloc = 0.0D0
        heart_inc_alloc = 0.0D0
        fx1 = 0.0D0

        x1 = leaf_inc_min

        x2 = (bminc_in_ind - (leaf_in_ind/ ltor - root_in_ind))/ (1. + 1. / ltor )

        dx = x2 - x1

        if (dx < 0.01) then !0.01 é a precisão da bisection.
            ! print*, '(dx < 0.01)' 

            !there seems to be rare cases where lminc_ind_min (x1) is almost equal to x2. In this case,
            !assume that the leafmass increment is equal to the midpoint between the values and skip 
            !the root finding procedure

            leaf_inc_alloc = x1 + 0.5 * dx

        else 
            ! print*, 'bisection method'
            !Find a root for non-negative lminc_ind, rminc_ind and sminc_ind using Bisection Method (Press et al., 1986, p 346)
            !There should be exactly one solution (no proof presented, but Steve has managed one).

            call positive_leaf_inc_min(leaf_in_ind, sap_in_ind, heart_in_ind,&
            root_in_ind, bminc_in_ind, sla_allom, wd_allom, dx, x1, x2, leaf_inc_alloc)        
            
        endif

        root_inc_alloc = (leaf_in_ind + leaf_inc_alloc) / ltor - root_in_ind
       

        sap_inc_alloc = bminc_in_ind - leaf_inc_alloc - root_inc_alloc
        

        ! if(sap_inc_alloc.lt.0.0D0) then           
        !     heart_inc_alloc = abs(sap_inc_alloc)
        !     sap_inc_alloc = sap_inc_alloc            
        ! endif
        ! print*, 'sap inc alloc', sap_inc_alloc, 'npp', bminc_in_ind,p
        ! print*, 'l', leaf_inc_alloc, 'r', root_inc_alloc, 'h_inc_alloc', heart_inc_alloc, p

    end subroutine normal_alloc

    function root_bisec_calc (leaf_in_ind, sap_in_ind, heart_in_ind, root_in_ind,&
        bminc_in_ind, x, sla_allom, wd_aux) result (fx1)

        real(r_8), intent(in) :: leaf_in_ind 
        real(r_8), intent(in) :: sap_in_ind
        real(r_8), intent(in) :: heart_in_ind 
        real(r_8), intent(in) :: root_in_ind 
        real(r_8), intent(in) :: bminc_in_ind 
        real(r_8), intent(in) :: x
        real(r_8), intent(in) :: sla_allom
        real(r_8), intent(in) :: wd_aux
        
        real(r_8) :: fx1 !output
        
        !internal variables
        real(r_8), parameter :: pi4 = pi/4
        real(r_8), parameter :: a1 = 2./ k_allom3
        real(r_8), parameter :: a2 = 1. + a1 !Essa é a forma correta !!CONFERIR ...ESTÁ DIFERENTE ENTRE NOSSO CÓDIGO E O lpjmlFIRE
        real(r_8), parameter :: a3 = k_allom2**a1
        real(r_8) :: wd_allom !var to transform wd unit
        
        !transforms WD from g/cm3 to g/m3
        wd_allom = wd_aux*1.D6
        
        !initializing variables
        fx1 = 0.0D0

        fx1 = a3 * ((sap_in_ind + bminc_in_ind - x - ((leaf_in_ind + x)/ltor) + root_in_ind + heart_in_ind) / wd_allom)/ pi4 - &
                    ((sap_in_ind + bminc_in_ind - x - ((leaf_in_ind + x)/ ltor) + root_in_ind) / ((leaf_in_ind + x)&
                    * sla_allom * wd_allom / klatosa)) ** a2
       

    end function root_bisec_calc

    subroutine positive_leaf_inc_min (leaf_in_ind, sap_in_ind, heart_in_ind,&
        root_in_ind, bminc_in_ind, sla_allom, wd_allom, dx2, x1_aux, x2_aux, leaf_inc_alloc)

        real(r_8), intent(in) :: leaf_in_ind 
        real(r_8), intent(in) :: sap_in_ind
        real(r_8), intent(in) :: heart_in_ind 
        real(r_8), intent(in) :: root_in_ind 
        real(r_8), intent(in) :: bminc_in_ind 
        real(r_8), intent(in) :: x1_aux, x2_aux 
        real(r_8), intent(in) :: dx2
        real(r_8), intent(in) :: sla_allom
        real(r_8), intent(in) :: wd_allom

        real(r_8), intent(out) :: leaf_inc_alloc

        !internal variable
        real(r_8) :: dx
        real(r_8) :: fx1
        real(r_8) :: fmid
        real(r_8) :: xmid
        real(r_8) :: x1
        real(r_8) :: x2
        real(r_8) :: sign
        real(r_8) :: rtbis


        integer(i_4) :: i

        x1 = x1_aux
        x2 = x2_aux

        ! print*,'bf', x1, x2

        dx = dx2 / real(nseg)
            
        fx1 = root_bisec_calc(leaf_in_ind, sap_in_ind, heart_in_ind,&
            root_in_ind, bminc_in_ind, x1, sla_allom, wd_allom)

        !Find approximate location of leftmost root on the interval (x1,x2).
        !Subdivide (x1,x2) into nseg equal segments seeking change in sign of f(xmid) relative to f(x1).

        fmid = fx1
        xmid = x1

        i = 1
            
        do 
            xmid = xmid + dx
            
            fmid = root_bisec_calc(leaf_in_ind, sap_in_ind, heart_in_ind,&
                root_in_ind, bminc_in_ind, xmid, sla_allom, wd_allom)

            
            
            if (fmid * fx1 .le. 0. .or. xmid .ge. x2) exit  !sign has changed or we are over the upper bound

            if (i > 20) print*, 'first alloc loop flag'
            if (i > 100) stop 'Too many iterations allocmod'
            i = i + 1
      

        end do
        !the interval that brackets zero in f(x) becomes the new bounds for the root search

        x1 = xmid - dx
        x2 = xmid

        !Apply bisection method to find root on the new interval (x1,x2)
        fx1 = root_bisec_calc(leaf_in_ind, sap_in_ind, heart_in_ind,&
        root_in_ind, bminc_in_ind, x1, sla_allom, wd_allom)

        if (fx1.ge.0.) then
            sign = -1
        else
            sign = 1
        endif

        rtbis = x1

        dx = x2 - x1

        !Bisection loop: search iterates on value of xmid until xmid lies within xacc of the root,
        !i.e. until |xmid-x| < xacc where f(x) = 0. the final value of xmid with be the leafmass increment

        i = 1

        do 
            dx   = 0.5 * dx
            xmid = rtbis + dx

            !calculate fmid = f(xmid) [eqn (22)]

            fmid = root_bisec_calc(leaf_in_ind, sap_in_ind, heart_in_ind,&
                root_in_ind, bminc_in_ind, xmid, sla_allom, wd_allom)

            if (fmid * sign .le. 0.) rtbis = xmid

            if (dx .lt. xacc .or. abs(fmid) .le. yacc) exit

            if (i > 20) print*,'second alloc loop flag'
            if (i > 50) stop 'Too many iterations allocmod'
  
            i = i + 1
        end do
        
        !Now rtbis contains numerical solution for lminc_ind given eqn (22)

        leaf_inc_alloc = rtbis

        end subroutine


    subroutine mortality_turnover (p, dt, leaf_in_ind, root_in_ind, sap_in_ind, heart_in_ind,sto_in_ind,&
        leaf_turn, root_turn, sap_turn, heart_turn, sto_turn)
        !ATENÇÃO: 1 ha
        
        real(r_8), dimension(ntraits),intent(in) :: dt  ! PLS attributes
        integer(i_4), intent(in) :: p
        !C in compartments previous the allocation
        real(r_8), intent(in) :: leaf_in_ind
        real(r_8), intent(in) :: sap_in_ind
        real(r_8), intent(in) :: root_in_ind
        real(r_8), intent(in) :: heart_in_ind
        real(r_8), intent(in) :: sto_in_ind

        !gC/m2
        real(r_8), intent(out) :: leaf_turn !amount of C to be lost by turnover
        real(r_8), intent(out) :: root_turn !amount of C to be lost by turnover
        real(r_8), intent(out) :: sap_turn !amount of C to be lost by turnover
        real(r_8), intent(out) :: heart_turn !amount of C to be lost by turnover
        real(r_8), intent(out) :: sto_turn !amount of C to be lost by turnover

        !internal
        real(r_8) :: leaf_turnover !variant leaf turnover (based on Darela's range)
        real(r_8) :: root_turnover !variant root turnover (based on Darela's range)
        real(r_8) :: sap_turnover
        real(r_8) :: heart_turnover

        !!Fixed turnover
        leaf_turn = leaf_in_ind*l_turnover 
        root_turn = root_in_ind*r_turnover
        sap_turn = sap_in_ind*s_turnover

        sto_turn = sto_in_ind*sto_turnover

        !heartwood incorporates the dead tissue from sapwood
        heart_turn = (heart_in_ind*h_turnover) + sap_turn

        ! !Variant turnover
        ! leaf_turnover = dt(3)
        ! root_turnover = dt(5)
        ! heart_turnover = dt(4)
        ! sap_turnover  = dt(20)
        
        ! ! print*,'sap turnover', sap_turnover

        ! leaf_turn = leaf_in_ind/leaf_turnover

        ! root_turn = root_in_ind/root_turnover

        ! sap_turn = sap_in_ind/sap_turnover
        
        ! !heartwood incorporates the dead tissue from sapwood
        ! heart_turn = (heart_in_ind/heart_turnover) + sap_turn
        ! ! heart_turn = (heart_in_ind*h_turnover) + sap_turn

        
        ! sto_turn = sto_in_ind*sto_turnover

       
        
    end subroutine

    subroutine storage_accumulation (bminc_in_ind, sto_inc_alloc)
        !ATENÇÃO: 1 ha
        
        !C in compartments previous the allocation
        real(r_8), intent(in) :: bminc_in_ind !non allocated NPP

        !gC/m2
        real(r_8), intent(out) :: sto_inc_alloc !carbon to storage


        sto_inc_alloc = bminc_in_ind
        
    end subroutine

    subroutine reallocation (bminc_in_ind,leaf_inc_min, root_inc_min,leaf_inc_alloc, &
        root_inc_alloc, sap_inc_alloc,heart_inc_alloc, sto_inc_alloc)
        
        !here for reallocation we sum the C available from NPP and storage to reallocate (only for leaves and fine roots)


        !inputs
        real(r_8), intent(in) :: leaf_inc_min
        real(r_8), intent(in) :: root_inc_min
        real(r_8), intent(in) :: bminc_in_ind

        !outputs
        real(r_8), intent(out) :: leaf_inc_alloc
        real(r_8), intent(out) :: root_inc_alloc
        real(r_8), intent(out) :: sap_inc_alloc
        real(r_8), intent(out) :: heart_inc_alloc
        real(r_8), intent(out) :: sto_inc_alloc

        
        !initialize variables
        leaf_inc_alloc = 0.0D0
        root_inc_alloc = 0.0D0
        sap_inc_alloc = 0.0D0
        heart_inc_alloc = 0.0D0
        sto_inc_alloc = 0.0D0


        leaf_inc_alloc = leaf_inc_min
            

        root_inc_alloc = root_inc_min

        !here if there is a c deficit, the bminc is lower than 0 (negative) -> discount this C from storage
        sto_inc_alloc = bminc_in_ind - (leaf_inc_alloc + root_inc_alloc)
 

        ! print*, 'sto_inc_alloc', sto_inc_alloc
        ! print*, 'bminc_in_ind', bminc_in_ind
        ! print*, 'sum leaf root inc min', leaf_inc_alloc + root_inc_alloc
        
    end subroutine

    subroutine mortality_greff()
    
    end subroutine
    

end module alloc2