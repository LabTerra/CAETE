Ecosystem Processes
===================

For the next sections, the symbols *i*, *y*, and *z* refer to a PLS, a grid cell, and a plant compartment, respectively.

Productivity
------------

The ecophysiological processes that determine potential primary productivity are calculated for each PLS within a grid cell. Cell-level plant biomass stocks and biogeochemical fluxes are obtained as abundance-weighted community-weighted means (CWMs) of all PLS present.

CAETÊ implements both C3 (Farquhar–von Caemmerer–Berry) and C4 (Chen et al., 1994) photosynthetic pathways. The maximum carboxylation rate of Rubisco depends on leaf nitrogen content (Domingues et al., 2010). These biochemical models respond to mesophyll temperature, atmospheric CO\ :sub:`2` concentration, and photosynthetically active radiation.

Photosynthesis
~~~~~~~~~~~~~~

The photosynthesis equation (:math:`GPP_{i,y}`) and the ones associated with it are based on Farquhar, Caemmerer, & Berry (1980) formulation, which takes into account three limiting factors: Rubisco carboxylation (:math:`J_{c}`), light (:math:`J_{L_i}`), and electron transport (:math:`J_{E}`). We also included water stress limitation (:math:`f_{5,i}`).

.. math::
   :label: eq_gpp

   GPP_{i,y}
   = 0.012 \times 31\,557\,600
     \times f_{1,i,y}^{\,4}
     \times f_{4,i,y}^{\,\text{sun}}
     \times f_{4,i,y}^{\,\text{shade}}
     \times f_{5,i}

where:

- :math:`f_{1,i,y}` is the leaf level gross photosynthesis;
- :math:`f_{4,i,y}^{\text{sun}}` and :math:`f_{4,i,y}^{\text{shade}}` are functions for upscaling the leaf level photosynthesis to the canopy level;
- :math:`f_{5,i}` is the water stress limitation.

In the following equations, :math:`k_n` are constants summarized in (:ref:`tab:all_vars`). The factor :math:`f_{1i,y}` is calculated as the smallest root among the three limiting rates: :math:`J_{C\,i,y}`, :math:`J_{L\,i,y}` and :math:`J_E`:

.. math::
   :label: eq_f1

   f_{1i,y} \;=\; \text{smallest root of}\;\bigl\{\,k_{1}J^{2}\;-\;J\,(J_{P\,i,y}+J_{E})\;+\;J_{P\,i,y}\,J_{E}\;=\;0\bigr\}

where :math:`J_{P\,i,y}` is itself the minimum between :math:`J_{C\,i,y}` and :math:`J_{L\,i,y}`, defined by:

.. math::
   :label: eq_jp

   J_{P\,i,y} \;=\; \text{smallest root of}\;\bigl\{\,k_{2}J_{P\,i,y}^{2}\;-\;J_{P\,i,y}\,(J_{C\,i,y}+J_{L\,i,y})\;+\;J_{C\,i,y}J_{L\,i,y}\;=\;0\bigr\}

The Rubisco‑limited carboxylation rate :math:`J_{C,i,y}` depends on several factors:

- the maximum carboxylation capacity :math:`V_{m,i,y}`,
- the internal CO\ :sub:`2` partial pressure :math:`C_{\text{press},i,y}`,
- the photorespiratory compensation point :math:`\Gamma_{i,y}`,
- and the Michaelis–Menten constants :math:`f_{2i,y}` for CO\ :sub:`2` and :math:`f_{3i,y}` for O\ :sub:`2`.

And is given by:

.. math::
   :label: eq_jc

   J_{C\,i,y} 
   = V_{m,i,y}\,
   \left\{\,C_{\text{press},i,y}
   -\frac{\Gamma_{i,y}}{C_{\text{press},i,y}}
   +f_{2i,y}\right\}
   \Biggl[\,1 + \Bigl(\tfrac{k_{3}}{f_{3i,y}}\Bigr)\Biggr]

The temperature dependencies to :math:`f_{2i,y}` and :math:`f_{3i,y}`:

.. math::
   :label: eq_f2

   f_{2i,y} = k_{12}\,k_{13}^{10\,(T_{y}-k_{11})}

.. math::
   :label: eq_f3

   f_{3i,y} = k_{14}\,k_{15}^{10\,(T_{y}-k_{11})}

The internal CO\ :sub:`2` pressure is adjusted for leaf water deficit :math:`r_{i,y}` as:

.. math::
   :label: eq_cpress

   C_{\text{press},i,y}
   = k_{16}\,\Bigl[\,1 - \bigl(\tfrac{r_{i,y}}{k_{17}}\bigr)\Bigr]\,(CO_{2y}-\Gamma_{i,y}) + \Gamma_{i,y}

The compensation point :math:`\Gamma_{i,y}` also depends on temperature:

.. math::
   :label: eq_gamma

   \Gamma_{i,y}
   = \Bigl(\tfrac{k_{3}}{k_{8}}\Bigr)\;k_{9}^{10\,(T_{y}-k_{11})}

Water stress modifies the effective CO\ :sub:`2` pressure via the leaf‑level moisture deficit :math:`r_{i,y}`, while :math:`C_{a,y}` denotes the ambient CO\ :sub:`2` concentration (input). The deficit is calculated as a fixed proportion of the saturation mixing ratio :math:`r_{\max,i,y}`:

.. math::
   :label: eq_r

   r_{i,y} = -0.315\,r_{\max,i,y}

Here, :math:`r_{\max,i,y}` is itself a function of the leaf vapor pressure :math:`w_{\text{press},i,y}` and the surface pressure :math:`P_{\text{surf},y}`.

Finally, :math:`r_{\max,i,y}` is a function of the partial pressure of water vapor :math:`w_{\text{press},i,y}` and the surface pressure :math:`P_{\text{surf},y}`:

.. math::
   :label: eq_rmax

   r_{\max,i,y} = 0.622\,\frac{w_{\text{press},i,y}}{P_{\text{surf},y}-w_{\text{press},i,y}}

The leaf interior vapor pressure :math:`w_{\text{press},i,y}` is calculated as a function of temperature :math:`T_y`:

.. math::
   :label: eq_wpress

   w_{\text{press},i,y} \;=\; 6.1121 \times \exp\!\Biggl\{\Bigl[\,18.678 \;-\;\frac{T_y}{234.5}\Bigr]\times\Bigl[\;\tfrac{T_y}{257.14 + T_y}\Bigr]\Biggr\}

The light‑limited photosynthetic rate :math:`J_{L,i,y}` depends on the incident photosynthetically active radiation (IPAR\ :sub:`i,y`); here IPAR is taken as 50% of the incoming shortwave radiation (input):

.. math::
   :label: eq_jl

   J_{L,i,y} \;=\; k_{4}\,(1 - k_{5})\,\mathrm{IPAR}_{i,y}

The electron‑transport‑limited rate :math:`J_{E,i,y}` is

.. math::
   :label: eq_je

   J_{E,i,y} \;=\; k_{7}\,V_{m,i,y}

where the Rubisco maximum rate :math:`V_{m,i,y}` itself is temperature‑dependent:

.. math::
   :label: eq_vm

   V_{m,i,y}
   = V_{c\max,i,y}\,\frac{k_{10}^{(T_y - k_{11})}}{k_{18}} \;+\; \exp\!\bigl(k_{19}\,(T_y - k_{20})\bigr)

and :math:`V_{c\max,i,y}` (:ref:`tab:all_vars`) is the maximum Rubisco carboxylation rate, with :math:`T_y` as input temperature.

The canopy‑scaling factor :math:`f_{4,i,y}` is split into sunlit and shaded components, :math:`f_{4,i,y}^{\mathrm{sun}}` and :math:`f_{4,i,y}^{\mathrm{shade}}`. The sunlit portion assumes direct radiation at a 90° incidence, while the shaded portion receives diffuse light at 20°:

.. math::
   :label: eq_f4_sun

   f_{4,i,y}^{\mathrm{sun}}
   = \frac{1 - \exp\bigl(-k_{21}\,\mathrm{LAI}_{\mathrm{sun},i,y}\bigr)}{k_{21}}

.. math::
   :label: eq_f4_shade

   f_{4,i,y}^{\mathrm{shade}}
   = \frac{1 - \exp\bigl(-k_{22}\,\mathrm{LAI}_{\mathrm{shade},i,y}\bigr)}{k_{22}}

Here :math:`\mathrm{LAI}_{i,y}` is partitioned into :math:`\mathrm{LAI}_{\mathrm{sun},i,y}` and :math:`\mathrm{LAI}_{\mathrm{shade},i,y}` according to canopy geometry. Total LAI is

.. math::
   :label: eq_lai

   \mathrm{LAI}_{i,y} = C_{\mathrm{leaf},i,y}\,\mathrm{SLA}_{i}

where :math:`C_{\mathrm{leaf},i,y}` is leaf carbon per unit ground area and :math:`\mathrm{SLA}_{i,y}` is the specific leaf area (a variant functional trait in this version).

Light availability and light competition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In CAETÊ, light competition is represented by a straightforward vertical-stratification scheme in which each PLS captures a different share of incident light. Each PLS within a grid cell is assigned to a vertical layer according to its height. This layer position determines the light available to the PLS, since canopy light attenuation obeys the Lambert–Beer law. Consequently, radiation‑dependent processes, such as photosynthesis and even allometric relationships, are modulated by the local light environment.

The Lambert–Beer law describes how the intensity of a light beam decreases exponentially as it travels through an absorbing medium (Oldham & Parnis, 2017) and is commonly applied in ecology to model light attenuation within forest canopies (Hirose, 2005).

To stratify PLS into discrete vertical layers, the model defines:

(i) ``N_layer``: the total number of layers, computed from the maximum height of each PLS (``H_max``, in meters);
(ii) ``Size_layer``: the uniform length of each layer (in meters) across the entire canopy profile.

.. math::
   :label: eq_nlayer

   \mathrm{N}_{\mathrm{layer}}
   \;=\;\frac{\mathrm{H}_{\mathrm{max}}}{5}

where 5 is a standardized value for all grid cells, used to set the vertical stratification.

.. math::
   :label: eq_sizelayer

   \mathrm{Size}_{\mathrm{layer}}
   \;=\;\frac{\mathrm{H}_{\mathrm{max}}}{\mathrm{N}_{\mathrm{layer}}}

The incident light (photosynthetically active radiation, IPAR; denoted :math:`\mathrm{Light}_{\mathrm{inc}}`) in each layer depends on the light arriving in the previous layer (:math:`\mathrm{Light}_{\mathrm{aval},\,l-1}`) minus the light consumed by the PLSs in that layer (:math:`\mathrm{Light}_{\mathrm{used},\,l}`):

.. math::
   :label: eq_light_inc

   \mathrm{Light}_{\mathrm{inc},\,l}
   \;=\;
   \mathrm{Light}_{\mathrm{aval},\,l-1}
   \;-\;
   \mathrm{Light}_{\mathrm{used},\,l}

In the topmost layer, :math:`\mathrm{Light}_{\mathrm{inc}}` is simply the absolute IPAR (in :math:`\mathrm{J}\,\mathrm{m}^{-2}\,\mathrm{s}^{-1}`).

:math:`\mathrm{Light}_{\mathrm{used}}` is calculated using an extinction coefficient of 0.59, typical for broadleaf forests (Binkley et al., 2013; Zhang et al., 2014), following the Lambert–Beer law (Sitch et al., 2003; Hirose, 2005):

.. math::
   :label: eq_light_used

   \mathrm{Light}_{\mathrm{used},\,l}
   \;=\;
   \mathrm{Light}_{\mathrm{aval},\,l}
   \bigl[\,1 - \exp\bigl(-0.59\,\mathrm{LAI}_{\mathrm{mean\ layer}}\bigr)\bigr]

The amount of light received by each PLS is fed into the photosynthesis functions (:math:`J_{L,i,y}`) mentioned above. All PLS within the same layer receive the same amount of available light.

Respiration
~~~~~~~~~~~

Autotrophic respiration, :math:`R_{a,i,y}`, is partitioned into growth respiration :math:`R_{g,i,y}` and maintenance respiration :math:`R_{m,i,y}` according to Ryan, 1991a,b.

.. math::
   :label: eq_ra

   R_{a,i,y} \;=\; R_{g,i,y} \;+\; R_{m,i,y}

Growth Respiration
~~~~~~~~~~~~~~~~~~

We assume that the metabolic cost of constructing new tissue consumes 25% of the carbon allocated. Denoting the carbon in compartment :math:`z` at time :math:`t` by :math:`C_{z,i,y}^t`, the total growth respiration across the three structural pools (leaves, fine roots, woody tissues) is:

.. math::
   :label: eq_rg

   R_{g,i,y}
   \;=\;
   \sum_{z=1}^{3}
   0.25\,\bigl(C_{z,i,y}^t \;-\; C_{z,i,y}^{t-1}\bigr)

Maintenance Respiration
~~~~~~~~~~~~~~~~~~~~~~~

Maintenance respiration is calculated from the nitrogen and carbon content of each tissue. Although roughly 60% of this respiration supports protein maintenance and replacement, the model imposes nitrogen limitation on maintenance costs according to C and N pools.

.. math::
   :label: eq_rm

   R_{m,i,y}
   \;=\;
   \sum_{z=1}^{3}
   \bigl[n_{c,z}\,C_{z,i,y}^{t}\,\exp\bigl(0.07\,T_{y}\bigr)\bigr]

Here:

- :math:`n_{c,z}` is the N:C ratio of compartment :math:`z` (leaves, sapwood, fine roots).
- :math:`C_{z,i,y}^{t}` is the carbon content of compartment :math:`z` at time :math:`t`.
- :math:`T_{y}` is the mean annual temperature (°C).

Because heartwood is largely non‑respiring, only 5% of aboveground woody carbon (sapwood) contributes to maintenance respiration, representing the metabolically active fraction (Pavlick et al., 2003). Typical N:C ratios used are 0.034 (leaves), 0.003 (sapwood), and 0.034 (fine roots, Levis et al., 2004; Sitch et al., 2003). In tropical forests, fine‑root respiration is driven by soil temperature :math:`T_{\mathrm{soil},y}` rather than air temperature (Oyama and Nobre, 2004).

Stomatal Conductance and Canopy Resistance
------------------------------------------

Stomatal conductance :math:`g_{s,i,y}` and canopy resistance :math:`C_{r,i,y}` couple the carbon assimilation (:math:`GPP_{i,y}`) to the water‑balance submodel:

.. math::
   :label: eq_cr

   C_{r,i,y}
   \;=\;
   \frac{1}{g_{s,i,y}}

.. math::
   :label: eq_gs

   g_{s,i,y}
   \;=\;
   g_{0}
   \;+\;
   1.6\,
   \Bigl(1 + \tfrac{g_{1}}{\sqrt{\mathrm{VPD}_{y}}}\Bigr)\,
   \frac{GPP_{i,y}}{C_{a,y}}

where:

- :math:`g_{0}=0.001` mol m\ :sup:`-2` s\ :sup:`-1` is the residual conductance,
- :math:`g_{1}` is the sensitivity to carbon assimilation (Medlyn et al.,2011) and a variant functional trait (:ref:`tab:params_k`),
- :math:`\mathrm{VPD}_{y}` is the vapor‑pressure deficit at the leaf surface.

The leaf‑surface vapor‑pressure deficit is computed as:

.. math::
   :label: eq_vpd

   \mathrm{VPD}_{y}
   \;=\;
   \frac{E_{\mathrm{vap},i,y}\,h_{y}}{10}

where :math:`h_{y}` is the ambient relative humidity and :math:`E_{\mathrm{vap},i,y}` is the transpiration flux.

Carbon, nitrogen, and phosphorus allocation
-------------------------------------------

In CAETÊ the process of carbon, nitrogen, and phosphorus allocation follows 8 main daily steps:

**1. Initial NPP partitioning**

Daily net primary production (NPP; :math:`\mathrm{g\,C\,m^{-2}\,day^{-1}}`) is divided among leaves (:math:`\ell`), wood (:math:`w`), and fine roots (:math:`r`) according to fixed allocation fractions :math:`\alpha_z` (:math:`\alpha_\ell+\alpha_w+\alpha_r=1`).

The potential carbon input for compartment :math:`z` is

.. math::
   :label: eq_cpot

   C_z^{\mathrm{pot}}
   = \alpha_z \,\mathrm{NPP}.

**2. Stoichiometric nutrient demand**

For each compartment :math:`z`, the required nitrogen and phosphorus are calculated as

.. math::
   :label: eq_np_demand

   N_z^{\mathrm{demand}}
   = \rho_i^{N\!:\!C}\,C_i^{\mathrm{pot}}, 
   \qquad
   P_z^{\mathrm{demand}}
   = \rho_z^{P\!:\!C}\,C_i^{\mathrm{pot}},

where :math:`\rho_i^{N\!:\!C}` and :math:`\rho_z^{P\!:\!C}` are PLS‑specific mass ratios.

**3. Daily nutrient availability**

Define internal reserves :math:`S_{N},S_{P}`, passive uptake :math:`U_{N}^{\mathrm{pas}},U_{P}^{\mathrm{pas}}`,
active uptake :math:`U_{N}^{\mathrm{act}},U_{P}^{\mathrm{act}}`, and biological N fixation :math:`F_{N}`.
Then

.. math::
   :label: eq_ndisp

   N_{\mathrm{disp}}
   = S_{N} + U_{N}^{\mathrm{pas}} + U_{N}^{\mathrm{act}} + F_{N},

.. math::
   :label: eq_pdisp

   P_{\mathrm{disp}}
   = S_{P} + U_{P}^{\mathrm{pas}} + U_{P}^{\mathrm{act}}

**4. N‑ and P‑Limited Carbon Allocation**

The real carbon allocated to compartment :math:`z` is the minimum of its potential and nutrient‑limited values:

.. math::
   :label: eq_creal

   C_{z}^{\mathrm{real}}
   =\min\!\Bigl(
   C_{z}^{\mathrm{pot}},\,
   \frac{N_{\mathrm{disp}}\,\alpha_{z}}{\rho_{z}^{N:C}},\,
   \frac{P_{\mathrm{disp}}\,\alpha_{z}}{\rho_{z}^{P:C}}
   \Bigr)

Total allocated carbon is

.. math::
   :label: eq_calloc

   C^{\mathrm{alloc}}
   =\sum_{z\in\{\ell,w,r\}}C_{z}^{\mathrm{real}}

**5. Carbon reserve dynamics and growth respiration**

Any unallocated NPP, is stored in a non‑structural carbon pool and incurs growth respiration equal to 25% of the carbon added during the time step.

.. math::
   :label: eq_dsc

   \Delta S_{C}
   =\mathrm{NPP} \;-\; C^{\mathrm{alloc}}

This reserve incurs growth respiration at 25% of the influx:

.. math::
   :label: eq_rg_reserve

   R_{\mathrm{growth}}^{\mathrm{reserve}}
   =0.25\,\Delta S_{C}

**6. Updating N and P reserves**

Surplus or deficit nutrients after biomass construction are added to reserves:
:math:`\Delta S_N` and :math:`\Delta S_P`, which may be positive (store) or negative (drawdown)

.. math::
   :label: eq_dsn

   \Delta S_{N}
   = N_{\mathrm{disp}}
    - \sum_{z}\rho_{z}^{N:C}\,C_{z}^{\mathrm{real}},

.. math::
   :label: eq_dsp

   \Delta S_{P}
   = P_{\mathrm{disp}}
    - \sum_{z}\rho_{z}^{P:C}\,C_{z}^{\mathrm{real}}

**7. Senescence, litter production, and resorption**

Leaf turnover exports litter carbon :math:`L_C^\ell` and returns a fraction of its N and P to the plant via resorption;

.. math::
   :label: eq_litter

   L_{C}^{\ell}
   = t_{\ell}\,M_{\ell},\\
   R_{N}^{\ell}
   = f_{\mathrm{res}}\,\bigl(\rho_{\ell}^{N:C}\,L_{C}^{\ell}\bigr),

Where: :math:`M_{\ell}` is leaf biomass,
:math:`t_{\ell}` is turnover rate, and
:math:`f_{\mathrm{res}}` is resorption fraction.

The remainder enters the soil as organic matter:

.. math::
   :label: eq_soil_remainder

   (1 - f_{\mathrm{res}})\,\rho_{\ell}^{N:C}\,L_{C}^{\ell}

**8. Energetic cost of nutrient acquisition**

Carbon costs for active uptake and resorption are computed by functions :math:`g_N, g_P` (active transport) and :math:`h_N, h_P` (resorption).

.. math::
   :label: eq_cost_n

   C_{\mathrm{cost}}^{N}
   = g_{N}\bigl(U_{N}^{\mathrm{act}}\bigr)
    + h_{N}\bigl(R_{N}^{\ell}\bigr)

.. math::
   :label: eq_cost_p

   C_{\mathrm{cost}}^{P}
   = g_{P}\bigl(U_{P}^{\mathrm{act}}\bigr)
    + h_{P}\bigl(R_{P}^{\ell}\bigr)

The total cost is subtracted from the plant’s carbon budget.

.. math::
   :label: eq_cost_total

   C_{\mathrm{cost}}^{\mathrm{total}}
   = C_{\mathrm{cost}}^{N} + C_{\mathrm{cost}}^{P}

Allometry and tree architecture
-------------------------------

Allometry in CAETÊ addresses both how carbon is allocated among leaves, fine roots, and non‑photosynthetic tissues (handled in the carbon‑allocation and residence‑time module; Rius et al. 2023) and the plant’s architectural growth. This approach is based on LPJ model (Sitch et al. 2003)

To PLS height :math:`H` (in meters):

.. math::
   :label: eq_height

   H \;=\; K_{\mathrm{alom2}}\;\;\mathrm{Diam}^{\mathrm{K_alom3}}

where :math:`K_{\mathrm{alom2}}` and :math:`K_{\mathrm{alom3}}` are allometric constants fixed (:ref:`tab:all_vars`), and :math:`\mathrm{Diam}` is the stem diameter in centimeters.

The crown area :math:`Ca` (in m\ :sup:`2`) is given by:

.. math::
   :label: eq_ca

   Ca \;=\; K_{\mathrm{alom1}}\;\mathrm{Diam}^{k_{\mathrm{rp}}}

where :math:`K_{\mathrm{alom1}}` and :math:`k_{\mathrm{rp}}` are allometric constants fixed (:ref:`tab:all_vars`), and :math:`\mathrm{Diam}` again is the stem diameter in centimeters.

Finally, the stem diameter itself (:math:`\mathrm{Diam}`, in cm) is determined from the stem carbon stock :math:`C_{\mathrm{stem}}` and wood density (:math:`\mathrm{WD}`):

.. math::
   :label: eq_diam

   \mathrm{Diam}
   \;=\;
   \frac{4\,C_{\mathrm{wood}}}
    {\bigl(\mathrm{WD}\ \pi\ 40\bigr)^{\tfrac{1}{2}+0.5}}
