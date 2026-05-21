Tables
======

Definitions, units, and equation tags for all variables described in CAETÊ (Eqs. 1-63)
--------------------------------------------------------------------------------------

.. list-table:: Definitions, units, and equation tags for all variables described in CAETÊ (Eqs. 1-63)
   :header-rows: 1
   :widths: 15 40 25 20
   :name: tab:all_vars
   :class: wrap-table

   * - **Symbol**
     - **Description**
     - **Unit**
     - **Equation**
   * - :math:`GPP_{i,y}`
     - Gross primary productivity
     - kg C m\ :sup:`-2` yr\ :sup:`-1`
     - :eq:`eq_gpp`
   * - :math:`f_{1,i,y}`
     - Rubisco–light–electron limitation factor
     - unitless
     - :eq:`eq_f1`
   * - :math:`J_{P,i,y}`
     - Minimum of :math:`J_{C,i,y}` and :math:`J_{L,i,y}`
     - mol CO\ :sub:`2` m\ :sup:`-2` s\ :sup:`-1`
     - :eq:`eq_jp`
   * - :math:`J_{C,i,y}`
     - Rubisco-limited carboxylation rate
     - mol CO\ :sub:`2` m\ :sup:`-2` s\ :sup:`-1`
     - :eq:`eq_jc`
   * - :math:`f_{2,i,y}`
     - Michaelis–Menten constant for CO\ :sub:`2`
     - Pa
     - :eq:`eq_f2`
   * - :math:`f_{3,i,y}`
     - Michaelis–Menten constant for O\ :sub:`2`
     - Pa
     - :eq:`eq_f3`
   * - :math:`C_{\text{press},i,y}`
     - Internal CO\ :sub:`2` partial pressure
     - Pa
     - :eq:`eq_cpress`
   * - :math:`\Gamma_{i,y}`
     - Photorespiration compensation point
     - Pa
     - :eq:`eq_gamma`
   * - :math:`r_{i,y}`
     - Leaf-level moisture deficit
     - g kg\ :sup:`-1`
     - :eq:`eq_r`
   * - :math:`r_{\max,i,y}`
     - Saturated mixing ratio
     - unitless
     - :eq:`eq_rmax`
   * - :math:`w_{\text{press},i,y}`
     - Leaf-interior vapor pressure
     - hPa
     - :eq:`eq_wpress`
   * - :math:`J_{L,i,y}`
     - Light-limited photosynthetic rate
     - mol CO\ :sub:`2` m\ :sup:`-2` s\ :sup:`-1`
     - :eq:`eq_jl`
   * - :math:`J_{E,i,y}`
     - Electron-transport-limited rate
     - mol CO\ :sub:`2` m\ :sup:`-2` s\ :sup:`-1`
     - :eq:`eq_je`
   * - :math:`V_{m,i,y}`
     - Temperature-dependent max. carboxylation rate
     - mol CO\ :sub:`2` m\ :sup:`-2` s\ :sup:`-1`
     - :eq:`eq_vm`
   * - :math:`f_{4,i,y}^{\text{sun}}`
     - Canopy upscaling (sunlit fraction)
     - unitless
     - :eq:`eq_f4_sun`
   * - :math:`f_{4,i,y}^{\text{shade}}`
     - Canopy upscaling (shaded fraction)
     - unitless
     - :eq:`eq_f4_shade`
   * - :math:`\text{LAI}_{i,y}`
     - Leaf area index
     - unitless
     - :eq:`eq_lai`
   * - :math:`N_{\text{layer}}`
     - Number of canopy layers
     - unitless
     - :eq:`eq_nlayer`
   * - :math:`\text{Size}_{\text{layer}}`
     - Layer length
     - m
     - :eq:`eq_sizelayer`
   * - :math:`\text{Light}_{\text{INC},l}`
     - Incoming light in layer :math:`l`
     - J m\ :sup:`-2` s\ :sup:`-1`
     - :eq:`eq_light_inc`
   * - :math:`\text{Light}_{\text{USED},l}`
     - Light used by layer :math:`l`
     - J m\ :sup:`-2` s\ :sup:`-1`
     - :eq:`eq_light_used`
   * - :math:`R_{a,i,y}`
     - Autotrophic respiration
     - kg C m\ :sup:`-2` yr\ :sup:`-1`
     - :eq:`eq_ra`
   * - :math:`R_{g,i,y}`
     - Growth respiration
     - kg C m\ :sup:`-2` yr\ :sup:`-1`
     - :eq:`eq_rg`
   * - :math:`R_{m,i,y}`
     - Maintenance respiration
     - kg C m\ :sup:`-2` yr\ :sup:`-1`
     - :eq:`eq_rm`
   * - :math:`C_{r,i,y}`
     - Canopy resistance
     - s m\ :sup:`-1`
     - :eq:`eq_cr`
   * - :math:`g_{s,i,y}`
     - Stomatal conductance
     - mol CO\ :sub:`2` m\ :sup:`-2` s\ :sup:`-1`
     - :eq:`eq_gs`
   * - :math:`\text{VPD}_{y}`
     - Vapor-pressure deficit
     - kPa
     - :eq:`eq_vpd`
   * - :math:`\text{NPP}`
     - Daily net primary production
     - g C m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_cpot`, :eq:`eq_dsc`
   * - :math:`\alpha_{\ell},\alpha_{w},\alpha_{r}`
     - Allocation fractions to leaves, wood, fine roots
     - unitless
     - :eq:`eq_cpot`
   * - :math:`C_{i}^{\text{pot}}`
     - Potential carbon input into compartment :math:`i`
     - g C m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_cpot`
   * - :math:`\rho_{i}^{N:C},\rho_{i}^{P:C}`
     - N:C and P:C stoichiometric ratios
     - g N / g C; g P / g C
     - :eq:`eq_np_demand`
   * - :math:`N_{i}^{\text{req}}`
     - Nitrogen demand of compartment :math:`i`
     - g N m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_np_demand`
   * - :math:`P_{i}^{\text{req}}`
     - Phosphorus demand of compartment :math:`i`
     - g P m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_np_demand`
   * - :math:`S_{N},S_{P}`
     - Internal N and P reserve pools
     - g N m\ :sup:`-2`; g P m\ :sup:`-2`
     - :eq:`eq_ndisp`, :eq:`eq_pdisp`
   * - :math:`U_{N}^{\text{pas}},U_{P}^{\text{pas}}`
     - Passive nutrient uptake
     - g nutrient m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_ndisp`, :eq:`eq_pdisp`
   * - :math:`U_{N}^{\text{act}},U_{P}^{\text{act}}`
     - Active nutrient uptake
     - g nutrient m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_ndisp`, :eq:`eq_pdisp`
   * - :math:`F_{N}`
     - Biological nitrogen fixation
     - g N m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_ndisp`
   * - :math:`N_{\text{disp}}`
     - Total available nitrogen
     - g N m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_ndisp`
   * - :math:`P_{\text{disp}}`
     - Total available phosphorus
     - g P m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_pdisp`
   * - :math:`C_{i}^{\text{real}}`
     - Realized carbon allocation
     - g C m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_creal`
   * - :math:`C^{\text{alloc}}`
     - Total allocated carbon
     - g C m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_calloc`
   * - :math:`\Delta S_{C}`
     - Change in carbon reserve
     - g C m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_dsc`
   * - :math:`R_{\text{growth}}^{\text{reserve}}`
     - Growth respiration on reserves
     - g C m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_rg_reserve`
   * - :math:`\Delta S_{N}`
     - Change in nitrogen reserve
     - g N m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_dsn`
   * - :math:`\Delta S_{P}`
     - Change in phosphorus reserve
     - g P m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_dsp`
   * - :math:`M_{\ell}`
     - Leaf biomass
     - g C m\ :sup:`-2`
     - :eq:`eq_litter`
   * - :math:`t_{\ell}`
     - Leaf turnover rate
     - day\ :sup:`-1`
     - :eq:`eq_litter`
   * - :math:`f_{\text{res}}`
     - Resorption fraction
     - unitless
     - :eq:`eq_litter`, :eq:`eq_soil_remainder`
   * - :math:`L_{C}^{\ell}`
     - Litter C flux from leaves
     - g C m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_litter`
   * - :math:`R_{N}^{\ell}`
     - N resorbed from leaves
     - g N m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_litter`
   * - :math:`C_{\text{cost}}^{N}`
     - Carbon cost of nitrogen acquisition
     - g C m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_cost_n`
   * - :math:`C_{\text{cost}}^{P}`
     - Carbon cost of phosphorus acquisition
     - g C m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_cost_p`
   * - :math:`g_{N},g_{P}`
     - Active-uptake cost functions
     - g C / nutrient
     - :eq:`eq_cost_n`, :eq:`eq_cost_p`
   * - :math:`h_{N},h_{P}`
     - Resorption cost functions
     - g C / nutrient
     - :eq:`eq_cost_n`, :eq:`eq_cost_p`
   * - :math:`C_{\text{cost}}^{\text{total}}`
     - Total carbon acquisition cost
     - g C m\ :sup:`-2` day\ :sup:`-1`
     - :eq:`eq_cost_total`
   * - :math:`H`
     - Plant height
     - m
     - :eq:`eq_height`
   * - :math:`Ca`
     - Crown area
     - m\ :sup:`2`
     - :eq:`eq_ca`
   * - :math:`\text{Diam}`
     - Stem diameter
     - cm
     - :eq:`eq_diam`
   * - :math:`W_{y,t}`
     - Soil water content
     - mm
     - :eq:`eq_ar`
   * - :math:`P_{\text{rec},y}`
     - Precipitation
     - mm day\ :sup:`-1`
     - :eq:`eq_ar`
   * - :math:`E_{\text{vap},i,y}`
     - Evapotranspiration
     - mm day\ :sup:`-1`
     - :eq:`eq_height`, :eq:`eq_wsat`
   * - :math:`R_{\text{off},i,y}`
     - Runoff
     - mm day\ :sup:`-1`
     - :eq:`eq_lambda`
   * - :math:`w_{\text{sat},y}`
     - Soil water saturation fraction
     - unitless
     - :eq:`eq_ca`
   * - :math:`\text{AWC}_{y}`
     - Available water capacity
     - m\ :sup:`3` m\ :sup:`-3`
     - :eq:`eq_awc`
   * - :math:`\theta_{\text{FC},y}`
     - Field capacity water content
     - m\ :sup:`3` m\ :sup:`-3`
     - :eq:`eq_fc`
   * - :math:`\theta_{\text{WP},y}`
     - Wilting point water content
     - m\ :sup:`3` m\ :sup:`-3`
     - :eq:`eq_wp`
   * - :math:`K_{\text{sat},y}`
     - Saturated hydraulic conductivity
     - mm h\ :sup:`-1`
     - :eq:`eq_ksat`
   * - :math:`K(\theta)_{y}`
     - Unsaturated hydraulic conductivity
     - mm h\ :sup:`-1`
     - :eq:`eq_ktheta`
   * - :math:`f_{5,i,y}`
     - Water-stress factor
     - unitless
     - :eq:`eq_f5`
   * - :math:`L_{i,y}`
     - Potential water supply for transpiration
     - mm day\ :sup:`-1`
     - :eq:`eq_L`
   * - :math:`D_{i,y}`
     - Unstressed transpiration demand
     - mm day\ :sup:`-1`
     - :eq:`eq_D`
   * - :math:`\gamma_{m}`
     - Temperature–evaporation coefficient
     - unitless
     - :eq:`eq_D`
   * - :math:`g_{m}`
     - Canopy-scaled stomatal conductance
     - mm s\ :sup:`-1`
     - :eq:`eq_D`
   * - :math:`g_{\text{pot},i,y}`
     - Potential canopy conductance
     - m s\ :sup:`-1`
     - :eq:`eq_gpot`
   * - :math:`r_{c,\min}`
     - Minimum canopy resistance
     - s m\ :sup:`-1`
     - :eq:`eq_gpot`


Trait definitions and ranges in CAETÊ
-------------------------------------

.. list-table:: Trait definitions and ranges in CAETÊ
   :header-rows: 1
   :widths: 15 55 30
   :name: tab:traits
   :class: wrap-table

   * - **Trait**
     - **Description**
     - **Range / Unit**
   * - `aleaf`
     - Allocation coefficient for leaves
     - 0–1 %
   * - `awood`
     - Allocation coefficient for wood
     - 0–1 %
   * - `aroot`
     - Allocation coefficient for fine roots
     - 0–1 %
   * - `leaf_N:C`
     - Nitrogen‑to‑carbon ratio in leaves
     - 0.001–0.05 g N / g C
   * - `wood_N:C`
     - Nitrogen‑to‑carbon ratio in wood
     - 0.001–0.01 g N / g C
   * - `froot_N:C`
     - Nitrogen‑to‑carbon ratio in fine roots
     - 0.001–0.06 g N / g C
   * - `leaf_P:C`
     - Phosphorus‑to‑carbon ratio in leaves
     - 0.0002–0.0095 g P / g C
   * - `wood_P:C`
     - Phosphorus‑to‑carbon ratio in wood
     - 3.12e-5–0.0035 g P / g C
   * - `froot_P:C`
     - Phosphorus‑to‑carbon ratio in fine roots
     - 0.0003–0.005 g P / g C
   * - :math:`g_{1}`
     - Stomatal conductance sensitivity to CO\ :sub:`2`
     - 0.1–19 kPa\ :sup:`0.5`
   * - `amp`
     - Fraction of NPP allocated to diazotrophs
     - 0.001–0.999 % NPP
   * - `WD`
     - Wood density
     - 0.5–1.0 g cm\ :sup:`-3`
   * - `pdia`
     - Fraction of daily NPP to biological N fixation
     - 0.01–0.10 % NPP
   * - `SLA`
     - Specific leaf area
     - 9–40 m\ :sup:`2` kg\ :sup:`-1` C
   * - `tleaf`
     - Leaf turnover time
     - 0.0833–4 years
   * - `twood`
     - Wood turnover time
     - 0.0833–80 years
   * - `troot`
     - Fine‑root turnover time
     - 0.0833–4 years
   * - `rsop_frac`
     - N and P resorption fraction during senescence
     - 0.2–0.7 %


Ecophysiological trade‑offs of functional traits in CAETÊ
---------------------------------------------------------

.. list-table:: Ecophysiological trade‑offs of functional traits in CAETÊ
   :header-rows: 1
   :widths: 20 80
   :name: tab:traits_tradeoffs
   :class: wrap-table

   * - **Trait**
     - **Ecophysiological trade‑offs**
   * - `aleaf`
     - Total plant carbon stock; Leaf Area Index; Growth and Maintenance respiration
   * - `awood`
     - Total plant carbon stock; Light Capture; Growth and Maintenance respiration
   * - `aroot`
     - Total plant carbon; Water stress; Growth and Maintenance respiration
   * - `leaf_N:C`
     - Higher leaf N boosts photosynthesis but raises respiration cost
   * - `wood_N:C`
     - Same trade‑off as leaf_N:C
   * - `froot_N:C`
     - Same trade‑off as leaf_N:C
   * - `leaf_P:C`
     - P demand and allocation to foliage
   * - `wood_P:C`
     - P demand and allocation to stem
   * - `froot_P:C`
     - P demand and allocation to roots
   * - :math:`g_{1}`
     - Balances carbon gain vs. water loss
   * - `amp`
     - Symbiosis trades C cost vs. N fixation benefit
   * - `WD`
     - Allocation, growth and turnover
   * - `pdia`
     - Fixed N reduces limitation; Carbon cost
   * - `SLA`
     - Leaf Area Index; Light capture efficiency
   * - `tleaf`
     - Mass balance between leaf production and loss
   * - `twood`
     - Woody carbon content; Total plant carbon stock
   * - `troot`
     - Nutrient uptake capacity; Water capacity
   * - `rsop_frac`
     - Efficiency of nutrient recovery vs. litter input