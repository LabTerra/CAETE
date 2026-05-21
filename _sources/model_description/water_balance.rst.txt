Water Balance
=============

CAETÊ includes a water-balance sub-model that operates at the grid-cell scale adapted from Oyama & Nobre (2004).

Soil water content and saturation
---------------------------------

CAETÊ separately computes soil water and snow budgets. Given the climate of our study area, we omit the snow budget here (see Oyama & Nobre, 2004). The actual soil water content :math:`W_{y,t}` in grid cell :math:`y` at time step :math:`t` is determined by precipitation (:math:`P_{\mathrm{rec},y}`), evapotranspiration (:math:`E_{\mathrm{vap},i,y}`; see Oyama & Nobre, 2004), and runoff (:math:`R_{\mathrm{off},i,y}`; see Oyama & Nobre, 2004):

.. math::
   :label: eq_water_balance

   \frac{\partial W_{y}}{\partial t}
   \;=\;
   P_{\mathrm{rec},y}
   \;-\;
   E_{\mathrm{vap},i,y}
   \;-\;
   R_{\mathrm{off},i,y}

The degree of soil water saturation in cell :math:`y` is then

.. math::
   :label: eq_wsat

   w_{\mathrm{sat},y}
   \;=\;
   \frac{W_{y}}{W_{\max}}

Soil heterogeneity
------------------

In CAETÊ, the soil profile is partitioned into two layers: a surface layer (0–30 cm) and a deeper layer (30–100 cm). For each layer, water-holding capacity (AWC), field capacity (FC), and the permanent wilting point (WP) are derived from soil texture and organic-matter content (Wieder et al., 2014). This two-layer scheme enables the model to capture edaphic heterogeneity in both water retention and hydraulic transport.

The water-holding capacity quantifies the volume of water retained in soil and accessible to plants. It is defined as the difference between the volumetric water content after gravitational drainage (field capacity) and the water content below which plants can no longer extract moisture (permanent wilting point).

.. math::
   :label: eq_awc

   \mathrm{AWC}_{y} \;=\;\theta_{\mathrm{FC}_{y}}\;-\;\theta_{\mathrm{WP}_{y}}

Field capacity is the volumetric water content retained in soil after free gravitational drainage, typically corresponding to a matric tension of 33 kPa (:math:`\theta_{33}` — model input).

.. math::
   :label: eq_fc

   \theta_{\mathrm{FC}_{y}} \;=\;\theta_{33}

The wilting point is the volumetric water content below which plants cannot extract water, associated with a tension of 1500 kPa (:math:`\theta_{1500}` — model input).

.. math::
   :label: eq_wp

   \theta_{\mathrm{WP}_{y}} \;=\;\theta_{1500}

According to Saxton and Rawls (2006) model, the exponent controls the shape of the soil water retention curve, defining the transition width between saturation and wilting.

.. math::
   :label: eq_lambda

   \lambda \;=\;\frac{1}{B}
   \quad,\quad
   B \;=\;\frac{\ln(1500)\;-\;\ln(33)}{\ln(\theta_{33})\;-\;\ln(\theta_{1500})}

The saturated hydraulic conductivity is an empirical estimate adjusted by the exponent :math:`\lambda`. While the unsaturated hydraulic conductivity defines how hydraulic conductivity decreases as the soil dries, based on :math:`K_{\mathrm{sat}}` and :math:`\lambda`.

.. math::
   :label: eq_ksat

   K_{\mathrm{sat},{y}}
   \;=\;
   1930 \;\times\; \bigl(\theta_{s} - \theta_{33}\bigr)^{\,3 - \lambda}

.. math::
   :label: eq_ktheta

   K(\theta)_{y}
   \;=\;
   K_{\mathrm{sat},{y}}
   \;\times\;
   \Bigl(\tfrac{\theta}{\theta_{s}}\Bigr)^{\,3 + \tfrac{2}{\lambda}}

Finally, the percolation determines the downward water movement or surface runoff based on the hydraulic conductivity

.. math::
   :label: eq_runoff

   R_{\mathrm{off},{y}}
   \;=\;
   \begin{cases}
   K_{\mathrm{sat},{y}}\times24, & \theta \ge \theta_{s}\quad(\text{saturated}),\\[6pt]
   K(\theta)_{y}\times24,       & \theta < \theta_{s}\quad(\text{unsaturated}).
   \end{cases}

Water Stress
------------

To incorporate water limitation into photosynthesis (:math:`GPP_{i,y}`) and to represent the investment trade-off for fine-root traits, we introduce a water‑stress factor :math:`f_{5,i,y}` based on the ratio of potential water supply for transpiration (:math:`L_{i,y}`) to atmospheric demand for transpiration (:math:`D_{i,y}`) (Pavlick et al., 2013):

.. math::
   :label: eq_f5

   f_{5,i,y}
   \;=\;
   1 \;-\;
   \exp\!\Bigl(\tfrac{L_{i,y}}{D_{i,y}}\Bigr)

Here, the supply term :math:`L_{i,y}` is proportional to the fine-root carbon stock (:math:`C_{\mathrm{root},i,y}`), a constant root uptake capacity :math:`c_{\mathrm{sru}} = 0.0005\,\mathrm{mmH_{2}O\,kgC^{-1}\,day^{-1}}`, and the previous day’s soil water saturation :math:`w_{\mathrm{sat},y,t-1}`:

.. math::
   :label: eq_L

   L_{i,y}
   \;=\;
   c_{\mathrm{sru}}
   \;C_{\mathrm{root},i,y}\;
   w_{\mathrm{sat},y,t-1}

Following Gerten et al. (2004), the variable :math:`D_{i,y}` represents the maximum transpiration rate under “unstressed” conditions, i.e., when stomatal opening is not limited by plant water potential:

.. math::
   :label: eq_D

   D_{i,y}
   \;=\;
   \bigl(1 - w_{\mathrm{sat},y}\bigr)\,
   E_{\mathrm{vap\,pot},y}\,
   \frac{\gamma_{m}}
        {1 + \dfrac{g_{m}}{g_{\mathrm{pot},i,y}}}

where:

- :math:`w_{\mathrm{sat},y}` is the soil water saturation fraction (Eq. 49),
- :math:`E_{\mathrm{vap\,pot},y}` is the potential evapotranspiration (see Oyama & Nobre, 2004),
- :math:`\gamma_{m}=1.391` is a dimensionless temperature–evaporation coefficient,
- :math:`g_{m}=3.26` mm s\ :sup:`-1` is the canopy‑scaled stomatal conductance,
- :math:`g_{\mathrm{pot},i,y}` is the potential canopy conductance under no water limitation, defined by the minimum stomatal resistance:

.. math::
   :label: eq_gpot

   g_{\mathrm{pot},i,y}
   \;=\;
   \frac{1}{r_{c\,\min}}

with :math:`r_{c\,\min}=100` s m\ :sup:`-1` as the minimum canopy resistance.