Grid-Cell Occupation and Performance
====================================

Functional‑trait values assigned to each PLS determine its ecophysiological behaviour and its interactions with the environment. Because every PLS is a unique trait combination, it embodies a distinct mode of carbon storage and of water and light capture; these traits therefore govern its performance and survival.

During the spin‑up phase the same set of PLSs is initialised in every grid cell, all starting from a bare‑soil condition. At that point each trait combination has an equal probability of establishment. Subsequent differential uptake of carbon, water, and light, driven by trait differences, produces variation in PLS abundance. Once a PLS is excluded from a cell it is removed from later timesteps, so a continuous environmental filter evaluates the suitability of each PLS under the prevailing conditions.

Environmental change can make certain trait combinations more productive, increasing their carbon‑storage capacity and thus their relative abundance (Eq. 43); less suitable combinations are suppressed.

The performance of the mean individual of PLS (:math:`i`) in cell (:math:`y`) is measured by its relative abundance :math:`A_{r,z,y}`, which depends on that PLS’s contribution to the cell’s total carbon stock :math:`C_{T,y}` and on the number of live PLSs (:math:`S`) at the timestep considered (:math:`T`). Accordingly, each grid cell is represented as a mosaic of PLSs whose areal occupancy is proportional to their percentage abundance.

.. math::
   :label: eq_ar

   A_{r,i,y} \;=\; \frac{C_{i,y}}{C_{T,y}}

.. math::
   :label: eq_ct

   C_{T,y} \;=\;\sum_{i=1}^{S} C_{i,y}

where :math:`C_{i,y}` is the total carbon stock of the :math:`i`\ th PLS in grid cell :math:`y`, itself the sum of carbon stored in each plant compartment (:math:`C_{z,i,y}`).

.. math::
   :label: eq_ci

   C_{i,y} \;=\;\sum_{z=1}^{5} C_{z,i,y}

The persistence of a PLS in a given grid cell requires that both its leaf and fine‑root compartments maintain more than 1 kg C m\ :sup:`-2` (to support photosynthesis). If either of these two structural pools falls below this threshold, the PLS cannot survive.

Ecosystem-scale processes and properties
----------------------------------------

Scaling biogeochemical fluxes from individual PLSs to the ecosystem level in CAETÊ follows the **biomass‑ratio hypothesis** (Grime, 1998). This hypothesis states that the immediate effect of a species’ functional traits on ecosystem functioning is proportional to its contribution to total community biomass. Both theoretical considerations and empirical evidence support the idea that a plant’s influence on ecosystem processes can be predicted from its biomass share.

Accordingly, CAETÊ simulates the properties and processes of each PLS (e.g., potential NPP) independently, without explicit interspecific competition. The grid‑cell value for any flux or pool is then obtained by weighting each PLS’s potential value by its relative abundance and summing across all PLSs. For net primary production in grid cell :math:`y`,

.. math::
   :label: eq_npp_cell

   \mathrm{NPP}_{y}
   = \sum_{z=1}^{S}
   \bigl(\mathrm{NPP}_{z}^{\text{pot}} \; A_{r,z,y}\bigr)

where :math:`S` is the number of extant PLSs in the cell and :math:`A_{r,z,y}` is the relative abundance of PLS :math:`z`.

Functional composition in a grid cell
-------------------------------------

At each time step (:math:`t`), the functional composition of grid cell (:math:`y`) is characterised by the **community‑weighted mean** (CWM) of every trait treated as variable in the simulation. For a given trait (:math:`F`), its grid‑cell value :math:`F_{y,t}` is

.. math::
   :label: eq_cwm

   F_{y,t} \;=\; \sum_{i=1}^{S} F_{i,y,t}\,A_{r,i,y,t},

where :math:`F_{i,y,t}` is trait value of PLS (:math:`i`) that is alive in cell (:math:`y`) at time (:math:`t`); :math:`A_{r,i,y,t}` is relative abundance of that PLS; and :math:`S` = number of extant PLSs in the community.

This CWM can be interpreted as the dominant trait expression within the community at the given time (Díaz et al., 2007), allowing temporal changes in functional composition to be tracked succinctly.
