Spin-Up
=======

The spin-up is executed to achieve a stable dynamic equilibrium and define the initial values of the plant carbon stocks and the PLS to be simulated. This initialization assigns each variable its first time-step value, estimated during the preparation of the initial simulation condition. Before running the model with input data, all C, N, and P pools in both soil and vegetation must be in dynamic equilibrium. These values form the initial boundary conditions.

Starting from a potential NPP value (:math:`1\,\mathrm{kg\,C\,m^{-2}\,year^{-1}}`), the spin-up allocates this carbon according to allocation fractions and residence times defined for each PLS. The process is iterative: In each cycle, the C stocks in leaves, fine roots, and woody tissues are recalculated until the balance between carbon input and output stabilizes. Before the spin-up proper, the average vegetation-to-soil C, N, and P fluxes are estimated through a three-year simulation (1979–1982) for all PLS. Then, the carbon decomposition model is applied iteratively to these average fluxes until the soil stocks reach dynamic equilibrium. Once stocks are balanced, the model runs 35 ten-year cycles using climatic conditions from 1979 to 1989 (a total of 350 years), maintaining a constant CO\ :sub:`2` concentration of :math:`342\,\mu\mathrm{mol}\,\mathrm{mol}^{-1}` (average value in the early 1980s).

Upon completion, final boundary conditions, soil and vegetation C, N, P, and water stocks, as well as initial PLS composition, are obtained, ready for subsequent simulations.
