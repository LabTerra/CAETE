# create_plsbin.py
import caete_module
import plsgen as pls
import numpy as np

seed = None

a = pls.table_gen(caete_module.global_par.npls, seed = seed)
np.savetxt("pls_ex.txt", a.T)
