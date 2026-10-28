# -*-coding:utf-8-*-
"""
Single gridcell test of the allometric version (run_caete_allom) with the
nutrient cycle - gridcell 185-240 (K34 / Manaus), ISIMIP input in ../k34.

USAGE (from src/, after compiling caete_module):

    printf 'c\\n1\\n' | python tests/allom_nutri/allom_nutri_test.py          # run and compare with the reference
    printf 'c\\n1\\n' | python tests/allom_nutri/allom_nutri_test.py --save   # run and (re)write the reference

(the printf answers the two questions caete.py asks when it is imported)

The script runs the same sequence twice from the same initial state:
    off: carbon only                 (run_caete_allom, nutri_cycle=False)
    on : nutrient cycle (N, P)       (run_caete_allom, nutri_cycle=True)

and prints/saves the mean of the last 365 days of each run. The reference
(allom_nutri_test_reference.csv) and the PLS table used to produce it
(allom_nutri_test_pls.csv) are stored in this folder. The number of PLSs is the one
caete_module was compiled with: the reference is only comparable if npls and
the PLS table are the same.
"""
import os
import sys
import bz2
import copy
import _pickle as pkl

import numpy as np
import pandas as pd
from joblib import load

# the model runs from src/ (relative paths to ../input, ../k34 and ../outputs)
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(SRC)
sys.path.insert(0, SRC)

import caete
from caete import grd

PLS_TABLE = os.path.join(HERE, "allom_nutri_test_pls.csv")
REFERENCE = os.path.join(HERE, "allom_nutri_test_reference.csv")

# Soil and hydraulics (same files used in model_driver.py)
tsoil = tuple(np.load(f"../input/soil/{n}.npy") for n in ("ws", "fc", "wp"))
ssoil = tuple(np.load(f"../input/soil/{n}.npy") for n in ("sws", "sfc", "swp"))
hsoil = tuple(np.load(f"../input/hydra/{n}.npy")
              for n in ("theta_sat", "psi_sat", "soil_text"))

with bz2.BZ2File("../k34/ISIMIP_HISTORICAL_METADATA.pbz2", mode='r') as fh:
    stime = copy.deepcopy(pkl.load(fh)[0])

with open("../input/co2/historical_CO2_annual_1765_2018.txt") as fh:
    co2_data = fh.readlines()


def get_pls_table():
    """Read the PLS table of the test. Create it if it does not exist"""
    if os.path.exists(PLS_TABLE):
        pls_table = pd.read_csv(PLS_TABLE).__array__().T
        assert pls_table.shape[1] == caete.npls, \
            f"{PLS_TABLE} has {pls_table.shape[1]} PLSs, caete_module was compiled with {caete.npls}"
    else:
        import plsgen as pls
        pls_table = pls.table_gen(caete.npls)
        pd.DataFrame(pls_table.T).to_csv(PLS_TABLE, index=False)
    return np.asfortranarray(pls_table, dtype=np.float64)


def run(mode, pls_table):
    """mode: 'off' (carbon only) or 'on' (nutrient cycle)"""
    grid = grd(240, 185, f"allom_nutri_test_{mode}")
    grid.init_caete_dyn("../k34", stime, co2_data, pls_table, tsoil, ssoil, hsoil)

    # soil pools spinup (same as model_driver.py)
    w, ll, cwd, rl, lnc = grid.bdg_spinup(
        start_date="19790101", end_date="19830101")
    grid.sdc_spinup(w, ll, cwd, rl, lnc)

    # carbon only spinup
    grid.run_caete_allom('19790101', '19891231', spinup=2,
                         fix_co2='1980', save=False, nutri_cycle=False)

    grid.run_caete_allom('19790101', '19891231', spinup=2,
                         fix_co2='1980', save=True, nutri_cycle=(mode == 'on'))

    return load(sorted(grid.outputs.values())[-1])


def summary(d, mode):
    """Mean (pools, daily rates) or sum (fluxes) of the last 365 days"""
    y = slice(-365, None)
    alive = d['area'][:, -1] > 0.0
    # limitation of each PLS in the last day, from the codes of leaf, wood
    # and root (1, 4: N; 2, 5: P; 6: co-limited; 0: no limitation)
    lim = d['lim_status'][:, :, -1]
    has_n = np.isin(lim, (1, 4)).any(axis=0)
    has_p = np.isin(lim, (2, 5)).any(axis=0)
    has_co = (lim == 6).any(axis=0)
    no_lim = (lim == 0).all(axis=0)
    n_lim = has_n & ~has_p & ~has_co
    p_lim = has_p & ~has_n & ~has_co
    co_lim = ~(no_lim | n_lim | p_lim)
    out = {'npp': d['npp'][y].mean(),
           'photo': d['photo'][y].mean(),
           'ar': d['ar'][y].mean(),
           'lai': d['lai'][y].mean(),
           'cleaf': d['cleaf'][y].mean(),
           'cfroot': d['cfroot'][y].mean(),
           'csap': d['csap'][y].mean(),
           'cheart': d['cheart'][y].mean(),
           'csto': d['csto'][y].mean(),
           'living_pls': d['ls'][-1],
           'nmin': d['nmin'][y].mean(),
           'pmin': d['pmin'][y].mean(),
           'nupt_mineral': d['nupt'][0, y].sum(),
           'nupt_organic': d['nupt'][1, y].sum(),
           'pupt_labile': d['pupt'][0, y].sum(),
           'pupt_sorbed': d['pupt'][1, y].sum(),
           'pupt_organic': d['pupt'][2, y].sum(),
           'litter_leaf_c': d['litter_l'][y].sum(),
           'litter_root_c': d['litter_fr'][y].sum(),
           'cwd_c': d['cwd'][y].sum(),
           'litter_n': d['lnc'][:3, y].sum(),
           'litter_p': d['lnc'][3:, y].sum(),
           'nsto': d['nsto'][y].mean(),
           'psto': d['psto'][y].mean(),
           'c_cost': d['c_cost'][y].sum(),
           'hresp': d['hresp'][y].mean(),
           'area_no_limitation': d['area'][alive & no_lim, -1].sum(),
           'area_n_limited': d['area'][alive & n_lim, -1].sum(),
           'area_p_limited': d['area'][alive & p_lim, -1].sum(),
           'area_colimited': d['area'][alive & co_lim, -1].sum()}
    return pd.Series(out, name=mode, dtype=np.float64)


if __name__ == "__main__":
    pls_table = get_pls_table()
    result = pd.concat([summary(run(mode, pls_table), mode)
                        for mode in ('off', 'on')], axis=1)
    result.index.name = 'variable'

    pd.set_option('display.float_format', '{:.6g}'.format)
    print("\nMEAN/SUM OF THE LAST 365 DAYS")
    print("carbon pools: kg(C) m-2; nsto, psto, nmin, pmin: g m-2;")
    print("npp, photo, ar: kg(C) m-2 yr-1; uptake, litter and c_cost: g m-2 yr-1;")
    print("hresp: g(C) m-2 day-1; area_*: fraction of the gridcell\n")
    print(result)

    if '--save' in sys.argv:
        result.to_csv(REFERENCE)
        print(f"\nReference saved: {REFERENCE}")
    elif os.path.exists(REFERENCE):
        ref = pd.read_csv(REFERENCE, index_col=0)
        diff = (result - ref).abs() / ref.abs().clip(lower=1e-12)
        print("\nRELATIVE DIFFERENCE TO THE REFERENCE")
        print(diff)
        print(f"\nmax relative difference: {diff.max().max():.3g}")
    else:
        print(f"\nNo reference found ({REFERENCE}). Use --save to create it")
