import os
import _pickle as pkl
import bz2
import copy
import multiprocessing as mp
from pathlib import Path
from random import shuffle

import argparse
import sys
import joblib
from netCDF4 import Dataset
import numpy as np
import re
from caete import grd, npls, print_progress, rbrk
import plsgen as pls


# ====================================
#            UTILS
# ====================================

def _convert_sombrero_char_to_bool(sombrero):
    if sombrero == "y":
        return True
    elif sombrero == "n":
        return False
    
def _convert_version_char_to_allom_bool(version):
    if version == '1':
        return True
    else:
        return False

def _check_valid_run_name(run_name, exit = False):
    if bool(re.match(r'^[A-Za-z][A-Za-z0-9]*$', run_name)):
        return True
    else:
        print(f"ERROR: Invalid run name! Please use only letters, numbers and no spaces!")
        if exit:
            sys.exit(1)
        return False


# ====================================
#              SET INPUTS
# ====================================

# Restrictions
ZONE_RESTRICTIONS = ['c', 's', 'e', 'nw']
MASKP_RESTRICTIONS = ['a', 'b', 'c']
CLIMATOLOGY_RESTRICTIONS = ['1', '2', '3', '4', '5']
SOMBRERO_RESTRICTIONS = ['y', 'n']
VERSION_RESTRICTIONS = ['1', '2']


def process_inputs(args=None):
    # --------------------------------------------
    # Parse arguments
    # --------------------------------------------
    parser = argparse.ArgumentParser(description='Model parameters')
    # Add arguments with short and long options
    parser.add_argument('--zone', '-z', type=str, choices=ZONE_RESTRICTIONS)
    parser.add_argument('--maskp', '-m', type=str, choices=MASKP_RESTRICTIONS)
    parser.add_argument('--sombrero', '-s', type=str, choices=SOMBRERO_RESTRICTIONS)
    parser.add_argument('--version', '-v', type=str, choices=VERSION_RESTRICTIONS)
    parser.add_argument('--run_name', '-n', type=str)
    parser.add_argument('--climatology', '-c', type=str, choices=CLIMATOLOGY_RESTRICTIONS)
    args = parser.parse_args(args)

    # --------------------------------------------
    # Assign variables
    # --------------------------------------------
    # Remain sombrero as None if not defined
    zone = args.zone if args.zone else None
    maskp = args.maskp if args.maskp else None
    sombrero = _convert_sombrero_char_to_bool(args.sombrero) if args.sombrero else None
    allom = _convert_version_char_to_allom_bool(args.version) if args.version else None
    climatology = args.climatology if args.climatology else None
    run_name = args.run_name if args.run_name and _check_valid_run_name(args.run_name, exit = True) else None


    print(f"sombrero: {sombrero}")
    print(f"maskp: {maskp}")
    print(f"allom: {allom}")
    print(f"run_name: {run_name}")
    print(f"zone: {zone}")
    print(f"climatology: {climatology}")


    '''
    Check sombrero
        Only asks interactively if sombrero was not set from arguments. Otherwise,
        ...argparse already checks SOMBRERO_RESTRICTIONS
    '''
    if sombrero is None:
        while True:
            i = input(f"▫️ RUN IN SOMBRERO? ({' / '.join(SOMBRERO_RESTRICTIONS)}): ")
            if i in SOMBRERO_RESTRICTIONS:
                sombrero = _convert_sombrero_char_to_bool(i)
                break
            else:
                pass


    '''
    Check maskp
        Only asks interactively if maskp was not set from arguments. Otherwise,
        ...argparse already checks MASKP_RESTRICTIONS
    '''
    if maskp is None:
        while True:
            maskp = input("▫️ THREE MASK OPTIONS: AMAZON BIOME (a); PAN-AMAZON (b) OR PLOT RUN (c): ")
            if maskp in MASKP_RESTRICTIONS:
                break
            else:
                pass

    '''
    Check version / allom
        Only asks interactively if allom was not set from arguments. Otherwise,
        ...argparse already checks VERSION_RESTRICTIONS
    '''
    if allom is None:
        while True:
            version = input(
                '▫️ Which version? \n' \
                '  - allom: Considers allometry constraints without nutrient cycle \n' \
                '  - nutri_cycle: Version with fix proportion to allocation and considering nutrient cycle \n' \
                '  (1: allom/2: nutri_cycle): ')
            if version in VERSION_RESTRICTIONS:
                allom = _convert_version_char_to_allom_bool(version)
                break
            else:
                pass


    # Check run name (if sombrero = False)
    # Logic:
    #    run_name=None and sombrero=True  --- OK
    #    run_name=None and sombrero=False --- Ask interactively
    #    run_name=any  and sombrero=True  --- ERROR
    #    run_name=any  and sombrero=False --- OK
    if run_name and sombrero:
        print("ERROR: You can only set run_name if you are not running on sombrero!")
        sys.exit(1)
    elif run_name is None and not sombrero:
        while True:
            run_name = input("▫️ Name to your run (folder to store outputs): ")
            # Check if it is a valid name
            if _check_valid_run_name(run_name, exit = False):
                break
            else:
                pass
    # Rename run_name
    outf = run_name

    # Check zone (if sombrero = False)
    # Logic:
    #    zone=None and sombrero=True  --- OK
    #    zone=None and sombrero=False --- Ask interactively
    #    zone=any  and sombrero=True  --- ERROR
    #    zone=any  and sombrero=False --- OK
    if zone and sombrero:
        print("ERROR: You can only set zone if you are not running on sombrero!")
        sys.exit(1)
    elif zone is None and not sombrero:
        while True:
            zone = input("▫️ Select a zone [c: central, s: south, e: east, nw: NW]: ")
            if zone in ZONE_RESTRICTIONS:
                break
            else:
                pass


    # Check climatology (if sombrero = True)
    # Logic:
    #    climatology=None and sombrero=True  --- Ask interactively
    #    climatology=None and sombrero=False --- OK
    #    climatology=any  and sombrero=True  --- OK
    #    climatology=any  and sombrero=False --- ERROR
    if climatology and not sombrero:
        print("ERROR: You can only set climatology if you are running on sombrero!")
        sys.exit(1)
    elif climatology is None and sombrero:
        clim_list = ["HISTORICAL-RUN",
                    "GFDL-ESM2M",
                    "HadGEM2-ES",
                    "IPSL-CM5A-LR",
                    "MIROC5"]
        
        CLIM_DATA_str = """
            You have the option to run any of the historical climatologies:

            HISTORICAL-RUN   1
            GFDL-ESM2M       2
            HadGEM2-ES       3
            IPSL-CM5A-LR     4
            MIROC5           5

            Option: """
        
        while True:
            climatology = input(CLIM_DATA_str)
            if climatology in CLIMATOLOGY_RESTRICTIONS:
                climatology = int(climatology)
                outf = clim_list[climatology - 1]
                break
            else:
                pass

    return zone, maskp, sombrero, allom, outf, climatology

if __name__ == "__main__":
    zone, maskp, sombrero, allom, outf, climatology = process_inputs()
    print(f"Zone: {zone}")
    print(f"Sombrero: {maskp}")
    print(f"sombrero: {sombrero}")
    print(f"allom: {allom}")
    print(f"outf: {outf}")
    print(f"climatology: {climatology}")