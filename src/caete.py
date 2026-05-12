# -*-coding:utf-8-*-
# "CAETÊ"
# Author:  João Paulo Darela Filho
"""
Copyright 2017- LabTerra

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import sys
import copy
import _pickle as pkl
import random as rd
from threading import Thread
from time import sleep
from pathlib import Path
import warnings
import bz2
import gc

from joblib import load, dump
import cftime
import numpy as np
from numpy import log as ln
from hydro_caete import soil_water
from caete_module import global_par as gp
from caete_module import budget as model
from caete_module import water as st
from caete_module import photo as m
from caete_module import soil_dec

NO_DATA = [-9999.0, -9999.0]
# print(f"RUNNING CAETÊ with {gp.npls} Plant Life Strategies")
# GLOBAL
out_ext = ".pkz"
npls = gp.npls
runplotp = False

# while True:
#     maskp = input(
#         "TWO MASK OPTIONS: AMAZON BIOME (a); PAN-AMAZON (b) OR PLOT RUN (c): ")
#     if maskp == 'b':
#         mask = np.load("../input/mask/mask_raisg-360-720.npy")
#         break
#     if maskp == 'a':
#         mask = np.load("../input/mask/mask_BIOMA.npy")
#         break
#     if maskp == 'c':
#         mask = np.load("../input/mask/mask_raisg-360-720.npy")
#         runplotp = True
#         break

mask = np.load("../input/mask/mask_raisg-360-720.npy")

Pan_Amazon_RECTANGLE = "y = 160:221 x = 201:272"

Pan_Amazon_CORNERS = {'ulc': (201, 160),
                      'lrc': (271, 220)}


def build_run_breaks(start, end, *, chunk_years=2, align="calendar"):
    """Build a list of (start, end) date-string tuples covering [start, end].

    Each tuple has the format ('YYYYMMDD', 'YYYYMMDD'), matching the
    convention used by `cf_date2str` and the HDF5 row keys.

    Parameters
    ----------
    start, end : str | datetime.date
        Inclusive bounds. Strings must be 'YYYYMMDD'.
    chunk_years : int, default 2
        Number of years per chunk.
    align : {'calendar', 'from_start'}, default 'calendar'
        - 'calendar': every chunk except possibly the first starts on
          Jan 1 and ends on Dec 31 of the last year in that chunk.
          The first chunk starts at `start`; if `start` is not Jan 1,
          the first chunk is shorter.
        - 'from_start': chunks of exactly `chunk_years` calendar years
          starting at `start` (i.e. `start + chunk_years*yr - 1 day`),
          with no Dec-31 snapping.
    """
    import datetime as _dt

    def _to_date(x):
        if isinstance(x, str):
            return _dt.date(int(x[0:4]), int(x[4:6]), int(x[6:8]))
        if isinstance(x, _dt.datetime):
            return x.date()
        if isinstance(x, _dt.date):
            return x
        raise TypeError(f"Unsupported date type: {type(x)!r}")

    def _fmt(d):
        return f"{d.year:04d}{d.month:02d}{d.day:02d}"

    if chunk_years < 1:
        raise ValueError("chunk_years must be >= 1")
    if align not in ("calendar", "from_start"):
        raise ValueError("align must be 'calendar' or 'from_start'")

    s = _to_date(start)
    e = _to_date(end)
    if e < s:
        raise ValueError("end must be on or after start")

    out = []
    cur = s
    while cur <= e:
        if align == "calendar":
            chunk_end = _dt.date(cur.year + chunk_years - 1, 12, 31)
        else:  # from_start
            chunk_end = _dt.date(cur.year + chunk_years, cur.month, cur.day) - _dt.timedelta(days=1)
        if chunk_end > e:
            chunk_end = e
        out.append((_fmt(cur), _fmt(chunk_end)))
        cur = chunk_end + _dt.timedelta(days=1)
    return out


run_breaks_hist = build_run_breaks('19010101', '20241231', chunk_years=1)

run_breaks_CMIP5_hist = build_run_breaks('19790101', '20051231')

run_breaks_CMIP5_proj = build_run_breaks('20060101', '20991231')


rbrk = [run_breaks_hist, run_breaks_CMIP5_hist, run_breaks_CMIP5_proj]

warnings.simplefilter("default")


# AUX FUNCS

def rwarn(txt='RuntimeWarning'):
    warnings.warn(f"{txt}", RuntimeWarning)


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=30):
    """FROM Stack Overflow/GIST, THANKS
    Call in a loop to create terminal progress bar

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    bar_utf = b'\xe2\x96\x88'  # bar -> unicode symbol = u'\u2588'
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' %
                     (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def neighbours_index(pos, matrix):
    neighbours = []
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    for i in range(max(0, pos[0] - 1), min(rows, pos[0] + 2)):
        for j in range(max(0, pos[1] - 1), min(cols, pos[1] + 2)):
            if (i, j) != pos:
                neighbours.append((i, j))
    return neighbours


# WARNING keep the lists of budget/carbon3 outputs updated with fortran code
def catch_out_budget(out):
    lst = ["evavg", "epavg", "phavg", "aravg", "nppavg",
           "laiavg", "rcavg", "f5avg", "rmavg", "rgavg", "cleafavg_pft", "cawoodavg_pft",
           "cfrootavg_pft", "stodbg", "ocpavg", "wueavg", "cueavg", "c_defavg", "vcmax",
           "specific_la", "nupt", "pupt", "litter_l", "cwd", "litter_fr", "npp2pay", "lnc", "delta_cveg",
           "limitation_status", "uptk_strat", 'cp', 'c_cost_cwm']

    return dict(zip(lst, out))


def catch_out_carbon3(out):
    lst = ['cs', 'snc', 'hr', 'nmin', 'pmin']

    return dict(zip(lst, out))


def find_coord(N, W):
    """ Given a pair of geographic (WGS84) coordinates (decimal degrees)
        returns the Y and X indices in the array (360,720//0.5° lon-lat)
        (C_contiguous) Tested only in south america"""
    Yc = round(N, 2)
    Xc = round(W, 2)

    if abs(Yc) > 89.75:
        if Yc < 0:
            Yc = -89.75
        else:
            Yc = 89.75

    if abs(Xc) > 179.75:
        if Xc < 0:
            Xc = -179.75
        else:
            Xc = 179.75

    Yind = 0
    Xind = 0

    lon = np.arange(-179.75, 180, 0.5)
    lat = np.arange(89.75, -90, -0.5)

    if True:
        while Yc < lat[Yind]:
            Yind += 1
    # else:
    #     Yind += lat.size // 2
    #     while Yc > lat[Yind]:
    #         Yind += 1
    if Xc <= 0:
        while Xc > lon[Xind]:
            Xind += 1
    else:
        Xind += lon.size // 2
        while Xc > lon[Xind]:
            Xind += 1

    return Yind, Xind


class grd:

    """
    Defines the gridcell object - This object stores all the input data,
    the data comming from model runs for each grid point, all the state variables and all the metadata
    describing the life cycle of the gridcell and the filepaths to the generated model outputs
    This class also provide several methods to apply the CAETÊ model with proper formated climatic and soil variables
    """

    def __init__(self, x, y, dump_folder):
        """Construct the gridcell object"""

        # CELL Identifiers
        self.x = x                            # Grid point x coordinate
        self.y = y                            # Grid point y coordinate
        self.xyname = str(y) + '-' + str(x)   # IDENTIFIES GRIDCELLS
        self.plot_name = dump_folder
        self.plot = None
        self.input_fname = f"input_data_{self.xyname}.pbz2"
        self.input_fpath = None
        self.data = None
        self.pos = (int(self.x), int(self.y))
        self.pls_table = None   # will receive the np.array with functional traits data
        self.outputs = {}       # dict, store filepaths of output data
        self.realized_runs = []
        self.experiments = 1
        # counts the execution of a time slice (a call of self.run_spinup)
        self.run_counter = 0
        self.neighbours = None

        self.ls = None          # Number of surviving plss//
        self.grid_filename = f"gridcell{self.xyname}" 
        self.out_dir = Path(
            "../outputs/{}/gridcell{}/".format(dump_folder, self.xyname)).resolve()
        self.flush_data = None

        # Time attributes
        self.time_index = None  # Array with the time stamps
        self.calendar = None    # Calendar name
        self.time_unit = None   # Time unit
        self.start_date = None
        self.end_date = None
        self.ssize = None
        self.sind = None
        self.eind = None

        # Input data
        self.filled = False     # Indicates when the gridcell is filled with input data
        self.pr = None
        self.ps = None
        self.rsds = None
        self.tas = None
        self.rhs = None

        # OUTPUTS
        self.soil_temp = None
        self.emaxm = None
        self.tsoil = None
        self.photo = None
        self.aresp = None
        self.npp = None
        self.lai = None
        self.csoil = None
        self.inorg_n = None
        self.inorg_p = None
        self.sorbed_n = None
        self.sorbed_p = None
        self.snc = None
        self.hresp = None
        self.rcm = None
        self.f5 = None
        self.runom = None
        self.evapm = None
        self.wsoil = None
        self.swsoil = None
        self.rm = None
        self.rg = None
        self.cleaf = None
        self.cawood = None
        self.cfroot = None
        self.area = None
        self.wue = None
        self.cue = None
        self.cdef = None
        self.nmin = None
        self.pmin = None
        self.vcmax = None
        self.specific_la = None
        self.nupt = None
        self.pupt = None
        self.litter_l = None
        self.cwd = None
        self.litter_fr = None
        self.lnc = None
        self.storage_pool = None
        self.lim_status = None
        self.uptake_strategy = None
        self.carbon_costs = None

        # WATER POOLS
        # Water content for each soil layer
        self.wp_water_upper_mm = None  # mm
        self.wp_water_lower_mm = None  # mm
        # Saturation point
        self.wmax_mm = None  # mm

        # SOIL POOLS
        self.input_nut = None
        self.sp_available_p = None
        self.sp_available_n = None
        self.sp_so_n = None
        self.sp_in_n = None
        self.sp_so_p = None
        self.sp_in_p = None
        self.sp_csoil = None
        self.sp_snr = None
        self.sp_uptk_costs = None
        self.sp_organic_n = None
        self.sp_sorganic_n = None
        self.sp_organic_p = None
        self.sp_sorganic_p = None

        # CVEG POOLS
        self.vp_cleaf = None
        self.vp_croot = None
        self.vp_cwood = None
        self.vp_dcl = None
        self.vp_dca = None
        self.vp_dcf = None
        self.vp_ocp = None
        self.vp_wdl = None
        self.vp_sto = None
        self.vp_lsid = None

        # Hydraulics
        self.theta_sat = None
        self.psi_sat = None
        self.soil_texture = None

    def _allocate_output_nosave(self, n):
        """Allocate minimal daily buffers for no-save runs.

        Creates only the arrays required to keep model state updates
        consistent when :meth:`run_caete` is called with ``save=False``.
        This mode is used during spinup phases where output files are not
        needed, but selected daily diagnostics are still required to update
        soil and vegetation pools.

        Parameters
        ----------
        n : int
            Number of simulated daily steps in the current run window.

        Returns
        -------
        None
            Buffers are allocated in place on ``self``.

        Notes
        -----
        Allocated attributes and shapes:

        - ``runom``: ``(n,)``
        - ``nupt``: ``(2, n)``
        - ``pupt``: ``(3, n)``
        - ``litter_l``: ``(n,)``
        - ``cwd``: ``(n,)``
        - ``litter_fr``: ``(n,)``
        - ``lnc``: ``(6, n)``
        - ``storage_pool``: ``(3, n)``
        - ``ls``: ``(n,)``

        All NumPy arrays are Fortran-contiguous (``order='F'``), matching the
        memory layout expected by the Fortran-bound workflow.
        """

        self.runom = np.zeros(shape=(n,), order='F')
        self.nupt = np.zeros(shape=(2, n), order='F')
        self.pupt = np.zeros(shape=(3, n), order='F')
        self.litter_l = np.zeros(shape=(n,), order='F')
        self.cwd = np.zeros(shape=(n,), order='F')
        self.litter_fr = np.zeros(shape=(n,), order='F')
        self.lnc = np.zeros(shape=(6, n), order='F')
        self.storage_pool = np.zeros(shape=(3, n), order='F')
        self.ls = np.zeros(shape=(n,), order='F')

    def _allocate_output(self, n, npls=npls):
        """Allocate full daily output buffers for save-enabled runs.

        Initializes all per-step arrays and per-PLS diagnostics required by
        :meth:`run_caete` when ``save=True``. These buffers are filled during
        the daily loop, then serialized by :meth:`_flush_output` /
        :meth:`_save_output`.

        Parameters
        ----------
        n : int
                Number of simulated daily steps in the current run window.

        npls : int, default ``npls``
                Total number of Plant Life Strategies represented in PLS-indexed
                output tensors (for example, area occupancy, limitation status,
                and uptake strategy).

        Returns
        -------
        None
                Buffers are allocated in place on ``self``.

        Notes
        -----
        This method resets and allocates both scalar-time-series outputs and
        PLS-resolved tensors. Key groups include:

        - Atmosphere/flux/state CWM series (for example ``photo``, ``npp``,
            ``lai``, ``evapm``, ``wue``, ``cue``)
        - Soil C/N/P pools and fluxes (for example ``csoil``, ``snc``,
            ``inorg_n``, ``sorbed_p``, ``nmin``, ``pmin``)
        - Nutrient uptake and litter inputs (``nupt``, ``pupt``, ``litter_l``,
            ``cwd``, ``litter_fr``, ``lnc``)
        - PLS-resolved diagnostics include ``area`` with shape ``(npls, n)``.
        - PLS-resolved diagnostics include ``lim_status`` with shape ``(3, npls, n)``.
        - PLS-resolved diagnostics include ``uptake_strategy`` with shape ``(2, npls, n)``.

        The rolling lists ``emaxm`` and ``tsoil`` are also reinitialized as
        empty Python lists and later converted to arrays during flushing.
        All NumPy arrays are Fortran-contiguous (``order='F'``) to preserve
        compatibility with downstream Fortran-oriented data handling.
        """
        
        self.emaxm = []
        self.tsoil = []
        self.photo = np.zeros(shape=(n,), order='F')
        self.aresp = np.zeros(shape=(n,), order='F')
        self.npp = np.zeros(shape=(n,), order='F')
        self.lai = np.zeros(shape=(n,), order='F')
        self.csoil = np.zeros(shape=(4, n), order='F')
        self.inorg_n = np.zeros(shape=(n,), order='F')
        self.inorg_p = np.zeros(shape=(n,), order='F')
        self.sorbed_n = np.zeros(shape=(n,), order='F')
        self.sorbed_p = np.zeros(shape=(n,), order='F')
        self.snc = np.zeros(shape=(8, n), order='F')
        self.hresp = np.zeros(shape=(n,), order='F')
        self.rcm = np.zeros(shape=(n,), order='F')
        self.f5 = np.zeros(shape=(n,), order='F')
        self.runom = np.zeros(shape=(n,), order='F')
        self.evapm = np.zeros(shape=(n,), order='F')
        self.wsoil = np.zeros(shape=(n,), order='F')
        self.swsoil = np.zeros(shape=(n,), order='F')
        self.rm = np.zeros(shape=(n,), order='F')
        self.rg = np.zeros(shape=(n,), order='F')
        self.cleaf = np.zeros(shape=(n,), order='F')
        self.cawood = np.zeros(shape=(n,), order='F')
        self.cfroot = np.zeros(shape=(n,), order='F')
        self.wue = np.zeros(shape=(n,), order='F')
        self.cue = np.zeros(shape=(n,), order='F')
        self.cdef = np.zeros(shape=(n,), order='F')
        self.nmin = np.zeros(shape=(n,), order='F')
        self.pmin = np.zeros(shape=(n,), order='F')
        self.vcmax = np.zeros(shape=(n,), order='F')
        self.specific_la = np.zeros(shape=(n,), order='F')
        self.nupt = np.zeros(shape=(2, n), order='F')
        self.pupt = np.zeros(shape=(3, n), order='F')
        self.litter_l = np.zeros(shape=(n,), order='F')
        self.cwd = np.zeros(shape=(n,), order='F')
        self.litter_fr = np.zeros(shape=(n,), order='F')
        self.lnc = np.zeros(shape=(6, n), order='F')
        self.storage_pool = np.zeros(shape=(3, n), order='F')
        self.ls = np.zeros(shape=(n,), order='F')
        self.carbon_costs = np.zeros(shape=(n,), order='F')

        self.area = np.zeros(shape=(npls, n), order='F')
        self.lim_status = np.zeros(
            shape=(3, npls, n), dtype=np.dtype('int16'), order='F')
        self.uptake_strategy = np.zeros(
            shape=(2, npls, n), dtype=np.dtype('int32'), order='F')

    def _flush_output(self, run_descr, index):
        """Package current run buffers and clear in-memory output attributes.

        Builds the per-run output dictionary consumed by :meth:`_save_output`,
        registers the destination filepath in ``self.outputs``, and flushes all
        per-step output buffers from this object so the next run/spin can
        allocate fresh arrays.

        Parameters
        ----------
        run_descr : str
            Prefix used to compose the logical output name for this flush
            (for example ``"spin"``). The final key is built as
            ``{run_descr}{counter:05d}{out_ext}`` for ``counter <= 99999``
            (zero-padded to 5 digits so filenames sort lexicographically in
            chronological order, e.g. ``spin00001.pkz``, ``spin00002.pkz``,
            ..., ``spin99999.pkz``). For ``counter > 99999`` the counter is
            written without padding, which breaks alphabetical ordering.

        index : tuple[int, int] | list[int]
            Two-element container with the inclusive numeric time bounds of
            the flushed window, stored as ``sind`` and ``eind`` in the output
            payload.

        Returns
        -------
        dict
            Serialized-ready payload containing daily outputs, PLS-resolved
            tensors, and metadata (calendar, time units, start/end indexes).
            This dictionary is intended to be written by :meth:`_save_output`.

        Notes
        -----
        - Increments ``self.run_counter`` on every call.
        - Adds an entry to ``self.outputs`` mapping the generated filename to
          its absolute path under ``self.out_dir``.
        - Converts list-backed fields (``emaxm``, ``tsoil``) to NumPy arrays in
          the returned payload.
        - Resets output attributes on ``self`` to ``None`` (or empty lists for
          ``emaxm`` and ``tsoil``) after packaging, reducing memory retention
          between runs.
        """
        to_pickle = {}
        self.run_counter += 1
        if self.run_counter < 10:
            spiname = run_descr + "0000" + str(self.run_counter) + out_ext
        elif self.run_counter < 100:
            spiname = run_descr + "000" + str(self.run_counter) + out_ext
        elif self.run_counter < 1000:
            spiname = run_descr + "00" + str(self.run_counter) + out_ext
        elif self.run_counter < 10000:
            spiname = run_descr + "0" + str(self.run_counter) + out_ext
        else:
            spiname = run_descr + str(self.run_counter) + out_ext

        self.outputs[spiname] = os.path.join(self.out_dir, spiname)
        to_pickle = {'emaxm': np.array(self.emaxm),
                     "tsoil": np.array(self.tsoil),
                     "photo": self.photo,
                     "aresp": self.aresp,
                     'npp': self.npp,
                     'lai': self.lai,
                     'csoil': self.csoil,
                     'inorg_n': self.inorg_n,
                     'inorg_p': self.inorg_p,
                     'sorbed_n': self.sorbed_n,
                     'sorbed_p': self.sorbed_p,
                     'snc': self.snc,
                     'hresp': self.hresp,
                     'rcm': self.rcm,
                     'f5': self.f5,
                     'runom': self.runom,
                     'evapm': self.evapm,
                     'wsoil': self.wsoil,
                     'swsoil': self.swsoil,
                     'rm': self.rm,
                     'rg': self.rg,
                     'cleaf': self.cleaf,
                     'cawood': self.cawood,
                     'cfroot': self.cfroot,
                     'area': self.area,
                     'wue': self.wue,
                     'cue': self.cue,
                     'cdef': self.cdef,
                     'nmin': self.nmin,
                     'pmin': self.pmin,
                     'vcmax': self.vcmax,
                     'specific_la': self.specific_la,
                     'nupt': self.nupt,
                     'pupt': self.pupt,
                     'litter_l': self.litter_l,
                     'cwd': self.cwd,
                     'litter_fr': self.litter_fr,
                     'lnc': self.lnc,
                     'ls': self.ls,
                     'lim_status': self.lim_status,
                     'c_cost': self.carbon_costs,
                     'u_strat': self.uptake_strategy,
                     'storage_pool': self.storage_pool,
                     'calendar': self.calendar,    # Calendar name
                     'time_unit': self.time_unit,   # Time unit
                     'sind': index[0],
                     'eind': index[1]}
        # Flush attrs
        self.emaxm = []
        self.tsoil = []
        self.photo = None
        self.aresp = None
        self.npp = None
        self.lai = None
        self.csoil = None
        self.inorg_n = None
        self.inorg_p = None
        self.sorbed_n = None
        self.sorbed_p = None
        self.snc = None
        self.hresp = None
        self.rcm = None
        self.f5 = None
        self.runom = None
        self.evapm = None
        self.wsoil = None
        self.swsoil = None
        self.rm = None
        self.rg = None
        self.cleaf = None
        self.cawood = None
        self.cfroot = None
        self.area = None
        self.wue = None
        self.cue = None
        self.cdef = None
        self.nmin = None
        self.pmin = None
        self.vcmax = None
        self.specific_la = None
        self.nupt = None
        self.pupt = None
        self.litter_l = None
        self.cwd = None
        self.litter_fr = None
        self.lnc = None
        self.storage_pool = None
        self.ls = None
        self.ls_id = None
        self.lim_status = None
        self.carbon_costs = None,
        self.uptake_strategy = None

        return to_pickle

    def _save_output(self, data_obj):
        """Persist one flushed run payload to a compressed pickle file.

        Writes the dictionary produced by :meth:`_flush_output` to disk using
        ``joblib.dump`` with zlib compression. The destination path is resolved
        from ``self.outputs`` using the filename pattern tied to the current
        ``self.run_counter``.

        Parameters
        ----------
        data_obj : dict
            Output payload generated by :meth:`_flush_output`, containing the
            daily series, PLS-resolved arrays, and run metadata (calendar,
            time unit, and index bounds).

        Returns
        -------
        None
            The method writes the compressed file and updates internal flush
            bookkeeping.

        Notes
        -----
        - Filename key selection follows the same numbering scheme used in
          :meth:`_flush_output`: the counter is zero-padded to 5 digits
          (``spin00001.pkz`` ... ``spin99999.pkz``) so that lexicographic
          ordering matches chronological ordering of the time slices.
        - Raises ``ValueError`` if ``self.run_counter`` exceeds ``99999``,
          since the fixed-width padding pattern is exhausted past that point.
        - Uses ``dump(data_obj, fh, compress=('zlib', 3), protocol=4)``.
        - Sets ``self.flush_data = 0`` after a successful write.
        """
        if self.run_counter < 10:
            fpath = "spin{}{}{}".format("0000", self.run_counter, out_ext)
        elif self.run_counter < 100:
            fpath = "spin{}{}{}".format("000", self.run_counter, out_ext)
        elif self.run_counter < 1000:
            fpath = "spin{}{}{}".format("00", self.run_counter, out_ext)
        elif self.run_counter < 10000:
            fpath = "spin{}{}{}".format("0", self.run_counter, out_ext)
        else:
            fpath = "spin{}{}".format(self.run_counter, out_ext)
        if self.run_counter > 99999:
            raise ValueError("run_counter exceeded 99999, filename pattern exhausted")
        
        with open(self.outputs[fpath], 'wb') as fh:
            dump(data_obj, fh, compress=('zlib', 3), protocol=4)
        self.flush_data = 0

    def init_caete_dyn(self, input_fpath, stime_i, co2, pls_table, tsoil, ssoil, hsoil):
        """Initialize this gridcell with forcing, traits, and state variables.

        Loads one gridcell input pickle (climate + soil nutrients), initializes
        the time metadata, stores model inputs (PLS table and atmospheric CO2
        series), and builds the initial water, vegetation, and soil pools used
        by :meth:`run_caete`.

        Parameters
        ----------
        input_fpath : str | pathlib.Path
            Directory containing ``input_data_{y-x}.pbz2`` files. The file used
            is derived from ``self.input_fname`` (set from this instance
            coordinates).

        stime_i : dict
            Time metadata dictionary with at least:

            - ``calendar`` : calendar name used by cftime
            - ``time_index`` : numeric time axis (daily)
            - ``units`` : CF-style time units string

            This object is deep-copied to ``self.stime`` and used to set
            ``self.start_date``/``self.end_date`` and numeric bounds
            (``self.sind``, ``self.eind``).

        co2 : list[str]
            Annual atmospheric CO2 records used later by :meth:`run_caete`.
            Expected line format is year/value text (e.g. ``"1901 296.3"`` or,
            for plot runs, comma-separated values).

        pls_table : numpy.ndarray
            Plant Life Strategy trait matrix consumed by the Fortran core. It is
            deep-copied into ``self.pls_table`` and used to create initial
            biomass/occupancy vectors.

        tsoil, ssoil, hsoil : numpy.ndarray
            Soil parameter arrays indexed as ``[layer_or_var, y, x]`` for this
            gridcell:

            - ``tsoil``: upper-layer hydraulic parameters (``ws1``, ``fc1``, ``wp1``)
            - ``ssoil``: lower-layer hydraulic parameters (``ws2``, ``fc2``, ``wp2``)
            - ``hsoil``: hydraulic descriptors (``theta_sat``, ``psi_sat``, texture)

        Returns
        -------
        None
            The method mutates ``self`` in place and marks the gridcell as
            initialized (``self.filled = True``).

        Notes
        -----
        - Guarded by ``assert self.filled == False``: this initializer is meant
          to run only once per gridcell instance.
        - Reads climate variables ``pr``, ``ps``, ``rsds``, ``tas``, ``hurs``
          and nutrient pools ``tn``, ``tp``, ``ap``, ``ip``, ``op`` from the
          compressed input file.
        - Initializes first-guess vegetation pools (leaf/root/wood carbon),
          computes living PLS indices, and creates baseline soil nutrient pools
          required by the daily loop.
        """

        assert self.filled == False, "already done"
        self.input_fpath = Path(os.path.join(input_fpath, self.input_fname))
        assert self.input_fpath.exists()

        with bz2.BZ2File(self.input_fpath, mode='r') as fh:
            self.data = pkl.load(fh)

        os.makedirs(self.out_dir, exist_ok=True)
        self.flush_data = 0

        self.pr = self.data['pr']
        self.ps = self.data['ps']
        self.rsds = self.data['rsds']
        self.tas = self.data['tas']
        self.rhs = self.data['hurs']

        # SOIL AND NUTRIENTS
        self.input_nut = []
        self.nutlist = ['tn', 'tp', 'ap', 'ip', 'op']
        for nut in self.nutlist:
            self.input_nut.append(self.data[nut])
        self.soil_dict = dict(zip(self.nutlist, self.input_nut))
        self.data = None

        # TIME
        self.stime = copy.deepcopy(stime_i)
        self.calendar = self.stime['calendar']
        self.time_index = self.stime['time_index']
        self.time_unit = self.stime['units']
        self.ssize = self.time_index.size
        self.sind = int(self.time_index[0])
        self.eind = int(self.time_index[-1])
        self.start_date = cftime.num2date(
            self.time_index[0], self.time_unit, calendar=self.calendar)
        self.end_date = cftime.num2date(
            self.time_index[-1], self.time_unit, calendar=self.calendar)

        # OTHER INPUTS
        self.pls_table = copy.deepcopy(pls_table)
        self.neighbours = neighbours_index(self.pos, mask)
        self.soil_temp = st.soil_temp_sub(self.tas[:1095] - 273.15)

        # Prepare co2 inputs (we have annually means)
        self.co2_data = copy.deepcopy(co2)

        self.tsoil = []
        self.emaxm = []

        # STATE
        # Water
        ws1 = tsoil[0][self.y, self.x].copy()
        fc1 = tsoil[1][self.y, self.x].copy()
        wp1 = tsoil[2][self.y, self.x].copy()

        ws2 = ssoil[0][self.y, self.x].copy()
        fc2 = ssoil[1][self.y, self.x].copy()
        wp2 = ssoil[2][self.y, self.x].copy()

        self.swp = soil_water(ws1, ws2, fc1, fc2, wp1, wp2)
        self.wp_water_upper_mm = self.swp.w1
        self.wp_water_lower_mm = self.swp.w2
        self.wmax_mm = np.float64(self.swp.w1_max + self.swp.w2_max)

        self.theta_sat = hsoil[0][self.y, self.x].copy()
        self.psi_sat = hsoil[1][self.y, self.x].copy()
        self.soil_texture = hsoil[2][self.y, self.x].copy()

        # Biomass
        self.vp_cleaf = np.zeros(shape=(npls,), order='F') + 1.0
        self.vp_croot = np.zeros(shape=(npls,), order='F') + 1.0
        self.vp_cwood = np.zeros(shape=(npls,), order='F') + 0.1
        self.vp_cwood[pls_table[6,:] == 0.0] = 0.0
        # self.vp_cleaf, self.vp_croot, self.vp_cwood = m.spinup2(
        #     1.0, self.pls_table)
        a, b, c, d = m.pft_area_frac(
            self.vp_cleaf, self.vp_croot, self.vp_cwood, self.pls_table[6, :])
        self.vp_lsid = np.where(a > 0.0)[0]
        self.ls = self.vp_lsid.size
        self.vp_dcl = np.zeros(shape=(npls,), order='F')
        self.vp_dca = np.zeros(shape=(npls,), order='F')
        self.vp_dcf = np.zeros(shape=(npls,), order='F')
        self.vp_ocp = np.zeros(shape=(npls,), order='F')
        self.vp_sto = np.zeros(shape=(3, npls), order='F')

        # # # SOIL
        self.sp_csoil = np.zeros(shape=(4,), order='F') + 1.0
        self.sp_snc = np.zeros(shape=(8,), order='F') + 0.1
        self.sp_available_p = self.soil_dict['ap']
        self.sp_available_n = 0.2 * self.soil_dict['tn']
        self.sp_in_n = 0.4 * self.soil_dict['tn']
        self.sp_so_n = 0.2 * self.soil_dict['tn']
        self.sp_so_p = self.soil_dict['tp'] - sum(self.input_nut[2:])
        self.sp_in_p = self.soil_dict['ip']
        self.sp_uptk_costs = np.zeros(npls, order='F')
        self.sp_organic_n = 0.1 * self.soil_dict['tn']
        self.sp_sorganic_n = 0.1 * self.soil_dict['tn']
        self.sp_organic_p = 0.5 * self.soil_dict['op']
        self.sp_sorganic_p = self.soil_dict['op'] - self.sp_organic_p

        self.outputs = dict()
        self.filled = True
        gc.collect()
        return None

    def clean_run(self, dump_folder, save_id):
        """Archive the current run's outputs and redirect this gridcell to a fresh dump folder.

        Snapshots the outputs produced so far under ``self.outputs`` into
        ``self.realized_runs`` (tagged with ``save_id``), then points
        ``self.out_dir`` at a brand-new directory under
        ``../outputs/{dump_folder}/gridcell{xyname}/`` and resets the per-run
        bookkeeping so the next call to :meth:`run_caete` writes into the new
        location. Intended to start a new experiment leg (for example, the
        transient or perturbation phase that follows a spinup) without losing
        the file registry of the previous leg.

        Parameters
        ----------
        dump_folder : str
            Name of the new top-level output folder (relative to
            ``../outputs/``) where subsequent runs will write their pickles.
            The directory ``../outputs/{dump_folder}/gridcell{xyname}/`` is
            created and **must not already exist**: if it does, the run is
            aborted, ``self.out_dir`` is restored to its previous value, and a
            ``RuntimeError`` is raised. This guard prevents accidentally
            overwriting an existing experiment.

        save_id : str
            Tag stored alongside the archived ``self.outputs`` snapshot in
            ``self.realized_runs`` (e.g. ``"init_cond"`` to mark the end of
            spinup). Used downstream to identify which experiment leg
            produced each pickled output.

        Returns
        -------
        None
            The method mutates ``self`` in place.

        Raises
        ------
        RuntimeError
            If ``../outputs/{dump_folder}/gridcell{xyname}/`` already exists.
            In that case ``self.out_dir`` is rolled back to its previous value
            before re-raising.

        Notes
        -----
        - Appends ``(save_id, self.outputs.copy())`` to ``self.realized_runs``
          so the previous leg's filepaths remain accessible.
        - Resets ``self.outputs`` to an empty dict and ``self.run_counter`` to
          ``0``, so the next flush starts at ``spin00001{out_ext}`` in the new
          folder.
        - Increments ``self.experiments`` to record that another experiment
          leg has been started on this gridcell.
        - ``self.out_dir`` is asserted to exist after the ``os.makedirs`` call
          to ensure the new destination is usable before any data is
          archived.
        """
        abort = False
        mem = str(self.out_dir)
        self.out_dir = Path(
            "../outputs/{}/gridcell{}/".format(dump_folder, self.xyname)).resolve()
        try:
            os.makedirs(str(self.out_dir), exist_ok=False)
        except FileExistsError:
            abort = True
            print(
                f"Folder {dump_folder} already exists. You cannot orerwrite its contents")
        finally:
            assert self.out_dir.exists(), f"Failed to create {self.out_dir}"

        if abort:
            print("ABORTING")
            self.out_dir = Path(mem)
            print(
                f"Returning the original grd_{self.xyname}.out_dir to {self.out_dir}")
            raise RuntimeError

        self.realized_runs.append((save_id, self.outputs.copy()))
        self.outputs = {}
        self.run_counter = 0
        self.experiments += 1

    def change_clim_input(self, input_fpath, stime_i, co2):
        """Swap the gridcell's climate forcing, time axis, and CO2 series in place.

        Reloads the per-gridcell input pickle (``input_data_{y-x}.pbz2``) from
        ``input_fpath``, replaces the climate arrays (``pr``, ``ps``, ``rsds``,
        ``tas``, ``hurs``) and soil nutrient pools on ``self``, rebuilds the
        time metadata from ``stime_i``, and stores a fresh atmospheric CO2
        record. Intended to be called on an already-initialized gridcell
        (typically after :meth:`init_caete_dyn` and at least one
        :meth:`run_caete` spinup) to chain a new forcing window — e.g. moving
        from the historical climate slab used for spinup to a CMIP-style
        scenario, or stepping through successive period chunks in the task5
        driver — while keeping the vegetation, soil, and water state pools
        already carried by the gridcell.

        Parameters
        ----------
        input_fpath : str | pathlib.Path
            Directory containing ``input_data_{y-x}.pbz2`` files. The actual
            file is composed from ``self.input_fname`` (set at construction
            from this gridcell's coordinates), so this argument selects the
            *source folder* for the new climate slab, not the file itself.

        stime_i : dict
            Time metadata dictionary for the new forcing window, with at
            least:

            - ``calendar`` : calendar name used by cftime
            - ``time_index`` : numeric time axis (daily)
            - ``units`` : CF-style time units string

            This object is deep-copied to ``self.stime`` and used to set
            ``self.start_date`` / ``self.end_date`` and the integer bounds
            ``self.sind`` / ``self.eind``.

        co2 : list[str]
            New annual atmospheric CO2 records, in the same format consumed by
            :meth:`run_caete` (year/value text lines, e.g. ``"1901 296.3"``;
            comma-separated for plot runs). Deep-copied into ``self.co2_data``,
            fully replacing the previous series.

        Returns
        -------
        None
            The method mutates ``self`` in place.

        Notes
        -----
        - Asserts that the resolved input file exists before loading.
        - Resets ``self.flush_data`` to ``0`` and clears ``self.data`` after
          the inputs are unpacked, mirroring :meth:`init_caete_dyn`.
        - Does **not** touch vegetation, soil, or water state pools, nor the
          PLS table, ``self.outputs``, or ``self.run_counter``: those carry
          over from the previous run leg so the gridcell can continue from
          its current state under the new forcing.
        - Unlike :meth:`init_caete_dyn`, no ``self.filled`` guard is applied
          — the method is meant to be called repeatedly on an initialized
          gridcell.
        """

        self.input_fpath = Path(os.path.join(input_fpath, self.input_fname))
        assert self.input_fpath.exists()

        with bz2.BZ2File(self.input_fpath, mode='r') as fh:
            self.data = pkl.load(fh)

        self.flush_data = 0

        self.pr = self.data['pr']
        self.ps = self.data['ps']
        self.rsds = self.data['rsds']
        self.tas = self.data['tas']
        self.rhs = self.data['hurs']

        # SOIL AND NUTRIENTS
        self.input_nut = []
        self.nutlist = ['tn', 'tp', 'ap', 'ip', 'op']
        for nut in self.nutlist:
            self.input_nut.append(self.data[nut])
        self.soil_dict = dict(zip(self.nutlist, self.input_nut))
        self.data = None

        # TIME
        self.stime = copy.deepcopy(stime_i)
        self.calendar = self.stime['calendar']
        self.time_index = self.stime['time_index']
        self.time_unit = self.stime['units']
        self.ssize = self.time_index.size
        self.sind = int(self.time_index[0])
        self.eind = int(self.time_index[-1])
        self.start_date = cftime.num2date(
            self.time_index[0], self.time_unit, calendar=self.calendar)
        self.end_date = cftime.num2date(
            self.time_index[-1], self.time_unit, calendar=self.calendar)

        # Prepare co2 inputs (we have annually means)
        self.co2_data = copy.deepcopy(co2)

        return None

    def run_caete(self,
                  start_date,
                  end_date,
                  spinup=0,
                  fix_co2=None,
                  save=True,
                  nutri_cycle=True,
                  afex=False):
        """Run the CAETÊ-DVM Fortran core for this gridcell over a date range.

        Drives the daily Fortran subroutines (`daily_budget`, `carbon3`,
        `soil_temp`, etc.), updates the gridcell state in place, and
        optionally writes per-chunk output pickles to ``self.out_dir``.

        Parameters
        ----------
        start_date : str
            Inclusive start date in ``"YYYYMMDD"`` format. Must lie within
            ``[self.start_date, self.end_date]`` (the climate-input span)
            and must be strictly before ``end_date``.

        end_date : str
            Inclusive end date in ``"YYYYMMDD"`` format. Must lie within
            ``[self.start_date, self.end_date]``.

        spinup : int, default 0
            Number of times the Fortran loop is repeated over the
            ``[start_date, end_date]`` window before returning.

            - ``0`` — single forward pass (transient run).
            - ``>0`` — that many repetitions; the gridcell state
              (vegetation pools, soil pools, water) is carried across
              repetitions, but climate is replayed from the start each
              time. Used by the spinup phases in the driver.

            When ``save=True``, every repetition flushes its own output
            pickle (``spin01.pkz``, ``spin02.pkz``, ...). When
            ``save=False``, no pickle is written but the in-memory state
            still evolves.

        fix_co2 : None | int | float | str, default None
            Controls the atmospheric CO2 forcing.

            - ``None`` — transient CO2: each year's value is read from
              ``self.co2_data`` and linearly interpolated to the next
              year on a daily basis.
            - ``int`` or ``float`` (must be > 0) — use this fixed CO2
              concentration (ppm) for every step. No interpolation.
            - ``str`` — a 4-digit year (e.g. ``"1901"``); the CO2 value
              for that year is looked up in ``self.co2_data`` and held
              constant for the whole run. Used by the spinup phases.

        save : bool, default True
            If ``True``, allocate full daily output buffers, populate
            them every step, and at the end of each spin repetition
            flush them to a compressed pickle in ``self.out_dir`` (via
            ``_flush_output`` / ``_save_output``) and append the path to
            ``self.outputs``. The pickles are what ``write_h5`` later
            reads.

            If ``False``, allocate only the minimal state buffers
            (``_allocate_output_nosave``) and write nothing. The daily
            outputs are still computed inside the Fortran call, just not
            stored. This is the spinup mode where you want pool states
            to evolve but do not need per-day records.

        nutri_cycle : bool, default True
            Enable the soil nutrient (N and P) cycle update each step:
            organic/inorganic pool transfers, sorption equilibria
            (``sorbed_n_equil`` / ``sorbed_p_equil``), solution
            equilibria, and plant uptake of organic N/P. When ``False``,
            those updates are skipped and the soil mineral pools stay
            (almost) frozen — used by phase-1 spinup to let vegetation
            equilibrate against a static nutrient background before the
            full biogeochemistry is switched on.

        afex : bool, default False
            Enable the AFEX (Amazon Fertilization EXperiment) nutrient
            addition pulse. When ``True``, on day-of-year 365 of every
            simulation year the file ``afex.cfg`` is read; its first
            line must be ``N``, ``P``, or ``NP`` and selects which
            available pool(s) receive an annual addition:

            - ``N``  → ``sp_available_n += 12.5`` g N m⁻² yr⁻¹
              (≈ 125 kg ha⁻¹ yr⁻¹)
            - ``P``  → ``sp_available_p += 5.0``  g P m⁻² yr⁻¹
              (≈  50 kg ha⁻¹ yr⁻¹)
            - ``NP`` → both of the above.

            Has no effect when ``False``; ``afex.cfg`` is not read in
            that case.

        Returns
        -------
        None
            All results are stored on ``self`` (state pools) and, when
            ``save=True``, in pickles registered in ``self.outputs``.

        Notes
        -----
        - If, during the run, every Plant Life Strategy (PLS) goes
          extinct, the gridcell is repopulated with a random subset
          of PLS templates only when ``save=False`` (spinup); during a
          transient run (``save=True``) the gridcell instead aborts
          the remaining steps for that call and emits a
          ``NO LIVING PLS - ABORT`` warning.
        - The valid CO2 lookup range is whatever ``self.co2_data``
          covers (the bundled file spans 1765-2024). The string form of
          ``fix_co2`` is validated only as "parseable as int", so any
          year outside that range will silently return ``None`` from
          ``find_co2``.
        """

        assert self.filled, "The gridcell has no input data"
        assert not fix_co2 or type(
            fix_co2) == str or fix_co2 > 0, "A fixed value for ATM[CO2] must be a positive number greater than zero or a proper string "
        ABORT = 0
        if self.plot is True:
            splitter = ","
        else:
            splitter = None  # split on any whitespace (tabs or spaces)

        def find_co2(year):
            for i in self.co2_data:
                parts = i.split(splitter) if splitter else i.split()
                if not parts or not parts[0].strip().lstrip('-').isdigit():
                    continue
                if int(parts[0]) == year:
                    return float(parts[1].strip())

        def find_index(start, end):
            result = []
            num = np.arange(self.ssize)
            ind = np.arange(self.sind, self.eind + 1)
            for r, i in zip(num, ind):
                if i == start:
                    result.append(r)
            for r, i in zip(num, ind):
                if i == end:
                    result.append(r)
            return result

        # Define start and end dates (read actual arguments)
        start = cftime.real_datetime(int(start_date[:4]), int(
            start_date[4:6]), int(start_date[6:]))
        end = cftime.real_datetime(int(end_date[:4]), int(
            end_date[4:6]), int(end_date[6:]))
        # Check dates sanity
        assert start < end, "start > end"
        assert start >= self.start_date
        assert end <= self.end_date

        # Define time index
        start_index = int(cftime.date2num(
            start, self.time_unit, self.calendar))
        end_index = int(cftime.date2num(end, self.time_unit, self.calendar))

        lb, hb = find_index(start_index, end_index)
        steps = np.arange(lb, hb + 1)
        day_indexes = np.arange(start_index, end_index + 1)
        spin = 1 if spinup == 0 else spinup

        # Climatic inputs and units conversions
        temp = self.tas[lb: hb + 1] - 273.15  # ! K to °C
        prec = self.pr[lb: hb + 1] * 86400  # kg m-2 s-1 to  mm/day
        # Pascal pra mbar (hPa)
        p_atm = self.ps[lb: hb + 1] * 0.01
        # W m-2 to mol m-2 s-1 ! 0.5 converts RSDS to PAR
        ipar = self.rsds[lb: hb + 1] * 0.198
        # Relative humidity (%) to unitless [0,1]
        ru = self.rhs[lb: hb + 1] / 100.0

        year0 = start.year
        co2 = find_co2(year0)
        count_days = start.dayofyr - 2
        loop = 0
        next_year = 0.0

        fix_co2_p = False
        if fix_co2 is None:
            fix_co2_p = False
        elif type(fix_co2) == int or type(fix_co2) == float:
            co2 = fix_co2
            fix_co2_p = True
        elif type(fix_co2) == str:
            assert type(int(
                fix_co2)) == int, "The string(\"yyyy\") for the fix_co2 argument must be an year between 1901-2016"
            co2 = find_co2(int(fix_co2))
            fix_co2_p = True

        for s in range(spin):
            if ABORT:
                pID = os.getpid()
                print(f'Closed process PID = {pID}\nGRD = {self.plot_name}\nCOORD = {self.pos}')
                break
            if save:
                self._allocate_output(steps.size)
                self.save = True
            else:
                self._allocate_output_nosave(steps.size)
                self.save = False
            for step in range(steps.size):
                if fix_co2_p:
                    pass
                else:
                    loop += 1
                    count_days += 1
                    # CAST DAILY CO2 ATM CONCENTRATION - LINEAR INTERPOLATION
                    days = 366 if m.leap(year0) == 1 else 365
                    if count_days == days:
                        count_days = 0
                        year0 = cftime.num2date(day_indexes[step],
                                                self.time_unit, self.calendar).year
                        co2 = find_co2(year0)
                        next_year = (find_co2(year0 + 1) - co2) / days

                    elif loop == 1 and count_days < days:
                        year0 = start.year
                        next_year = (find_co2(year0 + 1) - co2) / \
                            (days - count_days)

                    co2 += next_year

                # Update soil temperature
                self.soil_temp = st.soil_temp(self.soil_temp, temp[step])

                # AFEX
                if count_days == 364 and afex:
                    with open("afex.cfg", 'r') as afex_cfg:
                        afex_exp = afex_cfg.readlines()
                    afex_exp = afex_exp[0].strip()
                    if afex_exp == 'N':
                        # (12.5 g m-2 y-1 == 125 kg ha-1 y-1)
                        self.sp_available_n += 12.5
                    elif afex_exp == 'P':
                        # (5 g m-2 y-1 == 50 kg ha-1 y-1)
                        self.sp_available_p += 5.0
                    elif afex_exp == 'NP':
                        self.sp_available_n += 12.5
                        self.sp_available_p += 5.0

                # INFLATe VARS
                sto = np.zeros(shape=(3, npls), order='F')
                cleaf = np.zeros(npls, order='F')
                cwood = np.zeros(npls, order='F')
                croot = np.zeros(npls, order='F')
                dcl = np.zeros(npls, order='F')
                dca = np.zeros(npls, order='F')
                dcf = np.zeros(npls, order='F')
                uptk_costs = np.zeros(npls, order='F')

                sto[0, self.vp_lsid] = self.vp_sto[0, :]
                sto[1, self.vp_lsid] = self.vp_sto[1, :]
                sto[2, self.vp_lsid] = self.vp_sto[2, :]
                # Just Check the integrity of the data
                assert self.vp_lsid.size == self.vp_cleaf.size, 'different array sizes'
                c = 0
                for n in self.vp_lsid:
                    cleaf[n] = self.vp_cleaf[c]
                    cwood[n] = self.vp_cwood[c]
                    croot[n] = self.vp_croot[c]
                    dcl[n] = self.vp_dcl[c]
                    dca[n] = self.vp_dca[c]
                    dcf[n] = self.vp_dcf[c]
                    uptk_costs[n] = self.sp_uptk_costs[c]
                    c += 1
                ton = self.sp_organic_n #+ self.sp_sorganic_n
                top = self.sp_organic_p #+ self.sp_sorganic_p
                out = model.daily_budget(self.pls_table, self.wp_water_upper_mm, self.wp_water_lower_mm,
                                         self.soil_temp, temp[step], p_atm[step],
                                         ipar[step], ru[step], self.sp_available_n, self.sp_available_p,
                                         ton, top, self.sp_organic_p, co2, sto, cleaf, cwood, croot,
                                         dcl, dca, dcf, uptk_costs, self.wmax_mm)

                # del sto, cleaf, cwood, croot, dcl, dca, dcf, uptk_costs
                # Create a dict with the function output
                daily_output = catch_out_budget(out)

                self.vp_lsid = np.where(daily_output['ocpavg'] > 0.0)[0]
                self.vp_ocp = daily_output['ocpavg'][self.vp_lsid]
                self.ls[step] = self.vp_lsid.size

                if self.vp_lsid.size < 1 and not save:
                    self.vp_lsid = np.sort(
                        np.array(
                            rd.sample(list(np.arange(gp.npls)), int(gp.npls - 5))))
                    rwarn(
                        f"Gridcell {self.xyname} has no living Plant Life Strategies - Re-populating")
                    # REPOPULATE]
                    # UPDATE vegetation pools
                    self.vp_cleaf = np.zeros(shape=(self.vp_lsid.size,)) + 1.0
                    self.vp_cwood = np.zeros(shape=(self.vp_lsid.size,))
                    self.vp_croot = np.zeros(shape=(self.vp_lsid.size,)) + 1.0
                    awood = self.pls_table[6, :]
                    for i0, i in enumerate(self.vp_lsid):
                        if awood[i] > 0.0:
                            self.vp_cwood[i0] = 0.1

                    self.vp_dcl = np.zeros(shape=(self.vp_lsid.size,))
                    self.vp_dca = np.zeros(shape=(self.vp_lsid.size,))
                    self.vp_dcf = np.zeros(shape=(self.vp_lsid.size,))
                    self.vp_sto = np.zeros(shape=(3, self.vp_lsid.size))
                    self.sp_uptk_costs = np.zeros(shape=(self.vp_lsid.size,))

                    self.vp_ocp = np.zeros(shape=(self.vp_lsid.size,))
                    del awood
                    self.ls[step] = self.vp_lsid.size
                else:
                    if self.vp_lsid.size < 1:
                        ABORT = 1
                        rwarn(f"Gridcell {self.xyname} has"  + \
                               " no living Plant Life Strategies")
                    # UPDATE vegetation pools
                    self.vp_cleaf = daily_output['cleafavg_pft'][self.vp_lsid]
                    self.vp_cwood = daily_output['cawoodavg_pft'][self.vp_lsid]
                    self.vp_croot = daily_output['cfrootavg_pft'][self.vp_lsid]
                    self.vp_dcl = daily_output['delta_cveg'][0][self.vp_lsid]
                    self.vp_dca = daily_output['delta_cveg'][1][self.vp_lsid]
                    self.vp_dcf = daily_output['delta_cveg'][2][self.vp_lsid]
                    self.vp_sto = daily_output['stodbg'][:, self.vp_lsid]
                    self.sp_uptk_costs = daily_output['npp2pay'][self.vp_lsid]

                # UPDATE STATE VARIABLES
                # WATER CWM
                self.runom[step] = self.swp._update_pool(
                    prec[step], daily_output['evavg'])
                self.swp.w1 = np.float64(
                    0.0) if self.swp.w1 < 0.0 else self.swp.w1
                self.swp.w2 = np.float64(
                    0.0) if self.swp.w2 < 0.0 else self.swp.w2
                self.wp_water_upper_mm = self.swp.w1
                self.wp_water_lower_mm = self.swp.w2

                # Plant uptake and Carbon costs of nutrient uptake
                self.nupt[:, step] = daily_output['nupt']
                self.pupt[:, step] = daily_output['pupt']
                
                # CWM of STORAGE_POOL
                for i in range(3):
                    self.storage_pool[i, step] = np.sum(
                        self.vp_ocp * self.vp_sto[i])

                # OUTPUTS for SOIL CWM
                self.litter_l[step] = daily_output['litter_l'] + \
                    daily_output['cp'][3]
                self.cwd[step] = daily_output['cwd']
                self.litter_fr[step] = daily_output['litter_fr']
                self.lnc[:, step] = daily_output['lnc']
                wtot = self.wp_water_upper_mm + self.wp_water_lower_mm
                s_out = soil_dec.carbon3(self.soil_temp, wtot / self.wmax_mm, self.litter_l[step],
                                         self.cwd[step], self.litter_fr[step], self.lnc[:, step],
                                         self.sp_csoil, self.sp_snc)

                soil_out = catch_out_carbon3(s_out)

                # Organic C N & P
                self.sp_csoil = soil_out['cs']
                self.sp_snc = soil_out['snc']
                idx = np.where(self.sp_snc < 0.0)[0]
                if len(idx) > 0:
                    for i in idx:
                        self.sp_snc[i] = 0.0

                # IF NUTRICYCLE:
                if nutri_cycle:
                    # UPDATE ORGANIC POOLS
                    self.sp_organic_n = self.sp_snc[:2].sum()
                    self.sp_sorganic_n = self.sp_snc[2:4].sum()
                    self.sp_organic_p = self.sp_snc[4:6].sum()
                    self.sp_sorganic_p = self.sp_snc[6:].sum()
                    self.sp_available_p += soil_out['pmin']
                    self.sp_available_n += soil_out['nmin']
                    # NUTRIENT DINAMICS
                    # Inorganic N
                    self.sp_in_n += self.sp_available_n + self.sp_so_n
                    self.sp_so_n = soil_dec.sorbed_n_equil(self.sp_in_n)
                    self.sp_available_n = soil_dec.solution_n_equil(
                        self.sp_in_n)
                    self.sp_in_n -= self.sp_so_n + self.sp_available_n

                    # Inorganic P
                    self.sp_in_p += self.sp_available_p + self.sp_so_p
                    self.sp_so_p = soil_dec.sorbed_p_equil(self.sp_in_p)
                    self.sp_available_p = soil_dec.solution_p_equil(
                        self.sp_in_p)
                    self.sp_in_p -= self.sp_so_p + self.sp_available_p

                    # Sorbed P
                    if self.pupt[1, step] > 0.75:
                        rwarn(
                            f"Puptk_SO > soP_max - 987 | in spin{s}, step{step} - {self.pupt[1, step]}")
                        self.pupt[1, step] = 0.0

                    if self.pupt[1, step] > self.sp_so_p:
                        rwarn(
                            f"Puptk_SO > soP_pool - 992 | in spin{s}, step{step} - {self.pupt[1, step]}")

                    self.sp_so_p -= self.pupt[1, step]

                    try:
                        t1 = np.all(self.sp_snc > 0.0)
                    except:
                        if self.sp_snc is None:
                            self.sp_snc = np.zeros(shape=8,)
                            t1 = True
                        elif self.sp_snc is not None:
                            t1 = True
                        rwarn(f"Exception while handling sp_snc pool")
                    if not t1:
                        self.sp_snc[np.where(self.sp_snc < 0)[0]] = 0.0
                    # ORGANIC nutrients uptake
                    # N
                    if self.nupt[1, step] < 0.0:
                        rwarn(
                            f"NuptkO < 0 - 1003 | in spin{s}, step{step} - {self.nupt[1, step]}")
                        self.nupt[1, step] = 0.0
                    if self.nupt[1, step] > 2.5:
                        rwarn(
                            f"NuptkO  > max - 1007 | in spin{s}, step{step} - {self.nupt[1, step]}")
                        self.nupt[1, step] = 0.0

                    total_on = self.sp_snc[:4].sum()

                    if total_on > 0.0:
                        frsn = [i / total_on for i in self.sp_snc[:4]]
                    else:
                        frsn = [0.0, 0.0, 0.0, 0.0]

                    for i, fr in enumerate(frsn):
                        self.sp_snc[i] -= self.nupt[1, step] * fr

                    idx = np.where(self.sp_snc < 0.0)[0]
                    if len(idx) > 0:
                        for i in idx:
                            self.sp_snc[i] = 0.0

                    self.sp_organic_n = self.sp_snc[:2].sum()
                    self.sp_sorganic_n = self.sp_snc[2:4].sum()

                    # P
                    if self.pupt[2, step] < 0.0:
                        rwarn(
                            f"PuptkO < 0 - 1020 | in spin{s}, step{step} - {self.pupt[2, step]}")
                        self.pupt[2, step] = 0.0
                    if self.pupt[2, step] > 1.0:
                        rwarn(
                            f"PuptkO  > max - 1024 | in spin{s}, step{step} - {self.pupt[2, step]}")
                        self.pupt[2, step] = 0.0
                    total_op = self.sp_snc[4:].sum()
                    if total_op > 0.0:
                        frsp = [i / total_op for i in self.sp_snc[4:]]
                    else:
                        frsp = [0.0, 0.0, 0.0, 0.0]
                    for i, fr in enumerate(frsp):
                        self.sp_snc[i + 4] -= self.pupt[2, step] * fr

                    idx = np.where(self.sp_snc < 0.0)[0]
                    if len(idx) > 0:
                        for i in idx:
                            self.sp_snc[i] = 0.0

                    self.sp_organic_p = self.sp_snc[4:6].sum()
                    self.sp_sorganic_p = self.sp_snc[6:].sum()

                    # Raise some warnings
                    if self.sp_organic_n < 0.0:
                        self.sp_organic_n = 0.0
                        rwarn(f"ON negative in spin{s}, step{step}")
                    if self.sp_sorganic_n < 0.0:
                        self.sp_sorganic_n = 0.0
                        rwarn(f"SON negative in spin{s}, step{step}")
                    if self.sp_organic_p < 0.0:
                        self.sp_organic_p = 0.0
                        rwarn(f"OP negative in spin{s}, step{step}")
                    if self.sp_sorganic_p < 0.0:
                        self.sp_sorganic_p = 0.0
                        rwarn(f"SOP negative in spin{s}, step{step}")

                    # CALCULATE THE EQUILIBTIUM IN SOIL POOLS
                    # Soluble and inorganic pools
                    if self.pupt[0, step] > 1e2:
                        rwarn(
                            f"Puptk > max - 786 | in spin{s}, step{step} - {self.pupt[0, step]}")
                        self.pupt[0, step] = 0.0
                    self.sp_available_p -= self.pupt[0, step]

                    if self.nupt[0, step] > 1e3:
                        rwarn(
                            f"Nuptk > max - 792 | in spin{s}, step{step} - {self.nupt[0, step]}")
                        self.nupt[0, step] = 0.0
                    self.sp_available_n -= self.nupt[0, step]

                # END SOIL NUTRIENT DYNAMICS

                # # #  store (np.array) outputs
                if save:
                    assert self.save == True
                    self.carbon_costs[step] = daily_output['c_cost_cwm']
                    self.emaxm.append(daily_output['epavg'])
                    self.tsoil.append(self.soil_temp)
                    self.photo[step] = daily_output['phavg']
                    self.aresp[step] = daily_output['aravg']
                    self.npp[step] = daily_output['nppavg']
                    self.lai[step] = daily_output['laiavg']
                    self.rcm[step] = daily_output['rcavg']
                    self.f5[step] = daily_output['f5avg']
                    self.evapm[step] = daily_output['evavg']
                    self.wsoil[step] = self.wp_water_upper_mm
                    self.swsoil[step] = self.wp_water_lower_mm
                    self.rm[step] = daily_output['rmavg']
                    self.rg[step] = daily_output['rgavg']
                    self.wue[step] = daily_output['wueavg']
                    self.cue[step] = daily_output['cueavg']
                    self.cdef[step] = daily_output['c_defavg']
                    self.vcmax[step] = daily_output['vcmax']
                    self.specific_la[step] = daily_output['specific_la']
                    self.cleaf[step] = daily_output['cp'][0]
                    self.cawood[step] = daily_output['cp'][1]
                    self.cfroot[step] = daily_output['cp'][2]
                    self.hresp[step] = soil_out['hr']
                    self.csoil[:, step] = soil_out['cs']
                    self.inorg_n[step] = self.sp_in_n
                    self.inorg_p[step] = self.sp_in_p
                    self.sorbed_n[step] = self.sp_so_n
                    self.sorbed_p[step] = self.sp_so_p
                    self.snc[:, step] = soil_out['snc']
                    self.nmin[step] = self.sp_available_n
                    self.pmin[step] = self.sp_available_p
                    self.area[self.vp_lsid, step] = self.vp_ocp
                    self.lim_status[:, self.vp_lsid,
                                    step] = daily_output['limitation_status'][:, self.vp_lsid]
                    self.uptake_strategy[:, self.vp_lsid,
                                         step] = daily_output['uptk_strat'][:, self.vp_lsid]
                if ABORT:
                    rwarn("NO LIVING PLS - ABORT")
            gc.collect()
            if save:
                if s > 0:
                    while True:
                        if sv.is_alive():
                            sleep(0.5)
                        else:
                            break

                self.flush_data = self._flush_output(
                    'spin', (start_index, end_index))
                sv = Thread(target=self._save_output, args=(self.flush_data,))
                sv.start()
        if save:
            while True:
                if sv.is_alive():
                    sleep(0.5)
                else:
                    break
        gc.collect()
        return None

    def bdg_spinup(self, start_date, end_date):
        """Estimate steady-state litter and water inputs to feed the soil spinup.

        Runs a single, lightweight pass of the daily vegetation budget over
        ``[start_date, end_date]`` to obtain mean values of soil water and of
        the litter/CWD fluxes that the soil decomposition module needs to be
        spun up. No outputs are written and the per-step state-variable
        bookkeeping done by :meth:`run_caete` is intentionally skipped — only
        the values that drive :meth:`sdc_spinup` are accumulated. The result
        is then passed straight into :meth:`sdc_spinup` by the driver scripts
        to seed the soil C/N/P pools before the regular spinup begins.

        Parameters
        ----------
        start_date : str
            Inclusive start date in ``"YYYYMMDD"`` format. Must lie within
            ``[self.start_date, self.end_date]`` and be strictly before
            ``end_date``.

        end_date : str
            Inclusive end date in ``"YYYYMMDD"`` format. Must lie within
            ``[self.start_date, self.end_date]``.

        Returns
        -------
        tuple of (float, float, float, float, numpy.ndarray)
            Five summary statistics, each scaled by ``1.25`` (an empirical
            inflation factor used to bias the soil spinup toward observed
            tropical fluxes), unpacked as
            ``(water, litter_l, cwd, litter_fr, lnc)``:

            - ``water``    : mean total soil water content (mm) over the
              window, ``w1 + w2``.
            - ``litter_l`` : mean daily leaf litter input (g C m⁻² d⁻¹).
            - ``cwd``      : mean daily coarse-woody-debris input
              (g C m⁻² d⁻¹).
            - ``litter_fr``: mean daily fine-root litter input
              (g C m⁻² d⁻¹).
            - ``lnc``      : per-pool mean litter N/P concentrations,
              shape ``(6,)``, averaged along the time axis.

            These values are intended to be forwarded directly as the
            arguments of :meth:`sdc_spinup`.

        Notes
        -----
        - Sets ``self.budget_spinup = True`` as a marker for downstream
          code.
        - Uses the transient CO2 series from ``self.co2_data`` with the
          same daily linear-interpolation logic as :meth:`run_caete`; the
          ``find_co2`` parsing honours ``self.plot`` (comma-separated
          records when ``True``, whitespace otherwise).
        - Updates ``self.soil_temp``, ``self.wp_water_upper_mm``,
          ``self.wp_water_lower_mm``, and the underlying ``self.swp`` water
          pool on every step, but does **not** modify vegetation pools,
          soil C/N/P pools, or PLS occupancy.
        - The daily budget is called with the vegetation pools as-is
          (``self.vp_cleaf``, ``self.vp_cwood``, ``self.vp_croot``, etc.),
          so the gridcell must be initialized (``self.filled is True``).
        - No assertion is raised on out-of-bounds CO2 lookups; missing
          years can yield ``None`` from ``find_co2``.
        """

        assert self.filled, "The gridcell has no input data"
        self.budget_spinup = True

        if self.plot:
            splitter = ","
        else:
            splitter = None  # split on any whitespace (tabs or spaces)

        def find_co2(year):
            for i in self.co2_data:
                parts = i.split(splitter) if splitter else i.split()
                if not parts or not parts[0].strip().lstrip('-').isdigit():
                    continue
                if int(parts[0]) == year:
                    return float(parts[1].strip())

        def find_index(start, end):
            result = []
            num = np.arange(self.ssize)
            ind = np.arange(self.sind, self.eind + 1)
            for r, i in zip(num, ind):
                if i == start:
                    result.append(r)
            for r, i in zip(num, ind):
                if i == end:
                    result.append(r)
            return result

        # Define start and end dates
        start = cftime.real_datetime(int(start_date[:4]), int(
            start_date[4:6]), int(start_date[6:]))
        end = cftime.real_datetime(int(end_date[:4]), int(
            end_date[4:6]), int(end_date[6:]))
        # Check dates sanity
        assert start < end, "start > end"
        assert start >= self.start_date
        assert end <= self.end_date

        # Define time index
        start_index = int(cftime.date2num(
            start, self.time_unit, self.calendar))
        end_index = int(cftime.date2num(end, self.time_unit, self.calendar))

        lb, hb = find_index(start_index, end_index)
        steps = np.arange(lb, hb + 1)
        day_indexes = np.arange(start_index, end_index + 1)

        # Catch climatic input and make conversions
        temp = self.tas[lb: hb + 1] - 273.15  # ! K to °C
        prec = self.pr[lb: hb + 1] * 86400  # kg m-2 s-1 to  mm/day
        # transforamando de Pascal pra mbar (hPa)
        p_atm = self.ps[lb: hb + 1] * 0.01
        # W m-2 to mol m-2 s-1 ! 0.5 converts RSDS to PAR
        ipar = self.rsds[lb: hb + 1] * 0.5 / 2.18e5
        ru = self.rhs[lb: hb + 1] / 100.0

        year0 = start.year
        co2 = find_co2(year0)
        count_days = start.dayofyr - 2
        loop = 0
        next_year = 0
        wo = []
        llo = []
        cwdo = []
        rlo = []
        lnco = []

        sto = self.vp_sto
        cleaf = self.vp_cleaf
        cwood = self.vp_cwood
        croot = self.vp_croot
        dcl = self.vp_dcl
        dca = self.vp_dca
        dcf = self.vp_dcf
        uptk_costs = np.zeros(npls, order='F')

        for step in range(steps.size):
            loop += 1
            count_days += 1
            # CAST CO2 ATM CONCENTRATION
            days = 366 if m.leap(year0) == 1 else 365
            if count_days == days:
                count_days = 0
                year0 = cftime.num2date(day_indexes[step],
                                        self.time_unit, self.calendar).year
                co2 = find_co2(year0)
                next_year = (find_co2(year0 + 1) - co2) / days

            elif loop == 1 and count_days < days:
                year0 = start.year
                next_year = (find_co2(year0 + 1) - co2) / \
                    (days - count_days)

            co2 += next_year
            self.soil_temp = st.soil_temp(self.soil_temp, temp[step])

            out = model.daily_budget(self.pls_table, self.wp_water_upper_mm, self.wp_water_lower_mm,
                                     self.soil_temp, temp[step], p_atm[step],
                                     ipar[step], ru[step], self.sp_available_n, self.sp_available_p,
                                     self.sp_snc[:4].sum(
                                     ), self.sp_so_p, self.sp_snc[4:].sum(),
                                     co2, sto, cleaf, cwood, croot,
                                     dcl, dca, dcf, uptk_costs, self.wmax_mm)

            # Create a dict with the function output
            daily_output = catch_out_budget(out)
            runoff = self.swp._update_pool(prec[step], daily_output['evavg'])

            self.wp_water_upper_mm = self.swp.w1
            self.wp_water_lower_mm = self.swp.w2
            # UPDATE vegetation pools

            wo.append(np.float64(self.wp_water_upper_mm + self.wp_water_lower_mm))
            llo.append(daily_output['litter_l'])
            cwdo.append(daily_output['cwd'])
            rlo.append(daily_output['litter_fr'])
            lnco.append(daily_output['lnc'])

        f = np.array
        def x(a): return a * 1.25
        return x(f(wo).mean()), x(f(llo).mean()), x(f(cwdo).mean()), x(f(rlo).mean()), x(f(lnco).mean(axis=0,))

    def sdc_spinup(self, water, ll, cwd, rl, lnc):
        """Spin up the soil decomposition pools from steady-state litter inputs.

        Iterates the Fortran ``soil_dec.carbon3`` solver 3000 times with a
        fixed (mean) climatic and litter forcing so that the soil carbon and
        soil N/P pools (``self.sp_csoil`` and ``self.sp_snc``) converge
        toward an equilibrium consistent with those inputs. The forcing is
        normally produced by :meth:`bdg_spinup`, so the typical driver call
        is::

            w, ll, cwd, rl, lnc = grd.bdg_spinup(start, end)
            grd.sdc_spinup(w, ll, cwd, rl, lnc)

        After this call ``self.sp_csoil`` and ``self.sp_snc`` are ready to be
        used as initial conditions for the regular vegetation+soil spinup in
        :meth:`run_caete`.

        Parameters
        ----------
        water : float
            Mean total soil water content (mm) used to evaluate the soil
            moisture factor ``water / self.wmax_mm`` passed to
            ``soil_dec.carbon3``. Typically the first element of the tuple
            returned by :meth:`bdg_spinup`.

        ll : float
            Mean daily leaf-litter carbon input (g C m⁻² d⁻¹).

        cwd : float
            Mean daily coarse-woody-debris carbon input (g C m⁻² d⁻¹).

        rl : float
            Mean daily fine-root-litter carbon input (g C m⁻² d⁻¹).

        lnc : numpy.ndarray
            Per-pool mean litter N/P concentrations, shape ``(6,)``, as
            returned by :meth:`bdg_spinup`.

        Returns
        -------
        None
            The method mutates ``self.sp_csoil`` and ``self.sp_snc`` in
            place.

        Notes
        -----
        - The fixed iteration count (3000) is hard-coded and intended to be
          large enough for the litter pools to reach a numerical
          steady state under the constant forcing.
        - ``self.soil_temp`` is treated as constant during the spinup; no
          climate update is performed inside the loop.
        - Does **not** touch vegetation pools, water pools, or
          ``self.outputs``; only the soil C and N/P pools are evolved.
        """

        for x in range(3000):

            s_out = soil_dec.carbon3(self.soil_temp, water / self.wmax_mm, ll, cwd, rl, lnc,
                                     self.sp_csoil, self.sp_snc)

            soil_out = catch_out_carbon3(s_out)
            self.sp_csoil = soil_out['cs']
            self.sp_snc = soil_out['snc']


class plot(grd):
    """i and j are the latitude and longitude (in that order) of plot location in decimal degrees"""

    def __init__(self, i, j, dump_folder):
        y, x = find_coord(i, j)
        super().__init__(x, y, dump_folder)

        self.plot = True

    def init_plot(self, sdata, stime_i, co2, pls_table, tsoil, ssoil, hsoil):
        """ PREPARE A GRIDCELL TO RUN With PLOT OBSERVED DATA
            sdata : python dict with the proper structure - see the input files e.g. CAETE-DVM/input/central/input_data_175-235.pbz2
            stime_i:  python dict with the proper structure - see the input files e.g. CAETE-DVM/input/central/ISIMIP_HISTORICAL_METADATA.pbz2
            These dicts are build upon .csv climatic data in the file CAETE-DVM/src/k34_experiment.py where you can find an application of the plot class 
            co2: (list) a alist (association list) with yearly cCO2 ATM data(yyyy\t[CO2]\n)
            pls_table: np.ndarray with functional traits of a set of PLant life strategies
            tsoil, ssoil, hsoil: numpy arrays with soil parameters see the file CAETE-DVM/src/k34_experiment.py
        """

        assert self.filled == False, "already done"

        self.data = sdata

        os.makedirs(self.out_dir, exist_ok=True)
        self.flush_data = 0

        self.pr = self.data['pr']
        self.ps = self.data['ps']
        self.rsds = self.data['rsds']
        self.tas = self.data['tas']
        self.rhs = self.data['hurs']

        # SOIL AND NUTRIENTS
        self.input_nut = []
        self.nutlist = ['tn', 'tp', 'ap', 'ip', 'op']
        for nut in self.nutlist:
            self.input_nut.append(self.data[nut])
        self.soil_dict = dict(zip(self.nutlist, self.input_nut))
        self.data = None

        # TIME
        self.stime = copy.deepcopy(stime_i)
        self.calendar = self.stime['calendar']
        self.time_index = self.stime['time_index']
        self.time_unit = self.stime['units']
        self.ssize = self.time_index.size
        self.sind = int(self.time_index[0])
        self.eind = int(self.time_index[-1])
        self.start_date = cftime.num2date(
            self.time_index[0], self.time_unit, calendar=self.calendar)
        self.end_date = cftime.num2date(
            self.time_index[-1], self.time_unit, calendar=self.calendar)

        # OTHER INPUTS
        self.pls_table = pls_table.copy()
        self.neighbours = neighbours_index(self.pos, mask)
        self.soil_temp = st.soil_temp_sub(self.tas[:1095] - 273.15)

        # Prepare co2 inputs (we have annually means)
        self.co2_data = copy.deepcopy(co2)

        self.tsoil = []
        self.emaxm = []

        # STATE
        # Water
        ws1 = tsoil[0][self.y, self.x].copy()
        fc1 = tsoil[1][self.y, self.x].copy()
        wp1 = tsoil[2][self.y, self.x].copy()

        ws2 = ssoil[0][self.y, self.x].copy()
        fc2 = ssoil[1][self.y, self.x].copy()
        wp2 = ssoil[2][self.y, self.x].copy()

        self.swp = soil_water(ws1, ws2, fc1, fc2, wp1, wp2)
        self.wp_water_upper_mm = self.swp.w1
        self.wp_water_lower_mm = self.swp.w2
        self.wmax_mm = np.float64(self.swp.w1_max + self.swp.w2_max)

        self.theta_sat = hsoil[0][self.y, self.x].copy()
        self.psi_sat = hsoil[1][self.y, self.x].copy()
        self.soil_texture = hsoil[2][self.y, self.x].copy()

        # Biomass
        self.vp_cleaf = np.zeros(shape=(npls,), order='F') + 1.0
        self.vp_croot = np.zeros(shape=(npls,), order='F') + 1.0
        self.vp_cwood = np.zeros(shape=(npls,), order='F') + 0.1
        self.vp_cwood[pls_table[6,:] == 0.0] = 0.0

        a, b, c, d = m.pft_area_frac(
            self.vp_cleaf, self.vp_croot, self.vp_cwood, self.pls_table[6, :])
        self.vp_lsid = np.where(a > 0.0)[0]
        self.ls = self.vp_lsid.size
        del a, b, c, d
        self.vp_dcl = np.zeros(shape=(npls,), order='F')
        self.vp_dca = np.zeros(shape=(npls,), order='F')
        self.vp_dcf = np.zeros(shape=(npls,), order='F')
        self.vp_ocp = np.zeros(shape=(npls,), order='F')
        self.vp_sto = np.zeros(shape=(3, npls), order='F')

        # # # SOIL
        self.sp_csoil = np.zeros(shape=(4,), order='F') + 1.0
        self.sp_snc = np.zeros(shape=(8,), order='F') + 0.1
        self.sp_available_p = self.soil_dict['ap']
        self.sp_available_n = 0.2 * self.soil_dict['tn']
        self.sp_in_n = 0.4 * self.soil_dict['tn']
        self.sp_so_n = 0.2 * self.soil_dict['tn']
        self.sp_so_p = self.soil_dict['tp'] - sum(self.input_nut[2:])
        self.sp_in_p = self.soil_dict['ip']
        self.sp_uptk_costs = np.zeros(npls, order='F')
        self.sp_organic_n = 0.1 * self.soil_dict['tn']
        self.sp_sorganic_n = 0.1 * self.soil_dict['tn']
        self.sp_organic_p = 0.5 * self.soil_dict['op']
        self.sp_sorganic_p = self.soil_dict['op'] - self.sp_organic_p

        self.outputs = dict()
        self.filled = True

        return None
