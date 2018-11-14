import numpy as np
from collections import defaultdict, OrderedDict
import subprocess
import re
import ctypes
import sys
import os
import glob
import warnings

MKL = 'mkl'
OPENBLAS = 'openblas'


def get_index_duplicates_table(df, level):
    """Return a string table of duplicate index values and their locations."""

    assert level in df.index.names

    level_values = df.index.get_level_values(level)
    dupe_mask = level_values.duplicated(keep=False)

    dupes = level_values[dupe_mask]
    dupe_indices = np.flatnonzero(dupe_mask)

    dupe_dict = defaultdict(list)
    for index, value in zip(dupe_indices, dupes):
        dupe_dict[value].append(index)

    # hardcoded padding
    msg = f'Duplicates in index level {level}:\n\n'
    msg += '{0:<20} {1}\n'.format(level, 'Locations')
    for value, locations in sorted(dupe_dict.items()):
        locations = (str(item) for item in locations)
        msg += '{0:<20} {1}\n'.format(value, ', '.join(locations))

    return msg


def get_first_group(groupby):

    first_group_name = list(groupby.groups)[0]
    first_group = groupby.get_group(first_group_name)
    return first_group


def deduplicate_list(lst):
    return list(OrderedDict.fromkeys(lst))


class BLAS:
    def __init__(self, cdll, kind):

        if kind not in (MKL, OPENBLAS):
            raise ValueError(
                f'kind must be {MKL} or {OPENBLAS}, got {kind} instead.'
            )

        self.kind = kind
        self.cdll = cdll

        if kind == MKL:
            self.get_n_threads = cdll.MKL_Get_Max_Threads
            self.set_n_threads = cdll.MKL_Set_Num_Threads
        else:
            self.get_n_threads = cdll.openblas_get_num_threads
            self.set_n_threads = cdll.openblas_set_num_threads

    def __repr__(self):
        if self.kind == MKL:
            kind = 'MKL'
        if self.kind == OPENBLAS:
            kind = 'OpenBLAS'
        n_threads = self.get_n_threads()
        return f'{kind} @ {n_threads} threads'


def get_blas_mac(numpy_module):

    COMMAND = 'otool'
    FLAGS = '-L'
    PATTERN = r'^\t@loader_path/(?P<path>.*{}.*) \(.*\)$'

    NUMPY_PATH = os.path.join(numpy_module.__path__[0], 'core')
    MULTIARRAY_PATH = glob.glob(os.path.join(NUMPY_PATH, 'multiarray.*so'))[0]

    otool_result = subprocess.run(
        args=[COMMAND, FLAGS, MULTIARRAY_PATH],
        check=True,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )

    output = otool_result.stdout

    if MKL in output:
        kind = MKL
    elif OPENBLAS in output:
        kind = OPENBLAS
    else:
        return None

    pattern = PATTERN.format(kind)
    match = re.search(pattern, output, flags=re.MULTILINE)

    if match:
        rel_path = match.groupdict()['path']
        abs_path = os.path.join(NUMPY_PATH, rel_path)
        cdll = ctypes.CDLL(abs_path)
        return BLAS(cdll, kind)
    else:
        return None


def get_blas_linux(numpy_module):

    COMMAND = 'ldd'
    PATTERN = r'^\t.*{}.* => (?P<path>.*) \(0x.*$'

    NUMPY_PATH = os.path.join(numpy_module.__path__[0], 'core')
    MULTIARRAY_PATH = glob.glob(os.path.join(NUMPY_PATH, 'multiarray.*so'))[0]

    ldd_result = subprocess.run(
        args=[COMMAND, MULTIARRAY_PATH],
        check=True,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )

    output = ldd_result.stdout

    if MKL in output:
        kind = MKL
    elif OPENBLAS in output:
        kind = OPENBLAS
    else:
        return None

    pattern = PATTERN.format(kind)
    match = re.search(pattern, output, flags=re.MULTILINE)

    if match:
        path = match.groupdict()['path']
        cdll = ctypes.CDLL(path)
        return BLAS(cdll, kind)
    else:
        return None


def get_blas(numpy_module):
    """Return BLAS object or None if neither MKL nor OpenBLAS is found."""

    if sys.platform.startswith('linux'):
        return get_blas_linux(numpy_module)
    elif sys.platform == 'darwin':
        return get_blas_mac(numpy_module)

    warnings.warn(
        f'Searching for BLAS libraries on {sys.platform} is not supported.'
    )


class single_threaded:
    def __init__(self, numpy_module):
        self.blas = get_blas(numpy_module)

    def __enter__(self):
        if self.blas is not None:
            self.old_n_threads = self.blas.get_n_threads()
            self.blas.set_n_threads(1)
        else:
            warnings.warn(
                'No MKL/OpenBLAS found, assuming NumPy is single-threaded.'
            )

    def __exit__(self, *args):
        if self.blas is not None:
            self.blas.set_n_threads(self.old_n_threads)
            if self.blas.get_n_threads() != self.old_n_threads:
                message = (
                    f'Failed to reset {self.blas.kind} '
                    f'to {self.old_n_threads} threads (previous value).'
                )
                raise RuntimeError(message)
