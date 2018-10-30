import numpy as np
from collections import defaultdict
import subprocess
import re
import ctypes
import os
import glob

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
        return f'{kind} implementation of BLAS'


def get_blas_library(numpy_module):

    LDD = 'ldd'
    LDD_PATTERN = r'^\t(?P<lib>.*{}.*) => (?P<path>.*) \(0x.*$'

    NUMPY_PATH = os.path.join(numpy_module.__path__[0], 'core')
    MULTIARRAY_PATH = glob.glob(os.path.join(NUMPY_PATH, 'multiarray*.so'))[0]

    ldd_result = subprocess.run(
        args=[LDD, MULTIARRAY_PATH],
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
        raise RuntimeError('Failed to detect MKL/OpenBLAS.')

    pattern = LDD_PATTERN.format(kind)
    match = re.search(pattern, output, flags=re.MULTILINE)

    if match:
        lib = ctypes.CDLL(match.group(2))
        return BLAS(lib, kind)
    else:
        raise RuntimeError('Failed to detect MKL/OpenBLAS.')
