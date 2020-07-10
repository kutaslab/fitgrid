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
BLAS = 'blas'  # matches libopenblas, libcblas


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

        if kind not in (MKL, BLAS):
            raise ValueError(
                f'kind must be {MKL} or {BLAS}, got {kind} instead.'
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
        if self.kind == BLAS:
            kind = 'OpenBLAS'
        n_threads = self.get_n_threads()
        return f'{kind} @ {n_threads} threads'


def get_blas_osys(numpy_module, osys):

    NUMPY_PATH = os.path.join(numpy_module.__path__[0], 'core')
    MULTIARRAY_PATH = glob.glob(
        os.path.join(NUMPY_PATH, '_multiarray_umath*.so')
    )[0]

    if osys == 'linux':
        COMMAND = 'ldd'
        LDD_ARGS = [COMMAND, MULTIARRAY_PATH]
        PATTERN = r'^\t.*{}.* => (?P<path>.*) \(0x.*$'

    elif osys == 'darwin':
        COMMAND = 'otool'
        FLAGS = '-L'
        LDD_ARGS = [COMMAND, FLAGS, MULTIARRAY_PATH]

        # PATTERN = r'^\t@loader_path/(?P<path>.*{}.*) \(.*\)$'
        # MacOS 10.13.6 otools shows @rpath not @loader_path
        # for the conda installed mkl.
        PATTERN = r'^\t@.*path/(?P<path>.*{}.*) \(.*\)$'
    else:
        raise ValueError(f'get_blas_osys() does not support osys={osys}')

    ldd_result = subprocess.run(
        args=LDD_ARGS,
        check=True,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )

    output = ldd_result.stdout

    kinds = [MKL, BLAS]
    for kind in kinds:
        match = re.search(PATTERN.format(kind), output, flags=re.MULTILINE)
        if match:
            path = match.groupdict()['path']
            cdll = ctypes.CDLL(path)
            return BLAS(cdll, kind)

    # unknown kind
    return None


def get_blas(numpy_module):
    """Return BLAS object or None if neither MKL nor OpenBLAS is found."""

    if sys.platform.startswith('linux'):
        # return get_blas_linux(numpy_module)
        return get_blas_osys(numpy_module, 'linux')
    elif sys.platform == 'darwin':
        # return get_blas_mac(numpy_module)
        return get_blas_osys(numpy_module, 'darwin')

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


def design_matrix_is_constant(df, columns, time):
    """Check that values in columns of df do not change within any epoch.

    See Notes for more details.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to check
    columns : list of str
        list of column names to be checked
    time : str
        name of the time column

    Returns
    -------
    result : bool
        True if values in specified columns don't change, False otherwise


    Notes
    -----

    We check that from timepoint to timepoint, each epoch has the same value in
    a given column:


    .. table:: epoch1
       :widths: auto

       === ===
       A    B
       === ===
       1    x
       1    x
       1    x
       1    x
       1    x
       === ===


    .. table:: epoch2
       :widths: auto

       === ===
       A    B
       === ===
       2    y
       2    y
       2    y
       2    y
       2    y
       === ===


    This is helpful when performing linear regression on an epochs table where
    the predictors vary with epochs (as they are expected to) but stay constant
    from sample to sample, because we can do our modeling much faster.
    """
    gb = df.groupby(time)
    _, group = next(iter(gb))  # first group

    df_columns_values = df[columns].values
    group_columns_values = group[columns].values

    expected_df_values = np.repeat(group_columns_values, len(gb), axis=0)

    return (df_columns_values == expected_df_values).all()
