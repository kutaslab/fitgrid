import pandas as pd
import re
import sys

from .context import fitgrid
from fitgrid import tools


def test_duplicate_index_printer():

    df = pd.DataFrame(
        {'a': [1, 2, 3, 4, 5], 'b': [1, 1, 2, 3, 2], 'c': [0, 1, 2, 3, 4]}
    )

    single_level_df = df.set_index('b')

    table = tools.get_index_duplicates_table(single_level_df, 'b')

    assert re.search(r'1\s+0, 1\n', table) is not None
    assert re.search(r'2\s+2, 4\n', table) is not None


def test_blas_getter():

    import numpy

    blas = tools.get_blas(numpy)

    blas.set_n_threads(4)
    assert blas.get_n_threads() == 4

    blas.set_n_threads(2)
    assert blas.get_n_threads() == 2

def test_single_threaded_no_change():

    import numpy

    blas = tools.get_blas(numpy)
    old_n_threads = blas.get_n_threads()

    with tools.single_threaded(numpy):
        assert blas.get_n_threads() == 1

    assert blas.get_n_threads() == old_n_threads

def test_single_threaded_change_before():

    import numpy

    blas = tools.get_blas(numpy)

    BEFORE = 2
    blas.set_n_threads(BEFORE)

    assert blas.get_n_threads() == BEFORE

    with tools.single_threaded(numpy):
        assert blas.get_n_threads() == 1

    assert blas.get_n_threads() == BEFORE

    BEFORE = 3
    blas.set_n_threads(BEFORE)

    assert blas.get_n_threads() == BEFORE

    with tools.single_threaded(numpy):
        assert blas.get_n_threads() == 1

    assert blas.get_n_threads() == BEFORE
