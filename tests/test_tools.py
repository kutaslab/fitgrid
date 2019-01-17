import pandas as pd
import re
import os
import pytest

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


def test_deduplicate_list():

    # should remove duplicates and preserve order
    l = [1, 1, 2, 3, 1, 2, 4]
    assert tools.deduplicate_list(l) == [1, 2, 3, 4]


@pytest.mark.skipif(
    'TRAVIS' in os.environ,
    reason='https://github.com/kutaslab/fitgrid/issues/86',
)
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


@pytest.mark.skipif(
    'TRAVIS' in os.environ,
    reason='https://github.com/kutaslab/fitgrid/issues/86',
)
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


def test_design_matrix_is_constant():

    df = pd.DataFrame(
        dict(
            epoch_id=[
                0,  # epoch 0
                0,
                0,
                0,
                1,  # epoch 1
                1,
                1,
                1,
                2,  # epoch 2
                2,
                2,
                2,
            ],
            time=[
                0,  # epoch 0
                1,
                2,
                3,
                0,  # epoch 1
                1,
                2,
                3,
                0,  # epoch 2
                1,
                2,
                3,
            ],
            b=[
                'a',  # epoch 0
                'a',
                'a',
                'a',
                'b',  # epoch 1
                'b',
                'b',
                'b',
                'c',  # epoch 2
                'c',
                'c',
                'c',
            ],
            c=[
                10,  # epoch 0
                10,
                10,
                10,
                20,  # epoch 1
                20,
                20,
                20,
                30,  # epoch 2
                30,
                30,
                30,
            ],
        )
    ).set_index(['epoch_id', 'time'])

    assert tools.design_matrix_is_constant(df, ['b', 'c'], 'time')

    assert not tools.design_matrix_is_constant(
        df.sample(frac=1), ['b', 'c'], 'time'
    )

    assert tools.design_matrix_is_constant(
        df.sample(frac=1).sort_index(), ['b', 'c'], 'time'
    )

    df['c'] = [
        0,  # epoch 0
        1,
        2,
        3,
        4,  # epoch 1
        5,
        6,
        7,
        8,  # epoch 2
        9,
        10,
        11,
    ]
    assert not tools.design_matrix_is_constant(df, ['b', 'c'], 'time')

    df['c'] = [
        0,  # epoch 0
        0,
        0,
        0,
        1,  # epoch 1
        1,
        10000,
        1,
        2,  # epoch 2
        2,
        2,
        2,
    ]
    assert not tools.design_matrix_is_constant(df, ['b', 'c'], 'time')
