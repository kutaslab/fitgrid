import pytest
import numpy as np
import pandas as pd
import uuid
import os
from pathlib import Path
from .context import fitgrid, tpath
from fitgrid.errors import FitGridError
from fitgrid.fitgrid import FitGrid
from fitgrid import tools
import matplotlib

matplotlib.use('Agg')


def test__correct_channels_in_fitgrid():
    epochs = fitgrid.generate()
    LHS = ['channel0', 'channel1', 'channel2']
    grid = epochs.lm(LHS=LHS, RHS='categorical + continuous')
    assert grid.channels == LHS


def test__method_returning_dataframe_expands_correctly():

    epochs = fitgrid.generate()
    LHS = ['channel0', 'channel1', 'channel2']
    grid = epochs.lm(LHS=LHS, RHS='categorical + continuous')

    conf_int = grid.conf_int()
    assert (conf_int.columns == LHS).all()
    assert (
        conf_int.index.levels[1]
        == ['Intercept', 'categorical[T.cat1]', 'continuous']
    ).all()
    assert (conf_int.index.levels[2] == [0, 1]).all()
    assert (conf_int.dtypes == float).all()


def test__residuals_have_long_form_and_correct_index():

    n_epochs = 10
    n_categories = 2
    n_samples = 100

    epochs = fitgrid.generate(
        n_epochs=n_epochs, n_categories=n_categories, n_samples=n_samples
    )

    new_index = np.repeat(
        np.random.permutation(np.arange(n_categories * n_epochs)), n_samples
    )

    epochs.table.index = new_index

    grid = epochs.lm(
        LHS=['channel0', 'channel1', 'channel2'],
        RHS='categorical + continuous',
    )

    single_epoch = tools.get_first_group(epochs._snapshots)

    assert single_epoch.index.equals(grid.resid.index.levels[1])


def test__epoch_id_substitution():
    """When we get a numpy array/tuple/list, we try to use EPOCH_ID index.
    See github.com/kutaslab/fitgrid/issues/25.
    """

    from fitgrid import EPOCH_ID

    # create data with unusual index (shifted by 5)
    data, channels = fitgrid.fake_data._generate(10, 100, 2, 32)
    unusual_index = np.arange(20) + 5
    data.index.set_levels(unusual_index, level=EPOCH_ID, inplace=True)
    epochs = fitgrid.epochs_from_dataframe(data, channels)

    # remember epoch_index
    epoch_index = tools.get_first_group(epochs._snapshots).index
    assert (epoch_index == unusual_index).all()

    # take just two channels for speed
    LHS = channels[:2]
    grid = epochs.lm(LHS=LHS, RHS='categorical + continuous')

    # one additional level
    resid_pearson = grid.resid_pearson
    assert resid_pearson.index.levels[1].equals(epoch_index)
    assert (resid_pearson.index.levels[1] == epoch_index).all()
    assert resid_pearson.index.names[1] == EPOCH_ID

    # now we retrieve cooks_d and expect that EPOCH_ID is correct and named
    influence = grid.get_influence()

    # two additional levels
    cooks_d = influence.cooks_distance
    assert cooks_d.index.levels[2].equals(epoch_index)
    assert (cooks_d.index.levels[2] == epoch_index).all()
    assert cooks_d.index.names[2] == EPOCH_ID

    # one additional level
    cov_ratio = influence.cov_ratio
    assert cov_ratio.index.levels[1].equals(epoch_index)
    assert (cov_ratio.index.levels[1] == epoch_index).all()
    assert cov_ratio.index.names[1] == EPOCH_ID


def test__slicing():

    epochs = fitgrid.generate()
    grid = epochs.lm(
        LHS=['channel0', 'channel1', 'channel2'],
        RHS='categorical + continuous',
    )

    subgrid = grid[25, 'channel0']
    assert (subgrid._grid.columns == ['channel0']).all()
    assert (subgrid._grid.index == [25]).all()
    assert subgrid._grid is not grid._grid

    subgrid = grid[25:75, 'channel0']
    assert (subgrid._grid.columns == ['channel0']).all()
    assert (subgrid._grid.index == list(range(25, 76))).all()
    assert subgrid._grid is not grid._grid

    subgrid = grid[25:75, ['channel0', 'channel2']]
    assert (subgrid._grid.columns == ['channel0', 'channel2']).all()
    assert (subgrid._grid.index == list(range(25, 76))).all()
    assert subgrid._grid is not grid._grid

    with pytest.warns(UserWarning):
        subgrid = grid[25:75, ['channel1', 'channel0', 'channel1', 'channel2']]
    infl = subgrid.get_influence()
    infl.cooks_distance
    assert (
        subgrid._grid.columns == ['channel1', 'channel0', 'channel2']
    ).all()
    assert (subgrid._grid.index == list(range(25, 76))).all()
    assert subgrid._grid is not grid._grid

    subgrid = grid[:, :]
    assert (subgrid._grid.columns == grid._grid.columns).all()
    assert (subgrid._grid.index == grid._grid.index).all()
    assert subgrid._grid is not grid._grid

    with pytest.raises(FitGridError):
        grid[25]

    with pytest.raises(FitGridError):
        grid[25:75]

    with pytest.raises(FitGridError):
        grid['channel0']

    with pytest.raises(FitGridError):
        grid[['channel0', 'channel1']]


def test_grid_with_duplicate_channels():

    epochs = fitgrid.generate()
    grid = epochs.lm(
        LHS=['channel0', 'channel1', 'channel2'],
        RHS='categorical + continuous',
    )
    _grid_with_duplicate_channels = pd.concat(
        [grid._grid, grid._grid['channel0']], axis=1
    )
    with pytest.raises(FitGridError) as error:
        FitGrid(_grid_with_duplicate_channels, grid._epoch_index)

    assert "Duplicate column names" in str(error.value)


def test__smoke_influential_epochs():

    epochs = fitgrid.generate(n_channels=3, n_samples=10)
    grid = epochs.lm(
        LHS=['channel0', 'channel1', 'channel2'],
        RHS='categorical + continuous',
    )
    grid.influential_epochs()


def test__smoke_plot_betas():

    epochs = fitgrid.generate(n_channels=3, n_samples=10)
    grid = epochs.lm(
        LHS=['channel0', 'channel1', 'channel2'],
        RHS='categorical + continuous',
    )
    grid.plot_betas()


def test__smoke_plot_betas_legend_on_bottom():

    epochs = fitgrid.generate(n_channels=3, n_samples=10)
    grid = epochs.lm(
        LHS=['channel0', 'channel1', 'channel2'],
        RHS='categorical + continuous',
    )
    grid.plot_betas(legend_on_bottom=True)


# https://github.com/kutaslab/fitgrid/issues/51
def test__plot_betas_single_channel():

    epochs = fitgrid.generate(n_samples=2, n_channels=1)

    grid = epochs.lm(RHS='categorical + continuous')
    grid.plot_betas()


def test__smoke_plot_adj_rsquared():

    epochs = fitgrid.generate(n_channels=3, n_samples=10)
    grid = epochs.lm(
        LHS=['channel0', 'channel1', 'channel2'],
        RHS='categorical + continuous',
    )
    grid.plot_adj_rsquared()


def test__save_load_grid_lm(tpath):

    epochs = fitgrid.generate(n_samples=2, n_channels=2)
    grid = epochs.lm(RHS='categorical + continuous')

    TEST_FILENAME = tpath / 'data' / str(uuid.uuid4())
    grid.save(TEST_FILENAME)

    loaded_grid = fitgrid.load_grid(TEST_FILENAME)

    assert dir(grid) == dir(loaded_grid)
    assert grid.params.equals(loaded_grid.params)

    os.remove(TEST_FILENAME)


def test__save_load_grid_lmer(tpath):

    epochs = fitgrid.generate(n_samples=2, n_channels=1)
    grid = epochs.lmer(RHS='continuous + (continuous | categorical)')

    TEST_FILENAME = str(tpath / 'data' / str(uuid.uuid4()))
    grid.save(TEST_FILENAME)

    loaded_grid = fitgrid.load_grid(TEST_FILENAME)

    assert dir(grid) == dir(loaded_grid)
    assert grid.coefs.equals(loaded_grid.coefs)

    os.remove(TEST_FILENAME)
