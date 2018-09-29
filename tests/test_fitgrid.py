import pytest
import numpy as np
from .context import fitgrid
from fitgrid.errors import FitGridError
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

    single_epoch = epochs.snapshots.get_group(0)
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
    epoch_index = epochs.snapshots.get_group(0).index
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


def test__smoke_influential_epochs():

    epochs = fitgrid.generate()
    grid = epochs.lm(
        LHS=['channel0', 'channel1', 'channel2'],
        RHS='categorical + continuous',
    )
    grid.influential_epochs()


def test__smoke_plot_betas():

    epochs = fitgrid.generate()
    grid = epochs.lm(
        LHS=['channel0', 'channel1', 'channel2'],
        RHS='categorical + continuous',
    )
    grid.plot_betas()


def test__smoke_plot_adj_rsquared():

    epochs = fitgrid.generate()
    grid = epochs.lm(
        LHS=['channel0', 'channel1', 'channel2'],
        RHS='categorical + continuous',
    )
    grid.plot_adj_rsquared()
