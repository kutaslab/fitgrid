import pytest
import pandas as pd
from statsmodels.formula.api import ols
from .context import fitgrid
from fitgrid.errors import FitGridError
from fitgrid.fitgrid import LMFitGrid, LMERFitGrid
import matplotlib

matplotlib.use('Agg')


def test_epochs_lm_bad_inputs():

    epochs = fitgrid.generate()
    with pytest.raises(FitGridError):
        fitgrid.lm(epochs, LHS=['bad_channel'], RHS='categorical')


def test_lm_correctness():
    """Probe grid to check that correct results are in the right cells."""

    epochs = fitgrid.generate(n_samples=10)

    RHS = 'continuous + categorical'
    grid = fitgrid.lm(epochs, RHS=RHS)

    assert isinstance(grid, LMFitGrid)

    timepoints = [2, 3, 7]
    channels = ['channel1', 'channel2', 'channel4']

    rsquared = grid.rsquared
    params = grid.params

    table = epochs.table.reset_index().set_index('Time')

    for timepoint in timepoints:
        for channel in channels:
            data = table.loc[timepoint]
            fit = ols(channel + ' ~ ' + RHS, data).fit()
            assert fit.params.equals(params.loc[timepoint, channel])
            assert fit.rsquared == rsquared.loc[timepoint, channel]


def test_lm_correctness_parallel():
    """Probe grid to check that correct results are in the right cells."""

    epochs = fitgrid.generate(n_samples=10)

    RHS = 'continuous + categorical'
    grid = fitgrid.lm(epochs, RHS=RHS, parallel=True, n_cores=2)

    assert isinstance(grid, LMFitGrid)

    timepoints = [2, 3, 7]
    channels = ['channel1', 'channel2', 'channel4']

    rsquared = grid.rsquared
    params = grid.params

    table = epochs.table.reset_index().set_index('Time')

    for timepoint in timepoints:
        for channel in channels:
            data = table.loc[timepoint]
            fit = ols(channel + ' ~ ' + RHS, data).fit()
            assert fit.params.equals(params.loc[timepoint, channel])
            assert fit.rsquared == rsquared.loc[timepoint, channel]


def test_smoke_lmer():

    epochs = fitgrid.generate(n_samples=2, n_channels=2)
    grid = fitgrid.lmer(epochs, RHS='(continuous | categorical)')

    assert isinstance(grid, LMERFitGrid)

    assert grid.has_warning.dtypes.all() == bool
    assert grid.warning.dtypes.all() == object

    grid.coefs


def test_lmer_runs_correct_channels():

    epochs = fitgrid.generate(n_samples=2, n_channels=3)

    LHS = ['channel0', 'channel2']
    grid = fitgrid.lmer(epochs, LHS=LHS, RHS='(continuous | categorical)')

    assert isinstance(grid, LMERFitGrid)
    assert list(grid._grid.columns) == LHS


def test_lmer_no_REML():

    epochs = fitgrid.generate(n_samples=2, n_channels=2)
    grid = fitgrid.lmer(epochs, RHS='(continuous | categorical)', REML=False)

    assert isinstance(grid, LMERFitGrid)
    assert (grid._REML == False).all().all()


def test_lmer_correctness():
    """Probe grid to check that correct results are in the right cells."""

    from pymer4 import Lmer

    epochs = fitgrid.generate(n_samples=2, n_channels=2)

    RHS = 'continuous + (continuous | categorical)'
    grid = fitgrid.lmer(epochs, RHS=RHS)

    assert isinstance(grid, LMERFitGrid)

    timepoints = [0, 1]
    channels = ['channel0', 'channel1']

    coefs = grid.coefs
    aic = grid.AIC

    table = epochs.table.reset_index().set_index('Time')

    for timepoint in timepoints:
        for channel in channels:
            data = table.loc[timepoint]
            lmer = Lmer(channel + ' ~ ' + RHS, data)
            lmer.fit(summarize=False)
            pd.testing.assert_frame_equal(
                lmer.coefs,
                coefs.loc[timepoint, channel].unstack(),
                # Sig has dtype object and in the grid floats get upgraded to
                # obj, but up to dtype they are still equal
                check_dtype=False,
            )
            assert lmer.AIC == aic.loc[timepoint, channel]


def test_lmer_correctness_parallel():
    """Probe grid to check that correct results are in the right cells."""

    from pymer4 import Lmer

    epochs = fitgrid.generate(n_samples=2, n_channels=2)

    RHS = 'continuous + (continuous | categorical)'
    grid = fitgrid.lmer(epochs, RHS=RHS, parallel=True, n_cores=2)

    assert isinstance(grid, LMERFitGrid)

    timepoints = [0, 1]
    channels = ['channel0', 'channel1']

    coefs = grid.coefs
    aic = grid.AIC

    table = epochs.table.reset_index().set_index('Time')

    for timepoint in timepoints:
        for channel in channels:
            data = table.loc[timepoint]
            lmer = Lmer(channel + ' ~ ' + RHS, data)
            lmer.fit(summarize=False)
            pd.testing.assert_frame_equal(
                lmer.coefs,
                coefs.loc[timepoint, channel].unstack(),
                # Sig has dtype object and in the grid floats get upgraded to
                # obj, but up to dtype they are still equal
                check_dtype=False,
            )
            assert lmer.AIC == aic.loc[timepoint, channel]


def test_lm_patsy_formula_variable():

    epochs = fitgrid.generate(n_samples=10, n_channels=2)
    levels = ['cat0', 'cat1']

    fitgrid.lm(epochs, RHS='1 + C(categorical, levels=levels)')
