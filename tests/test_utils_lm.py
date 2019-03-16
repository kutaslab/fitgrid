from .context import fitgrid
from numpy import allclose
from pandas import DataFrame
from statsmodels.stats.outliers_influence import OLSInfluence

from fitgrid.utils.lm import _OLS_INFLUENCE_ATTRS

PARALLEL = True
N_CORES = 4
SEED = 0


def get_seeded_lm_grid_infl(seed):
    """frozen test data and influence measures"""

    epochs = fitgrid.generate(
        n_samples=5, n_channels=2, n_categories=2, seed=seed
    )
    RHS = 'continuous + categorical'
    lm_grid = fitgrid.lm(
        epochs,
        LHS=epochs.channels,
        RHS=RHS,
        parallel=PARALLEL,
        n_cores=N_CORES,
    )

    infl = lm_grid.get_influence()

    return lm_grid, infl


def test_smoke_get_vifs():

    epochs = fitgrid.generate()
    RHS = 'continuous + categorical'
    fitgrid.utils.lm.get_vifs(epochs, RHS)


def test__OLS_INFLUENCE_ATTRS():
    # check the fitgrid lm influence attr database is current

    lm_grid, infl = get_seeded_lm_grid_infl(SEED)
    infl_attrs = dir(infl)  # from the mother ship

    assert set(infl_attrs) == set(_OLS_INFLUENCE_ATTRS.keys())
    for infl_attr, spec in _OLS_INFLUENCE_ATTRS.items():
        calc, val_type = spec[0], spec[1]
        assert calc in [None, 'nobs', 'nobs_loop', 'nobs_k']
        assert val_type in [None, float]


def test__check_infl_attr():
    lm_grid, infl = get_seeded_lm_grid_infl(SEED)

    for infl_attr in dir(infl):
        fitgrid.utils.lm._check_influence_attr(lm_grid, infl_attr)

def test__get_infl_dffits_internal():

    lm_grid, infl = get_seeded_lm_grid_infl(SEED)

    crit_val = 0.93
    infl_data_df, infl_idxs = fitgrid.utils.lm._get_infl_dffits_internal(
        infl, crit_val
    )

    test_vals = {
        "Epoch_idx": [3, 6, 14, 18],
        "Time": [2, 0, 0, 2],
        "channel": ["channel0", "channel0", "channel1", "channel1"],
        "diagnostic": "dffits_internal",
        'value': [1.002137, 0.958969, 1.049066, 0.943897],
        "critical": crit_val,
    }

    for col, result in test_vals.items():
        if col == 'value':
            assert allclose(infl_data_df[col], result)
        else:
            assert all(infl_data_df[col] == result)


def test__get_infl_cooks_distance():

    lm_grid, infl = get_seeded_lm_grid_infl(SEED)

    crit_val = 0.3
    infl_data_df, infl_idxs = fitgrid.utils.lm._get_infl_cooks_distance(
        infl, crit_val
    )

    # compare with seeded test run
    assert infl_data_df.shape == (4, 6)
    test_vals = {
        "Epoch_idx": [3, 6, 14, 18],
        "Time": [2, 0, 0, 1],
        "channel": ["channel0", "channel0", "channel1", "channel1"],
        "diagnostic": "cooks_distance",
        'value': [0.334759, 0.306540, 0.366846, 0.331196],
        "critical": crit_val,
    }

    for col, result in test_vals.items():
        if col == 'value':
            # float64
            assert allclose(infl_data_df[col], result)
        else:
            assert all(infl_data_df[col] == result)


def test_smoke_get_influential_data():
    """wrapper"""

    def _check_infl_data_df(iddf, with_cval=False, cval=None):
        assert isinstance(infl_data_df, DataFrame)
        assert all(
            infl_data_df.columns == fitgrid.utils.lm._FG_LM_DIAGNOSTIC_COLUMNS
        )
        assert all(infl_data_df["diagnostic"] == diagnostic)

    lm_grid, _ = get_seeded_lm_grid_infl(SEED)

    # default, no critical value
    for diagnostic in ["dffits_internal", "cooks_distance"]:
        infl_data_df = fitgrid.utils.lm.get_influential_data(
            lm_grid, diagnostic
        )
        _check_infl_data_df(infl_data_df, with_cval=False)

        # explicit critical value
        for cval in [None, 0.1]:
            infl_data_df = fitgrid.utils.lm.get_influential_data(
                lm_grid, diagnostic, crit_val=cval
            )
            _check_infl_data_df(infl_data_df, with_cval=True, cval=cval)
