from .context import fitgrid
from numpy import allclose
from pandas import DataFrame

PARALLEL = True
N_CORES = 4
SEED = 0


def test_smoke_get_vifs():

    epochs = fitgrid.generate()
    RHS = 'continuous + categorical'
    fitgrid.utils.lm.get_vifs(epochs, RHS)


def test__get_infl_dffits_internal():

    epochs = fitgrid.generate(
        n_samples=5, n_channels=2, n_categories=2, seed=SEED
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

    crit_val = 0.93
    infl_data_df, infl_idxs = fitgrid.utils.lm._get_infl_dffits_internal(
        infl, crit_val
    )

    test_vals = {
        "Epoch_idx": [0, 0, 2, 2],
        "Time": [6, 14, 3, 18],
        "channel": ["channel0", "channel1", "channel0", "channel1"],
        "diagnostic": "dffits_internal",
        'value': [0.958969, 1.049066, 1.002137, 0.943897],
        "critical": crit_val,
    }

    for col, result in test_vals.items():
        if col == 'value':
            assert allclose(infl_data_df[col], result)
        else:
            assert all(infl_data_df[col] == result)


def test__get_infl_cooks_distance():

    epochs = fitgrid.generate(
        n_samples=5, n_channels=2, n_categories=2, seed=SEED
    )
    RHS = "continuous + categorical"

    lm_grid = fitgrid.lm(
        epochs,
        LHS=epochs.channels,
        RHS=RHS,
        parallel=PARALLEL,
        n_cores=N_CORES,
    )

    infl = lm_grid.get_influence()

    crit_val = 0.3
    infl_data_df, infl_idxs = fitgrid.utils.lm._get_infl_cooks_distance(
        infl, crit_val
    )

    # compare with seeded test run
    assert infl_data_df.shape == (4, 6)
    test_vals = {
        "Epoch_idx": [0, 0, 1, 2],
        "Time": [6, 14, 18, 3],
        "channel": ["channel0", "channel1", "channel1", "channel0"],
        "diagnostic": "cooks_distance",
        'value': [0.306540, 0.366846, 0.331196, 0.334759],
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
        # print(infl_data_df.head())
        # print(infl_data_df.tail())
        # print(f"get_influential data({diagnostic}) ok")

    epochs = fitgrid.generate(
        n_samples=5, n_channels=2, n_categories=2, seed=SEED
    )
    RHS = "continuous + categorical"

    lm_grid = fitgrid.lm(
        epochs,
        LHS=epochs.channels,
        RHS=RHS,
        parallel=PARALLEL,
        n_cores=N_CORES,
    )

    # test the crit_val param
    for diagnostic in ["dffits_internal", "cooks_distance"]:
        infl_data_df = fitgrid.utils.lm.get_influential_data(
            lm_grid, diagnostic
        )
        _check_infl_data_df(infl_data_df, with_cval=False)

        for cval in [None, 0.1]:
            infl_data_df = fitgrid.utils.lm.get_influential_data(
                lm_grid, diagnostic, crit_val=cval
            )
            _check_infl_data_df(infl_data_df, with_cval=True, cval=cval)
