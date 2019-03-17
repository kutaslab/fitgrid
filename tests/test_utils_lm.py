from .context import fitgrid
from numpy import allclose
from pandas import DataFrame, MultiIndex

from fitgrid.utils.lm import _OLS_INFLUENCE_ATTRS, FLOAT_TYPE, INT_TYPE


PARALLEL = True
N_CORES = 4


def get_seeded_lm_grid_infl(
        n_samples=5,
        n_epochs=10,
        n_channels=2,
        n_categories=2,
        seed=0,
        parallel=True,
        n_cores=4
):
    """frozen test data and influence measures"""

    epochs = fitgrid.generate(
        n_epochs=n_epochs,
        n_samples=n_samples,
        n_channels=n_channels,
        n_categories=n_categories,
        seed=seed
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
    """fitgrid influence attr database must match statsmodels OLSInfluence"""

    lm_grid, infl = get_seeded_lm_grid_infl()
    infl_attrs = dir(infl)  # from the mother ship

    assert set(infl_attrs) == set(_OLS_INFLUENCE_ATTRS.keys())
    for infl_attr, spec in _OLS_INFLUENCE_ATTRS.items():
        calc, val_type = spec[0], spec[1]
        assert calc in [None, 'nobs', 'nobs_loop', 'nobs_k']
        assert val_type in [None, FLOAT_TYPE, INT_TYPE]


def test__check_infl_attr():
    """prerun all available OLSInfluence attrs through the checker"""

    lm_grid, infl = get_seeded_lm_grid_infl()
    for infl_attr in dir(infl):
        fitgrid.utils.lm._check_influence_attr(infl, infl_attr)


def test__get_infl_attr_vals():
    """grab values for real"""
    lm_grid, infl = get_seeded_lm_grid_infl()

    implemented_diagnostics = [
        attr
        for attr, spec in _OLS_INFLUENCE_ATTRS.items()
        if spec[0] is not None  # 'nobs', 'nobs_k', 'nobs_loop'
    ]
    for infl_attr in implemented_diagnostics:
        infl_data_df = fitgrid.utils.lm._get_infl_attr_vals(
            lm_grid, infl_attr, do_nobs_loop=True
        )
        print(infl_attr, type(infl_data_df))
        print(infl_data_df.iloc[0:5, :])


def test__get_infl_attr_vals_big():
    """grab values for large data set"""

    print('fitting a big grid ... be patient')
    lm_grid, infl = get_seeded_lm_grid_infl(
        n_samples=20,
        n_epochs=10000,
        n_channels=24,
        parallel=True,
        n_cores=4,
    )
    print('done')

    nobs_diagnostics = [
        attr
        for attr, spec in _OLS_INFLUENCE_ATTRS.items()
        if spec[0] == 'nobs'
    ]
    for infl_attr in nobs_diagnostics:
        infl_data_df = fitgrid.utils.lm._get_infl_attr_vals(
            lm_grid, infl_attr
        )
        print(infl_attr, infl_data_df.shape)


def test__get_infl_cooks_distance():

    lm_grid, infl = get_seeded_lm_grid_infl()

    crit_val = 0.3
    infl_data_df, infl_data_idxs = fitgrid.utils.lm._get_infl_cooks_distance(
        lm_grid, crit_val
    )

    # compare with seeded test run and enforce index structure
    test_index = MultiIndex.from_arrays(
        names=['Time', 'Epoch_idx', 'Channels'],
        arrays=[
            [0, 0, 1, 2],
            [6, 14, 18, 3],
            ["channel0", "channel1", "channel1", "channel0"],
        ],
    )

    test_vals = {
        'cooks_distance_0': [0.306540, 0.366846, 0.331196, 0.334759],
        "cooks_distance_1": crit_val,
    }

    test_idxs = (([12, 29, 77, 86]),)

    assert infl_data_df.shape == (4, 2)
    assert infl_data_df.columns.name == 'cooks_distance'
    assert all(test_index == infl_data_df.index)
    assert all(test_vals == infl_data_df)
    assert all([all(x[0] == x[1]) for x in zip(test_idxs, infl_data_idxs)])


def test__get_infl_dffits_internal():

    lm_grid, infl = get_seeded_lm_grid_infl()

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


def test_smoke_get_influential_data():
    """wrapper"""

    def _check_infl_data_df(iddf, with_cval=False, cval=None):
        assert isinstance(infl_data_df, DataFrame)
        assert all(
            infl_data_df.columns == fitgrid.utils.lm._FG_LM_DIAGNOSTIC_COLUMNS
        )
        assert all(infl_data_df["diagnostic"] == diagnostic)

    lm_grid, _ = get_seeded_lm_grid_infl()

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
