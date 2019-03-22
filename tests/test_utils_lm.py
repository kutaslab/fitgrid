import warnings
import pytest
from .context import fitgrid
from pandas import DataFrame

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


# _OLS_INFLUENCE_ATTRS lists statsmodels diagnostics known to lm.py
def test__OLS_INFLUENCE_ATTRS():
    """verify _OLS_INFLUENCE_ATTRS matches statsmodels OLSInfluence"""

    lm_grid, infl = get_seeded_lm_grid_infl()
    infl_attrs = dir(infl)  # from the mothership

    assert set(infl_attrs) == set(_OLS_INFLUENCE_ATTRS.keys())
    for infl_attr, spec in _OLS_INFLUENCE_ATTRS.items():
        calc, val_type = spec[0], spec[1]
        assert calc in [None, 'nobs', 'nobs_loop', 'nobs_k']
        assert val_type in [None, FLOAT_TYPE, INT_TYPE]


# OLSInfluence diagnostics are a grab bag, lots of outcomes

# not gridded, gridable should fail NotImplemented
not_imp = [
    pytest.param(att, tv, marks=pytest.mark.xfail)
    for att, spec in _OLS_INFLUENCE_ATTRS.items()
    if spec[0] is None
    for tv in (True, False)
]

# nobs_loop should fail with do_nobs_loop=False, run with True
nobs_loop_false = [
    pytest.param(att, False, marks=pytest.mark.xfail)
    for att, spec in _OLS_INFLUENCE_ATTRS.items()
    if spec[0] == 'nobs_loop'
]
nobs_loop_true = [
    pytest.param(att, True)
    for att, spec in _OLS_INFLUENCE_ATTRS.items()
    if spec[0] == 'nobs_loop'
]

# the rest should run +/- do_nobs_loop
rest = [
    pytest.param(att, tv)
    for att, spec in _OLS_INFLUENCE_ATTRS.items()
    if spec[0] not in [None, 'nobs_loop']
    for tv in (True, False)
]


# parametrize the tests
def pytest_generate_tests(metafunc):
    # _check_infl_attr parametrizer
    if (
            metafunc.function is test__get_attr_df
            and 'attr' in metafunc.fixturenames
    ):
        metafunc.parametrize(
            "attr,do_nobs_loop",
            not_imp + nobs_loop_false + nobs_loop_true + rest
        )


# actual test fixture (finally)
def test__get_attr_df(attr, do_nobs_loop):
    lm_grid, infl = get_seeded_lm_grid_infl()
    fitgrid.utils.lm._get_attr_df(infl, attr, do_nobs_loop)


def test_get_nobs_diagnostics_big_grid():
    # nobs diagnostics on a large dataset

    print('fitting a big grid ... be patient')
    lm_grid, infl = get_seeded_lm_grid_infl(
        n_samples=20,
        n_epochs=10,  # 10000
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
        diag_df = fitgrid.utils.lm.get_diagnostic(
            lm_grid, infl_attr
        )
        print(infl_attr, diag_df.shape)


def test_get_ess_press():
    # nobs x 1 length values, no special handling
    lm_grid, infl = get_seeded_lm_grid_infl()
    crit_val = 0.3
    infl_df = fitgrid.utils.lm.get_diagnostic(
        lm_grid,
        'ess_press',
        select_by=crit_val,
        direction='above',
    )
    # raise NotImplementedError("check values")
    infl_df.head()
    warnings.warn("TO DO: add values check")


def test__get_dfbetas_distance():

    lm_grid, infl = get_seeded_lm_grid_infl()
    crit_val = 0.3

    infl_df = fitgrid.utils.lm.get_diagnostic(
        lm_grid,
        'cooks_distance',
        select_by=crit_val,
        direction='above',
    )
    # raise NotImplementedError("check values")
    infl_df.head()
    warnings.warn("TO DO: add values check")


# ------------------------------------------------------------
# cooks_distance and dffits need special handling because
# statsmodels returns a tuple
def test_get_cooks_distance():

    lm_grid, infl = get_seeded_lm_grid_infl()
    crit_val = 0.3
    infl_df = fitgrid.utils.lm.get_diagnostic(
        lm_grid,
        'cooks_distance',
        select_by=crit_val,
        direction='above',
    )

    test_vals = {
        "Time": [0, 0, 1, 2],
        "Epoch_idx": [6, 14, 18, 3],
        "Channel": ["channel0", "channel1", "channel1", "channel0"],
        'cooks_distance': [0.306540, 0.366846, 0.331196, 0.334759],
    }
    test_df = DataFrame.from_dict(test_vals).set_index(
        ['Time', 'Epoch_idx', 'Channel']
    )

    assert all(infl_df == test_df)


def test_get_dffits_internal():

    lm_grid, infl = get_seeded_lm_grid_infl()

    crit_val = 0.93

    infl_df = fitgrid.utils.lm.get_diagnostic(
        lm_grid,
        'dffits_internal',
        select_by=crit_val,
        direction='above',
    )

    test_vals = {
        "Time": [0, 0, 2, 2],
        "Epoch_idx": [6, 14, 3, 18],
        "Channel": ["channel0", "channel1", "channel0", "channel1"],
        "dffits_internal": [0.958969, 1.049066, 1.002137, 0.943897],
    }

    test_df = DataFrame.from_dict(test_vals).set_index(
        ['Time', 'Epoch_idx', 'Channel']
    )
    assert all(test_df == infl_df)


# ------------------------------------------------------------
# UI wrappers
# ------------------------------------------------------------
def test_list_diagnostics():
    fitgrid.utils.lm.list_diagnostics()


def demo_fun(x): return None
@pytest.mark.parametrize("select_by", [None, 'sm', 0.1, demo_fun])
@pytest.mark.parametrize("direction", ["above", "below"])
def test_smoke_get_diagnostics(select_by, direction):

    lm_grid, _ = get_seeded_lm_grid_infl()
    do_nobs_loop = False
    for diagnostic, spec in _OLS_INFLUENCE_ATTRS.items():
        calc, val_type, index_names = spec
        if calc is None:
            continue
        if calc == 'nobs_loop':
            do_nobs_loop = True

        fitgrid.utils.lm.get_diagnostic(
                lm_grid, diagnostic, select_by,  direction, do_nobs_loop
            )
