import warnings
import pytest
from .context import fitgrid
from pandas import DataFrame, concat
import fitgrid.utils as fgutil

PARALLEL = False
N_CORES = 4  # for dev/testing


def get_seeded_lm_grid_infl(
    n_samples=5,
    n_epochs=10,
    n_channels=2,
    n_categories=2,
    seed=0,
    parallel=True,
    n_cores=4,
):
    """frozen test data and influence measures"""

    epochs = fitgrid.generate(
        n_epochs=n_epochs,
        n_samples=n_samples,
        n_channels=n_channels,
        n_categories=n_categories,
        seed=seed,
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
    fgutil.lm.get_vifs(epochs, RHS)


# _OLS_INFLUENCE_ATTRS lists statsmodels diagnostics known to lm.py
def test__OLS_INFLUENCE_ATTRS():
    """verify _OLS_INFLUENCE_ATTRS matches statsmodels OLSInfluence"""

    lm_grid, infl = get_seeded_lm_grid_infl()
    infl_attrs = dir(infl)  # from the mothership

    assert set(fgutil.lm._OLS_INFLUENCE_ATTRS.keys()).issubset(set(infl_attrs))
    for infl_attr, spec in fgutil.lm._OLS_INFLUENCE_ATTRS.items():
        calc, val_type = spec[0], spec[1]
        assert calc in [None, 'nobs', 'nobs_loop', 'nobs_k']
        assert val_type in [None, fgutil.lm._FLOAT_TYPE, fgutil.lm._INT_TYPE]


# ------------------------------------------------------------
# OLSInfluence diagnostics are a grab bag, lots of outcomes
# ------------------------------------------------------------
# not gridded, gridable should fail NotImplemented
not_imp = [
    pytest.param(att, tv, marks=pytest.mark.xfail)
    for att, spec in fgutil.lm._OLS_INFLUENCE_ATTRS.items()
    if spec[0] is None
    for tv in (True, False)
]

# nobs_loop should fail with do_nobs_loop=False, run with True
nobs_loop_false = [
    pytest.param(att, False, marks=pytest.mark.xfail(strict=True))
    for att, spec in fgutil.lm._OLS_INFLUENCE_ATTRS.items()
    if spec[0] == 'nobs_loop'
]
nobs_loop_true = [
    pytest.param(att, True)
    for att, spec in fgutil.lm._OLS_INFLUENCE_ATTRS.items()
    if spec[0] == 'nobs_loop'
]

# the rest should run +/- do_nobs_loop
rest = [
    pytest.param(att, tv)
    for att, spec in fgutil.lm._OLS_INFLUENCE_ATTRS.items()
    if spec[0] not in [None, 'nobs_loop']
    for tv in (True, False)
]

# ------------------------------------------------------------
# smoke tests for diagnostic data frame filtering options
# ------------------------------------------------------------
filt_1_side = [
    pytest.param(how, b0, b1, shape)
    for how in ["above", "below"]
    for b0 in [-1.0, 1.0]
    for b1 in [None, -1.0, 1.0]
    for shape in ['long', 'wide']
]
filt_1_side_x = [
    pytest.param(
        how,
        b0,
        b1,
        "long",
        marks=pytest.mark.xfail(strict=True, raises=(ValueError, TypeError)),
    )
    for how in ["above", "below"]
    for b0 in [None]
    for b1 in [None, -1.0, 1.0]
]
filt_interval = [
    pytest.param(how, b0, b1, shape)
    for how in ["inside", "outside"]
    for b0 in [0, 1.0]
    for b1 in [1.0, 2.0]
    for shape in ['long', 'wide']
]
filt_interval_x = [
    pytest.param(
        how,
        b0,
        b1,
        "long",
        marks=pytest.mark.xfail(strict=True, raises=(ValueError, TypeError)),
    )
    for how in ["inside", "outside"]
    for b0 in [None, -1.0, 1.0]
    for b1 in [None]
]


# ------------------------------------------------------------
# pytest metafunc parametrization
# ------------------------------------------------------------
def pytest_generate_tests(metafunc):

    # backend grid scraper
    if metafunc.function in [test__get_diagnostic, test_get_diagnostic]:
        metafunc.parametrize(
            "attr,do_nobs_loop",
            not_imp + nobs_loop_false + nobs_loop_true + rest,
        )

    # diagnostic filter function
    if metafunc.function is test_smoke_filter_diagnostic:
        metafunc.parametrize(
            "how,b0,b1,shape",
            filt_1_side + filt_1_side_x + filt_interval + filt_interval_x,
        )


# ------------------------------------------------------------
# backend
# ------------------------------------------------------------
def test__get_diagnostic(attr, do_nobs_loop):
    # getter for native fitgrid dataframe
    lm_grid, __ = get_seeded_lm_grid_infl()
    _ = fgutil.lm._get_diagnostic(lm_grid, attr, do_nobs_loop)


lmg, _ = get_seeded_lm_grid_infl()
lmgx, _ = get_seeded_lm_grid_infl()
lmgx.tester = []


@pytest.mark.parametrize(
    "lm_grid",
    [
        lmg,
        pytest.param(lmgx, marks=pytest.mark.xfail(strict=True)),
        pytest.param([], marks=pytest.mark.xfail(strict=True)),
    ],
)
@pytest.mark.parametrize(
    "diag",
    [
        "cooks_distance",
        pytest.param("Cooks_D", marks=pytest.mark.xfail(strict=True)),
        pytest.param(True, marks=pytest.mark.xfail(strict=True)),
    ],
)
@pytest.mark.parametrize(
    "dnl",
    [
        True,
        pytest.param("True", marks=pytest.mark.xfail(strict=True)),
        pytest.param(123.4, marks=pytest.mark.xfail(strict=True)),
    ],
)
def test__check_get_diagnostic_args(lm_grid, diag, dnl):

    fgutil.lm._check_get_diagnostic_args(lm_grid, diag, dnl)


# ------------------------------------------------------------
# UI wrappers
# ------------------------------------------------------------
def test_list_diagnostics():
    fgutil.lm.list_diagnostics()


def test_get_diagnostic(attr, do_nobs_loop):
    # UI getter labels indexes and (ugh) splits special case
    # statsmodels 2-ple returns ... ugh.
    lm_grid, _ = get_seeded_lm_grid_infl()
    d_df, sm_df = fgutil.lm.get_diagnostic(lm_grid, attr, do_nobs_loop)
    assert attr == d_df.columns.unique('diagnostic')[0]
    assert all(d_df.columns.unique('channel') == ['channel0', 'channel1'])
    assert sm_df is None or sm_df.shape == d_df.shape


def Xtest_get_nobs_diagnostics_big_grid():
    # nobs diagnostics on a large dataset
    print('fitting a big grid ... be patient')
    lm_grid, infl = get_seeded_lm_grid_infl(
        n_samples=20, n_epochs=10000, n_channels=24, parallel=True, n_cores=4
    )
    print('done')

    nobs_diagnostics = [
        attr
        for attr, spec in fgutil.lm._OLS_INFLUENCE_ATTRS.items()
        if spec[0] == 'nobs'
    ]
    for infl_attr in nobs_diagnostics:
        diag_df, _ = fgutil.lm.get_diagnostic(lm_grid, infl_attr)
        print(infl_attr, diag_df.shape)


def test_get_ess_press():
    lm_grid, infl = get_seeded_lm_grid_infl()
    infl_df, _ = fgutil.lm.get_diagnostic(lm_grid, 'ess_press')
    warnings.warn("TO DO: add values check")


def test_get_dfbetas():
    lm_grid, infl = get_seeded_lm_grid_infl()
    infl_df, _ = fgutil.lm.get_diagnostic(
        lm_grid, 'dfbetas', do_nobs_loop=True
    )
    warnings.warn("TO DO: add values check")


# ------------------------------------------------------------
# cooks_distance and dffits need special handling because
# statsmodels returns a tuple
def test_get_cooks_distance():

    test_vals = {
        "Time": [0, 0, 1, 2],
        "Epoch_idx": [6, 14, 18, 3],
        "channel": ["channel0", "channel1", "channel1", "channel0"],
        'cooks_distance': [0.306540, 0.366846, 0.331196, 0.334759],
    }
    test_df = DataFrame.from_dict(test_vals).set_index(
        ['Time', 'Epoch_idx', 'channel']
    )

    lm_grid, infl = get_seeded_lm_grid_infl()
    infl_df, sm_1_df = fgutil.lm.get_diagnostic(lm_grid, 'cooks_distance')

    crit_val = 0.3
    selected_df = fgutil.lm.filter_diagnostic(infl_df, "above", crit_val)
    assert all(test_df.index == selected_df.index)
    assert all(test_df == selected_df)


def test_get_dffits_internal():

    test_vals = {
        "Time": [0, 0, 2, 2],
        "Epoch_idx": [6, 14, 3, 18],
        "channel": ["channel0", "channel1", "channel0", "channel1"],
        "dffits_internal": [0.958969, 1.049066, 1.002137, 0.943897],
    }

    test_df = DataFrame.from_dict(test_vals).set_index(
        ['Time', 'Epoch_idx', 'channel']
    )

    lm_grid, infl = get_seeded_lm_grid_infl()
    infl_df, sm_df = fgutil.lm.get_diagnostic(lm_grid, 'dffits_internal')

    crit_val = 0.93
    selected_df = fgutil.lm.filter_diagnostic(infl_df, "above", crit_val)

    assert all(test_df.index == selected_df.index)
    assert all(test_df == selected_df)


def test_smoke_filter_diagnostic(how, b0, b1, shape):

    lm_grid, _ = get_seeded_lm_grid_infl()
    for diagnostic, spec in fgutil.lm._OLS_INFLUENCE_ATTRS.items():
        calc, _, _ = spec
        if calc is None:
            continue
        diag_df, sm_1_df = fgutil.lm.get_diagnostic(
            lm_grid, diagnostic, do_nobs_loop=True
        )
        fd = fgutil.lm.filter_diagnostic(diag_df, how, b0, b1, shape)
        if shape == "wide":
            assert diag_df.shape == fd.shape


@pytest.mark.parametrize(
    'b0,b1',
    [
        (0, 0),
        (-10, 10),
        pytest.param(
            10, -10, marks=pytest.mark.xfail(strict=True, raises=ValueError)
        ),
        pytest.param(
            0, [1], marks=pytest.mark.xfail(strict=True, raises=TypeError)
        ),
    ],
)
def test_filter_diagnostic_interval(b0, b1):

    lm_grid, _ = get_seeded_lm_grid_infl()

    diag_df, sm_1_df = fgutil.lm.get_diagnostic(
        lm_grid, "influence", do_nobs_loop=True
    )

    in_df = fgutil.lm.filter_diagnostic(diag_df, 'inside', b0, b1)
    out_df = fgutil.lm.filter_diagnostic(diag_df, 'outside', b0, b1)

    if not all(diag_df.stack(-1) == concat([in_df, out_df]).sort_index()):
        raise ValueError("inside + outside != all")
