import pytest
import re
import warnings
import hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

# import at module level or multiprocessing + pymer4 + rpy2 errors
from pymer4 import Lmer

import fitgrid
from fitgrid import DATA_DIR
from fitgrid.utils.summary import INDEX_NAMES, KEY_LABELS
from .context import FIT_ATOL, FIT_RTOL, FIT_ATOL_FAIL, FIT_RTOL_FAIL

_EPOCH_ID = fitgrid.defaults.EPOCH_ID
_TIME = fitgrid.defaults.TIME

pd.set_option("display.width", 256)
PARALLEL = True
N_CORES = 4

# ------------------------------------------------------------
#
# "new" LMER fit data files for epochs = _get_epochs(seed=0)
#  after (late 2019) with Py36 releases of r-lme4, r-matrix
#
# numpy                     1.16.4           py36h7e9f1db_0
# numpy-base                1.16.4           py36hde5b4d6_0
# pandas                    0.25.3           py36he6710b0_0
# pymer4                    0.6.0                    py36_0    kutaslab
# python                    3.6.9                h265db76_0
# r-lme4                    1.1_21            r36h29659fb_0
# r-lmertest                3.1_0             r36h6115d3f_0
# r-matrix                  1.2_17            r36h96ca727_0
#
# "old" checksums (2018ish) were for prior r-lme4, r-matrix releases
#
# numpy                     1.16.4           py36h7e9f1db_0
# numpy-base                1.16.4           py36hde5b4d6_0
# pandas                    0.25.3           py36he6710b0_0
# pymer4                    0.6.0                    py36_0    kutaslab
# python                    3.6.9                h265db76_0
# r-lme4                    1.1_17           r351h29659fb_0
# r-lmertest                3.0_1            r351h6115d3f_0
# r-matrix                  1.2_14           r351h96ca727_0

# gold standard data files
TEST_SUMMARIZE = {
    'lm': {
        # unchanged from v0.4.11 v0.5.0
        'fname': DATA_DIR / "test_summarize_lm.v0.5.0.tsv",
        'md5sum': 'b3b16d7b44d3ce4591cfc07695e14a56',
    },
    'lmer': {
        # for v0.4.11
        # 'fname': DATA_DIR / "test_summarize_lmer.tsv",
        # 'md5sum': 'c082d5fd2634992ae7f2fb381155f340',
        'fname': DATA_DIR / "test_summarize_lmer.v0.5.0.tsv",
        'md5sum': 'bdf5a0b8c0cb82004b2982cacd17ba89',
    },
}

# ------------------------------------------------------------
# pytest metafunc parametrization
# ------------------------------------------------------------
def pytest_generate_tests(metafunc):

    # check figrid default epoch_id and time index names and others
    if metafunc.function in [
        test__lm_get_summaries_df,
        test__lmer_get_summaries_df,
    ]:
        metafunc.parametrize(
            "epoch_id,time",
            [
                (ep, ti)
                for ep in [_EPOCH_ID, _EPOCH_ID + "_ALT"]
                for ti in [_TIME, _TIME + "_ALT"]
            ],
        )


def _get_epochs_fg(seed=None, epoch_id=_EPOCH_ID, time=_TIME):
    # pretend we are starting the pipeline with user epochs dataframe

    # generate fake data
    fake_epochs = fitgrid.generate(
        n_samples=5,
        n_channels=2,
        n_categories=2,
        epoch_id=epoch_id,
        time=time,
        seed=seed,
    )
    epochs_df = fake_epochs.table
    chans = fake_epochs.channels

    # convert to fitgrid epochs object
    epochs_fg = fitgrid.epochs_from_dataframe(
        epochs_df.reset_index().set_index([epoch_id, time]),
        channels=chans,
        epoch_id=epoch_id,
        time=time,
    )

    return epochs_fg


def test__check_summary_df():
    # all bad summaries

    fg_epochs = fitgrid.generate(
        n_epochs=3, n_samples=2, n_categories=2, n_channels=2
    )
    sumry = fitgrid.utils.summary.summarize(
        fg_epochs,
        modeler="lm",
        LHS=fg_epochs.channels,
        RHS=["categorical", "continuous"],
    )
    sumry_bad_idx = sumry.rename_axis(index=sumry.index.names[::-1])
    sumry_bad_keys = sumry.copy()
    sumry_bad_keys.index.set_levels(
        fitgrid.utils.summary.KEY_LABELS[::-1], -1, inplace=True
    )
    for df in [
        "not_a_dataframe",
        pd.DataFrame(),
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).set_index("a"),
        sumry_bad_idx,
        sumry_bad_keys,
    ]:
        with pytest.raises(ValueError):
            fitgrid.utils.summary._check_summary_df(df, None)


def test__lm_get_summaries_df(epoch_id, time):

    fgrid_lm = fitgrid.lm(
        _get_epochs_fg(epoch_id=epoch_id, time=time),
        RHS="1 + continuous + categorical",
        parallel=PARALLEL,
        n_cores=N_CORES,
    )

    summaries_df = fitgrid.utils.summary._lm_get_summaries_df(fgrid_lm)
    fitgrid.utils.summary._check_summary_df(summaries_df, fgrid_lm)


def test__lmer_get_summaries_df(epoch_id, time):

    fgrid_lmer = fitgrid.lmer(
        _get_epochs_fg(epoch_id=epoch_id, time=time),
        RHS="1 + continuous + (1 | categorical)",
        parallel=PARALLEL,
        n_cores=N_CORES,
    )
    summaries_df = fitgrid.utils.summary._lmer_get_summaries_df(fgrid_lmer)
    fitgrid.utils.summary._check_summary_df(summaries_df, fgrid_lmer)


# summary.summarize args
bad_epochs_mark = pytest.mark.xfail(reason=TypeError, strict=True)


@pytest.mark.parametrize(
    "epoch_arg",
    [
        pytest.param(np.array([]), marks=bad_epochs_mark),
        pytest.param(pd.DataFrame(), marks=bad_epochs_mark),
    ],
)
def test_summarize_args(epoch_arg):
    """test summary.summarize argument guards"""
    fitgrid.utils.summary.summarize(epoch_arg, None, None, None, None, None)


def test_summarize():
    """test main wrapper to scrape summaries lm and lmer grids"""

    for mdlr, finfo in TEST_SUMMARIZE.items():
        with open(finfo['fname']) as stream:
            md5sum = hashlib.md5(stream.read().encode('utf8')).hexdigest()
            assert finfo['md5sum'] == md5sum

    # modelers and RHSs
    tests = {
        "lm": [
            "1 + continuous + categorical",
            "0 + continuous + categorical",
            "1 + continuous",
            "0 + continuous",
            "1 + categorical",
            "0 + categorical",
            "1",
        ],
        # formula whitespace got clobbered in lme4/pymer4
        "lmer": [
            "1+continuous+(continuous|categorical)",
            "0+continuous+(continuous|categorical)",
            "1+continuous+(1|categorical)",
            "0+continuous+(1|categorical)",
            "1+(continuous|categorical)",
            "1+(1|categorical)",
        ],
    }

    epochs_fg = fitgrid.generate(
        n_samples=2, n_epochs=3, n_channels=2, n_categories=2, seed=0
    )

    # do it
    summary_dfs = []
    for modler, RHSs in tests.items():
        summaries_df = fitgrid.utils.summary.summarize(
            epochs_fg,
            modler,
            LHS=epochs_fg.channels,
            RHS=RHSs,
            parallel=PARALLEL,
            n_cores=N_CORES,
        )

        assert RHSs == summaries_df.index.unique('model').to_list()
        assert summaries_df.index.names[0] == epochs_fg.time
        assert summaries_df.index.names[1:] == INDEX_NAMES[1:]
        assert all(summaries_df.index.levels[-1] == KEY_LABELS)
        fitgrid.utils.summary._check_summary_df(summaries_df, epochs_fg)

        # verify data and select the modler
        if modler == 'lm':
            modler_ = fitgrid.lm
        elif modler == 'lmer':
            modler_ = fitgrid.lmer
        else:
            raise ValueError('bad modler')

        # read gold standard summary data
        expected_df = pd.read_csv(
            TEST_SUMMARIZE[modler]['fname'], sep='\t'
        ).set_index(summaries_df.index.names)

        # check numerical values, skip warnings and warning strings
        # which changed substantially in lme4::lmer 0.21 -> 0.22
        actual_vals = (
            summaries_df.query('key not in ["has_warning", "warnings"]')
            .stack()
            .astype(float)
        )
        actual_vals.name = 'actual'

        expected_vals = (
            expected_df.query('key not in ["has_warning", "warnings"]')
            .stack()
            .astype(float)
        )
        expected_vals.name = 'expected'

        deltas = abs(actual_vals - expected_vals)
        deltas.name = 'abs_delta'

        # find numpy deviations from close
        oo_tol = ~np.isclose(
            actual_vals, expected_vals, atol=FIT_ATOL, rtol=FIT_RTOL
        )

        if oo_tol.any():

            # lookup which tolerance deviations are out of spec for fits
            oo_tol_fail = ~np.isclose(
                actual_vals,
                expected_vals,
                atol=FIT_ATOL_FAIL,
                rtol=FIT_RTOL_FAIL,
            )
            fails = pd.DataFrame(
                ['X' if x else '' for x in oo_tol_fail],
                index=deltas.index,
                columns=['fail'],
            )

            discrepancies = pd.concat(
                [
                    actual_vals[oo_tol],
                    expected_vals[oo_tol],
                    deltas[oo_tol],
                    fails[oo_tol],
                ],
                axis=1,
            ).sort_values(by='abs_delta', axis=0, ascending=True)

            # dump long form
            with pd.option_context('display.max_rows', None):
                n_d = len(discrepancies)
                n = len(oo_tol)
                msg = (
                    f'\n------------------------------------------------------------\n'
                    f'{n_d} / {n} fitted vals out of tolerance: +/- ({FIT_ATOL} + {FIT_RTOL} * expected)\n'
                    f'{discrepancies}\n'
                    f'------------------------------------------------------------\n'
                )
                warnings.warn(msg)

                if oo_tol_fail.any():
                    fail_msg = (
                        f'Fitted values marked X are too far out of tolerance '
                        f'+/- ({FIT_ATOL} + {FIT_RTOL} * expected)'
                    )
                    raise Exception(fail_msg)

        # on to other checks ...

        # ensure the one and only per-model values are broadcast
        # correctly across the betas within a model
        per_model_keys = ['AIC', 'SSresid', 'has_warning', 'logLike', 'sigma2']
        for pmk in per_model_keys:
            for time, models in summaries_df.groupby(
                summaries_df.index.names[0]
            ):
                for model, model_data in models.groupby('model'):
                    # each
                    assert all(
                        model_data.query('key==@pmk').apply(
                            lambda x: len(np.unique(x))
                        )
                        == 1
                    )

        # refit the models and check against slices of the summary stack
        # to guard against slicing-induced swizzled row indexes
        for rhs in RHSs:
            grid_fg = modler_(
                epochs_fg,
                RHS=rhs,
                LHS=epochs_fg.channels,
                parallel=PARALLEL,
                n_cores=N_CORES,
            )

            # reconstruct a partial summary for this fit
            param_keys = ['Estimate', 'SE']
            if modler == 'lm':
                Estimate = grid_fg.params.copy()
                Estimate.insert(0, 'key', 'Estimate')

                SE = grid_fg.bse.copy()
                SE.insert(0, 'key', 'SE')

                summary = pd.concat([Estimate, SE])
                summary.index.names = [grid_fg.time, 'beta']

            elif modler == 'lmer':
                summary = grid_fg.coefs.loc[pd.IndexSlice[:, :, param_keys], :]
                summary.index.names = [grid_fg.time, 'beta', 'key']

            # add the model and index
            summary.insert(0, 'model', rhs)
            summary = summary.reset_index().set_index(
                [grid_fg.time] + INDEX_NAMES[1:]
            )

            # slice the summary stack and check against the re-fitted grid
            for key, model_key in summaries_df.query('model==@rhs').groupby(
                'key'
            ):
                if key in param_keys:
                    assert all(model_key == summary.query('key==@key'))

        # made it
        summary_dfs.append(summaries_df)

    return summary_dfs


# ------------------------------------------------------------
# lmer kwarg test values from frozen random number generator
# ------------------------------------------------------------
Estimate = pd.DataFrame(
    {
        "channel0": [12.501_775, 8.695_407, -0.687_732, 5.780_693, 2.560_008],
        "channel1": [
            -12.797_436,
            -5.100_620,
            -0.130_712,
            -5.296_107,
            0.637_531,
        ],
    }
)

AIC_REML_True = pd.DataFrame(
    {
        "channel0": [
            187.238_526,
            185.522_789,
            185.038_999,
            188.017_987,
            186.134_492,
        ],
        "channel1": [
            180.067_965,
            181.492_657,
            182.632_950,
            191.689_899,
            187.709_303,
        ],
    }
)

AIC_REML_False = pd.DataFrame(
    {
        "channel0": [
            199.401_434,
            197.185_610,
            196.598_625,
            199.734_402,
            197.751_776,
        ],
        "channel1": [
            191.365_958,
            194.114_479,
            194.065_942,
            203.599_573,
            200.262_021,
        ],
    }
)


@pytest.mark.parametrize(
    "kw,est,aic",
    [
        ({}, Estimate, AIC_REML_True),
        ({"REML": True}, Estimate, AIC_REML_True),
        ({"REML": False}, Estimate, AIC_REML_False),
    ],
)
def test_summarize_lmer_kwargs(kw, est, aic):

    epochs_fg = _get_epochs_fg(seed=0)  # freeze data to test values

    LHS = epochs_fg.channels
    RHS = "1 +(1 | categorical)"  # for fitgrid.lmer
    RHS = re.sub(r"\s+", "", RHS)  # new for pymer4 7.0+

    # what the fitgrid modeler returns
    lmer_fit = fitgrid.lmer(epochs_fg, LHS=LHS, RHS=RHS, **kw)
    lmer_fit_betas = lmer_fit.coefs
    lmer_fit_betas.index.names = [lmer_fit.time, 'beta', 'key']

    # what the summarize wrapper scrapes from the grid
    summaries_df = fitgrid.utils.summary.summarize(
        epochs_fg, "lmer", LHS=LHS, RHS=RHS, **kw
    )
    fitgrid.utils.summary._check_summary_df(summaries_df, lmer_fit)

    # compare results
    summary_keys = set(summaries_df.index.unique('key'))
    lmer_fit_betas_keys = set(lmer_fit_betas.index.unique('key'))

    # from the grid.params ... lmer specific
    shared_keys = summary_keys.intersection(lmer_fit_betas_keys)

    # other grid.attr
    attr_keys = [key for key in summary_keys if key in dir(lmer_fit.tester)]

    for key in shared_keys.union(attr_keys):

        if key == "warnings":
            # LMERFitGrid.warnings are irregular and incomensurable with
            # their rendition as canonical summary time x channel gridded "_"
            # joined or empty strings
            continue

        # these come from the coefs dataframe
        if key in shared_keys:
            modeler_vals = lmer_fit_betas.query("key==@key").reset_index(
                drop=True
            )

            summarize_vals = summaries_df.query(
                "model==@RHS and key==@key"
            ).reset_index(drop=True)

        # these come from grid attributes
        elif key in attr_keys:
            modeler_vals = getattr(lmer_fit, key).reset_index(drop=True)
            summarize_vals = summaries_df.query("key==@key").reset_index(
                drop=True
            )
        else:
            raise ValueError(f"unknown summary key: {key}")

        try:
            all(modeler_vals == summarize_vals)
        except Exception as fail:
            msg = f"kwargs: {kw} key: {key}"
            print(msg)
            raise fail

        # smoke test that the REML=True v. False is not changing Estimate
        # and is changing AIC
        if key == 'Estimate':
            assert all(summarize_vals == est)

        if key == 'AIC':
            assert all(summarize_vals == aic)


def test__get_AICs():

    RHSs = [
        "1 + continuous + categorical",
        "1 + continuous",
        "1 + categorical",
        "0 + categorical",
    ]

    epochs_fg = _get_epochs_fg(seed=0)
    summaries_df = fitgrid.utils.summary.summarize(
        epochs_fg, 'lm', LHS=epochs_fg.channels, RHS=RHSs
    )

    # first check AIC summaries are OK
    # summaries_df is for a stack of models at each time
    for time, aics in summaries_df.query('key=="AIC"').groupby(epochs_fg.time):
        for model, model_aic in aics.groupby('model'):
            assert all(model_aic.apply(lambda x: len(np.unique(x))) == 1)

    aics = fitgrid.utils.summary._get_AICs(summaries_df)

    assert (
        RHSs
        == summaries_df.index.unique('model').tolist()
        == aics.index.unique('model').tolist()
    )

    for (time, chan), tc_aics in aics.groupby([epochs_fg.time, 'channel']):
        # mins at each time, channel
        min = tc_aics['AIC'].min()
        assert np.allclose(
            tc_aics['AIC'].astype('float') - min,
            tc_aics['min_delta'].astype('float'),
            atol=0,
        )

    return aics


def test_lm_plot_betas_AICmin_deltas():
    """test lm summary plotting"""

    # data in common for lm and lmer
    epochs_fg = fitgrid.generate(n_samples=10, n_channels=10, seed=32)
    channels = [
        column for column in epochs_fg.table.columns if "channel" in column
    ]

    lm_rhs = ["1 + categorical", "1 + continuous"]
    lm_summaries = fitgrid.utils.summary.summarize(
        epochs_fg, "lm", LHS=channels, RHS=lm_rhs, parallel=False
    )

    # ------------------------------------------------------------
    # plot AIC min deltas

    # default
    fitgrid.utils.summary.plot_AICmin_deltas(lm_summaries)
    plt.close("all")

    # plotting kwargs here, warnings checked w/ LMER
    fitgrid.utils.summary.plot_AICmin_deltas(
        lm_summaries,
        figsize=(12, 8),
        gridspec_kw={"width_ratios": [1, 1, 0.1]},
        subplot_kw={"ylim": (0, 50)},
    )
    plt.close('all')

    # ------------------------------------------------------------
    # plot betas, prevent matplotlib too many figures warning
    with mpl.rc_context({"figure.max_open_warning": 41}):

        fitgrid.utils.summary.plot_betas(lm_summaries)
        plt.close('all')

        # deprecated kwargs < v0.5.0
        with pytest.warns(FutureWarning):
            fitgrid.utils.summary.plot_betas(lm_summaries, figsize=(12, 3))
            plt.close("all")

        # kwargs v0.5.0
        fitgrid.utils.summary.plot_betas(
            lm_summaries,
            LHS=["channel0", "channel1"],
            fdr_kw={"method": "BH", "rate": 0.10, "plot_pvalues": True},
            beta_plot_kwargs={"ylim": (-100, 100)},
            models=["1 + categorical"],
            betas=["Intercept"],
            interval=[2, 6],
        )
        plt.close('all')


def test_lmer_warnings_plot_betas_AICmin_deltas():
    """test lmer warnings and summary plotting"""

    # ------------------------------------------------------------
    # setup the data and models for nice variety of warnings
    epochs_fg = fitgrid.generate(n_samples=8, n_channels=4, seed=32)
    channels = [
        column for column in epochs_fg.table.columns if "channel" in column
    ]

    lmer_rhs = [
        "1 + continuous + (1 | categorical)",
        "1 + categorical + (continuous | categorical)",
    ]

    # ------------------------------------------------------------
    # lmer summaries, plotting betas, AICmin delta
    lmer_summaries = fitgrid.utils.summary.summarize(
        epochs_fg,
        "lmer",
        LHS=channels,
        RHS=lmer_rhs,
        parallel=True,
        n_cores=2,
    )

    # lmer plot_beta w/ warnings
    fitgrid.utils.summary.plot_betas(lmer_summaries)
    plt.close("all")

    fitgrid.utils.summary.plot_betas(
        lmer_summaries,
        fdr_kw={"method": "BH", "rate": 0.10, "plot_pvalues": False},
        beta_plot_kwargs={"ylim": (-100, 100)},
        models=["1+continuous+(1|categorical)"],
        betas=["(Intercept)"],
        show_warnings=False,
        interval=[2, 6],
        df_func=lambda x: x,
    )

    # AIC default
    fitgrid.utils.summary.plot_AICmin_deltas(lmer_summaries)

    # warning display options
    for shwarn in [
        "no_labels",
        "labels",
        ["converge"],
        ["converge", "singular"],
    ]:
        fitgrid.utils.summary.plot_AICmin_deltas(
            lmer_summaries, show_warnings=shwarn
        )

    with pytest.warns(UserWarning):
        fitgrid.utils.summary.plot_AICmin_deltas(
            lmer_summaries, show_warnings="converge"
        )
