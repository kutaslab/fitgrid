import pytest
import re
import warnings
import hashlib
from numpy import log10
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import fitgrid
from fitgrid.utils.summary import INDEX_NAMES, KEY_LABELS, PER_MODEL_KEY_LABELS
from .context import tpath, FIT_ATOL, FIT_RTOL

pd.set_option("display.width", 256)
PARALLEL = True
N_CORES = 4

# ------------------------------------------------------------
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


TEST_SUMMARIZE = {
    'lm': {
        'fname': 'tests/data/test_summarize_lm.tsv',
        'md5sum': '43b6a8b0fc621f1b0ca0ea71270241ac',
    },
    'lmer': {
        'fname': 'tests/data/test_summarize_lmer.tsv',
        'md5sum': 'ee7721553cdcbe3c4a137e9ddc78ffba',
    },
}


def _get_epochs_fg(seed=None):
    # pretend we are starting the pipeline with user epochs dataframe

    # generate fake data
    fake_epochs = fitgrid.generate(
        n_samples=5, n_channels=2, n_categories=2, seed=seed
    )
    epochs_df = fake_epochs.table
    chans = fake_epochs.channels

    # convert to fitgrid epochs object
    epochs_fg = fitgrid.epochs_from_dataframe(
        epochs_df.reset_index().set_index(['Epoch_idx', 'Time']),
        channels=chans,
        epoch_id="Epoch_idx",
        time='Time',
    )

    return epochs_fg


def test__lm_get_summaries_df():

    fgrid_lm = fitgrid.lm(
        _get_epochs_fg(),
        RHS="1 + continuous + categorical",
        parallel=PARALLEL,
        n_cores=N_CORES,
    )

    summaries_df = fitgrid.utils.summary._lm_get_summaries_df(fgrid_lm)
    fitgrid.utils.summary._check_summary_df(summaries_df)


def test__lmer_get_summaries_df():

    fgrid_lmer = fitgrid.lmer(
        _get_epochs_fg(),
        RHS="1 + continuous + (1 | categorical)",
        parallel=PARALLEL,
        n_cores=N_CORES,
    )

    summaries_df = fitgrid.utils.summary._lmer_get_summaries_df(fgrid_lmer)
    fitgrid.utils.summary._check_summary_df(summaries_df)


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
        "lmer": [
            "1 + continuous + (continuous | categorical)",
            "0 + continuous + (continuous | categorical)",
            "1 + continuous + (1 | categorical)",
            "0 + continuous + (1 | categorical)",
            "1 + (continuous | categorical)",
            "1 + (1 | categorical)",
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
        assert summaries_df.index.names == INDEX_NAMES
        assert all(summaries_df.index.levels[-1] == KEY_LABELS)
        fitgrid.utils.summary._check_summary_df(summaries_df)

        aics = fitgrid.utils.summary._get_AICs(summaries_df)

        # verify data and select the modler
        if modler == 'lm':
            modler_ = fitgrid.lm
        elif modler == 'lmer':
            modler_ = fitgrid.lmer
        else:
            raise ValueError('bad modler')

        # read gold standard summary data
        expected = pd.read_csv(
            TEST_SUMMARIZE[modler]['fname'], sep='\t'
        ).set_index(summaries_df.index.names)

        # handle lme4 warnings separately, these changed substantially
        # at some point
        if not expected.query('key == "has_warning"').equals(
            summaries_df.query('key == "has_warning"')
        ):
            warnings.warn(f'{modler} has_warning values have changed')

        expected_vals = expected.query('key != "has_warning"').copy()
        expected_vals['val'] = 'expected'
        expected_vals.set_index('val', append=True, inplace=True)

        fit_vals = summaries_df.query('key != "has_warning"').copy()
        fit_vals['val'] = 'fitted'
        fit_vals.set_index('val', append=True, inplace=True)

        test_vals = pd.concat([expected_vals, fit_vals])

        # verify values and warn of discrepancies
        for (model, beta, key), vals in test_vals.groupby(
            ['model', 'beta', 'key']
        ):

            # compare actual, expected
            in_tol = np.isclose(
                vals.query('val == "expected"'),
                vals.query('val == "fitted"'),
                atol=FIT_ATOL,
                rtol=FIT_RTOL,
            )

            # display if actual are not within tolerance
            if not in_tol.all():
                msg = (
                    f'\n------------------------------------------------------------\n'
                    f'fitted vals out of tolerance: {FIT_ATOL} + {FIT_RTOL} * expected\n'
                    f'{modler} {model} {beta} {key}\n'
                    f'{in_tol}\n'
                    f'{vals.unstack(-1)}\n'
                    f'------------------------------------------------------------\n'
                )
                warnings.warn(msg)

        # on to other checks ...

        # ensure the one and only per-model values are broadcast
        # correctly across the betas within a model
        per_model_keys = ['AIC', 'SSresid', 'has_warning', 'logLike', 'sigma2']
        for pmk in per_model_keys:
            for time, models in summaries_df.groupby('Time'):
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
                summary.index.names = ['Time', 'beta']

            elif modler == 'lmer':
                summary = grid_fg.coefs.loc[pd.IndexSlice[:, :, param_keys], :]
                summary.index.names = ['Time', 'beta', 'key']

            # add the model and index
            summary.insert(0, 'model', rhs)
            summary = summary.reset_index().set_index(INDEX_NAMES)

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
    RHS = "1 + (1 | categorical)"  # for fitgrid.lmer

    # what the fitgrid modeler returns
    lmer_fit = fitgrid.lmer(epochs_fg, LHS=LHS, RHS=RHS, **kw)
    lmer_fit_betas = lmer_fit.coefs
    lmer_fit_betas.index.names = ['Time', 'beta', 'key']

    # what the summarize wrapper scrapes from the grid
    summaries_df = fitgrid.utils.summary.summarize(
        epochs_fg, "lmer", LHS=LHS, RHS=RHS, **kw
    )
    fitgrid.utils.summary._check_summary_df(summaries_df)

    # compare results
    summary_keys = set(summaries_df.index.unique('key'))
    lmer_fit_betas_keys = set(lmer_fit_betas.index.unique('key'))

    # from the grid.params ... lmer specific
    shared_keys = summary_keys.intersection(lmer_fit_betas_keys)

    # other grid.attr
    attr_keys = [key for key in summary_keys if key in dir(lmer_fit.tester)]
    for key in shared_keys.union(attr_keys):

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
            raise ValueError(f"unknown key: {key}")

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

    pass


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
    for time, aics in summaries_df.query('key=="AIC"').groupby('Time'):
        for model, model_aic in aics.groupby('model'):
            assert all(model_aic.apply(lambda x: len(np.unique(x))) == 1)

    aics = fitgrid.utils.summary._get_AICs(summaries_df)

    assert (
        RHSs
        == summaries_df.index.unique('model').tolist()
        == aics.index.unique('model').tolist()
    )

    for (time, chan), tc_aics in aics.groupby(['Time', 'channel']):
        # mins at each time, channel
        min = tc_aics['AIC'].min()
        assert np.allclose(
            tc_aics['AIC'].astype('float') - min,
            tc_aics['min_delta'].astype('float'),
            atol=0,
        )

    return aics


def test_smoke_plot_betas():
    """TO DO: needs argument testing"""

    for summary_df in test_summarize():
        cols = [col for col in summary_df.columns if "channel" in col]
        for fdr in [None, 'BY', 'BH']:
            _ = fitgrid.utils.summary.plot_betas(
                summary_df=summary_df, LHS=cols, fdr=fdr
            )
            plt.close('all')

        for df_func in [None, log10]:
            _ = fitgrid.utils.summary.plot_betas(
                summary_df=summary_df, LHS=cols, df_func=df_func
            )
            plt.close('all')


def test_smoke_plot_AICs():

    for summary_df in test_summarize():
        f, axs = fitgrid.utils.summary.plot_AICmin_deltas(summary_df)
        plt.close('all')
