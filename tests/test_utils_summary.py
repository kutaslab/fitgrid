import pytest
import re
from numpy import log10
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import fitgrid
from fitgrid.utils.summary import INDEX_NAMES, KEY_LABELS, PER_MODEL_KEY_LABELS
from .context import tpath, FIT_RTOL

pd.set_option("display.width", 256)
PARALLEL = True
N_CORES = 4


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
    """test main wrapper to scrape summaries from either lm or lmer grids"""

    # column sums ... guard against unexpected changes
    # lm_checksum = np.array([65721.957_801_9, 97108.199_717_59])
    # lmer_checksum = np.array([27917.386_516_15, 63191.613_344_82])

    lm_checksum = np.array([120_135.560_853_52, 172_375.489_486_64])
    lmer_checksum = np.array([41748.779_227, 90712.637_260])  # mkl?
    # lmer_checksum_2 = np.array([41756.165_766_74, 90723.291_317_1]) # blas

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

        # verify checksums and select the modler
        if modler == 'lm':
            assert np.allclose(summaries_df.apply(sum), lm_checksum, atol=0)
            modler_ = fitgrid.lm
        elif modler == 'lmer':

            # check the summary is correct for a known LMER build
            assert np.allclose(
                summaries_df.apply(sum), lmer_checksum, atol=0, rtol=FIT_RTOL
            )
            modler_ = fitgrid.lmer
        else:
            raise ValueError('bad modler')

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
        # ... guard against slicing-induced swizzled row indexes
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
            rtol=FIT_RTOL,
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
