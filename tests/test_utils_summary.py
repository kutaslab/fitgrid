import pytest
from numpy import log10
import pandas as pd
from matplotlib import pyplot as plt
import fitgrid
from fitgrid.utils.summary import INDEX_NAMES  # , KEY_LABELS

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

    # modelers and RHSs
    tests = {
        "lm": [
            "1 + continuous + categorical",
            "1 + continuous",
            "1 + categorical",
            "1",
        ],
        "lmer": [
            "1 + continuous + (1 | categorical)",
            "1 + (1 | categorical)",
        ],
    }

    epochs_fg = _get_epochs_fg()

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
        assert summaries_df.index.names == INDEX_NAMES
        fitgrid.utils.summary._check_summary_df(summaries_df)
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
    attr_keys = summary_keys.difference(lmer_fit_betas_keys)
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
    """stub"""

    RHSs = [
        "1 + continuous + categorical",
        "1 + continuous",
        "1 + categorical",
    ]

    epochs_fg = _get_epochs_fg()
    summaries_df = fitgrid.utils.summary.summarize(
        epochs_fg, 'lm', LHS=epochs_fg.channels, RHS=RHSs
    )

    aics = fitgrid.utils.summary._get_AICs(summaries_df)
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
        _ = fitgrid.utils.summary.plot_AICmin_deltas(summary_df)
        plt.close('all')
