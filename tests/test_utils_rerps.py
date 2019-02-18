import fitgrid
from fitgrid.utils.rerps import INDEX_NAMES, KEY_LABELS

def _get_epochs_fg():
    # pretend we are starting the pipeline with user epochs dataframe

    # generate fake data
    fake_epochs = fitgrid.generate(
        n_samples=5,
        n_channels=2,
        n_categories=2
    )
    epochs_df = fake_epochs.table
    chans = fake_epochs.channels

    # convert to fitgrid epochs object
    epochs_fg = fitgrid.epochs_from_dataframe(
        epochs_df.reset_index().set_index(['Epoch_idx', 'Time']),
        channels=chans,
        epoch_id="Epoch_idx",
        time='Time'
    )

    return epochs_fg


def test__lm_get_coefs_df():

    fgrid_lm = fitgrid.lm(
        _get_epochs_fg(),
        RHS="1 + continuous + categorical",
        n_cores=8
    )

    coefs_df = fitgrid.utils.rerps._lm_get_coefs_df(fgrid_lm)
    fitgrid.utils.rerps._check_rerps_df(coefs_df)


def test__lmer_get_coefs_df():

    fgrid_lmer = fitgrid.lmer(
        _get_epochs_fg(),
        RHS="1 + continuous + (1 | categorical)",
        n_cores=8
    )

    coefs_df = fitgrid.utils.rerps._lmer_get_coefs_df(fgrid_lmer)
    fitgrid.utils.rerps._check_rerps_df(coefs_df)


def test_get_rerps():
    """test main wrapper to scrape rerps from either lm or lmer grids"""

    n_cores = 12

    # modelers and RHSs
    tests = {
        "lm": [
            "1 + continuous + categorical",
            "1 + continuous",
            "1 + categorical",
            "1"
        ],
        "lmer": [
            "1 + continuous + (1 | categorical)",
            "1 + (1 | categorical)",
        ]
    }
    
    epochs_fg = _get_epochs_fg()

    # do it
    for modler, RHSs in tests.items():
        rerps_df = fitgrid.utils.rerps.get_rerps(
            epochs_fg,
            modler,
            LHS=epochs_fg.channels,
            RHS=RHSs,
            n_cores=n_cores
        )
        assert rerps_df.index.names == INDEX_NAMES
        assert set(KEY_LABELS).issubset(set(rerps_df.index.levels[-1]))

    return rerps_df

    
def test_get_AICs():
    """stub"""

    RHSs = [
        "1 + continuous + categorical",
        "1 + continuous",
        "1 + categorical",
    ]

    epochs_fg = _get_epochs_fg()
    rerps_df = fitgrid.utils.rerps.get_rerps(
        epochs_fg,
        'lm',
        LHS=epochs_fg.channels,
        RHS=RHSs
    )

    aics = fitgrid.utils.rerps.get_AICs(rerps_df)
    return aics



def test_smoke_plot_chans():
    """TO DO: needs argument testing"""

    rerps_df = test_get_rerps()
    fitgrid.utils.rerps.plot_chans(
        rerps_df=rerps_df,
        LHS=[col for col in rerps_df.columns if "channel" in col]
    )


def test_plot_AICs():

   aics = test_get_AICs()
   f = fitgrid.utils.rerps.plot_AICs(aics)
