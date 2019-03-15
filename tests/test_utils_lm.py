import pandas as pd
from .context import fitgrid

PARALLEL = True
N_CORES = 4
SEED = 0


def test_smoke_get_vifs():

    epochs = fitgrid.generate()
    RHS = 'continuous + categorical'
    fitgrid.utils.lm.get_vifs(epochs, RHS)


def test_smoke__get_infl_dffits():

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
    infl_data_df, infl_idxs = fitgrid.utils.lm._get_infl_dffits_internal(infl)


def test_smoke__get_infl_cooks_distance():

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
    infl_data_df, infl_idxs = fitgrid.utils.lm._get_infl_cooks_distance(infl)


def test_smoke_get_influential_data():
    """wrapper"""

    def _check_infl_data_df(iddf, with_cval=False, cval=None):
        assert isinstance(infl_data_df, pd.DataFrame)
        assert all(
            infl_data_df.columns == fitgrid.utils.lm._FG_LM_DIAGNOSTIC_COLUMNS
        )
        assert all(infl_data_df['diagnostic'] == diagnostic)
        # print(infl_data_df.head())
        # print(infl_data_df.tail())
        # print(f"get_influential data({diagnostic}) ok")

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

    # test the crit_val param
    for diagnostic in ['dffits_internal', 'cooks_distance']:
        infl_data_df = fitgrid.utils.lm.get_influential_data(
            lm_grid, diagnostic
        )
        _check_infl_data_df(infl_data_df, with_cval=False)

        for cval in [None, 0.1]:
            infl_data_df = fitgrid.utils.lm.get_influential_data(
                lm_grid, diagnostic, crit_val=cval
            )
            _check_infl_data_df(infl_data_df, with_cval=True, cval=cval)
