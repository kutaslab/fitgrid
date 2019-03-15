import numpy as np
from scipy import stats
import pandas as pd
import patsy
from tqdm import tqdm
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor as vif,
)
from statsmodels.regression.linear_model import RegressionResultsWrapper

from fitgrid.fitgrid import FitGrid

_FG_LM_DIAGNOSTIC_COLUMNS = [
    'Epoch_idx',
    'Time',
    'channel',
    'diagnostic',
    'value',
    'critical',
]


def get_vifs(epochs, RHS):
    def get_single_vif(group, RHS):
        dmatrix = patsy.dmatrix(formula_like=RHS, data=group)
        vifs = {
            name: vif(dmatrix, index)
            for name, index in dmatrix.design_info.column_name_indexes.items()
        }
        return pd.Series(vifs)

    tqdm.pandas(desc="Time")

    return epochs._snapshots.progress_apply(get_single_vif, RHS=RHS)


def _get_infl_cooks_distance(infl, crit_val=None):
    """backend Cook's D grid scraper, returns 2-D row, col indexs

    statmodels returns a D and p-value, we want the critical D

    Since n epochs, and p model params are constant across
    a grid, we can get away with one critical value, all cells

    TPU
    """

    if not (crit_val is None or isinstance(crit_val, float)):
        msg = f"crit_val must be a floating point number"
        raise TypeError(msg)

    if crit_val is None:
        # fall back to calculated median critical D = F(n, n - p )

        dfn = int(np.unique(infl.k_vars)[0])
        assert isinstance(dfn, int)

        dfd = int(np.unique(infl.results.df_resid)[0])
        assert isinstance(dfd, int)

        # look up the approx F_0.5 for the model
        # median F (p, n - p)  about 1.0, for large n
        fcdf = stats.f.cdf(np.linspace(0, 100, num=1000), dfn, dfd)
        crit_val = np.median(fcdf)

    # cooks D grid ... hanging slice is black's idea not mine
    infl_vals_grid = infl.cooks_distance.loc[
        pd.IndexSlice[:, 0, :], :
    ].reset_index(1, drop=True)

    assert crit_val is not None

    # grid indices, not values
    infl_idxs = np.where(infl_vals_grid > crit_val)

    infl_data = []  # list of tuples
    for row, col in zip(*infl_idxs):
        infl_data.append(
            infl_vals_grid.index[row]
            + (infl_vals_grid.columns[col],)
            + ('cooks_distance',)
            + (infl_vals_grid.iloc[row, col], crit_val)
        )

    infl_data_df = pd.DataFrame(infl_data, columns=_FG_LM_DIAGNOSTIC_COLUMNS)

    return infl_data_df, infl_idxs


def _get_infl_dffits_internal(infl, crit_val=None):
    """backend dffit grid scraper ... use the wrapper """

    if crit_val is None:
        # fall back to statsmodels default

        # else blackened at the 0 index ... too much of a good thing
        slicer = pd.IndexSlice
        crit_val = np.unique(infl.dffits_internal.loc[slicer[:, 1], :])[0]

    assert isinstance(crit_val, float)

    # TO DO ... these could get big ... iterate over grid cells to save space?

    # reindex with infl FitGrid epoch_ids

    # Unlike Cook's D, all epoch dffit values are nested as a list
    # in each Time x Chan grid cell. Munge back into a epoch_id indexed
    # dataframe like so ...
    infl_vals_grid = (
        infl.dffits_internal.loc[pd.IndexSlice[:, 0], :]
        .reset_index(-1, drop=True)
        .stack(-1)
        .apply(lambda x: pd.Series(x, index=infl.epoch_index))
        .unstack(0)
    ).T

    # the munging and testing could be one line but oh boy
    infl_idxs = np.where(infl_vals_grid > crit_val)

    # slice out just the influential data points
    infl_data = []  # list of tuples
    for row, col in zip(*infl_idxs):

        infl_data.append(
            infl_vals_grid.index[row]
            + (infl_vals_grid.columns[col],)
            + ('dffits_internal',)
            + (infl_vals_grid.iloc[row, col], crit_val)
        )

    infl_data_df = pd.DataFrame(infl_data, columns=_FG_LM_DIAGNOSTIC_COLUMNS)

    return infl_data_df, infl_idxs


def get_influential_data(lm_grid, diagnostic, crit_val=None):
    """all the FitGrid's influential datapoints as an Epoch, Time, Chan dataframe

    Parameters
    ----------
    lm_grid : fitgrid.FitGrid
        as returned by fg.lm()

    diagnostic : {cooks_distance, dffits_internal}
        see `statsmodels.OLSInfluence` docs

    Returns
    -------
        infl_data_df

    Notes
    -----

        diagnostic critical values are defined in statsmodels.OLSInfluence

    """

    # modicum of guarding
    msg = None
    if not isinstance(lm_grid, FitGrid):
        msg = f"lm_grid must be a FitGrid not {type(lm_grid)}"

    if not isinstance(lm_grid.tester, RegressionResultsWrapper):
        msg = f"lm_grid must be fit with fitgrid.lm()"

    if msg is not None:
        raise TypeError(msg)

    # ask statsmodels about influence
    infl = lm_grid.get_influence()

    if diagnostic == 'cooks_distance':
        infl_data_df, _ = _get_infl_cooks_distance(infl, crit_val)

    elif diagnostic == 'dffits_internal':
        infl_data_df, _ = _get_infl_dffits_internal(infl, crit_val)

    else:
        raise NotImplementedError(f"{diagnostic}")

    return infl_data_df
