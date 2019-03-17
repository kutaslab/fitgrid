import warnings
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


# data bank of what there is, how it runs, data type
# test_lm_utils.py guards the attrs and data types
#
#   nobs = number of observations
#   nobs_k = number of observations x model regressors
#   nobs_loop = nobs iteration ... slow
#

FLOAT_TYPE = np.float64
INT_TYPE = np.int64

# attr : calc_type, value_dtype, index_names returned by fitgrid
_OLS_INFLUENCE_ATTRS = {
    '_get_drop_vari': (None, None),
    '_res_looo': (None, None),
    '_ols_xnoti': (None, None),
    'aux_regression_exog': (None, None),
    'aux_regression_endog': (None, None),
    'cooks_distance': ('nobs', FLOAT_TYPE, ['Time', None, 'Epoch_idx']),
    'cov_ratio': ('nobs_loop', FLOAT_TYPE, ['Time', 'Epoch_idx']),
    'det_cov_params_not_obsi': (
        'nobs_loop',
        FLOAT_TYPE,
        ['Time', 'Epoch_idx'],
    ),
    'dfbetas': ('nobs_loop', FLOAT_TYPE, ['Time', 'Epoch_idx', None]),
    'dffits': ('nobs_loop', FLOAT_TYPE, ['Time', None, 'Epoch_idx']),
    'dffits_internal': ('nobs', FLOAT_TYPE, ['Time', None, 'Epoch_idx']),
    'endog': (None, None),  # ('nobs', FLOAT_TYPE),  # from data
    'ess_press': ('nobs', FLOAT_TYPE, ['Time']),
    'exog': (None, None),  # 'nobs_k', FLOAT_TYPE),  # from data
    'get_resid_studentized_external': (None, None),  # method
    'hat_diag_factor': ('nobs', FLOAT_TYPE, ['Time', 'Epoch_idx']),
    'hat_matrix_diag': ('nobs', FLOAT_TYPE, ['Time', 'Epoch_idx']),
    'influence': ('nobs', FLOAT_TYPE, ['Time', 'Epoch_idx']),
    'k_vars': ('nobs', INT_TYPE, ['Time']),
    'model_class': (None, None),  # not a DataFrame
    'nobs': ('nobs', INT_TYPE, ['Time']),
    'params_not_obsi': ('nobs_loop', FLOAT_TYPE, ['Time', 'Epoch_idx', None]),
    'resid_press': ('nobs', FLOAT_TYPE, ['Time', 'Epoch_idx']),
    'resid_std': ('nobs', FLOAT_TYPE, ['Time', 'Epoch_idx']),
    'resid_studentized_external': (
        'nobs_loop',
        FLOAT_TYPE,
        ['Time', 'Epoch_idx'],
    ),
    'resid_studentized_internal': ('nobs', FLOAT_TYPE, ['Time', 'Epoch_idx']),
    'resid_var': ('nobs', FLOAT_TYPE, ['Time', 'Epoch_idx']),
    'results': (None, None),  # not a DataFrame
    'save': (None, None),  # not a DataFrame
    'sigma2_not_obsi': ('nobs_loop', FLOAT_TYPE, ['Time', 'Epoch_idx']),
    'sigma_est': ('nobs', FLOAT_TYPE, ['Time']),
    'summary_frame': (None, None),  # not a DataFrame
    'summary_table': (None, None),  # not a DataFrame
}


_FG_LM_DIAGNOSTIC_COLUMNS = [
    'Epoch_idx',
    'Time',
    'channel',
    'diagnostic',
    'value',
    'critical',
]


def _check_influence_attr(infl, infl_attr, do_nobs_loop):

    if infl_attr not in _OLS_INFLUENCE_ATTRS:
        raise ValueError(f"unknown OLSInfluence attribute {infl_attr}")

    infl_calc, infl_dtype, index_names = _OLS_INFLUENCE_ATTRS[infl_attr]

    if infl_calc is None:
        raise ValueError(f"fitgrid cannot calculate {infl_attr}")

    if infl_calc == "nobs_loop":
        if not do_nobs_loop:
            msg = f"{infl_attr} is slow, to calculate set do_nob_loop=True"
            raise ValueError(msg)

    attr_grid = getattr(infl, infl_attr)
    if not isinstance(attr_grid, pd.DataFrame):
        raise TypeError(f"{infl_attr} grid is not a pandas DataFrame")

    actual_type = type(getattr(infl, infl_attr).iloc[0, 0])
    if actual_type is not infl_dtype:
        raise TypeError(f"gridded {infl_attr} dtype should be {infl_dtype}")

    if not index_names == attr_grid.index.names:
        raise TypeError(
            f" OLS_INFLUENCE_ATTRS thinks {infl_attr} index"
            f" names should be be {index_names},  the grid"
            f" index names are {attr_grid.index.names}"
        )


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


def _get_infl_attr_vals(lm_grid, attr, crit_val=None, do_nobs_loop=False):
    """scrape OLSInfluence attribute data into long form Time, Epoch, Chan

    Parameters
    ----------
    lm_grid : LMFitGrid

    attr : string
       see _OLS_INFLUENCE_ATTRS for supported attributes

    crit_val : TBD

    do_nobs_loop : bool
       if True, calculate nobs loop measures. Default false

    Return
    ------
    attr_df : pd.DataFrame
        row index names: time, epoch_id, channel
        columns vary by attribute

    """

    infl = lm_grid.get_influence()
    _check_influence_attr(infl, attr, do_nobs_loop)

    attr_df = getattr(infl, attr)  # .loc[vals_slicer, :]

    # switch everything to long form

    # standardize the row index,
    #  - fill in None with attr
    #  - propogate attr name to index level labels
    index_names, none_count, attr_idxs = [], 0, []
    for i, name in enumerate(attr_df.index.names):
        if name is not None:
            index_names.append(name)
        else:
            # process unnamed leels

            # 1. capture the offset for unstacking
            attr_idxs.append(i)

            # 2. rename name the column from the attr, extend if needed
            nc_str = '' if none_count == 0 else str(none_count)
            attr_idx_name = f"{attr}{nc_str}"
            index_names.append(attr_idx_name)

            # 2. update this level labels with the new name
            new_labels = [
                f"{attr_idx_name}_{j}" for j in attr_df.index.levels[i]
            ]
            attr_df.index.set_levels(new_labels, i, inplace=True)

            # increment on the way out
            none_count += 1

    attr_df.index.names = index_names
    if len(attr_idxs) > 0:
        attr_df = attr_df.unstack(attr_idxs)
    attr_df = attr_df.stack(0)
    attr_df.index.rename(attr_df.index.names[:-1] + ['Channel'], inplace=True)
    # promote series to frame
    if isinstance(attr_df, pd.Series):
        attr_df.name = attr
        attr_df = attr_df.to_frame()

    # standard-ish long Time, Epoch,  Channel dataframe
    return attr_df


def _get_infl_cooks_distance(lm_grid, crit_val=None):
    """backend Cook's D grid scraper, returns 2-D row, col indexs

    statmodels returns a D and p-value, we want the critical D

    n epochs, and p model params are constant across a grid, so we can
    get away with one critical value, all grid cells

    """

    if not (crit_val is None or isinstance(crit_val, float)):
        msg = f"crit_val must be a floating point number"
        raise TypeError(msg)

    if crit_val is None:
        # fall back to calculated median critical D = F(n, n - p )
        infl = lm_grid.get_influence()

        dfn = int(np.unique(infl.k_vars)[0])
        assert isinstance(dfn, int)

        dfd = int(np.unique(infl.results.df_resid)[0])
        assert isinstance(dfd, int)

        # look up the approx F_0.5 for the model
        # median F (p, n - p)  about 1.0, for large n
        fcdf = stats.f.cdf(np.linspace(0, 100, num=1000), dfn, dfd)
        crit_val = np.median(fcdf)

    # cooks D grid ... hanging slice is black's idea not mine
    infl_vals = _get_infl_attr_vals(lm_grid, 'cooks_distance')
    infl_val_idxs = np.where(infl_vals['cooks_distance_0'] > crit_val)

    infl_vals = infl_vals.iloc[infl_val_idxs]
    infl_vals['cooks_distance_1'] = crit_val
    infl_vals.index = infl_vals.index.remove_unused_levels()
    infl_vals.sort_values(by=['Time', 'Epoch_idx', 'Channel'])

    return infl_vals, infl_val_idxs


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

    # infl_vals_grid = (
    #     infl.dffits_internal.loc[pd.IndexSlice[:, 0], :]
    #     .reset_index(-1, drop=True)
    #    .stack(-1)
    #    .apply(lambda x: pd.Series(x, index=infl.epoch_index))
    #    .unstack(0)
    # ).T

    infl_vals_grid = infl.dffits_internal.loc[
        pd.IndexSlice[:, 0, :], :
    ].reset_index(1, drop=True)

    # the munging and testing could be one line but oh boy
    infl_idxs = np.where(infl_vals_grid > crit_val)

    # slice out just the influential data points
    infl_data = []  # list of tuples
    for row, col in zip(*infl_idxs):

        infl_data.append(
            infl_vals_grid.index[row][::-1]
            + (infl_vals_grid.columns[col],)
            + ('dffits_internal',)
            + (infl_vals_grid.iloc[row, col], crit_val)
        )

    infl_data_df = pd.DataFrame(
        infl_data, columns=_FG_LM_DIAGNOSTIC_COLUMNS
    ).sort_values(['Epoch_idx', 'Time', 'channel'])

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
