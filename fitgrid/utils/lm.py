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

# ------------------------------------------------------------
# r.e. statsmodels OLS influence attributes
#
# fitgrid's database of what there is, how it runs, and data type.
#
# used for checking and to guard user input
#
# suitable for generating a docs table, but not implemented
#
#   nobs = number of observations
#   nobs_k = number of observations x model regressors
#   nobs_loop = nobs iteration ... slow
#
FLOAT_TYPE = np.float64
INT_TYPE = np.int64

# attr : (calc_type, value_dtype, index_names) as returned by fitgrid
_OLS_INFLUENCE_ATTRS = {
    '_get_drop_vari': (None, None, None),
    '_res_looo': (None, None, None),
    '_ols_xnoti': (None, None, None),
    'aux_regression_exog': (None, None, None),
    'aux_regression_endog': (None, None, None),
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
    'endog': (None, None, None),  # ('nobs', FLOAT_TYPE),  # from data
    'ess_press': ('nobs', FLOAT_TYPE, ['Time']),
    'exog': (None, None, None),  # 'nobs_k', FLOAT_TYPE),  # from data
    'get_resid_studentized_external': (None, None, None),  # method
    'hat_diag_factor': ('nobs', FLOAT_TYPE, ['Time', 'Epoch_idx']),
    'hat_matrix_diag': ('nobs', FLOAT_TYPE, ['Time', 'Epoch_idx']),
    'influence': ('nobs', FLOAT_TYPE, ['Time', 'Epoch_idx']),
    'k_vars': ('nobs', INT_TYPE, ['Time']),
    'model_class': (None, None, None),  # not a DataFrame
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
    'results': (None, None, None),  # not a DataFrame
    'save': (None, None, None),  # not a DataFrame
    'sigma2_not_obsi': ('nobs_loop', FLOAT_TYPE, ['Time', 'Epoch_idx']),
    'sigma_est': ('nobs', FLOAT_TYPE, ['Time']),
    'summary_frame': (None, None, None),  # not a DataFrame
    'summary_table': (None, None, None),  # not a DataFrame
}


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


# ------------------------------------------------------------
# getter function argument and getter checkers
# ------------------------------------------------------------
def _check_get_infl_args(lm_grid, infl_attr, direction, crit_val):
    """argument type, value checker, doesn't run anything"""

    if not isinstance(lm_grid, pd.DataFrame):
        raise TypeError(f"lm_grid is not a pandas DataFrame")

    if not (
        isinstance(infl_attr, str) and infl_attr in _OLS_INFLUENCE_ATTRS
    ):
        raise ValueError(f"unknown OLSInfluence attribute {infl_attr}")

    if not (
            (crit_val is None)
            or (crit_val != 'sm')
            or (isinstance(crit_val, float))
            or (hasattr(crit_val, '__call__'))
    ):
        msg = "crit_val must be None, 'sm', a float or a function"
        raise TypeError(msg)

    # need a direction with critical values
    if crit_val is not None:
        if direction not in ['above', 'below']:
            raise ValueError(
                f"crit_val requires a direction 'above' or 'below"
            )

    return 0


def _check_influence_attr(infl, infl_attr, do_nobs_loop):
    """general purpose checker and raw grid getter, may be slow"""

    if infl_attr not in _OLS_INFLUENCE_ATTRS:
        raise ValueError(f"unknown OLSInfluence attribute {infl_attr}")

    infl_calc, infl_dtype, index_names = _OLS_INFLUENCE_ATTRS[infl_attr]

    if infl_calc is None:
        raise ValueError(f"fitgrid cannot calculate {infl_attr}")

    if infl_calc == "nobs_loop" and not do_nobs_loop:
        msg = f"{infl_attr} is slow, to calculate anyway set do_nob_loop=True"
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

    return attr_grid


# ------------------------------------------------------------
# back end workers ...
# ------------------------------------------------------------
def _get_infl_attr_vals(lm_grid, attr, do_nobs_loop=False):
    """scrape OLSInfluence attribute data into long form Time, Epoch, Chan

    Parameters
    ----------
    lm_grid : LMFitGrid

    attr : string
       see _OLS_INFLUENCE_ATTRS for supported attributes

    direction : {'above', 'below'}
       data points to return relative to critical value, if set

    do_nobs_loop : bool
       if True, calculate slow leave-one-out nobs loop, default=False

    Return
    ------
    attr_df : pd.DataFrame
        grid_time, grid_epoch_id, channel
        columns vary by attribute

    """

    infl = lm_grid.get_influence()
    attr_df = _check_influence_attr(infl, attr, do_nobs_loop)

    # switch the 2-D Time, Epoch x Channels grid
    # to long format dataframe Time, Epoch, Channels

    # standardize the row index with attribut name and labels
    index_names, none_idxs, none_count = [], [], 0
    for i, name in enumerate(attr_df.index.names):
        if name is not None:
            index_names.append(name)
        else:
            # capture the index offset for unstacking
            none_idxs.append(i)

            # rename name the None column from the attr
            nc_str = '' if none_count == 0 else str(none_count)
            attr_idx_name = f"{attr}{nc_str}"
            index_names.append(attr_idx_name)

            # index labels are 0, 1, ... prepend the attr
            new_labels = [
                f"{attr_idx_name}_{j}" for j in attr_df.index.levels[i]
            ]
            attr_df.index.set_levels(new_labels, i, inplace=True)

            # increment on the way out
            none_count += 1

    attr_df.index.names = index_names
    if len(none_idxs) > 0:
        attr_df = attr_df.unstack(none_idxs)
    attr_df = attr_df.stack(0)
    attr_df.index.names = attr_df.index.names[:-1] + ['Channel']

    # promote series so all returns are data frames
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

    infl_df, infl_idxs = None, None
    infl_df = _get_infl_attr_vals(lm_grid, 'cooks_distance')

    if crit_val is not None:
        infl_idxs = np.where(infl_df['cooks_distance_0'] > crit_val)
        infl_df = infl_df.iloc[infl_idxs]
        infl_df['cooks_distance_1'] = crit_val
        infl_df.index = infl_df.index.remove_unused_levels()

    infl_df.sort_values(by=['Time', 'Epoch_idx', 'Channel'])
    return infl_df, infl_idxs


def _get_infl_dffits_internal(lm_grid, crit_val='sm', direction='above'):
    """backend dffit grid scraper"""

    infl_df = _get_infl_attr_vals(lm_grid, 'dffits_internal')

    # no way around special critical value handling for dffits
    # b.c. it returns a crit val scalar by default in second
    # column
    if crit_val == 'sm':
        pass  # use statsmodels default
    elif crit_val is None or isinstance(crit_val, float):
        infl_df['dffits_internal_1'] = crit_val
    elif hasattr(crit_val, '__call__'):
        # crit_val(lm_grid)
        raise NotImplementedError('TO DO')
    else:
        raise TypeError('crit_val must be None, sm, float, or function')

    if crit_val is not None:
        if direction == 'above':
            cond = infl_df['dffits_internal_0'] > crit_val
        elif direction == 'below':
            cond = infl_df['dffits_internal_0'] < crit_val
        else:
            raise ValueError('bug in lm.py illegal direction={direction}')

        infl_idxs = np.where(cond)
        infl_df = infl_df.iloc[infl_idxs]
        infl_df.index = infl_df.index.remove_unused_levels()

    infl_df.sort_values(by=['Time', 'Epoch_idx', 'Channel'])
    return infl_df, infl_idxs


def get_influential_data(lm_grid, diagnostic, crit_val=None):
    """all the FitGrid's influential datapoints as an Epoch, Time, Chan dataframe

    Parameters
    ----------
    lm_grid : fitgrid.FitGrid
        as returned by fg.lm()

    diagnostic : {cooks_distance, dffits_internal}
        see `statsmodels.OLSInfluence` docs

    crit_val : {None, float, 'sm', func}
       critical value cutoff for filtering returned data points

       `None` return all, may be very large for data point influence measures

       `float` is explicit value, e.g., from a user calculation

       `sm` is the statsmodels default, e.g., for cook's D, dffits

       `func` is a function that takes `lm_grid`, `attr` and
            returns one critical val float or same shape grid of them

    direction : {'above','below'}
       which side of the critical value to return

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
