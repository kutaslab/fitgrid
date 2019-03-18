import numpy as np
from scipy import stats
import pandas as pd
import patsy
from tqdm import tqdm
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor as vif,
)
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.outliers_influence import OLSInfluence
import warnings

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
        index is grid_time, grid_epoch_id, channel
        columns vary by attribute

    sm_1 : pd.DataFrame
        emtpy except when statsmodels return is a 2-ple (vals_0, vals_1),
        then vals_1

    """

    attr_df, sm_1 = None, None
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


    import pdb
    pdb.set_trace()

    # special case handling to split up dataframe from statsmodel tuples
    if attr in ["cooks_distance", "dffits_internal", "dffits_external"]:
        sm_1 = attr_df[attr_df.columns[-1]].to_frame()
        sm_1.columns = [f"{col}_sm" for col in sm_1.columns]
        del attr[attr_df.columns[-1]

    # standard-ish long Time, Epoch,  Channel dataframes
    return attr_df, sm_1

# ------------------------------------------------------------
# idiosyncratic backends for UI getter
# ------------------------------------------------------------
# def _get_infl_cooks_distance(lm_grid, crit_val=None):
#     """backend Cook's D grid scraper, returns 2-D row, col indexs

#     statmodels returns a D and p-value, we want the critical D

#     n epochs, and p model params are constant across a grid, so we can
#     get away with one critical value, all grid cells

#     """
#     infl_df, infl_idxs = None, None
#     infl_df = _get_infl_attr_vals(lm_grid, 'cooks_distance')

#     infl_df.sort_values(by=['Time', 'Epoch_idx', 'Channel'])
#     return infl_df, infl_idxs


# def _get_infl_dffits_internal(lm_grid, crit_val='sm', direction='above'):
#     """backend dffit grid scraper"""

#     infl_df, infl_idxs = None, None
#     infl_df = _get_infl_attr_vals(lm_grid, 'dffits_internal')
#     infl_df = _crit_val_handler(infl, diagnostic, crit_val)

#     infl_df.sort_values(by=['Time', 'Epoch_idx', 'Channel'])
#     return infl_df, infl_idxs


def _crit_val_handler(infl_df, infl, diagnostic, direction, crit_val):

    # No way around special critical value handling b.c. statsmodels
    # returns a critical value or p for cooks_distance, dffits, but
    # not others
    # 
    # There are 16 (4 x 4) i, j cases:
    #   4 crit_val:  None, float, 'sm', function(infl)
    #   4 diagnostics:
    #      cooks_distance, dffits_internal, dffits_external, the rest
    #
    # Approach
    #   * set crit_val_ according to the crit_val i, diagnostic j case
    #   * slice infl_df with crit_val_ and return

    # deviations should be caught in the UI, check anyhow
    assert isinstance(infl_df, pd.DataFrame)
    assert isinstance(infl, FitGrid)
    assert diagnostic in _OLS_INFLUENCE_ATTRS.keys()
    assert direction in ['above', 'below']

    # default: assume we aren't thresholding by crit_val
    infl_idxs, crit_val_ = np.ndarray(shape=(0,)), np.ndarray(shape=(0,))

    import pdb

    infl_idxs, crit_val_ = None, None

    # first strip oddball diagnostic dataframes down to just the diagnostic
    # values, like all the rest, but retain the statsmodels computations
    # for the 'sm' option
    special_diagnostics = [
        'cooks_distance', 'dffits_internal', 'dffits_external'
    ]
    if diagnostic in special_diagnostics:
        # these infl_df have an extra colum froom the statsmodels
        # (values, crit) tuple, strip it from the data frame
        assert infl_df.columns[1] == f"{diagnostic}_1"
        crit_val_ = infl_df[infl_df.columns[1]].to_frame()
        crit_val_.columns = [f"{col}_sm_crit_val" for col in crit_val_.columns]
        infl_df.drop(columns=[infl_df.columns[1]], inplace=True)

    # now all infl_dfs are the same format, handle crit_val by case

    # ------------------------------------------------------------
    # case i==0 crit_val is None
    # j = 0,1,2,3 all diagnostics
    if crit_val is None:
        # do not subset results. special case crit_val_ is returned
        return infl_df, infl_idxs, crit_val_

    # ------------------------------------------------------------
    # case i==2 crit_val == 'sm' for statsmodel default
    # case j=1,2,3 special diagnostic crit_val were handled above
    if crit_val == 'sm':
        # case j=4 non-special case diagnostics have no sm crit_values
        if diagnostic not in special_diagnostics:
            msg = (
                f"ignoring crit_val='sm', statsmodels does not define"
                " a critical value for {diagnostic}"
            )
            warnings.warn(msg)
            infl_idxs, crit_val_ = None, None

    # ------------------------------------------------------------
    # case i==1 crit_val is float-like
    try:
        # convertible to float?
        crit_val_ = float(crit_val)
    except Exception:
        pass

    # ------------------------------------------------------------
    # i==3 crit_val is a function
    if hasattr(crit_val, '__call__'):
        crit_val_ = crit_val(infl)
        warnings.warn('functions not implemented')

    if crit_val_ is None:
        infl_idxs = None
        return infl_df, infl_idxs, crit_val_

    # assert isinstance(crit_val_, pd.DataFrame)
    # assert crit_val_.shape[0] == 1 or crit_val_.shape == infl_df.shape

    # wrap up ... most infl_df dataframes now are all just the
    # calculated values, having the special case special case critical
    # value columns in case (1, 1:3)
    # The exceptions are special cases where statsmodels returns a
    # a tuple (values, crit) and the crit got stripped above. So
    # every
    # was pre-decorated with statsmodels critical value h
    try:
        if direction == 'above':
            infl_idxs = np.where(infl_df > crit_val_.to_numpy())
        elif direction == 'below':
            infl_idxs = np.where(infl_df < crit_val_.to_numpy())
        else:
            msg = 'lm.py bad direction={direction} please report an issue'
            raise ValueError(msg)
    except Exception:
        pdb.set_trace()

    infl_df = infl_df.iloc[infl_idxs]
    infl_df.index = infl_df.index.remove_unused_levels()

    return infl_df, crit_val_, infl_idxs

    # # DEPRECATED ------

    # if crit_val is not None:
    #     infl_idxs = np.where(infl_df['cooks_distance_0'] > crit_val)
    #     infl_df = infl_df.iloc[infl_idxs]
    #     infl_df['cooks_distance_1'] = crit_val
    #     infl_df.index = infl_df.index.remove_unused_levels()

    # # dffits internal ----------------------------------------
    # if crit_val == 'sm':
    #     pass  # use statsmodels default
    # elif crit_val is None or isinstance(crit_val, float):
    #     infl_df['dffits_internal_1'] = crit_val
    # elif hasattr(crit_val, '__call__'):
    #     # crit_val(lm_grid)
    #     raise NotImplementedError('TO DO')
    # else:
    #     raise TypeError('crit_val must be None, sm, float, or function')

    # if crit_val is not None:

    # # general case ----------------------------------------
    # # statsmodels doesn't define a critical value for most diagnostics
    # if crit_val in ['sm', None] or isinstance(crit_val, float):
    #     infl_df['critical_value'] = crit_val
    # elif hasattr(crit_val, '__call__'):
    #     # crit_val(lm_grid)
    #     raise NotImplementedError('TO DO')
    # else:
    #     raise TypeError('crit_val must be None, sm, float, or function')

    # if crit_val is not None:
    #     if direction == 'above':
    #         infl_idxs = np.where(infl_df > crit_val)
    #     elif direction == 'below':
    #         infl_idxs = np.where(infl_df < crit_val)
    #     else:
    #         msg = 'lm.py bad direction={direction} please report an issue'
    #         raise ValueError(msg)

    #     infl_df = infl_df.iloc[infl_idxs]
    #     infl_df.index = infl_df.index.remove_unused_levels()


# ------------------------------------------------------------
# User wrappers
# ------------------------------------------------------------
def list_diagnostics():
    """brief description and usage"""

    fast = [
        f"  get_diagnostic(lm_grid, {attr}, direction, crit_val)"
        for attr, spec in _OLS_INFLUENCE_ATTRS.items()
        if spec[0] not in [None, 'nobs_loop']
    ]

    slow = [
        (
            f"  get_diagnostic(lm_grid, {attr}, direction, crit_val,"
            " do_nobs_loop=True)"
        )
        for attr, spec in _OLS_INFLUENCE_ATTRS.items()
        if spec[0] == 'nobs_loop'
    ]
    not_implemented = [
        f"  {attr}: not implemented"
        for attr, spec in _OLS_INFLUENCE_ATTRS.items()
        if spec[0] is None
    ]

    print("Fast:\nThese are caclulated quickly from the fitted grid,"
          " usable for large data sets\n")
    for usage in fast:
        print(usage)

    print("\nSlow:\nThese recompute a new model for each data point,"
          " disabled by default but can be forced like so\n")
    for usage in slow:
        print(usage)

    print("\nNot implemented:\nThese are not available from fitgrid\n")
    for usage in not_implemented:
        print(usage)


def get_diagnostic(
    lm_grid, diagnostic, direction=None, crit_val=None, do_nobs_loop=False
):
    """statsmodels diagnostic measures for the grid's models and data

    .. Warning::

       The size of data diagnostic measures like `cooks_distance`,
       `dffits`, and `dfbetas` is a multiple of the original data.

       Use the `crit_val` and `direction` option to get smaller
       subsets of the measures above or below the critical value.

    For a list of `statsmodels` diagnostics available in fitgrid run

        ```python
        fitgrid.utils.lm.list_diagnostics()
        ```

    For details about the diagnostic measures visit

    www.statsmodels.org/statsmodels.stats.outliers_influence.OLSInfluence


    Parameters
    ----------
    lm_grid : fitgrid.FitGrid
        as returned by fg.lm()

    diagnostic : string
        e.g., "est_std", "cooks_distance", "dffits_internal",

    crit_val : {None, float, 'sm', func}
       critical value cutoff for filtering returned data points

       `None` return all, may be a multiple of the number of observations

       `float` is explicit value, e.g., from a user calculation

       `sm` is the statsmodels default, e.g., for cook's D, dffits

       `func` is a function that takes `lm_grid`, `attr` and
            returns one critical val float or same shape grid of them
            NOT IMPLEMENTED

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

    infl_data_df, infl_idxs = None, None

    # scrape the grid
    infl_df, infl = _get_infl_attr_vals(lm_grid, diagnostic, do_nobs_loop)
    print('before crit_val', diagnostic, infl_df.shape)

    # prune by crit_val if any
    infl_df, infl_idxs, crit_val = _crit_val_handler(
        infl_df, infl, diagnostic, direction, crit_val
    )

    print('after crit_val', diagnostic, infl_df.shape)
    return infl_data_df, infl_idxs
