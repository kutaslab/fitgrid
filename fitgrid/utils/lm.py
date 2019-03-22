import numpy as np
import pandas as pd
import patsy
from tqdm import tqdm
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor as vif,
)
from statsmodels.regression.linear_model import RegressionResultsWrapper
import warnings
import pdb
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
def _check_get_diagnostic_args(
        lm_grid,
        diagnostic,
        select_by=None,
        direction=None,
        do_nobs_loop=False
):

    """argument type, value checker, doesn't run anything"""

    # types
    msg = None
    if not isinstance(lm_grid, FitGrid):
        msg = f"lm_grid must be a FitGrid not {type(lm_grid)}"

    if not isinstance(lm_grid.tester, RegressionResultsWrapper):
        msg = f"lm_grid must be fit with fitgrid.lm()"

    if not isinstance(do_nobs_loop, bool):
        msg = f"do_nobs_loop must be True or False"

    if msg is not None:
        raise TypeError(msg)

    # values
    if not (
        isinstance(diagnostic, str) and diagnostic in _OLS_INFLUENCE_ATTRS
    ):
        raise ValueError(f"unknown OLSInfluence attribute {diagnostic}")

    if not (
            (select_by is None)
            or (select_by == 'sm')
            or (isinstance(select_by, float))
            or (hasattr(select_by, '__call__'))
    ):
        msg = "select_by must be None, 'sm', a float or a function"
        raise TypeError(msg)

    # need a direction with critical values
    if select_by is not None:
        if direction not in ['above', 'below']:
            raise ValueError(
                f"crit_val requires a direction 'above' or 'below"
            )

    return 0


# ------------------------------------------------------------
# back end workers ...
# ------------------------------------------------------------
def _get_attr_df(infl, infl_attr, do_nobs_loop):
    """general purpose checker and raw grid getter, may be slow"""

    if infl_attr not in _OLS_INFLUENCE_ATTRS:
        raise ValueError(f"unknown OLSInfluence attribute {infl_attr}")

    infl_calc, infl_dtype, index_names = _OLS_INFLUENCE_ATTRS[infl_attr]

    if infl_calc is None:
        raise ValueError(f"fitgrid cannot calculate {infl_attr}")

    if infl_calc == "nobs_loop" and not do_nobs_loop:
        msg = f"{infl_attr} is slow, to calculate anyway set do_nobs_loop=True"
        raise ValueError(msg)

    attr_df = getattr(infl, infl_attr)
    if not isinstance(attr_df, pd.DataFrame):
        raise TypeError(f"{infl_attr} grid is not a pandas DataFrame")

    actual_type = type(getattr(infl, infl_attr).iloc[0, 0])
    if actual_type is not infl_dtype:
        raise TypeError(f"gridded {infl_attr} dtype should be {infl_dtype}")

    if not index_names == attr_df.index.names:
        raise TypeError(
            f" OLS_INFLUENCE_ATTRS thinks {infl_attr} index"
            f" names should be be {index_names},  the grid"
            f" index names are {attr_df.index.names}"
        )

    # Special case handling is unavoidable b.c. some OLSInflunce methods
    # return 2 kinds of values, others don't. Split up those that do.
    sm_1_df = None
    if infl_attr in ["cooks_distance", "dffits_internal", "dffits"]:
        assert len(attr_df.index.names) == 3
        assert attr_df.index.names[1] is None  # diagnostic index

        # values returned as 2nd item in diagnostic 2-ple
        sm_1_df = attr_df.loc[pd.IndexSlice[:, 1, :], :].reset_index(
            1, drop=True
        )
        sm_1_df.columns.name = f"{infl_attr}_sm_1"

        # diagnostic measures
        attr_df = attr_df.loc[pd.IndexSlice[:, 0, :], :].reset_index(
            1, drop=True
        )

    # name unlabeled index from fitgrid
    if len(attr_df.index.names) == 3 and attr_df.index.names[2] is None:
        attr_df.index.names = [
            name if name is not None else f"{infl_attr}_id"
            for name in attr_df.index.names
        ]

    # decorate the columns, diagnostic spans channels
    attr_df.columns = pd.MultiIndex.from_product(
        [[infl_attr], attr_df.columns],
        names=['diagnostic', 'Channels']
    )

    # attr_df.columns.name must == diagnostic by construction
    if attr_df.columns.unique('diagnostic')[0] != infl_attr:
        msg = (
            "uh oh diagnostic dataframe bug please report an issue"
            "and reproducible example."
        )
        raise ValueError(msg)

    return attr_df, sm_1_df


def _get_attr_crit_val(diagnostic, attr_df, sm_1_df, select_by):
    """handler for the different kinds of diagnostic select_by

    Parameters
    ----------
    see `get_diagnostics()`

    four cases:

        None = not selecting diagnostic values, passing all through

        crit_val = float-like, float-array-like, 'sm', function,

        float-like is a user-defined constant or array same shape as attr_df

        'sm' flags use the precomputed statsmodels value(s) in
             `sm_1_df` as returned by get_attr_df()

        function is a user defined function of attr_df, infl.

    Return
    ------
    crit_vals_df
        shape == attr_df.shape or None

    """

    crit_vals_df = None  # empty

    # case 1 float-like -> dataframe
    try:
        crit_vals_df = pd.DataFrame(
            np.full(shape=attr_df.shape, fill_value=float(select_by))
        )
        crit_vals_df.index = attr_df.index.copy()
    except Exception:
        pass

    # case 2 float-array-like -> dataframe
    try:
        crit_vals_df = pd.DataFrame(
            np.array(select_by).astype(float)
        )
        crit_vals_df.index = attr_df.index.copy()
        crit_vals_df.columns = attr_df.columns.copy()
        crit_vals_df.columns.name = [f"{diagnostic}_crit_val"]

    except Exception:
        pass

    # case 3 statsmodels default from tuple return
    if select_by == 'sm':
        if sm_1_df is not None:
            crit_vals_df = sm_1_df
        else:
            warnings.warn('statsmodels has no default for {diagnostic}')
            crit_vals_df = None

    # case 4 function
    if hasattr(select_by, '__call__'):
        warnings.warn('functions not yet implemented')
        crit_vals_df = None

    assert crit_vals_df is None or (
        crit_vals_df.shape == attr_df.shape
        and all(crit_vals_df.index == attr_df.index)
    )

    return crit_vals_df


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
    lm_grid, diagnostic, select_by=None, direction=None, do_nobs_loop=False
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

    select_by : {None, float, float-array, 'sm', func}
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
    _check_get_diagnostic_args(
        lm_grid,
        diagnostic,
        select_by,
        direction,
        do_nobs_loop
    )


    # a FitGrid
    attr_df, sm_1_df = _get_attr_df(
        lm_grid.get_influence(), diagnostic, do_nobs_loop
    )

    # critical values for this diagnostic, if any
    crit_vals_df = _get_attr_crit_val(
        diagnostic, attr_df, sm_1_df, select_by
    )

    # prune diagnostic values according to critical values
    if crit_vals_df is not None:
        # prune
        if direction == 'above':
            m = attr_df > crit_vals_df.to_numpy()
        elif direction == 'below':
            m = attr_df < crit_vals_df.to_numpy()
        else:
            msg = f"{direction} bad crit_val direction please report an issue"
            raise ValueError(msg)
        attr_df = attr_df.where(m, np.nan)

    # pivot columns to long format, explict dropna is the default
    attr_df = attr_df.stack(1, dropna=True)
    if attr_df.size > 0:
        assert isinstance(attr_df, pd.DataFrame)

    return attr_df
