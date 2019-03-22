import numpy as np
import pandas as pd
import patsy
from tqdm import tqdm
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor as vif,
)
from statsmodels.regression.linear_model import RegressionResultsWrapper
import warnings
from fitgrid.fitgrid import FitGrid

# ------------------------------------------------------------
# fitgrid's database of statsmodels OLSInfluence diagnostics:
#
#  * what there is, how it runs, and data type, like so
#
#  *  attr : (calc_type, value_dtype, df.index.names)
#
#  df.index.names are as returned by LMFitGrid attribute getter
#  nobs = number of observations
#  nobs_k = number of observations x model regressors
#  nobs_loop = nobs re-fitting ... slow
# ------------------------------------------------------------
FLOAT_TYPE = np.float64
INT_TYPE = np.int64
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
# OLSInfluence diagnostic helpers TPU
# ------------------------------------------------------------
def _check_get_diagnostic_args(
        lm_grid=None,
        diagnostic=None,
        select_by=None,
        direction=None,
        do_nobs_loop=False
):
    # type, value checking doesn't run anything, for args see get_diagnostic()

    # types ------------------------------------------------------
    msg = None
    if not isinstance(lm_grid, FitGrid):
        msg = f"lm_grid must be a FitGrid not {type(lm_grid)}"

    if not isinstance(lm_grid.tester, RegressionResultsWrapper):
        msg = f"lm_grid must be fit with fitgrid.lm()"

    if not isinstance(diagnostic, str):
        msg = f"{diagnostic} must be a string"

    if not (isinstance(direction, str) or direction is None):
        msg = f"{direction} must be a string"

    if not (
            (select_by is None)
            or (select_by == 'sm')
            or (isinstance(select_by, float))
            or (hasattr(select_by, '__call__'))
    ):
        msg = "select_by must be None, 'sm', a float or a function"

    if not isinstance(do_nobs_loop, bool):
        msg = f"do_nobs_loop must be True or False"

    if msg is not None:
        raise TypeError(msg)

    # values ------------------------------------------------------
    if not (
        diagnostic in _OLS_INFLUENCE_ATTRS
    ):
        msg = f"unknown OLSInfluence attribute {diagnostic}"

    # need a direction to select
    if select_by is not None:
        if direction not in ['above', 'below']:
            msg = f"select_by requires a direction 'above' or 'below"

    if msg is not None:
        raise ValueError(msg)


def _get_attr_df(infl, infl_attr, do_nobs_loop):
    """general purpose checker and raw grid getter, may be slow"""

    if not (
        infl_attr in _OLS_INFLUENCE_ATTRS
    ):
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
    see get_diagnostics()

    Four cases: select_by = None, float-like, 'sm', function

        None = not selecting diagnostic values, passing all through

        float-like is a user-defined constant or array same shape as attr_df

        'sm' flags use the precomputed statsmodels value(s) in
             `sm_1_df` as returned by get_attr_df()

        function is a user defined function of attr_df, infl.

    Returns
    -------
    crit_vals_df
        where crit_vals_df.shape == attr_df.shape or crit_vals is None

    """

    crit_vals_df = None  # empty

    # case 1 float-like scalar
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

    return crit_vals_df  # may be None


# ------------------------------------------------------------
# UI wrappers
# ------------------------------------------------------------
def list_diagnostics():
    """Show fast, slow, and not implemented statsmodels diagnostics"""

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
    """Fetch and optionally prune a `statsmodels` diagnostic measure.

    `statsmodels` implements a variety of data and model diagnostic
    measures. For some, it also computes a version of a recommended
    critical value or :math:`p`-value. Use these at your own risk
    after careful study of the `statsmodels` source code. For details
    visit :sm_docs:`statsmodels.stats.outliers_influence.OLSInfluence.html`

    For a catalog of the measures available for `fitgrid.lm()` run
    this in Python

    .. code-block:: python

       >>>fitgrid.utils.lm.list_diagnostics()

    .. Warning:: Data diagnostics can be very large and very slow, see
       Notes for details.

       * By default **all** values of the diagnostics are computed, these
         can be pruned with the `select_by` and `direction` options.

       * By default slow diagnostics are **not** computed, this can be
         forced by setting `do_nobs_loop=True`.


    Parameters
    ----------
    lm_grid : fitgrid.LMFitGrid
        as returned by `figrid.lm()`

    diagnostic : string
        as implemented in `statsmodels`, e.g., "cooks_distance",
        "dffits_internal", "est_std".

    select_by : one of {None, float, float-array, 'sm', func}
        critical value cutoff for filtering returned data points

        * `None` returns all, may be a multiple of the number of observations

        * `float` is a float-like number or numpy.array of float-like
          the same length as the number of observations, e.g., from
          a user calculation

        * `sm` is the statsmodels default, where there is one, e.g., for
          `cooks_distance`, `dffits_internal`, `dffits`.

        * `func` NOT IMPLEMENTED. A function that computes critical
          value(s)

    direction : {"above", "below"}
       which side of the critical value to return

    Returns
    -------
        diagnostic_df : pandas.DataFrame

    Notes
    -----

    * Size
      `diagnostic_df` values for data measures like `cooks_distance`
      and `hat_matrix_diagonal` are the size of the original data plus
      a row index and for some data measures like `dfbetas`, they are the
      size of the data multiplied by the number of regressors in the
      model.

    * Speed
      Leave-one-observation-out (LOOO) model refitting take as long as
      it takes to fit one model multiplied by the number of
      observations. This can be intractable for large datasets. Diagnostic
      measures calculated from the original fit like `cooks_distance`
      and `dffits_internal` are tractable even for large data sets.

    Examples
    --------

    .. code-block:: python

       # fake data
       epochs_fg = fitgrid.generate()
       lm_grid = fitgrid.lm(
           epochs_fg,
           LHS=epochs_fg.channels,
           RHS='continuous + categorical',
           parallel=True,
           n_cores=4,
       )

       press_crit_val = 12000.0  # made up value
       diagnostic_df = fitgrid.utils.lm.get_diagnostic(
           lm_grid,
           'ess_press',
           select_by=press_crit_val,
           direction='above',
       )

       # Some folks think this is reasonable for large N,
       # references witheld by design. What do you think?
       cooks_D_crit_val = 1.0

       influential_cooks_Ds_df = fitgrid.utils.lm.get_diagnostic(
           lm_grid,
           'ess_press',
           select_by=cooks_D_crit_val,
           direction='above',
       )

       un_influential_cooks_Ds_df = fitgrid.utils.lm.get_diagnostic(
           lm_grid,
           'ess_press',
           select_by=cooks_D_crit_val,
           direction='below',
       )

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
    diagnostic_df, sm_1_df = _get_attr_df(
        lm_grid.get_influence(), diagnostic, do_nobs_loop
    )

    # critical values for this diagnostic, if any
    crit_vals_df = _get_attr_crit_val(
        diagnostic, diagnostic_df, sm_1_df, select_by
    )

    # prune diagnostic values according to critical values
    if crit_vals_df is not None:
        # prune
        if direction == 'above':
            m = diagnostic_df > crit_vals_df.to_numpy()
        elif direction == 'below':
            m = diagnostic_df < crit_vals_df.to_numpy()
        else:
            msg = f"{direction} bad crit_val direction please report an issue"
            raise ValueError(msg)
        diagnostic_df = diagnostic_df.where(m, np.nan)

    # pivot columns to long format, explict dropna is the default
    diagnostic_df = diagnostic_df.stack(1, dropna=True)
    if diagnostic_df.size > 0:
        assert isinstance(diagnostic_df, pd.DataFrame)

    return diagnostic_df
