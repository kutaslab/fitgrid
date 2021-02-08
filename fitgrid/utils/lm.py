import copy
import warnings
import numpy as np
import pandas as pd
import patsy
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor as vif,
)
from tqdm import tqdm
from statsmodels.regression.linear_model import RegressionResultsWrapper
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
#  TPU 03/19
# ------------------------------------------------------------

# '_TIME' and '_EPOCH_ID' are used to compare indexes returned
# by the diagnostics with with those in the grid.
_FLOAT_TYPE = np.float64
_INT_TYPE = np.int64
_OLS_INFLUENCE_ATTRS = {
    'cooks_distance': ('nobs', _FLOAT_TYPE, ['_TIME', None, '_EPOCH_ID']),
    'cov_ratio': ('nobs_loop', _FLOAT_TYPE, ['_TIME', '_EPOCH_ID']),
    'dfbetas': ('nobs_loop', _FLOAT_TYPE, ['_TIME', '_EPOCH_ID', None]),
    'dffits': ('nobs_loop', _FLOAT_TYPE, ['_TIME', None, '_EPOCH_ID']),
    'dffits_internal': ('nobs', _FLOAT_TYPE, ['_TIME', None, '_EPOCH_ID']),
    'ess_press': ('nobs', _FLOAT_TYPE, ['_TIME']),
    'hat_matrix_diag': ('nobs', _FLOAT_TYPE, ['_TIME', '_EPOCH_ID']),
    'influence': ('nobs', _FLOAT_TYPE, ['_TIME', '_EPOCH_ID']),
    'k_vars': ('nobs', _INT_TYPE, ['_TIME']),
    'nobs': ('nobs', _INT_TYPE, ['_TIME']),
    'resid_press': ('nobs', _FLOAT_TYPE, ['_TIME', '_EPOCH_ID']),
    'resid_std': ('nobs', _FLOAT_TYPE, ['_TIME', '_EPOCH_ID']),
    'resid_studentized_external': (
        'nobs_loop',
        _FLOAT_TYPE,
        ['_TIME', '_EPOCH_ID'],
    ),
    'resid_studentized_internal': (
        'nobs',
        _FLOAT_TYPE,
        ['_TIME', '_EPOCH_ID'],
    ),
    'resid_var': ('nobs', _FLOAT_TYPE, ['_TIME', '_EPOCH_ID']),
}


def get_vifs(epochs, RHS, quiet=False):
    def get_single_vif(group, RHS):
        dmatrix = patsy.dmatrix(formula_like=RHS, data=group)
        vifs = {
            name: vif(dmatrix, index)
            for name, index in dmatrix.design_info.column_name_indexes.items()
        }
        return pd.Series(vifs)

    tqdm.pandas(desc="Time", disable=quiet)
    # return epochs._snapshots.progress_apply(get_single_vif, RHS=RHS)
    return epochs._snapshots.apply(get_single_vif, RHS=RHS)


# ------------------------------------------------------------
# OLSInfluence diagnostic helpers TPU 03/19
# ------------------------------------------------------------
def _check_get_diagnostic_args(lm_grid, diagnostic, do_nobs_loop):
    # type, value checking doesn't run anything, for args see get_diagnostic()

    # types ------------------------------------------------------
    msg = None
    if not isinstance(lm_grid, FitGrid):
        msg = f"lm_grid must be a FitGrid not {type(lm_grid)}"

    if not isinstance(lm_grid.tester, RegressionResultsWrapper):
        msg = f"lm_grid must be fit with fitgrid.lm()"

    if not isinstance(diagnostic, str):
        msg = f"{diagnostic} must be a string"

    if not isinstance(do_nobs_loop, bool):
        msg = f"do_nobs_loop must be True or False"

    if msg is not None:
        raise TypeError(msg)

    # values ------------------------------------------------------
    infl_calc, infl_dtype, index_names = _OLS_INFLUENCE_ATTRS[diagnostic]

    if diagnostic not in _OLS_INFLUENCE_ATTRS:
        msg = f"unknown OLSInfluence attribute {diagnostic}"

    if infl_calc == "nobs_loop" and not do_nobs_loop:
        msg = f"{diagnostic} is slow, set do_nobs_loop=True to calculate"

    if msg is not None:
        raise ValueError(msg)

    if infl_calc is None:
        msg = (
            f"get_diagnostic({diagnostic}), try"
            " lm_grid.get_influence().{diagnostic}"
        )
        raise NotImplementedError(msg)


def _get_diagnostic(lm_grid, diag, do_nobs_loop):
    """grid scraper with a modicum of validation"""

    # modicum of guarding
    _check_get_diagnostic_args(
        lm_grid=lm_grid, diagnostic=diag, do_nobs_loop=do_nobs_loop
    )

    infl_calc, infl_dtype, index_names = _OLS_INFLUENCE_ATTRS[diag]
    attr_df = getattr(lm_grid.get_influence(), diag).copy()

    if not isinstance(attr_df, pd.DataFrame):
        raise TypeError(f"{diag} grid is not a pandas DataFrame")

    actual_type = type(attr_df.iloc[0, 0])
    if actual_type is not infl_dtype:
        raise TypeError(f"gridded {diag} dtype should be {infl_dtype}")

    # swap in grid values for diagnostic _TIME and _EPOCH_ID and check

    _index_names = copy.copy(index_names)  # else _OLS_INFLUENCE is modified
    for idx, index_name in enumerate(_index_names):
        if index_name == "_TIME":
            _index_names[idx] = lm_grid.time

        if index_name == "_EPOCH_ID":
            _index_names[idx] = lm_grid.epoch_index.name

    if not _index_names == attr_df.index.names:
        import pdb

        pdb.set_trace()
        raise TypeError(
            f"_OLS_INFLUENCE_ATTRS thinks {diag} index names"
            f" should be {index_names} not {attr_df.index.names}"
        )

    return attr_df


# ------------------------------------------------------------
# UI wrappers
# ------------------------------------------------------------
def list_diagnostics():
    """Display `statsmodels` diagnostics implemented in fitgrid.utils.lm"""

    fast = [
        f"  get_diagnostic(lm_grid, '{attr}')"
        for attr, spec in _OLS_INFLUENCE_ATTRS.items()
        if spec[0] not in [None, 'nobs_loop']
    ]

    slow = [
        f"  get_diagnostic(lm_grid, '{attr}', do_nobs_loop=True)"
        for attr, spec in _OLS_INFLUENCE_ATTRS.items()
        if spec[0] == 'nobs_loop'
    ]

    not_implemented = [
        (
            f"  {attr}: not implemented for get_diagnostic(), "
            "try querying the grid directly"
        )
        for attr, spec in _OLS_INFLUENCE_ATTRS.items()
        if spec[0] is None
    ]

    print(
        "Fast: These are caclulated quickly with get_diagnostic(),"
        " usable for large data sets\n"
    )
    for usage in fast:
        print(usage)

    print(
        "\nSlow: These are available with get_diagnostic() but"
        " refit a model without each data point. Disabled by"
        " default but can be forced like so\n"
    )
    for usage in slow:
        print(usage)

    print(
        "\nNot implemented:\nThese are not available for get_diagnostic() but"
        "may be available in the fitted grid.\n"
    )
    for usage in not_implemented:
        print(usage)


def get_diagnostic(lm_grid, diagnostic, do_nobs_loop=False):
    """Fetch `statsmodels` diagnostic as a Time  x Channel dataframe

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

       * By default **all** values of the diagnostics are computed,
         this dataframe can be pruned with
         :meth:`fitgrid.utils.lm.filter_diagnostic` function.

       * By default slow diagnostics are **not** computed, this can be
         forced by setting `do_nobs_loop=True`.


    Parameters
    ----------
    lm_grid : fitgrid.LMFitGrid
        As returned by :meth:`fitgrid.lm`.

    diagnostic : string
        As implemented in `statsmodels`, e.g., "cooks_distance",
        "dffits_internal", "est_std", "dfbetas".

    do_nobs_loop : bool
        `True` forces slow leave-one-observation-out model refitting.

    Returns
    -------
    diagnostic_df : pandas.DataFrame
        Channels are in columns. Model measures are row indexed by
        time; data measures add an epoch row index; parameter measures add
        a parameter row index.

    sm_1_df : pandas.DataFrame
        The supplemenatary values `statsmodels` returns, or `None`,
        same shape as diagnostic_df.

    Notes
    -----
    * **Size:** `diagnostic_df` values for data measures like
      `cooks_distance` and `hat_matrix_diagonal` are the size of the
      original data plus a row index and for some data measures like
      `dfbetas`, they are the size of the data multiplied by the
      number of regressors in the model.

    * **Speed:** Leave-one-observation-out (LOOO) model refitting takes
      as long as it takes to fit one model multiplied by the number of
      observations. This can be intractable for large
      datasets. Diagnostic measures calculated from the original fit
      like `cooks_distance` and `dffits_internal` are tractable even
      for large data sets.

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

       # data diagnostic, one dataframe with the values
       ess_press, _ = fitgrid.utils.lm.get_diagnostic(
           lm_grid,
           'ess_press'
       )

       # Cook's D dataframe AND the p-values statsmodels computes
       cooks_Ds, sm_pvals = fitgrid.utils.lm.get_diagnostic(
           lm_grid,
           'cooks_distance'
       )

       # this fails because it requires LOOO loop
       dfbetas_df, _  = fitgrid.utils.lm.get_diagnostic(
           lm_grid,
           'dfbetas'
       )

       # this succeeds by forcing LOOO loop calculation
       dfbetas_df, _  = fitgrid.utils.lm.get_diagnostic(
           lm_grid,
           'dfbetas',
           do_nobs_loop=True
       )

    """

    diag_df = _get_diagnostic(lm_grid, diagnostic, do_nobs_loop)

    # special case diagnostic handling is unavoidable b.c. some
    # OLSInflunce methods return 2-ples, most don't. Extract
    # both parts for those that do.
    sm_1_df = None
    if diagnostic in ["cooks_distance", "dffits_internal", "dffits"]:
        assert len(diag_df.index.names) == 3
        assert diag_df.index.names[1] is None  # diagnostic index

        # values returned as 2nd item in diagnostic 2-ple
        sm_1_df = diag_df.loc[pd.IndexSlice[:, 1, :], :].reset_index(
            1, drop=True
        )

        # label the columns
        sm_1_df.columns = pd.MultiIndex.from_product(
            [[f"{diagnostic}_sm_1"], diag_df.columns],
            names=['diagnostic', 'channel'],
        )

        # diagnostic measures
        diag_df = diag_df.loc[pd.IndexSlice[:, 0, :], :].reset_index(
            1, drop=True
        )

    # name unlabeled index from fitgrid
    if len(diag_df.index.names) == 3 and diag_df.index.names[2] is None:
        diag_df.index.names = [
            name if name is not None else f"{diagnostic}_id"
            for name in diag_df.index.names
        ]

    # label the columns
    diag_df.columns = pd.MultiIndex.from_product(
        [[diagnostic], diag_df.columns], names=['diagnostic', 'channel']
    )

    # wide columns == channels diagnostic dataframe
    return diag_df, sm_1_df


def filter_diagnostic(
    diagnostic_df, how, bound_0, bound_1=None, format='long'
):
    """Select a subset of a fitgrid `statsmodels` diagnostic dataframe by value.

    Use this to identify time ponts, epochs, parameters, channels with
    outlying or potentially influential data.

    Parameters
    ----------
    diagnostic_df : pandas.DataFrame
        As returned by :meth:`fitgrid.utils.lm.get_diagnostic`

    how : {'above', 'below', 'inside', 'outside'}
        slice `diagnostic_df` above or below `bound_0` or inside or
        outside the closed interval `(bound_0, bound_1)`.

    bound_0 : scalar or array-like
        `bound_0` is the mandatory boundary for all `how`.  See
        `pandas.DataFrame.gt` and `pandas.DataFrame.lt` documents
        for binary comparisons with dataframes.

    bound_1: scalar or array-like
        `bound_1 is the mandatory upper bound for `how="inside"` and
        `how="outside"`.

    format : {'long', 'wide'}
        The `long` format pivots the channel columns into a row index
        and returns just those times, (epochs, parameters), channels
        that pass the filter. The `wide` format returns `filtered_df`
        with the same shape as `diagnostic_df`, those datapoints that
        pass the filter in their original row, column location, `nans`
        elsewhere.

    Returns
    -------
        selected_df : pandas.DataFrame

    """

    if how in ["above", "below"]:
        try:
            bound_0 > 0
        except Exception as fail:
            fail.args = ("bound_0", *fail.args)
            raise fail

        if bound_1 is not None:
            msg = "bound_1 is ignored with how=above and how=below"
            warnings.warn(msg)
            bound_1 = None

    elif how in ["inside", "outside"]:
        # are bounds comparable, legal
        try:
            bound_1 < bound_0
        except Exception as fail:
            fail.args = ("bound_1, bound_0", *fail.args)
            raise fail

        if np.array(bound_1).__lt__(bound_0):
            msg = "upper bound_1 value(s) less than bound_0"
            raise ValueError(msg)

    else:
        msg = f"how must be above, below, inside, outside"
        raise ValueError(msg)

    # all and only four cases, pandas handles failures
    if how == "above":
        diagnostic_df = diagnostic_df.where(diagnostic_df.gt(bound_0))

    if how == "below":
        diagnostic_df = diagnostic_df.where(diagnostic_df.lt(bound_0))

    if how == "inside":
        diagnostic_df = diagnostic_df.where(
            diagnostic_df.gt(bound_0) & diagnostic_df.lt(bound_1)
        )

    if how == "outside":
        diagnostic_df = diagnostic_df.where(
            diagnostic_df.lt(bound_0) | diagnostic_df.gt(bound_1)
        )

    if format == "long":
        return diagnostic_df.stack(1, dropna=True).sort_index()
    elif format == "wide":
        return diagnostic_df
    else:
        raise ValueError(f"format must be long or wide, not {format}")
