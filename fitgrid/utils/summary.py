import itertools
import copy
import warnings
import re
from cycler import cycler as cy
from collections import defaultdict
import pprint as pp
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import fitgrid


# enforce some common structure for summary dataframes
# scraped out of different fit objects.
# _TIME is a place holder and replaced by the grid.time value on the fly

INDEX_NAMES = ['_TIME', 'model', 'beta', 'key']

# each model, beta combination has all these values,
# some are per-beta, some are per-model
KEY_LABELS = [
    '2.5_ci',
    '97.5_ci',
    'AIC',
    'DF',
    'Estimate',
    'P-val',
    'SE',
    'SSresid',
    'T-stat',
    'has_warning',
    'logLike',
    'sigma2',
    'warnings',
]

# special treatment for per-model values ... broadcast to all params
PER_MODEL_KEY_LABELS = [
    'AIC',
    'SSresid',
    'has_warning',
    'warnings',
    'logLike',
    'sigma2',
]


def summarize(
    epochs_fg,
    modeler,
    LHS,
    RHS,
    parallel=False,
    n_cores=2,
    quiet=False,
    **kwargs,
):
    """Fit the data with one or more model formulas and return summary information.

    Convenience wrapper, useful for keeping memory use manageable when
    gathering betas and fit measures for a stack of models.


    Parameters
    ----------
    epochs_fg : fitgrid.epochs.Epochs
       as returned by `fitgrid.epochs_from_dataframe()` or
       `fitgrid.from_hdf()`, *NOT* a `pandas.DataFrame`.

    modeler : {'lm', 'lmer'}
       class of model to fit, `lm` for OLS, `lmer` for linear mixed-effects.
       Note: the RHS formula language must match the modeler.

    LHS : list of str
       the data columns to model

    RHS : model formula or list of model formulas to fit
       see the Python package `patsy` docs for `lm` formula langauge
       and the R library `lme4` docs for the `lmer` formula langauge.

    parallel : bool
       If True, model fitting is distributed to multiple cores

    n_cores : int
       number of cores to use. See what works, but golden rule if running
       on a shared machine.

    quiet : bool
       Show progress bar default=True

    **kwargs : key=value arguments passed to the modeler, optional

    Returns
    -------
    summary_df : `pandas.DataFrame`
        indexed by `timestamp`, `model_formula`, `beta`, and `key`,
        where the keys are `ll.l_ci`, `uu.u_ci`, `AIC`, `DF`, `Estimate`,
        `P-val`, `SE`, `T-stat`, `has_warning`, `logLike`.


    Examples
    --------

    >>> lm_formulas = [
        '1 + fixed_a + fixed_b + fixed_a:fixed_b',
        '1 + fixed_a + fixed_b',
        '1 + fixed_a,
        '1 + fixed_b,
        '1',
    ]
    >>> lm_summary_df = fitgrid.utils.summarize(
        epochs_fg,
        'lm',
        LHS=['MiPf', 'MiCe', 'MiPa', 'MiOc'],
        RHS=lmer_formulas,
        parallel=True,
        n_cores=4
    )

    >>> lmer_formulas = [
        '1 + fixed_a + (1 + fixed_a | random_a) + (1 | random_b)',
        '1 + fixed_a + (1 | random_a) + (1 | random_b)',
        '1 + fixed_a + (1 | random_a)',
    ]
    >>> lmer_summary_df = fitgrid.utils.summarize(
        epochs_fg,
        'lmer',
        LHS=['MiPf', 'MiCe', 'MiPa', 'MiOc'],
        RHS=lmer_formulas,
        parallel=True,
        n_cores=12,
        REML=False
    )

    """

    warnings.warn(
        'fitgrid summaries are in early days, subject to change', FutureWarning
    )

    # modicum of guarding
    msg = None
    if isinstance(epochs_fg, pd.DataFrame):
        msg = (
            "Convert dataframe to fitgrid epochs with "
            "fitgrid.epochs_from_dataframe()"
        )
    elif not isinstance(epochs_fg, fitgrid.epochs.Epochs):
        msg = f"epochs_fg must be a fitgrid.Epochs not {type(epochs_fg)}"
    if msg is not None:
        raise TypeError(msg)

    # select modler
    if modeler == 'lm':
        _modeler = fitgrid.lm
        _scraper = _lm_get_summaries_df
    elif modeler == 'lmer':
        _modeler = fitgrid.lmer
        _scraper = _lmer_get_summaries_df
    else:
        raise ValueError("modeler must be 'lm' or 'lmer'")

    # promote RHS scalar str to singleton list
    RHS = np.atleast_1d(RHS).tolist()

    # loop through model formulas fitting and scraping summaries
    summaries = []
    for _rhs in RHS:
        summaries.append(
            _scraper(
                _modeler(
                    epochs_fg,
                    LHS=LHS,
                    RHS=_rhs,
                    parallel=parallel,
                    n_cores=n_cores,
                    quiet=quiet,
                    **kwargs,
                )
            )
        )

    summary_df = pd.concat(summaries)
    _check_summary_df(summary_df, epochs_fg)

    return summary_df


# ------------------------------------------------------------
# private-ish summary helpers for scraping summary info from fits
# ------------------------------------------------------------
def _check_summary_df(summary_df, fg_obj):
    """check summary df structure, and against the fitgrid object if any"""
    # fg_obj can be fitgrid.Epochs, LMGrid or LMERGrid, they all have a time attribute

    # check for fatal error conditions
    error_msg = None  # set on error

    # check summary
    if not isinstance(summary_df, pd.DataFrame):
        error_msg = "summary data is not a pandas.DataFrame"

    elif not len(summary_df):
        error_msg = "summary data frame is empty"

    elif not summary_df.index.names[1:] == INDEX_NAMES[1:]:
        # first name is _TIME, set from user epochs data
        error_msg = (
            f"summary index names do not match INDEX_NAMES: {INDEX_NAMES}"
        )

    elif not all(summary_df.index.levels[-1] == KEY_LABELS):
        error_msg = (
            f"summary index key levels dot match KEY_LABELS: {KEY_LABELS}"
        )

    else:
        # TBD
        pass

    # does summary of an object agree with its object?
    if fg_obj:
        assert any(
            [
                isinstance(fg_obj, fgtype)
                for fgtype in [
                    fitgrid.epochs.Epochs,
                    fitgrid.fitgrid.LMFitGrid,
                    fitgrid.fitgrid.LMERFitGrid,
                ]
            ]
        )

        if not summary_df.index.names == [fg_obj.time] + INDEX_NAMES[1:]:
            error_msg = (
                f"summary fitgrid object index mismatch: "
                f"summary_df.index.names: {summary_df.index.names} "
                f"fitgrd object: {[fg_obj.time] + INDEX_NAMES[1:]}"
            )

    if error_msg:
        raise ValueError(error_msg)

    # check for non-fatal issues
    if "warnings" not in summary_df.index.unique("key"):
        msg = (
            "Summaries are from fitgrid version < 0.5.0, use that version or re-fit the"
            f" models with this one fitgrid.utils.summarize() v{fitgrid.__version__}"
        )
        raise RuntimeError(msg)


def _update_INDEX_NAMES(lxgrid, index_names):
    """use the grid time column name for the summary index"""
    assert index_names[0] == '_TIME'
    _index_names = copy.copy(index_names)
    _index_names[0] = lxgrid.time
    return _index_names


def _stringify_lmer_warnings(fg_lmer):
    """create grid w/ _ separated string of lme4::lmer warning list items, else "" """

    warning_grids = fitgrid.utils.lmer.get_lmer_warnings(
        fg_lmer
    )  # dict of indicator dataframes
    warning_string_grid = pd.DataFrame(
        np.full(fg_lmer._grid.shape, ""),
        index=fg_lmer._grid.index.copy(),
        columns=fg_lmer._grid.columns.copy(),
    )

    # collect multiple warnings into single sorted "_" separated strings
    # on a tidy time x channel grid
    for warning, warning_grid in warning_grids.items():
        for idx, row_vals in warning_grid.iterrows():
            for jdx, col_val in row_vals.iteritems():
                if col_val:
                    if len(warning_string_grid.loc[idx, jdx]) == 0:
                        warning_string_grid.loc[idx, jdx] = warning
                    else:
                        # split, sort, reassemble
                        wrns = "_".join(
                            sorted(
                                warning_string_grid.loc[idx, jdx].split("_")
                                + [warning]
                            )
                        )
                        warning_string_grid.loc[idx, jdx] = wrns
    return warning_string_grid


# def _unstringify_lmer_warnings(lmer_summaries):
#     """convert stringfied lmer warning grid back into dict of indicator grids as in get_lmer_warnings()"""
#     string_warning_grid = lmer_summaries.query("key=='warnings'")
#     warnings = []
#     for warning in np.unique(string_warning_grid):
#         if len(warning) > 0:
#             warnings += warning.split("_")

#     warning_grids = {}
#     for warning in sorted(warnings):
#         warning_grids[warning] = string_warning_grid.applymap(
#             lambda x: 1 if warning in x else 0
#         )
#     return warning_grids


def _lm_get_summaries_df(fg_ols, ci_alpha=0.05):
    """scrape fitgrid.LMFitgrid OLS info into a tidy dataframe

    Parameters
    ----------
    fg_ols : fitgrid.LMFitGrid

    ci_alpha : float {.05}
       alpha for confidence interval


    Returns
    -------
    summaries_df : pd.DataFrame
       index.names = [`_TIME`, `model`, `beta`, `key`] where
       `_TIME` is the `fg_ols.time` and columns are the `fg_ols` columns


    Notes
    -----
    The `summaries_df` row and column indexes are munged to match
    fitgrid.lmer._get_summaries_df()

    """

    # set time column from the grid, always index.names[0]
    _index_names = _update_INDEX_NAMES(fg_ols, INDEX_NAMES)
    _time = _index_names[0]

    # grab and tidy the formula RHS
    rhs = fg_ols.tester.model.formula.split('~')[1].strip()
    rhs = re.sub(r"\s+", " ", rhs)

    # fitgrid returns them in the last column of the index
    param_names = fg_ols.params.index.get_level_values(-1).unique()

    # fetch a master copy of the model info
    model_vals = []
    model_key_attrs = [
        ("DF", "df_resid"),
        ("AIC", "aic"),
        ("logLike", 'llf'),
        ("SSresid", 'ssr'),
        ("sigma2", 'mse_resid'),
    ]

    for (key, attr) in model_key_attrs:
        vals = None
        vals = getattr(fg_ols, attr).copy()
        if vals is None:
            raise AttributeError(f"model: {rhs} attribute: {attr}")
        vals['key'] = key
        model_vals.append(vals)

    # statsmodels result wrappers have different versions of llf!
    aics = (-2 * fg_ols.llf) + 2 * (fg_ols.df_model + fg_ols.k_constant)
    if not np.allclose(fg_ols.aic, aics):
        msg = (
            "uh oh ...statsmodels OLS aic and llf calculations have changed."
            " please report an issue to fitgrid"
        )
        raise ValueError(msg)

    # handle warnings
    # build model has_warnings with False for ols
    has_warnings = pd.DataFrame(
        np.zeros(model_vals[0].shape).astype('bool'),
        columns=model_vals[0].columns,
        index=model_vals[0].index,
    )
    has_warnings['key'] = 'has_warning'
    model_vals.append(has_warnings)

    # build empty warning string to match has_warnings == False
    warnings = has_warnings.applymap(lambda x: "")
    warnings["key"] = "warnings"
    model_vals.append(warnings)

    model_vals = pd.concat(model_vals)

    # constants across the model
    model_vals['model'] = rhs

    # replicate the model info for each beta
    # ... horribly redundant but mighty handy when slicing later
    pmvs = []
    for p in param_names:
        pmv = model_vals.copy()
        # pmv['param'] = p
        pmv['beta'] = p
        pmvs.append(pmv)

    pmvs = (
        pd.concat(pmvs).reset_index().set_index(_index_names)
    )  # INDEX_NAMES)

    # lookup the param_name specifc info for this bundle
    summaries = []

    # select model point estimates mapped like so (key, OLS_attribute)
    sv_attrs = [
        ('Estimate', 'params'),  # coefficient value
        ('SE', 'bse'),
        ('P-val', 'pvalues'),
        ('T-stat', 'tvalues'),
    ]

    for idx, (key, attr) in enumerate(sv_attrs):
        attr_vals = getattr(fg_ols, attr).copy()  # ! don't mod the _grid
        if attr_vals is None:
            raise AttributeError(f"not found: {attr}")

        attr_vals.index.set_names('beta', level=-1, inplace=True)
        attr_vals['model'] = rhs
        attr_vals['key'] = key

        # update list of beta bundles
        summaries.append(
            attr_vals.reset_index().set_index(_index_names)
        )  # INDEX_NAMES))

    # special handling for confidence interval
    ci_bounds = [
        f"{bound:.1f}_ci"
        for bound in [100 * (1 + (b * (1 - ci_alpha))) / 2.0 for b in [-1, 1]]
    ]
    cis = fg_ols.conf_int(alpha=ci_alpha)

    cis.index = cis.index.rename([_time, 'beta', 'key'])
    cis.index = cis.index.set_levels(ci_bounds, 'key')
    cis['model'] = rhs

    summaries.append(cis.reset_index().set_index(_index_names))
    summaries_df = pd.concat(summaries)

    # add the parmeter model info
    # summaries_df = pd.concat([summaries_df, pmvs]).sort_index().astype(float)
    summaries_df = pd.concat([summaries_df, pmvs]).sort_index()

    _check_summary_df(summaries_df, fg_ols)

    return summaries_df


def _lmer_get_summaries_df(fg_lmer):
    """scrape a single model fitgrid.LMERFitGrid into a standard summary format

    Note: some values are fitgrid attributes (via pymer), others are derived

    Parameters
    ----------
    fg_lmer : fitgrid.LMERFitGrid

    """

    def scrape_sigma2(fg_lmer):
        # sigma2 is extracted from fg_lmer.ranef_var ...
        # residuals should be in the last row of ranef_var at each Time
        ranef_var = fg_lmer.ranef_var

        # set the None index names
        assert ranef_var.index.names == [fg_lmer.time, None, None]
        ranef_var.index.set_names([fg_lmer.time, 'key', 'value'], inplace=True)

        assert 'Residual' == ranef_var.index.get_level_values(1).unique()[-1]
        assert all(
            ['Name', 'Var', 'Std']
            == ranef_var.index.get_level_values(2).unique()
        )

        # slice out the Residual Variance at each time point
        # and drop all but the Time indexes to make Time x Chan
        sigma2 = ranef_var.query(
            'key=="Residual" and value=="Var"'
        ).reset_index(['key', 'value'], drop=True)

        return sigma2

    # set time column from the grid, always index.names[0]
    _index_names = _update_INDEX_NAMES(fg_lmer, INDEX_NAMES)
    _time = _index_names[0]

    # look these up directly
    pymer_attribs = ['AIC', 'has_warning', 'logLike']

    #  x=lmer_fg caclulate or extract from other attributes
    derived_attribs = {
        # since pymer4 0.7.1 the Lmer model.resid are renamed
        # model.residuals and come back as a well-behaved
        # dataframe of floats rather than rpy2 objects
        "SSresid": lambda lmer: lmer.residuals.apply(lambda x: x ** 2)
        .groupby([fg_lmer.time])
        .sum(),
        'sigma2': lambda x: scrape_sigma2(x),
        "warnings": lambda x: _stringify_lmer_warnings(x),
    }

    # grab and tidy the formulat RHS from the first grid cell
    rhs = fg_lmer.tester.formula.split('~')[1].strip()
    rhs = re.sub(r"\s+", "", rhs)

    # coef estimates and stats ... these are 2-D
    summaries_df = fg_lmer.coefs.copy()  # don't mod the original

    summaries_df.index.names = [_time, 'beta', 'key']
    summaries_df = summaries_df.query("key != 'Sig'")  # drop the stars
    summaries_df.index = summaries_df.index.remove_unused_levels()

    summaries_df.insert(0, 'model', rhs)
    summaries_df.set_index('model', append=True, inplace=True)
    summaries_df.reset_index(['key', 'beta'], inplace=True)

    # scrape AIC and other useful 1-D fit attributes into summaries_df
    for attrib in pymer_attribs + list(derived_attribs.keys()):
        # LOGGER.info(attrib)

        # lookup or calculate model measures
        if attrib in pymer_attribs:
            attrib_df = getattr(fg_lmer, attrib).copy()
        else:
            attrib_df = derived_attribs[attrib](fg_lmer)

        attrib_df.insert(0, 'model', rhs)
        attrib_df.insert(1, 'key', attrib)

        # propagate attributes to each beta ... wasteful but tidy
        # when grouping by beta
        for beta in summaries_df['beta'].unique():
            beta_attrib = attrib_df.copy().set_index('model', append=True)
            beta_attrib.insert(0, 'beta', beta)
            summaries_df = summaries_df.append(beta_attrib)

    summaries_df = (
        summaries_df.reset_index()
        .set_index(_index_names)  # INDEX_NAMES)
        .sort_index()
        #        .astype(float)
    )

    _check_summary_df(summaries_df, fg_lmer)
    return summaries_df


def _get_AICs(summary_df):
    """collect AICs, AIC_min deltas, and lmer warnings from summary_df

    Parameters
    ----------

    summary_df : multi-indexed pandas.DataFrame
       as returned by `fitgrid.summary.summarize()`

    Returns
    -------
    aics : multi-indexed pandas pd.DataFrame

    """

    # AIC and lmer warnings are 1 per model, pull from the first
    # model coefficient only, e.g., (Intercept)
    aic_cols = ["AIC", "has_warning", "warnings"]
    aics = []
    # for model, model_data in summary_df.groupby('model'):
    # groupby processes models in alphabetical sort order
    for model in summary_df.index.unique('model'):
        model_data = summary_df.query("model==@model")
        first_param = model_data.index.get_level_values('beta').unique()[0]
        aic = pd.DataFrame(
            summary_df.loc[pd.IndexSlice[:, model, first_param, aic_cols], :]
            .stack(-1)
            .unstack("key")
            .reset_index(["beta"], drop=True),
            columns=aic_cols,
        )
        aic.index.names = aic.index.names[:-1] + ["channel"]
        aics += [aic]
    AICs = pd.concat(aics)
    assert set(summary_df.index.unique('model')) == set(
        AICs.index.unique('model')
    )

    # sort except model, channel
    AICs.sort_index(
        axis=0,
        level=[l for l in AICs.index.names if not l in ['model', 'channel']],
        sort_remaining=False,
        inplace=True,
    )

    # calculate AIC_min for the fitted models at each time, channel
    AICs['min_delta'] = np.inf  # init to float

    # time label is the first index level, may not be fitgrid.defaults.TIME
    assert AICs.index.names == summary_df.index.names[:2] + ["channel"]
    for time in AICs.index.get_level_values(0).unique():
        for chan in AICs.index.get_level_values('channel').unique():
            slicer = pd.IndexSlice[time, :, chan]
            AICs.loc[slicer, 'min_delta'] = AICs.loc[slicer, 'AIC'] - min(
                AICs.loc[slicer, 'AIC']
            )

    FutureWarning('fitgrid AICs are in early days, subject to change')

    assert set(summary_df.index.unique('model')) == set(
        AICs.index.unique('model')
    )
    return AICs


def summaries_fdr_control(
    model_summary_df, method="BY", rate=0.05, plot_pvalues=True,
):
    """False discovery rate control for non-zero betas in model summary dataframes

    The family of tests for FDR control is assumed to be **all and
    only** the channels, models, and :math:`\hat{\beta}_i` in the
    summary dataframe.  The user must select the appropriate family of
    tests by slicing or stacking summary dataframes before running the
    FDR calculator.

    Parameters
    ----------
    model_summary_df : pandas.DataFrame
        As returned by `fitgrid.utils.summary.summarize`.
    method : str {"BY", "BH"}
        BY (default) is from Benjamini and Yekatuli [1]_, BH is Benjamini and
        Hochberg [2]_.
    rate : float {0.05}
        The target rate for controlling false discoveries.
    plot_pvalues : bool {True, False} 
        Display a plot of the family of $p$-values and critical value for FDR control.


    References
    ----------
    .. [1] Benjamini, Y., & Yekutieli, D. (2001). The control of
           the false discovery rate in multiple testing under
           dependency.The Annals of Statistics, 29, 1165-1188.

    .. [2] Benjamini, Y., & Hochberg, Y. (1995). Controlling the
           false discovery rate: A practical and powerful approach to
           multiple testing. Journal of the Royal Statistical
           Society. Series B (Methodological), 57, 289-300.

    """

    _check_summary_df(model_summary_df, None)
    pvals_df = model_summary_df.query("key == 'P-val'")  # fetch pvals
    pvals = np.sort(pvals_df.to_numpy().flatten())
    m = len(pvals)
    ks = list()

    if method == 'BH':
        # Benjamini & Hochberg ... restricted
        c_m = 1
    elif method == 'BY':
        # Benjamini & Yekatuli general case
        c_m = np.sum([1 / i for i in range(1, m + 1)])
    else:
        raise ValueError("method must be 'BH' or 'BY'")

    for k, p in enumerate(pvals):
        kmcm = k / (m * c_m)
        if p <= kmcm * rate:
            ks.append(k)

    if len(ks) > 0:
        crit_p = pvals[max(ks)]
        crit_p_idx = np.where(pvals < crit_p)[0].max()
    else:
        crit_p = 0.0
        crit_p_idx = 0

    n_pvals = len(pvals)

    fdr_specs = {
        "method": method,
        "rate": rate,
        "crit_p": crit_p,
        "n_pvals": n_pvals,
        "models": list(pvals_df.index.unique('model')),
        "betas": list(pvals_df.index.unique('beta')),
        "channels": list(pvals_df.columns),
    }

    fig, ax = None, None
    if plot_pvalues:

        fig, ax = plt.subplots()
        ax.set_title("Distribution of $p$-values")
        ax.plot(np.arange(m), pvals, color="k")
        ax.axhline(crit_p, xmax=crit_p_idx, ls="--", color="k")
        ax.axvline(crit_p_idx, ymax=0.5, ls="--", color="k")
        ax.annotate(
            xy=(crit_p_idx, 0.525),
            text=f"critcal $p$={crit_p:0.5f} for {method} FDR {rate}",
            ha="left",
        )
        ax.text(
            x=0.0,
            y=-0.15,
            s=pp.pformat(fdr_specs, compact=True),
            va="top",
            ha="left",
            transform=ax.transAxes,
            wrap=True,
        )
    else:
        fig, ax = None, None

    return fdr_specs, fig, ax


def plot_betas(
    summary_df,
    LHS=[],
    models=[],
    betas=[],
    interval=[],
    beta_plot_kw={},
    show_se=True,
    show_warnings=True,
    fdr_kw={},
    fig_kw={},
    df_func=None,
    scatter_size=75,
    **kwargs,
):

    """Plot  model parameter estimates for model, beta, and channel LHS

    The time course of estimated betas and standard errors is plotted
    by channel for the models, betas, and channels in the data
    frame. Channels, models, betas and time intervals may be selected
    from the summary dataframe. Plots are marked with model fit
    warnings by default and may be tagged to indicate differences from
    0 controlled for false discovery rate (FDR).


    Parameters
    ----------
    summary_df : pd.DataFrame
       as returned by fitgrid.utils.summary.summarize

    LHS : list of str or []
       column names of the data, [] default = all channels

    models : list of str or []
       select model or model betas to display, [] default = all models

    betas : list of str [] or []
       select beta or betas to plot,  [] default = all betas

    interval : [start, stop] list of two ints
       time interval to plot

    beta_plot_kw : dict
       keyword arguments passed to matplotlib.axes.plot()

    show_se : bool
       toggle display of standard error shading (default = True)

    show_warnings : bool
       toggle display of model warnings (default = True)

    fdr_kw : dict (default empty)
        One or more keyword arguments passed to ``summaries_fdr_control()`` to trigger
        to tag plots for FDR controlled differences from 0.

    fig_kw : dict
       keyword args passed to pyplot.subplots()

    df_func : {None, function}
        toggle degrees of freedom line plot via function, e.g.,
        ``np.log10``, ``lambda x: x``

    scatter_size : float
       scatterplot marker size for FDR (default = 75) and warnings (= 1.5 scatter_size)


    Returns
    -------
    figs : list of matplotlib.Figure


    Note
    ----

    The FDR family of tests is given by all channels, models, betas,
    and times in the summary data frame regardless of which of these
    are selected for plotting. To specify a different family of tests,
    construct a summary dataframe with all and only the tests for that
    family before plotting the betas.

    """
    _check_summary_df(summary_df, None)

    # fitgrid < 0.5.0
    for kwarg in ["figsize", "fdr", "alpha", "s"]:
        if kwarg in kwargs.keys():
            msg = (
                "keyword {kwarg} is deprecated in fitgrid 0.5.0, has no effect and "
                "will be removed. See figrid.utils.summary.plot_betas() documentation."
            )
            warnings.warn(msg, FutureWarning)

    # ------------------------------------------------------------
    # validate kwargs
    error_msg = None

    # LHS defaults to all channels
    if isinstance(LHS, str):
        LHS = [LHS]
    if LHS == []:
        LHS = list(summary_df.columns)

    if not all([isinstance(col, str) for col in LHS]):
        error_msg = "LHS must be a list of channel name strings"
    for channel in LHS:
        if channel not in summary_df.columns:
            error_msg = f"channel {channel} not found in the summary columns"

    # model, beta
    for key, vals in {"model": models, "beta": betas}.items():
        if vals and not (
            isinstance(vals, list)
            and all([isinstance(itm, str) for itm in vals])
        ):
            error_msg = f"{key} must be a list of strings"

        unique_vals = list(summary_df.index.unique(key))
        for val in vals:
            if val not in unique_vals:
                error_msg = (
                    f"{val} not found, check the summary index: "
                    f"name={key}, labels={unique_vals}"
                )

    # validate interval
    if interval:
        t_min = summary_df.index[0][0]
        t_max = summary_df.index[-1][0]
        if not (
            isinstance(interval, list)
            and all([isinstance(t, int) for t in interval])
            and interval[0] < interval[1]
            and interval[0] >= t_min
            and interval[1] <= t_max
        ):
            error_msg = (
                "interval must be a list of increasing integers "
                f"in the summary time range between {t_min} and {t_max}."
            )
    # fail on any error
    if error_msg:
        raise ValueError(error_msg)

    # ------------------------------------------------------------
    # filter summary for selections, if any

    # summary_df.sort_index(inplace=True)
    model_summary_df = summary_df  # a reference may be all we need

    if not LHS == list(summary_df.columns):
        model_summary_df = summary_df[LHS].copy()
    if models:
        model_summary_df = model_summary_df.query("model in @models").copy()
    if betas:
        model_summary_df = model_summary_df.query("beta in @betas").copy()

    if interval:
        model_summary_df.sort_index(inplace=True)
        model_summary_df = model_summary_df.loc[
            # Index = time, model, beta, key
            pd.IndexSlice[interval[0] : interval[1], :, :, :,],
            :,
        ]

    models = list(model_summary_df.index.unique("model"))
    _time = model_summary_df.index.names[0]

    # ------------------------------------------------------------
    # optional FDR calc
    if fdr_kw:
        # the family of tests for FDR is given by the summary data, *not*
        # which slices happen to be selected for plotting.
        fdr_specs, fdr_fig, fdr_ax = summaries_fdr_control(
            summary_df, **fdr_kw
        )
        if not summary_df.equals(model_summary_df):
            fdr_msg = (
                "FDR test family is for **ALL** models, betas, and channels in "
                "the summary dataframe not just those selected for plotting."
            )
            warnings.warn(fdr_msg)
        print(pp.pformat(fdr_specs, compact=True))

    # ------------------------------------------------------------
    # set up to plot various warnings consistently
    warning_kinds = np.unique(
        np.hstack(
            [
                w.split("_") if len(w) else []
                for w in np.unique(summary_df.query("key=='warnings'"))
            ]
        )
    )
    warning_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    warning_cycler = cy(color=warning_colors) + cy(
        marker=Line2D.filled_markers[: len(warning_colors)]
    )  # [1:len(warning_colors) + 1])
    # build the dict as warning keys are encountered, then use them as styled
    cy_iter = iter(warning_cycler)
    warning_styles = defaultdict(lambda: next(cy_iter))
    for warning_kind in warning_kinds:
        print(warning_kind)
    # ------------------------------------------------------------
    # set up figures
    figs = list()

    for model, col in itertools.product(models, LHS):

        # select beta for this model
        for beta in model_summary_df.query("model == @model").index.unique(
            "beta"
        ):

            # start the fig, ax
            if "figsize" not in fig_kw.keys():
                fig_kw["figsize"] = (8, 3)  # default

            f, ax_beta = plt.subplots(nrows=1, ncols=1, **fig_kw)

            # unstack this beta as a column for plotting
            fg_beta = (
                model_summary_df.loc[pd.IndexSlice[:, model, beta], col]
                .unstack(level='key')
                .reset_index(_time)  # time label for this model_summary_df
            )

            fg_beta.plot(
                x=_time,
                y='Estimate',
                ax=ax_beta,
                color='black',
                alpha=0.5,
                label=beta,
                **beta_plot_kw,
            )

            # optional +/- SE band
            if show_se:
                beta_hat = fg_beta['Estimate']
                ax_beta.fill_between(
                    x=fg_beta[_time],
                    y1=(beta_hat + fg_beta["SE"]).astype(float),
                    y2=(beta_hat - fg_beta["SE"]).astype(float),
                    alpha=0.2,
                    color='black',
                )

            # optional (transformed) degrees of freedom
            if df_func is not None:
                try:
                    func_name = getattr(df_func, "__name__")
                except AttributeError:
                    func_name = str(df_func)

                fg_beta['DF_'] = fg_beta['DF'].apply(lambda x: df_func(x))
                fg_beta.plot(
                    x=_time, y='DF_', ax=ax_beta, label=f"{func_name}(df)"
                )

            # FDR controlled differences from 0
            if fdr_kw:
                fdr_mask = fg_beta["P-val"] < fdr_specs["crit_p"]
                ax_beta.scatter(
                    fg_beta[_time][fdr_mask],
                    fg_beta['Estimate'][fdr_mask],
                    color='black',
                    zorder=3,
                    label=f"{fdr_specs['method']} FDR p < crit {fdr_specs['crit_p']:0.2}",
                    s=scatter_size,
                )

            # warnings
            if show_warnings and any(
                [len(warning) for warning in fg_beta["warnings"]]
            ):
                warn_strs = np.hstack(
                    [
                        np.array(wrn.split("_"))  # lengths vary
                        for wrn in fg_beta["warnings"].unique()
                        if len(wrn) > 0
                    ]
                )
                warn_strs = sorted(np.unique(warn_strs))

                for warn_str in warn_strs:
                    # separate warnings by 1/4 major tick interval
                    sep = np.abs(
                        (ax_beta.get_yticks()[:2] * [0.25, -0.25]).sum()
                    )
                    # warn_offset = (warning_kinds.index(warn_str) + 1) * sep
                    warn_offset = (
                        np.where(warning_kinds == warn_str)[0] + 1
                    ) * sep

                    warn_mask = fg_beta["warnings"].apply(
                        lambda x: warn_str in x
                    )
                    ax_beta.scatter(
                        fg_beta[_time][warn_mask],
                        fg_beta['Estimate'][warn_mask] + warn_offset,
                        zorder=4,
                        label=warn_str,
                        **warning_styles[
                            warn_str
                        ],  # cycler + defaultdict voodoo
                        alpha=0.75,
                        s=scatter_size * 1.5,
                    )

            ax_beta.axhline(y=0, linestyle='--', color='black')
            ax_beta.legend(loc='upper left', bbox_to_anchor=(0.0, -0.25))

            formula = fg_beta.index.get_level_values('model').unique()[0]
            ax_beta.set_title(f'{col} {beta}: {formula}', loc='left')

            figs.append(f)

    return figs


def plot_AICmin_deltas(
    summary_df,
    show_warnings="no_labels",
    figsize=None,
    gridspec_kw=None,
    subplot_kw=None,
):
    r"""plot FitGrid min delta AICs and fitter warnings

    Thresholds of AIC_min delta at 2, 4, 7, 10 are from Burnham &
    Anderson 2004, see Notes.

    Parameters
    ----------
    summary_df : pd.DataFrame
       as returned by fitgrid.utils.summary.summarize

    show_warnings : {"no_labels", "labels", str, list of str}
       "no_labels" (default) highlights everywhere there is any warning in
       red, the default behavior in fitgrid < v0.5.0. "labels" display
       all warning strings the axes titles.  A `str` or list of `str` selects 
       and display only warnings that (partial) match a model warning string.

    figsize : 2-ple
       pyplot.figure figure size parameter

    gridspec_kw : dict
       matplotlib.gridspec keyword args passed to ``pyplot.subplots(...,
       gridspec_kw=gridspec_kw})``

    subplot_kw : dict
       keyword args passed to ``pyplot.subplots(..., subplot_kw=subplot_kw))``

    Returns
    -------
    f, axs : matplotlib.pyplot.Figure

    Notes
    -----

       [BurAnd2004]_ p. 270-271. Where :math:`AIC_{min}` is the
       lowest AIC value for "a set of a priori candidate models
       well-supported by the underlying science :math:`g_{i}, i = 1,
       2, ..., R)`",

       .. math:: \Delta_{i} = AIC_{i} - AIC_{min}

       "is the information loss experienced if we are using fitted
       model :math:`g_{i}` rather than the best model, :math:`g_{min}`
       for inference." ...

       "Some simple rules of thumb are often useful in assessing the
       relative merits of models in the set: Models having
       :math:`\Delta_{i} <= 2` have substantial support (evidence), those
       in which :math:`\Delta_{i} 4 <= 7` have considerably less support,
       and models having :math:`\Delta_{i} > 10` have essentially no
       support."

    """

    def _get_warnings_grid(model_warnings, show_warnings):
        """look up warnings according to aic and user kwarg value """

        # split the "_" separated multiple warning strings into unique types
        warning_kinds = np.unique(
            np.hstack(
                [
                    w.split("_") if len(w) else []
                    for w in np.unique(model_warnings)
                ]
            )
        )
        warning_kinds = list(warning_kinds)

        # optionally filter by user keyword matching
        if show_warnings not in ["no_labels", "labels"]:

            user_kinds = []
            for kw_warning in show_warnings:
                found_kinds = [
                    warning_kind
                    for warning_kind in warning_kinds
                    if kw_warning in warning_kind  # string matches
                ]

                # collect the matching kinds or warn
                if found_kinds:
                    user_kinds += found_kinds
                else:
                    msg = (
                        f"show_warnings '{kw_warning}' not found in model "
                        f"{m} warnings: [{', '.join(warning_kinds)}]"
                    )
                    warnings.warn(msg)

            # update filtered kinds
            warning_kinds = user_kinds

        # build indicator grid for matching warning kinds
        warnings_grid = model_warnings.applymap(
            lambda x: 1
            if any([warning_kind in x for warning_kind in warning_kinds])
            else 0
        )
        return warning_kinds, warnings_grid

    # ------------------------------------------------------------
    # validate kwarg
    if show_warnings not in ["no_labels", "labels"]:
        # promote string to list
        show_warnings = list(np.atleast_1d(show_warnings))
        if not all([isinstance(wrn, str) for wrn in show_warnings]):
            msg = (
                "show_warnings must be 'all', 'kinds', or a string or list of strings "
                "that partial match warnings"
            )
            raise ValueError(msg)

    # validate summary dataframe
    _check_summary_df(summary_df, None)
    _time = summary_df.index.names[0]

    # fetch the AIC min delta data
    aics = _get_AICs(summary_df)  # long format
    models = aics.index.unique('model')
    channels = aics.index.unique('channel')

    # ------------------------------------------------------------
    # figure setup
    if figsize is None:
        figsize = (12, 8)

    # reasonable default, update w/ user kwargs if any
    gspec_kw = {'width_ratios': [0.46, 0.46, 0.015]}
    if gridspec_kw:
        gspec_kw.update(gridspec_kw)

    # main figure, axes: number of models rows x 3 columns: traces, raster, colorbar
    f, axs = plt.subplots(
        len(models),  # 1 axis row per model
        3,
        squeeze=False,  # keep axes shape (1, 3), tho singleton model is pointless
        figsize=figsize,
        gridspec_kw=gspec_kw,
        subplot_kw=subplot_kw,
    )

    # plot each model on an axes row
    for i, m in enumerate(models):
        traces = axs[i, 0]
        heatmap = axs[i, 1]
        colorbar = axs[i, 2]

        # ------------------------------------------------------------
        # slice this model min delta values and warnings
        _min_deltas = (
            aics.loc[pd.IndexSlice[:, m, :], 'min_delta']
            .reset_index('model', drop=True)
            .unstack('channel')
            .astype(float)
        )
        # unstack() alphanum sorts the channel index ... ugh
        _min_deltas = _min_deltas.reindex(columns=channels)

        model_warnings = (
            aics.loc[pd.IndexSlice[:, m, :], 'warnings']
            .reset_index('model', drop=True)
            .unstack('channel')
        )
        model_warnings = model_warnings.reindex(columns=channels)

        # fetch warnings for heatmapping
        warning_kinds, warnings_grid = _get_warnings_grid(
            model_warnings, show_warnings
        )

        # ------------------------------------------------------------
        # plot traces and warnings in left column

        # left column title is model with optional list of warnings
        title_str = f"{m}"
        if warning_kinds and not show_warnings == "no_labels":
            title_str += "\n" + "\n".join(warning_kinds)
        traces.set_title(title_str, loc="left")

        for chan in channels:
            traces.plot(
                _min_deltas.reset_index()[_time], _min_deltas[chan], label=chan
            )

            # warning mask
            chan_mask = (
                _min_deltas[chan].where(warnings_grid[chan] == 1).dropna()
            )
            traces.scatter(
                chan_mask.index, chan_mask, c="crimson", label=None,
            )

        if i == 0:
            # first channel legend left of the main plot
            traces.legend()
            traces.legend(
                loc='upper right',
                bbox_to_anchor=(-0.2, 1.0),
                handles=traces.get_legend().legendHandles[::-1],
            )

        aic_min_delta_bounds = [0, 2, 4, 7, 10]
        # for y in aic_min_delta_bounds:
        for y in aic_min_delta_bounds[1:]:
            traces.axhline(y=y, color='black', linestyle='dotted')

        # ------------------------------------------------------------
        # heatmap

        # colorbrewer 2.0 Blues color blind safe n=5
        # http://colorbrewer2.org/#type=sequential&scheme=Blues&n=5
        pal = ['#eff3ff', '#bdd7e7', '#6baed6', '#3182bd', '#08519c']

        cmap = mpl.colors.ListedColormap(pal)
        # cmap.set_over(color='#fcae91')
        cmap.set_over(color='#08306b')  # darkest from Blues n=7
        cmap.set_under(color='lightgray')

        bounds = aic_min_delta_bounds
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        ylabels = _min_deltas.columns
        heatmap.yaxis.set_major_locator(
            mpl.ticker.FixedLocator(np.arange(len(ylabels)))
        )
        heatmap.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(ylabels))
        im = heatmap.pcolormesh(
            _min_deltas.index,
            np.arange(len(ylabels)),
            _min_deltas.T,
            cmap=cmap,
            norm=norm,
            shading='nearest',
        )

        # any non-zero warnings are red
        if warnings_grid.to_numpy().max():
            assert (warnings_grid.index == _min_deltas.index).all()
            assert (warnings_grid.columns == _min_deltas.columns).all()
            heatmap.pcolormesh(
                warnings_grid.index,
                np.arange(len(ylabels)),
                np.ma.masked_equal(warnings_grid.T.to_numpy(), 0),
                shading="nearest",
                cmap=mpl.colors.ListedColormap(['red']),
            )

        colorbar = mpl.colorbar.Colorbar(colorbar, im, extend='max')
    return f, axs
