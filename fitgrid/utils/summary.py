import warnings
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import fitgrid

# enforce some common structure for summary dataframes
# scraped out of different fit objects.
INDEX_NAMES = ['Time', 'model', 'beta', 'key']

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
]

# special treatment for per-model values ... broadcast to all params
PER_MODEL_KEY_LABELS = ['AIC', 'SSresid', 'has_warning', 'logLike', 'sigma2']


def summarize(
    epochs_fg,
    modeler,
    LHS,
    RHS,
    parallel=True,
    n_cores=4,
    save_as=None,
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

    n_cores : int
       number of cores to use. See what works, but golden rule if running
       on a shared machine.

    save_as : 2-ple of str, optional
       write the summary dataframe to disk with
       `pd.to_hdf(path, key, format='fixed')`

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

    FutureWarning('fitgrid summaries are in early days, subject to change')

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

    # single formula -> singleton list
    if isinstance(RHS, str):
        RHS = [RHS]

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
                    **kwargs,
                )
            )
        )

    summary_df = pd.concat(summaries)
    _check_summary_df(summary_df)

    del summaries

    if save_as is not None:
        try:
            fname, group = save_as
            summary_df.to_hdf(fname, group)
        except Exception as fail:
            warnings.warn(
                f"save_as={save_as} failed: {fail}. You can try to "
                "save the returned dataframe with pandas.to_hdf()"
            )

    return summary_df


# ------------------------------------------------------------
# private-ish summary helpers for scraping summary info from fits
# ------------------------------------------------------------
def _check_summary_df(summary_df):
    # order matters
    if not (
        summary_df.index.names == INDEX_NAMES
        and all(summary_df.index.levels[-1] == KEY_LABELS)
    ):
        raise ValueError(
            "uh oh ... fitgrid summary dataframe bug, please post an issue"
        )


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
       index.names = [`Time`, `model`, `beta`, `key`]
       columns are the `fg_ols` columns


    Notes
    -----
    The `summaries_df` row and column indexes are munged to match
    fitgrid.lmer._get_summaries_df()

    """

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

    # build model has_warnings with False for ols
    warnings = pd.DataFrame(
        np.zeros(model_vals[0].shape).astype('bool'),
        columns=model_vals[0].columns,
        index=model_vals[0].index,
    )
    warnings['key'] = 'has_warning'
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

    pmvs = pd.concat(pmvs).reset_index().set_index(INDEX_NAMES)

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

        attr_vals.index = attr_vals.index.rename(['Time', 'beta'])
        attr_vals['model'] = rhs
        attr_vals['key'] = key

        # update list of beta bundles
        summaries.append(attr_vals.reset_index().set_index(INDEX_NAMES))

    # special handling for confidence interval
    ci_bounds = [
        f"{bound:.1f}_ci"
        for bound in [100 * (1 + (b * (1 - ci_alpha))) / 2.0 for b in [-1, 1]]
    ]
    cis = fg_ols.conf_int(alpha=ci_alpha)

    cis.index = cis.index.rename(['Time', 'beta', 'key'])
    cis.index = cis.index.set_levels(ci_bounds, 'key')
    cis['model'] = rhs
    summaries.append(cis.reset_index().set_index(INDEX_NAMES))

    summaries_df = pd.concat(summaries)

    # add the parmeter model info
    summaries_df = pd.concat([summaries_df, pmvs]).sort_index().astype(float)

    _check_summary_df(summaries_df)

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
        assert ranef_var.index.names == ['Time', None, None]
        ranef_var.index.names = ['Time', 'key', 'value']

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

    # look these up directly
    pymer_attribs = ['AIC', 'has_warning', 'logLike']

    #  x=lmer_fg caclulate or extract from other attributes
    derived_attribs = {
        # fg_lmer.resid comes from pymer wrapping lme4 function resid(object)
        'SSresid': lambda x: x.resid.groupby('Time').apply(
            lambda y: np.sum(y ** 2)
        ),
        'sigma2': lambda x: scrape_sigma2(x),
    }

    # grab and tidy the formulat RHS from the first grid cell
    rhs = fg_lmer.tester.formula.split('~')[1].strip()
    rhs = re.sub(r"\s+", " ", rhs)

    # coef estimates and stats ... these are 2-D
    summaries_df = fg_lmer.coefs.copy()  # don't mod the original

    summaries_df.index.names = ['Time', 'beta', 'key']
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
        .set_index(INDEX_NAMES)
        .sort_index()
        .astype(float)
    )

    _check_summary_df(summaries_df)

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
    aic_cols = ["AIC", "has_warning"]
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
    for time in AICs.index.get_level_values('Time').unique():
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


def plot_betas(
    summary_df,
    LHS,
    alpha=0.05,
    fdr=None,
    figsize=None,
    s=None,
    df_func=None,
    **kwargs,
):

    """Plot model parameter estimates for each data column in LHS

    Parameters
    ----------
    summary_df : pd.DataFrame
       as returned by fitgrid.utils.summary.summarize

    LHS : list of str
       column names of the data fitgrid.fitgrid docs

    alpha : float
       alpha level for false discovery rate correction

    fdr : str {None, 'BY', 'BH'}
        Add markers for FDR adjusted significant :math:`p`-values. BY
        is Benjamini and Yekatuli, BH is Benjamini and Hochberg, None
        supresses the markers.
    df_func : {None, function}
        plot `function(degrees of freedom)`, e.g., `np.log10`,  `lambda x: x`

    s : float
       scatterplot marker size for BH and lmer decorations

    kwargs : dict
       keyword args passed to pyplot.subplots()


    Returns
    -------
    figs : list

    """

    def _do_fdr(_fg_beta):
        # FDR helper
        m = len(_fg_beta)
        pvals = _fg_beta['P-val'].copy().sort_values()
        ks = list()

        if fdr == 'BH':
            # Benjamini & Hochberg ... restricted
            c_m = 1
        if fdr == 'BY':
            # Benjamini & Yekatuli general case
            c_m = np.sum([1 / i for i in range(1, m + 1)])

        for k, p in enumerate(pvals):
            kmcm = k / (m * c_m)
            if p <= kmcm * alpha:
                ks.append(k)

        if len(ks) > 0:
            crit_p = pvals.iloc[max(ks)]
        else:
            crit_p = 0.0

        _fg_beta['sig_fdr'] = _fg_beta['P-val'] < crit_p

        # slice out sig ps for plotting
        sig_ps = _fg_beta.loc[_fg_beta['sig_fdr'], :]
        return sig_ps, crit_p
        # --------------------

    figs = list()

    if isinstance(LHS, str):
        LHS = [LHS]
    assert all([isinstance(col, str) for col in LHS])

    if fdr not in ['BH', 'BY', None]:
        raise ValueError(f"fdr must be 'BH', 'BY' or None")

    # defaults
    if figsize is None:
        figsize = (8, 3)

    betas = summary_df.index.get_level_values('beta').unique()
    for col in LHS:
        # for idx, beta in enumerate(betas):
        for beta in betas:

            # f, ax_beta = plt.subplots(len(betas), ncol=1, **kwargs)
            f, ax_beta = plt.subplots(
                nrows=1, ncols=1, figsize=figsize, **kwargs
            )

            # if len(betas) == 1:
            #    ax_beta = [ax_beta]

            # unstack this beta, column for plotting
            fg_beta = (
                summary_df.loc[pd.IndexSlice[:, :, beta], col]
                .unstack(level='key')
                .reset_index('Time')
            )

            # lmer SEs
            fg_beta['mn+SE'] = (fg_beta['Estimate'] + fg_beta['SE']).astype(
                float
            )
            fg_beta['mn-SE'] = (fg_beta['Estimate'] - fg_beta['SE']).astype(
                float
            )

            fg_beta.plot(
                x='Time',
                y='Estimate',
                # ax=ax_beta[idx],
                ax=ax_beta,
                color='black',
                alpha=0.5,
                label=beta,
            )

            # ax_beta[idx].fill_between(
            ax_beta.fill_between(
                x=fg_beta['Time'],
                y1=fg_beta['mn+SE'],
                y2=fg_beta['mn-SE'],
                alpha=0.2,
                color='black',
            )

            # plot transformed df
            if df_func is not None:
                fg_beta['DF_'] = fg_beta['DF'].apply(lambda x: df_func(x))
                fg_beta.plot(
                    x='Time', y='DF_', ax=ax_beta, label=f"{df_func}(df)"
                )

            if s is not None:
                my_kwargs = {'s': s}
            else:
                my_kwargs = {}

            # mark FDR sig ps
            if fdr is not None:
                # optionally fetch FDR adjusted sig ps
                sig_ps, crit_p = _do_fdr(fg_beta)
                ax_beta.scatter(
                    sig_ps['Time'],
                    sig_ps['Estimate'],
                    color='black',
                    zorder=3,
                    label=f'{fdr} FDR p < crit {crit_p:0.2}',
                    **my_kwargs,
                )

            try:
                # color warnings last to mask sig ps
                warn_ma = np.ma.where(fg_beta['has_warning'] > 0)[0]
                # ax_beta[idx].scatter(
                ax_beta.scatter(
                    fg_beta['Time'].iloc[warn_ma],
                    fg_beta['Estimate'].iloc[warn_ma],
                    color='red',
                    zorder=4,
                    label='model warnings',
                    **my_kwargs,
                )
            except Exception:
                pass

            # ax_beta[idx].axhline(y=0, linestyle='--', color='black')
            # ax_beta[idx].legend()

            ax_beta.axhline(y=0, linestyle='--', color='black')
            ax_beta.legend(loc=(1.0, 0.0))

            # title
            # rhs = fg_lmer.formula[col].unique()[0]
            formula = fg_beta.index.get_level_values('model').unique()[0]
            assert isinstance(formula, str)
            # ax_beta[idx].set_title(f'{col} {beta}: {formula}')
            ax_beta.set_title(f'{col} {beta}: {formula}')

            figs.append(f)

    return figs


def plot_AICmin_deltas(summary_df, figsize=None, gridspec_kw=None, **kwargs):
    r"""plot FitGrid min delta AICs and fitter warnings

    Thresholds of AIC_min delta at 2, 4, 7, 10 are from Burnham &
    Anderson 2004, see Notes.

    Parameters
    ----------
    summary_df : pd.DataFrame
       as returned by fitgrid.utils.summary.summarize

    figsize : 2-ple
       pyplot.figure figure size parameter

    gridspec_kw : dict
       matplotlib.gridspec key : value parameters

    kwargs : dict
       keyword args passed to plt.subplots(...)

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

    aics = _get_AICs(summary_df)  # long format
    models = aics.index.unique('model')
    channels = aics.index.unique('channel')

    nrows = len(models)

    if figsize is None:
        figsize = (12, 8)

    f, axs = plt.subplots(
        nrows, 2, **kwargs, figsize=figsize, gridspec_kw=gridspec_kw
    )

    for i, m in enumerate(models):

        # debugging
        # print(f"i: {i} m: {m}")

        # channel traces
        if len(models) == 1:
            traces = axs[0]
            heatmap = axs[1]
        else:
            traces = axs[i, 0]
            heatmap = axs[i, 1]

        traces.set_title(f'aic min delta: {m}', loc='left')
        for c in channels:
            min_deltas = aics.loc[
                pd.IndexSlice[:, m, c], ['min_delta', 'has_warning']
            ].reset_index('Time')
            traces.plot(min_deltas['Time'], min_deltas['min_delta'], label=c)
            warn_ma = np.ma.where(min_deltas['has_warning'] > 0)
            traces.scatter(
                min_deltas['Time'].iloc[warn_ma],
                min_deltas['min_delta'].iloc[warn_ma],
                color='red',
                label=None,
            )

        if i == 0:
            # first channel legend left of the main plot
            traces.legend(loc='upper right', bbox_to_anchor=(-0.2, 1.0))

        aic_min_delta_bounds = [0, 2, 4, 7, 10]
        for y in aic_min_delta_bounds:
            traces.axhline(y=y, color='black', linestyle='dotted')

        # for heat map
        _min_deltas = (
            aics.loc[pd.IndexSlice[:, m, :], 'min_delta']
            .reset_index('model', drop=True)
            .unstack('channel')
            .astype(float)
        )
        # unstack() alphanum sorts the channel index ... ugh
        _min_deltas = _min_deltas.reindex(columns=channels)

        _has_warnings = (
            aics.loc[pd.IndexSlice[:, m, :], 'has_warning']
            .reset_index('model', drop=True)
            .unstack('channel')
            .astype(bool)
        )
        _has_warnings = _has_warnings.reindex(columns=channels)

        # colorbrewer 2.0 Blues color blind safe n=5
        # http://colorbrewer2.org/#type=sequential&scheme=Blues&n=5
        pal = ['#eff3ff', '#bdd7e7', '#6baed6', '#3182bd', '#08519c']

        cmap = mpl.colors.ListedColormap(pal)
        # cmap.set_over(color='#fcae91')
        cmap.set_over(color='#08306b')  # darkest from Blues n=7
        cmap.set_under(color='lightgray')

        bounds = aic_min_delta_bounds
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        im = heatmap.pcolormesh(
            _min_deltas.index,
            np.arange(len(_min_deltas.columns) + 1),
            _min_deltas.T,
            cmap=cmap,
            norm=norm,
        )

        # fitter warnings mask
        pal_ma = ['k']
        bounds_ma = [0.5]
        cmap_ma = mpl.colors.ListedColormap(pal_ma)
        cmap_ma.set_over(color='crimson')
        cmap_ma.set_under(alpha=0.0)  # don't show goods
        norm_ma = mpl.colors.BoundaryNorm(bounds_ma, cmap_ma.N)
        heatmap.pcolormesh(
            _has_warnings.index,
            np.arange(len(_has_warnings.columns) + 1),
            _has_warnings.T,
            cmap=cmap_ma,
            norm=norm_ma,
        )
        yloc = mpl.ticker.IndexLocator(base=1, offset=0.5)
        heatmap.yaxis.set_major_locator(yloc)
        heatmap.set_yticklabels(_min_deltas.columns)
        plt.colorbar(im, ax=heatmap, extend='max')

    return f, axs
