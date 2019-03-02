import warnings
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import fitgrid

import pdb

# enforce some common structure for summary dataframes
# scraped out of different fit objects.
INDEX_NAMES = ['Time', 'model', 'beta', 'key']
KEY_LABELS = [
    '2.5_ci',
    '97.5_ci',
    'AIC',
    'DF',
    'Estimate',
    'P-val',
    'SE',
    'T-stat',
    'has_warning',
]


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
    """Fit a list of one or more models and return summary information.

    Convenience wrapper for fitting a stack of models
    while keeping memory use manageable.


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

    **kwargs : key=value arguments passed to the modeler


    Returns
    -------
    summary_df : `pandas.DataFrame`
        indexed by `timestamp`, `model_formula`, `beta`, and `key`,
        where the keys are `ll.l_ci`, `uu.u_ci`, `AIC`, `DF`, `Estimate`,
        `P-val`, `SE`, `T-stat`, `has_warning`.


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
        n_cores=24,
        REML=False
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
        n_cores=24,
        REML=False
    )

    """

    FutureWarning('fitgrid summaries are in early days, subject to change')

    # modicum of guarding
    msg = None
    if isinstance(epochs_fg, pd.DataFrame):
        msg = ("Convert dataframe to fitgrid epochs with "
               "fitgrid.epochs_from_dataframe()")
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
                    n_cores=n_cores
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


def _lm_get_summaries_df(fg_ols, ci_alpha=.05):
    """scrape fitgrid.LMFitgrid OLS info into a tidy dataframe

    Parameters
    ----------
    fg_ols : fitgrid.LMFitGrid

    ci_alpha : float {.05}
       alpha for confidence interval


    Returns
    -------
    summaries_df : pd.DataFrame
       index.names = [`Time`, `model`, `param`, `key`]
       columns are the `fg_ols` columns


    Notes
    -----
    The `summaries_df` dataframe is row and column indexed the same
    as for fitgrid.lmer._get_summaries_df()

    """

    rhs = fg_ols[
        0,
        fg_ols._grid.columns[0]
    ].model.formula.iat[0, 0].split('~')[1]

    # fitgrid returns them in the last column of the index
    param_names = fg_ols.params.index.get_level_values(-1).unique()

    # fetch a master copy of the model info
    model_vals = []
    model_key_attrs = [("DF", "df_resid"), ("AIC", "aic")]
    for (key, attr) in model_key_attrs:
        vals = None
        vals = getattr(fg_ols, attr).copy()
        if vals is None:
            raise AttributeError(f"model: {rhs} attribute: {attr}")
        vals['key'] = key
        model_vals.append(vals)

    # build model has_warnings with False for ols
    warnings = pd.DataFrame(
        np.zeros(model_vals[0].shape).astype('bool'),
        columns=model_vals[0].columns,
        index=model_vals[0].index
    )
    warnings['key'] = 'has_warning'
    model_vals.append(warnings)

    model_vals = pd.concat(model_vals)
    model_vals['model'] = rhs

    # replicate the model info for each parameter
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

    # select model point estimates (summary_name, OLS_attribute)
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

        # attr_vals.index = attr_vals.index.rename(['Time', 'param'])
        attr_vals.index = attr_vals.index.rename(['Time', 'beta'])
        attr_vals['model'] = rhs
        attr_vals['key'] = key

        # update list of param bundles
        summaries.append(attr_vals.reset_index().set_index(INDEX_NAMES))

    # special handling for confidence interval
    ci_bounds = [
        f"{bound:.1f}_ci"
        for bound in [
            100 * (1 + (b * (1 - ci_alpha))) / 2.0 for b in [-1, 1]
        ]
    ]
    cis = fg_ols.conf_int(alpha=ci_alpha)
    # cis.index = cis.index.rename(['Time', 'param', 'key'])
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
    """scrape fitgrid.LMERFitGrid.summaries into a standardized format dataframe

    Parameters
    ----------
    fg_lmer : fitgrid.LMERFitGrid

    """

    # INDEX_NAMES = ['Time', 'model', 'param', 'key']

    attribs = ['AIC', 'has_warning']

    rhs = fg_lmer.formula.iloc[0, 0].split('~')[1].strip()
    rhs = re.sub(r"[ ]{1,}", r" ", rhs)

    # coef estimates and stats ... these are 2-D
    summaries_df = fg_lmer.coefs.copy()  # don't mod the original

    # summaries_df.index.names = ['Time', 'param', 'key']
    summaries_df.index.names = ['Time', 'beta', 'key']
    summaries_df = summaries_df.query("key != 'Sig'")  # drop the silly stars
    summaries_df.insert(0, 'model', rhs)
    summaries_df.set_index('model', append=True, inplace=True)

    # summaries_df.reset_index(['key', 'param'], inplace=True)
    summaries_df.reset_index(['key', 'beta'], inplace=True)

    # LOGGER.info('collecting fit attributes into summaries dataframe')
    # scrape AIC and other useful 1-D fit attributes into summaries_df
    for attrib in attribs:
        # LOGGER.info(attrib)
        attrib_df = getattr(fg_lmer, attrib).copy()
        attrib_df.insert(0, 'model', rhs)
        attrib_df.insert(1, 'key', attrib)

        # propagate attributes to each param ... wasteful but tidy
        for beta in summaries_df['beta'].unique():
            beta_attrib = attrib_df.copy().set_index('model', append=True)
            beta_attrib.insert(0, 'beta', beta)
            summaries_df = summaries_df.append(beta_attrib)

    summaries_df = (
        summaries_df
        .reset_index()
        .set_index(INDEX_NAMES)
        .sort_index().astype(float)
    )
    _check_summary_df(summaries_df)

    return summaries_df


def get_AICs(summary_df):
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
    # model coefficient only, typically (Intercept)
    first_param = summary_df.index.get_level_values('beta').unique()[0]
    AICs = pd.DataFrame(
        (
            summary_df.loc[pd.IndexSlice[:, :, first_param, 'AIC'], :]
            .reset_index(['key', 'beta'], drop=True)
            .stack(0)
        ),
        columns=['AIC'],
    )

    AICs.index.names = AICs.index.names[:-1] + ['channel']
    AICs['min_delta'] = None
    AICs.sort_index(inplace=True)

    # calculate AIC_min for the fitted models at each time, channel
    for time in AICs.index.get_level_values('Time').unique():
        for chan in AICs.index.get_level_values('channel').unique():
            slicer = pd.IndexSlice[time, :, chan]
            AICs.loc[slicer, 'min_delta'] = AICs.loc[slicer, 'AIC'] - min(
                AICs.loc[slicer, 'AIC']
            )

    # merge in corresponding column of lmer warnings
    has_warnings = pd.DataFrame(
        summary_df.loc[pd.IndexSlice[:, :, first_param, 'has_warning'], :]
        .reset_index(['key', 'beta'], drop=True)
        .stack(0),
        columns=['has_warning'],
    )

    has_warnings.index.names = has_warnings.index.names[:-1] + ['channel']
    has_warnings.sort_index(inplace=True)
    AICs = AICs.merge(has_warnings, left_index=True, right_index=True)
    FutureWarning('coef AICs are in early days, subject to change')
    return AICs

def plot_betas(
        summary_df,
        LHS,
        alpha=0.05,
        fdr='BY',
        figsize=None,
        s=None,
        **kwargs
):

    """Plot model parameter estimates for data columns in LHS

    Parameters
    ----------
    summary_df : pd.DataFrame
       as returned by fitgrid.utils.summary.summarize

    LHS : list of str
       column names of the data fitgrid.fitgrid docs

    alpha : float
       alpha level for false discovery rate correction

    fdr : str {'BY', 'BH'}
        BY is Benjamini and Yekatuli FDR, BH is Benjamini and Hochberg

    s : float
       scatterplot marker size for BH and lmer decorations

    kwargs : dict
       keyword args passed to pyplot.subplots()

    Returns
    -------
    f : matplotlib.pyplot.Figure

    """

    figs = list()

    if isinstance(LHS, str):
        LHS = [LHS]
    assert all([isinstance(col, str) for col in LHS])

    # defaults
    if figsize is None:
        figsize = (8, 3)

    coefs = summary_df.index.get_level_values('beta').unique()
    for col in LHS:
        # for idx, coef in enumerate(coefs):
        for coef in coefs:

            # f, ax_coef = plt.subplots(len(coefs), ncol=1, **kwargs)
            f, ax_coef = plt.subplots(
                nrows=1, ncols=1, figsize=figsize, **kwargs
            )

            # if len(coefs) == 1:
            #    ax_coef = [ax_coef]

            # unstack this coef, column for plotting
            fg_coef = (
                summary_df.loc[pd.IndexSlice[:, :, coef], col]
                .unstack(level='key')
                .reset_index('Time')
            )

            # log scale DF
            fg_coef['log10DF'] = fg_coef['DF'].apply(
                lambda x: np.log10(x)
            )

            # calculate B-H FDR
            m = len(fg_coef)
            pvals = fg_coef['P-val'].copy().sort_values()
            ks = list()

            if fdr not in ['BH', 'BY']:
                raise ValueError(f"fdr must be BH or BY")
            if fdr == 'BH':
                # Benjamini & Hochberg ... restricted
                c_m = 1
            elif fdr == 'BY':
                # Benjamini & Yekatuli general case
                c_m = np.sum([1 / i for i in range(1, m + 1)])

            for k, p in enumerate(pvals):
                kmcm = (k / (m * c_m))
                if p <= kmcm * alpha:
                    ks.append(k)

            if len(ks) > 0:
                crit_p = pvals.iloc[max(ks)]
            else:
                crit_p = 0.0
            fg_coef['sig_fdr'] = fg_coef['P-val'] < crit_p

            # slice out sig ps for plotting
            sig_ps = fg_coef.loc[fg_coef['sig_fdr'], :]

            # lmer SEs
            fg_coef['mn+SE'] = (
                fg_coef['Estimate'] + fg_coef['SE']
            ).astype(float)
            fg_coef['mn-SE'] = (
                fg_coef['Estimate'] - fg_coef['SE']
            ).astype(float)

            fg_coef.plot(
                x='Time',
                y='Estimate',
                # ax=ax_coef[idx],
                ax=ax_coef,
                color='black',
                alpha=0.5,
                label=coef,
            )

            # ax_coef[idx].fill_between(
            ax_coef.fill_between(
                x=fg_coef['Time'],
                y1=fg_coef['mn+SE'],
                y2=fg_coef['mn-SE'],
                alpha=0.2,
                color='black',
            )

            # plot log10 df
            fg_coef.plot(x='Time', y='log10DF', ax=ax_coef)

            if s is not None:
                my_kwargs = {'s': s}
            else:
                my_kwargs = {}

            # color sig ps
            ax_coef.scatter(
                sig_ps['Time'],
                sig_ps['Estimate'],
                color='black',
                zorder=3,
                label=f'{fdr} FDR p < crit {crit_p:0.2}',
                **my_kwargs,
            )

            try:
                # color warnings last to mask sig ps
                warn_ma = np.ma.where(fg_coef['has_warning'] > 0)[0]
                # ax_coef[idx].scatter(
                ax_coef.scatter(
                    fg_coef['Time'].iloc[warn_ma],
                    fg_coef['Estimate'].iloc[warn_ma],
                    color='red',
                    zorder=4,
                    label='lmer warnings',
                    **my_kwargs,
                )
            except Exception:
                pass

            # ax_coef[idx].axhline(y=0, linestyle='--', color='black')
            # ax_coef[idx].legend()

            ax_coef.axhline(y=0, linestyle='--', color='black')
            ax_coef.legend(loc=(1.0, 0.0))

            # title
            # rhs = fg_lmer.formula[col].unique()[0]
            formula = fg_coef.index.get_level_values('model').unique()[0]
            assert isinstance(formula, str)
            # ax_coef[idx].set_title(f'{col} {coef}: {formula}')
            ax_coef.set_title(f'{col} {coef}: {formula}')

            figs.append(f)

    return figs


def plot_AICs(aics, figsize=None, gridspec_kw=None, **kwargs):
    """plot FitGrid min delta AICs and fitter warnings

    Thresholds of AIC_min delta at 2, 4, 7, 10 are from Burnham &
    Anderson 2004, p. 271.

    Parameters
    ----------
    aics : pd.DataFrame as returned by summary.get_AICs()

    figsize : 2-ple
       pyplot.figure figure size parameter

    gridspec=kw : dict
       matplotlib.gridspec key : value parameters

    kwargs : dict
       keyword args passed to plt.subplots(...)

    Returns
    -------
    f : matplotlib.pyplot.Figure

    Notes
    -----

    From the article:

       "Some simple rules of thumb are often useful in assessing the
        relative merits of models in the set: Models having
        delta_{i} <= 2 have substantial support (evidence), those
        in which delta_{i} 4 <= 7 have considerably less support,
        and models having delta_{i} > 10 have essentially no
        support."


    References
    ----------

    .. [1] Burnham, K. P., & Anderson, D. R. (2004). Multimodel
       inference - understanding AIC and BIC in model
       selection. Sociological Methods & Research, 33(2),
       261-304. doi:10.1177/0049124104268644

    """
    models = aics.index.get_level_values('model').unique()
    channels = aics.index.get_level_values('channel').unique()

    if 'nrows' not in kwargs.keys():
        nrows = len(models)

    if figsize is None:
        figsize = (8, 3)

    f, axs = plt.subplots(
        nrows,  # len(models),
        2,
        **kwargs,
        figsize=figsize,
        gridspec_kw=gridspec_kw,
    )

    for i, m in enumerate(models):
        # channel traces
        if len(models) == 1:
            traces = axs[0]
            heatmap = axs[1]
        else:
            traces = axs[i, 0]
            heatmap = axs[i, 1]

        traces.set_title(f'aic min delta: {m}')
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
        traces.legend()

        aic_min_delta_bounds = [0, 2, 4, 7, 10]
        for y in aic_min_delta_bounds:
            traces.axhline(y=y, color='black', linestyle='dotted')

        # heat map
        _min_deltas = (
            aics.loc[pd.IndexSlice[:, m, :], 'min_delta']
            .reset_index('model', drop=True)
            .unstack('channel')
            .astype(float)
        )

        _has_warnings = (
            aics.loc[pd.IndexSlice[:, m, :], 'has_warning']
            .reset_index('model', drop=True)
            .unstack('channel')
            .astype(bool)
        )

        # _min_deltas_ma = np.ma.where(_has_warnings)

        # bluish color blind friendly
        pal = ['#eff3ff', '#bdd7e7', '#6baed6', '#3182bd', '#08519c']

        # redish color blind friendly
        # pal = ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15']

        # grayscale
        # pal = ['#f7f7f7', '#cccccc', '#969696', '#636363', '#252525']

        cmap = mpl.colors.ListedColormap(pal)
        cmap.set_over(color='#fcae91')
        # cmap.set_under('0.75')
        bounds = aic_min_delta_bounds
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # cb2 = mpl.colorbar.ColorbarBase(cmap=cmap,
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

    return f
