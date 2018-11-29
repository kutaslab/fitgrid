import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import fitgrid


def fit_lmers(fg_epochs, LHS, RHSs, parallel=True, n_cores=4, save_as=None):
    """Fit a set of lmer models and return rERPs, AICs, and lmer warnings

    Parameters
    ----------
    LHS : fitgrid.lmer LHS specification

    RHSs : list of fitgrid.lmer RHS specifications

    parallel : bool

    Returns
    -------
    lmer_coefs : multi-indexed pandas.DataFrame
       Time, model, param, key x LHS 

    Raises
    ------
    FutureWarning
    
    Examples
    --------

    LHS = ['MiPf', 'MiCe', 'MiPa', 'MiOc', 'cproi']
    LHS = ['cproi']

    RHSs = [
      'a_cloze_c + (a_cloze_c | sub_id) + (a_cloze_c | m_item_id)',
      'a_cloze_c + (a_cloze_c | sub_id) + (1 | m_item_id)',
      'a_cloze_c + (1 | sub_id) + (a_cloze | m_item_id)',
      'a_cloze_c + (1 | sub_id) + (1 | m_item_id)',

    """

    # container to hold model information scraped from the fits
    lmer_coefs = pd.DataFrame()
    attribs = ['AIC', 'has_warning']
    for rhs in RHSs:
        fg_lmer = fg_epochs.lmer(
            LHS=LHS, RHS=rhs, parallel=parallel, n_cores=n_cores
        )
        fg_lmer.coefs.index.names = ['Time', 'param', 'key']

        # coef estimates and stats ... these are 2-D
        coefs_df = fg_lmer.coefs.copy()
        coefs_df.insert(0, 'model', rhs)
        coefs_df.set_index('model', append=True, inplace=True)
        coefs_df.reset_index(['key', 'param'], inplace=True)

        # LOGGER.info('collecting fit attributes into coefs dataframe')
        # scrape AIC and other useful 1-D fit attributes into coefs_df
        for attrib in attribs:
            # LOGGER.info(attrib)
            attrib_df = getattr(fg_lmer, attrib).copy()
            attrib_df.insert(0, 'model', rhs)
            attrib_df.insert(1, 'key', attrib)

            # propagate attributes to each param ... wasteful but tidy
            for param in coefs_df['param'].unique():
                param_attrib = attrib_df.copy().set_index('model', append=True)
                param_attrib.insert(0, 'param', param)
                coefs_df = coefs_df.append(param_attrib)

        # update main container
        lmer_coefs = lmer_coefs.append(coefs_df)
        del (fg_lmer)

    # refresh index
    lmer_coefs.set_index(['param', 'key'], append=True, inplace=True)
    lmer_coefs.sort_index(inplace=True)

    if save_as is not None:
        # filepath, h5 group slashpath
        coefs.to_hdf(save_as[0], save_as[1])

    FutureWarning('lmer_coefs are in early days, subject to change')
    return lmer_coefs


def get_lmer_AICs(lmer_coefs):
    """collect AICs, AIC_min deltas, and lmer warnings from lmer_coefs

    Parameters
    ----------
    
    lmer_coefs : multi-indexed pandas.DataFrame
       Time, model, param, key x LHS, as returned by fit_lmers()

    Returns
    -------
    aics : multi-indexed pandas pd.DataFrame
    
    """
    # AIC and lmer warnings are 1 per model, pull from the first
    # model coefficient only, typically (Intercept)
    first_param = lmer_coefs.index.get_level_values('param').unique()[0]
    AICs = pd.DataFrame(
        (
            lmer_coefs.loc[pd.IndexSlice[:, :, first_param, 'AIC'], :]
            .reset_index(['key', 'param'], drop=True)
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
        lmer_coefs.loc[pd.IndexSlice[:, :, first_param, 'has_warning'], :]
        .reset_index(['key', 'param'], drop=True)
        .stack(0),
        columns=['has_warning'],
    )

    has_warnings.index.names = has_warnings.index.names[:-1] + ['channel']
    has_warnings.sort_index(inplace=True)
    AICs = AICs.merge(has_warnings, left_index=True, right_index=True)
    FutureWarning('lmer_coef AICs are in early days, subject to change')
    return AICs


# ------------------------------------------------------------
# plotting
# ------------------------------------------------------------
def plot_lmer_AICs(aics):
    """plot FitGrid AICs and lmer warnings

    Thresholds of AIC_min delta at 2, 4, 7, 10 are from Burnham &
    Anderson 2004, p. 271.

    Parameters
    ----------
    aics : pd.DataFrame as returned by get_lmer_AICs()

    Returns
    -------
    f : matplotlib.pyplot.Figure

    """
    models = aics.index.get_level_values('model').unique()
    channels = aics.index.get_level_values('channel').unique()

    f, axs = plt.subplots(
        len(models),
        2,
        figsize=(16, 6 * len(models)),
        gridspec_kw={'hspace': 0.3},
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

        _min_deltas_ma = np.ma.where(_has_warnings)

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

        # lmer warnings mask
        pal_ma = ['k']
        bounds_ma = [0.5]
        cmap_ma = mpl.colors.ListedColormap(pal_ma)
        cmap_ma.set_over(color='crimson')
        cmap_ma.set_under(alpha=0.0)  # don't show goods
        norm_ma = mpl.colors.BoundaryNorm(bounds_ma, cmap_ma.N)
        im_ma = heatmap.pcolormesh(
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


def plot_lmer_rERPs(LHS, lmer_coefs, alpha=0.05):
    """
    Parameters
    ----------
    lmer_coefs : pd.DataFrame see fitgrid.utils.lmer.fit_lmers()

    LHS : fitgrid.LHS specification, see fitgrid.fitgrid.lmer()

    alpha : float
       one-tailed > 0 alpha level

    Returns
    -------
    f : matplotlib.pyplot.Figure


    """
    figs = list()

    if isinstance(LHS, str):
        LHS = [LHS]
    assert all([isinstance(col, str) for col in LHS])

    # coefs = fg_lmer.coefs.index.get_level_values('param').unique()
    coefs = lmer_coefs.index.get_level_values('param').unique()
    for col in LHS:
        f, ax_coef = plt.subplots(
            len(coefs), figsize=(18, 12), gridspec_kw={'hspace': 0.3}
        )
        if len(coefs) == 1:
            ax_coef = [ax_coef]

        for idx, coef in enumerate(coefs):

            # unstack this coef, column for plotting
            fg_lmer_coef = (
                lmer_coefs.loc[pd.IndexSlice[:, :, coef], col]
                .unstack(level='key')
                .reset_index('Time')
            )

            # log scale DF
            fg_lmer_coef['log10DF'] = fg_lmer_coef['DF'].apply(
                lambda x: np.log10(x)
            )

            # calculate B-H FDR
            m = len(fg_lmer_coef)
            pvals = fg_lmer_coef['P-val'].copy().sort_values()
            ks = list()
            for k, p in enumerate(pvals):
                if p <= k / m * alpha:
                    ks.append(k)

            if len(ks) > 0:
                crit_p = pvals.iloc[max(ks)]
            else:
                crit_p = 0.0
            fg_lmer_coef['sig_fdr'] = fg_lmer_coef['P-val'] < crit_p

            # slice out sig ps for plotting
            sig_ps = fg_lmer_coef.loc[fg_lmer_coef['sig_fdr'], :]

            # lmer SEs
            fg_lmer_coef['mn+SE'] = (
                fg_lmer_coef['Estimate'] + fg_lmer_coef['SE']
            ).astype(float)
            fg_lmer_coef['mn-SE'] = (
                fg_lmer_coef['Estimate'] - fg_lmer_coef['SE']
            ).astype(float)

            fg_lmer_coef.plot(
                x='Time',
                y='Estimate',
                ax=ax_coef[idx],
                color='black',
                alpha=0.5,
                label=coef,
            )

            ax_coef[idx].fill_between(
                x=fg_lmer_coef['Time'],
                y1=fg_lmer_coef['mn+SE'],
                y2=fg_lmer_coef['mn-SE'],
                alpha=0.2,
                color='black',
            )

            # plot log10 df
            fg_lmer_coef.plot(x='Time', y='log10DF', ax=ax_coef[idx])

            # color sig ps
            ax_coef[idx].scatter(
                sig_ps['Time'],
                sig_ps['Estimate'],
                color='black',
                zorder=3,
                label=f'BH FDR p < crit {crit_p:0.2}',
            )

            # color warnings last to mask sig ps
            warn_ma = np.ma.where(fg_lmer_coef['has_warning'] > 0)[0]
            ax_coef[idx].scatter(
                fg_lmer_coef['Time'].iloc[warn_ma],
                fg_lmer_coef['Estimate'].iloc[warn_ma],
                color='red',
                zorder=4,
                label='lmer warnings',
            )

            ax_coef[idx].axhline(y=0, linestyle='--', color='black')
            ax_coef[idx].legend()

            # title
            # rhs = fg_lmer.formula[col].unique()[0]
            formula = fg_lmer_coef.index.get_level_values('model').unique()[0]
            assert isinstance(formula, str)
            ax_coef[idx].set_title(f'{col} {coef}: {formula}')
        figs.append(f)
    return figs
