"""wrappers around fitgrid fitters for rERP analysis"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_AICs(rerps):
    """collect AICs, AIC_min deltas, and lmer warnings from rerps

    Parameters
    ----------

    rerps : multi-indexed pandas.DataFrame
       Time, model, param, key x LHS, as returned by fit_lmers()

    Returns
    -------
    aics : multi-indexed pandas pd.DataFrame

    """
    # AIC and lmer warnings are 1 per model, pull from the first
    # model coefficient only, typically (Intercept)
    first_param = rerps.index.get_level_values('param').unique()[0]
    AICs = pd.DataFrame(
        (
            rerps.loc[pd.IndexSlice[:, :, first_param, 'AIC'], :]
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
        rerps.loc[pd.IndexSlice[:, :, first_param, 'has_warning'], :]
        .reset_index(['key', 'param'], drop=True)
        .stack(0),
        columns=['has_warning'],
    )

    has_warnings.index.names = has_warnings.index.names[:-1] + ['channel']
    has_warnings.sort_index(inplace=True)
    AICs = AICs.merge(has_warnings, left_index=True, right_index=True)
    FutureWarning('coef AICs are in early days, subject to change')
    return AICs


def plot_chans(
        LHS,
        rerps,
        alpha=0.05,
        fdr='BY',
        figsize=None,
        s=None,
        **kwargs):

    """Plot single channel rERPs from an rERP format dataframe with matplotlib

    Parameters
    ----------
    rerps : pd.DataFrame
       as returned by fitgrid.utils.lmer._get_rerps and
       fitgrid.utils.lm._get_rerps

    LHS : fitgrid.LHS specification
       see fitgrid.fitgrid docs

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

    # coefs = fg_lmer.coefs.index.get_level_values('param').unique()
    coefs = rerps.index.get_level_values('param').unique()
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
                rerps.loc[pd.IndexSlice[:, :, coef], col]
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
