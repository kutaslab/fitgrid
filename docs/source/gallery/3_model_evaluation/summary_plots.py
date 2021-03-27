"""Model summary visualization
==============================
"""

# %%
#
# These examples illustrate LMER and OLS model summary visualizations.
# For LMER models, this includes the model fitting warnings that are often encountered and
# other interesting information like the estimated degrees of freedom.
#
# The fitgrid utilities for summary plotting all return matplot.Figure and matplotlib.Axes objects
# or a list of them so they can be fine tuned.
#
# The examples use a small simulated data set to speed up
# processing and silly models to generate interesting warnings.

# %%
# .. warning::
#
#    In practice, a model summary data frame may have large numbers of channels, models, and betas
#    and the default plotting behavior is to plot the entire dataframe. To avoid
#    overgenerating plots, select what you want to plot with the optional keyword
#    arguments and/or prune the summary dataframe before plotting.
#

from matplotlib import pyplot as plt
import fitgrid

# %%
# A small random data set
epochs_fg = fitgrid.generate(n_samples=8, n_channels=4, seed=32)
channels = [
    column for column in epochs_fg.table.columns if "channel" in column
]

# %%
# Summarize a stack of two LMER models
lmer_rhs = [
    "1 + categorical + (continuous | categorical)",
    "1 + continuous + (1 | categorical)",
]

lmer_summaries = fitgrid.utils.summary.summarize(
    epochs_fg,
    "lmer",
    LHS=channels,
    RHS=lmer_rhs,
    parallel=True,
    n_cores=2,
    quiet=True,
)
lmer_summaries

# %%
# Select and plot one model intercept at a channels. Model warnings,
# if any, are plotted by default.
figs = fitgrid.utils.summary.plot_betas(
    lmer_summaries,
    LHS=["channel2"],
    models=["1+categorical+(continuous|categorical)"],
    betas=["(Intercept)"],
    beta_plot_kwargs={"ylim": (-75, 75)},
)

# %%
# Degrees of freedom for mixed-effects models is somewhat controversial, you can plot those returned
# by lmerTest, scaled by a function of your choosing such as ``numpy.log10()`` when the df are much larger
# than the betas. The identity function ``dof()`` below shows how to define your own.
def dof(x):
    return x


figs = fitgrid.utils.summary.plot_betas(
    lmer_summaries,
    LHS=["channel2"],
    models=["1+categorical+(continuous|categorical)"],
    betas=["(Intercept)"],
    beta_plot_kwargs={"ylim": (-100, 100)},
    df_func=dof,  # overplot degrees of freedom
)


# %%
# FDR controlled beta differences from zero can be plotted as well, though
# for LMER models, *p* values are somewhat controversial. For this toy
# data set with random data, none of the tests survive the FDR control
# procedure.
figs = fitgrid.utils.summary.plot_betas(
    lmer_summaries,
    LHS=["channel2"],
    models=["1+categorical+(continuous|categorical)"],
    betas=["(Intercept)"],
    beta_plot_kwargs={"ylim": (-100, 100)},
    fdr_kw={"method": "BY"},
)


# %%
# For AIC :math:`\Delta_\mathsf{min}` plots the default
# is to highlight all grid cells with warnings.
fig, axs = fitgrid.utils.summary.plot_AICmin_deltas(
    lmer_summaries, figsize=(12, 5),
)
fig.tight_layout()


# %%
# Plot all warnings and display the warning types
plt.close("all")
fig, ax = fitgrid.utils.summary.plot_AICmin_deltas(
    lmer_summaries, show_warnings="labels"
)
fig.tight_layout()

# %%
# Select specific warning types to plot
plt.close("all")
fig, ax = fitgrid.utils.summary.plot_AICmin_deltas(
    lmer_summaries, show_warnings=["converge"]
)
fig.tight_layout()


# %%
# OLS Summaries
# ----------------------------------------
#
# Summaries of OLS models and models stacks are computed the same way as LMER models.

# Compute OLS fit summaries for two models
lm_rhs = ["1 + categorical", "1 + continuous"]
lm_summaries = fitgrid.utils.summary.summarize(
    epochs_fg, "lm", LHS=channels, RHS=lm_rhs, parallel=False, quiet=True,
)
lm_summaries


# %%
#
# Since summary data frames are indexed alike for LMER and OLS, the same
# beta and AIC :math:`\Delta_\mathsf{min}` plotting are used and work
# the same way.

plt.close('all')
figs = fitgrid.utils.summary.plot_betas(
    lm_summaries,
    LHS=["channel0", "channel1"],
    models=["1 + categorical"],
    betas=["Intercept"],
    beta_plot_kwargs={"ylim": (-100, 100)},
    interval=[2, 6],
)

# %%
# AIC :math:`\Delta_\mathsf{min}`

plt.close("all")
fig, ax = fitgrid.utils.summary.plot_AICmin_deltas(lm_summaries)
fig.tight_layout()

#%%
# Some matplotlib options can be passed through.
#
# The matplotlib.Figure and matplotlib.Axes are returned so they can be customized.

plt.close("all")
fig, axs = fitgrid.utils.summary.plot_AICmin_deltas(
    lm_summaries,
    figsize=(12, 8),
    gridspec_kw={"width_ratios": [1, 1, 0.1]},  # column widths
)

# example Axes tuning
for ax in axs:
    ax[0].set(ylim=(0, 12))
    for line in [hl for hl in ax[0].get_lines()]:
        if "channel" not in line.get_label():
            line.set(linewidth=1.5, color="lightgray")
        else:
            line.set(linewidth=3)

axs[1][1].annotate(
    text=(
        "For these $\mathcal{N}(0, 1)$ random data,\n"
        "$\Delta_{\mathsf{min}}$ > 4 are rare by chance."
    ),
    xy=(1, 0.6),
    xytext=(1.75, 1.25),
    arrowprops={"width": 2},
    # multialign="left",
)
fig.tight_layout()
