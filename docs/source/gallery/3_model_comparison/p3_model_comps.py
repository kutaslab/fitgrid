"""AIC and likelihood ratio model comparison
=========================================

There are many approaches to comparing models and evaluating relative
goodness-of-fit. The `FitGrid[time, channe]` query mechanism is
designed to streamline the computations for whatever approach is
deemed appropriate for the research question at hand..


The AIC example demonstrates the `fitgrid.utils.summarize` function, a
convenience wrapper that fits a list of models and returns key summary
information for each as an indexed data frame. The summary may be
passed to a function for visualizing the AIC
:math:`\mathsf{\Delta_{min}}` model comparison ([BurAnd2004]_) as
shown, or processed by custom user routines as needed.

For cases where the summary information is insufficient, the
likelihood ratio example illustrates how to compute and visualize a
model comparison measure derived from the original fitted grids. Other
measures may be computed in a similar manner.

"""

# %%
# Prepare sample data
# -------------------

import pandas as pd
from matplotlib import pyplot as plt
import fitgrid as fg
from fitgrid import DATA_DIR, sample_data

sample_data.get_file("sub000p3.ms1500.epochs.feather")
p3_epochs_df = pd.read_feather(DATA_DIR / "sub000p3.ms1500.epochs.feather")

# drop calibration pulses for these examples
p3_epochs_df = p3_epochs_df.query("stim != 'cal'")

# look up the data QC flags and select the good epochs
good_epochs = p3_epochs_df.query("match_time == 0 and log_flags == 0")[
    "epoch_id"
]
p3_epochs_df = p3_epochs_df.query("epoch_id in @good_epochs")

# rename the time stamp column
p3_epochs_df.rename(columns={"match_time": "time_ms"}, inplace=True)

# select columns of interest for modeling
indices = ["epoch_id", "time_ms"]
predictors = ["stim", "tone"]  # categorical with 2 levels: standard, target
channels = ["MiPf", "MiCe", "MiPa", "MiOc"]  # midline electrodes
p3_epochs_df = p3_epochs_df[indices + predictors + channels]

# set the epoch and time column index for fg.Epochs
p3_epochs_df.set_index(["epoch_id", "time_ms"], inplace=True)

# "baseline", i.e., center each epoch on the 200 ms pre-stimulus interval
centered = []
for epoch_id, vals in p3_epochs_df.groupby("epoch_id"):
    centered.append(
        vals[channels]
        - vals[channels].query("time_ms >= -200 and time_ms < 0").mean()
    )
p3_epochs_df[channels] = pd.concat(centered)

# slice epochs down to a shorter interval
p3_epochs_df.query("time_ms >= -200 and time_ms <= 600", inplace=True)
p3_epochs_df

# %%
# Ingest as `fitgrid.Epochs`
# --------------------------
p3_epochs_fg = fg.epochs_from_dataframe(
    p3_epochs_df, epoch_id="epoch_id", time="time_ms", channels=channels
)


# %%
# Model summaries
# ---------------
#
# The `fitgrid.utils.summarize` function is a convenience wrapper that fits
# a list of models and collects a few key summary measures into
# a single dataframe indexed by model, beta estimate, and the measure.
# It supports OLS and LME model fitting and the summaries are
# returned are in the same format.
#
# This experimental design is a fully crossed 2 (stim: standard, target)
# :math:`\times` 2 (tone: hi, lo). The predictors are treatment coded (`patsy` default).
#
# Here is a stack of 5 models to summarize and compare:

rhss_T = [
    "1 + stim + tone + stim:tone",  # long form of "stim * tone"
    "1 + stim + tone",
    "1 + stim",
    "1 + tone",
    "1",
]

# %%
from fitgrid.utils.summary import summarize

lm_summary_T = summarize(
    p3_epochs_fg, modeler="lm", RHS=rhss_T, LHS=channels, parallel=False
)
lm_summary_T


# %%
# AIC model comparison: :math:`\Delta_{i} = \mathsf{AIC}_{i} - \mathsf{AIC_{min}}`
# --------------------------------------------------------------------------------
#
# Akiakie's information criterion (AIC) increases with residual error
# and the number of model parameters so comparison on AIC favors the
# better fitting, more parsimonious models with lower AIC values.
#
# This example visualizes the channel-wise time course of
# :math:`\Delta_{i} = \mathsf{AIC}_{i} - \mathsf{AIC_{min}}`, a
# measure of the AIC of model *i* vs. the lowest AIC of any model in
# the set. Burnham and Anderson propose
# heuristics where models with :math:`\Delta_{i}` around 4 are less well
# supported by the data than the alternative(s) and models with :math:`\Delta_{i}`
# > 7 substantially less so.
#
# In the next figure, the line plots (left column) and raster plots
# (right column) show the same data in different ways.  Higher
# amplitude line plots and corresponding darker shades of blue in the
# raster plot indicate that the model's AIC is higher than the best
# candidate in the set.
#
# **Prestimulus.** Prior to stimulus onset at time = 0, the more
# parsimonious models (bottom three rows) have systematically lower
# AIC values (broad regions of lighest blue) than the more complex
# models (top two rows). This indicates that during this interval, the
# predictor variables alone and in combination do not soak up enough
# variability to offset the penalty for increasing the model
# complexity. In terms of this AIC measure, none of the models appear
# systematically better supported by the data than the null model
# (bottom row) in the prestimulus interval.
#
# **Post-stimulus.** In the interval between around 300 - 375 ms
# poststimulus, the full model that includes `stim` and `tone`
# predictors and their interaction has the minimum AIC among the
# models compared at all channels except the prefrontal MiPf.  The
# sharp transients in the magntiude of the AIC differences (> 7) at
# these channels in this interval indicates substantially less support
# for the alternative models.
#

from fitgrid.utils.summary import plot_AICmin_deltas

fig, axs = plot_AICmin_deltas(lm_summary_T, figsize=(12, 12))
fig.tight_layout()
for ax_row in range(len(axs)):
    axs[ax_row, 0].set(ylim=(0, 50))

# %%
# Likelihood Ratio
# ----------------
#
# This example fits a full and reduced
# model, computes and then visualizes the time course of likelihood ratios
# with a few lines of code.

# %%
# Fit the full model. The log likelihood dataframe is returned by querying the `FitGrid`.
lmg_full = fg.lm(p3_epochs_fg, RHS="1 + stim + tone + stim:tone", LHS=channels)
lmg_full.llf

# %%
# Fit the reduced model likewise.
lmg_reduced = fg.lm(p3_epochs_fg, RHS="1 + stim + tone", LHS=channels)
lmg_reduced.llf

# %%
# **Calculate.** The likelihood ratio is the difference of the log likelihoods.
likelihood_ratio = lmg_full.llf - lmg_reduced.llf
likelihood_ratio

# %%
# **Visualize.** This comparison shows that stimulus x tone interaction
# term in the model has little systematic impact on the
# goodness-of-fit as given by the likelihood except around 300 - 375 ms poststimulus,
# largest over central scalp (MiCe).

fig, ax = plt.subplots(figsize=(12, 3))

# render
im = ax.imshow(likelihood_ratio.T, interpolation="none", aspect=16)
cb = fig.colorbar(im, ax=ax)

# label
ax.set_title("Likelihood ratio")
ax.set(xlabel="Time (ms)", ylabel="Channel")

xtick_labels = range(-200, 600, 100)
ax.set_xticks([likelihood_ratio.index.get_loc(tick) for tick in xtick_labels])
ax.set_xticklabels(xtick_labels)

ax.set_yticks(range(len(likelihood_ratio.columns)))
ax.set_yticklabels(likelihood_ratio.columns)
fig.tight_layout()

# %%
fig, ax = plt.subplots(figsize=(8, 3))
ax.set_title("Likelihood Ratio")
_ = (lmg_full.llf - lmg_reduced.llf).plot(ax=ax)
