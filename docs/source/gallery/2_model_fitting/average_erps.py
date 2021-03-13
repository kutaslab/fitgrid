""".. _average_erps:

Average ERPs with `fitgrid`
===========================
"""

# %%
# For designed EEG experiments with categorical variables, a useful
# range of Hunt-Dawson average ERPs and ERP effects fall out of
# ordinary least squares regression modeling by selecting the
# appropriate categorical predictor variable coding with the `patsy`
# formula language. The results are identical to the addition,
# subtraction of average ERP waveforms without programming
# ad hoc algebraic manipulations.

# %%
# Prepare and load epochs
# -----------------------

import pandas as pd
import fitgrid as fg
from fitgrid import DATA_DIR, sample_data

sample_data.get_file("sub000p3.ms1500.epochs.feather")
p3_epochs_df = pd.read_feather(DATA_DIR / "sub000p3.ms1500.epochs.feather")

# select 3 types of stimulus event: standards, targets, and bioamp calibration triggers
p3_epochs_df = p3_epochs_df.query("stim in ['standard', 'target', 'cal']")

# look up the data QC flags and select the good epochs
good_epochs = p3_epochs_df.query("match_time == 0 and log_flags == 0")[
    "epoch_id"
]
p3_epochs_df = p3_epochs_df.query("epoch_id in @good_epochs")

# rename the time stamp column
p3_epochs_df.rename(columns={"match_time": "time_ms"}, inplace=True)

# select columns of interest for modeling
indices = ["epoch_id", "time_ms"]
predictors = ["stim"]  # categorical with 2 levels: standard, target
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

# load data into fitgrid.Epochs
p3_epochs_fg = fg.epochs_from_dataframe(
    p3_epochs_df, epoch_id="epoch_id", time="time_ms", channels=channels
)

#%%
# average ERPs by condition: :math:`\sim \mathsf{0 + stim}`
# ---------------------------------------------------------
#
# Supressing the intercept term in the `patsy` model formula triggers
# full-rank dummy (indicator) coding of the two-level categorical variable.
# The estimated coefficients are identical to the average ERPs in
# each condition. The minimal design matrix illustrates dummy
# coding for one categorical variable with two levels.

# %%
lmg_0_stim = fg.lm(p3_epochs_fg, RHS="0 + stim")


# %%
# Parameter estimates = Smith & Kutas (2015) regression ERPs
beta_hats = lmg_0_stim.params
beta_hats

# %%
# Parameter estimate standard errors
bses = lmg_0_stim.bse
bses


# %%
# Visualize parameter estimates +/- standard error
from matplotlib import pyplot as plt

# label index columns for pandas groupby
for attr_df in [beta_hats, bses]:
    attr_df.index.set_names(["time_ms", "beta_hats"], inplace=True)

for beta_hat, vals in beta_hats.groupby("beta_hats"):
    vals.reset_index('beta_hats', inplace=True, drop=True)
    times = vals.index.to_numpy()
    bse = bses.query("beta_hats==@beta_hat")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(beta_hat)
    ax.set(
        xlabel="Time (ms)",
        xlim=(times[0], times[-1]),
        ylabel=r"$\mu$V",
        ylim=(-15, 15),
    )
    ax.axhline(0, color="lightgray", lw=1)
    ax.axvline(0, color="gray", lw=1)

    for jdx, chan in enumerate(vals.columns):
        ax.plot(times, vals[chan], label=chan)
        ax.fill_between(
            times, vals[chan] - bse[chan], vals[chan] + bse[chan], alpha=0.2
        )
    ax.legend(loc=(1.05, 0.5))


# %%
# Why this works
# --------------
#
# Here is a small ("right hand side") design matrix for 9 observations
# of a categorical variable with 3 levels.  There is no intercept
# (constant) and when one of the 3 regressors is 1, the others are
# 0. The :math:`\hat{\beta}` weights that minimize overall error are
# the means of the data at each level of the categorical variable.

from patsy import demo_data, dmatrix

cat_2 = demo_data("a", nlevels=3, min_rows=8)
dmatrix("0 + a", data=cat_2, return_type="dataframe")

# %%
# For EEG data, the "means of the data at each level of the categorical variable"
# are the time-domain average ERPs.  In the sample data, the categorical stimulus
# variable has three levels: `standard`, `target`, and `cal` for the
# 10 :math:`\mu\mathsf{V}` calibration square wave.
#
# We can reach into one cell of the `FitGrid` at time = 0 and channel
# = `MiPa` and pull out the design matrix. The three column indicator
# coding is the same as the `demo_data` example except for the column
# labels and hundreds observations instead of 9.

# %%
lmg_0_stim[0, "MiPa"].model.exog_names.unstack(-2)

# %%
lmg_0_stim[0, "MiPa"].model.exog.unstack(-1)
