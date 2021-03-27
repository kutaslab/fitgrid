"""
False Discovery Rate (FDR) control for estimated coefficents
============================================================
"""

# %%
# The fitgrid summary utilities include an FDR critical value
# calculator for special case of estimated predictor coefficients
# :math:`\hat{\beta}_i \neq 0`. The family of tests is assumed to be
# all betas in the model summary dataframe so the size of the family
# will shrink if a summary data frame is sliced down by channels,
# models, betas, or times and grow if summaries are stacked
# together. The test is based on the *p*-values returned by the
# statsmodels for fitgrid.lm() and the lmerTest *p*-values for
# fitgrid.lmer().  The methods implemented are from [BenYek2001]_ and
# [BenHoc1995]_.

# %%
# FDR control is illustrated here for some sample EEG data.
import numpy as np
import pandas as pd
import fitgrid as fg
from fitgrid import DATA_DIR, sample_data

sample_data.get_file("sub000p3.ms1500.epochs.feather")
p3_epochs_df = pd.read_feather(DATA_DIR / "sub000p3.ms1500.epochs.feather")

# select stimulus types
p3_epochs_df = p3_epochs_df.query(
    "stim in ['standard', 'target'] and tone in ['hi', 'lo']"
)

# look up the data QC flags and select the good epochs
good_epochs = p3_epochs_df.query("match_time == 0 and log_flags == 0")[
    "epoch_id"
]
p3_epochs_df = p3_epochs_df.query("epoch_id in @good_epochs")

# rename the time stamp column
p3_epochs_df.rename(columns={"match_time": "time"}, inplace=True)

# select columns of interest for modeling
indices = ["epoch_id", "time"]
predictors = ["stim", "tone"]  # categorical with 2 levels: standard, target
channels = ["MiPf", "MiCe", "MiPa", "MiOc"]  # midline electrodes
p3_epochs_df = p3_epochs_df[indices + predictors + channels]

# set the epoch and time column index for fg.Epochs
p3_epochs_df.set_index(["epoch_id", "time"], inplace=True)

# "baseline", i.e., center each epoch on the 200 ms pre-stimulus interval
centered = []
for epoch_id, vals in p3_epochs_df.groupby("epoch_id"):
    centered.append(
        vals[channels]
        - vals[channels].query("time >= -200 and time < 0").mean()
    )
p3_epochs_df[channels] = pd.concat(centered)

# load data into fg.Epochs
p3_epochs_fg = fg.epochs_from_dataframe(
    p3_epochs_df, epoch_id="epoch_id", time="time", channels=channels
)

# %%
# Summarize a simple model with one categorical predictor: stim (2 levels: standard, target).
lm_summary = fg.utils.summary.summarize(
    p3_epochs_fg, modeler="lm", LHS=channels, RHS=["1 + stim",], quiet=True,
)
lm_summary

# %%
# The summary dataframe includes the *t* statistic for the test
# :math:`\hat{\beta} \neq 0` and corresponding
# *p*-values, uncorrected for multiple comparisons.
lm_summary.query("key in ['T-stat', 'P-val']")

# %%
# Here, for example is Benjamini and Hochberg FDR control at 0.05 (=default) for **all** estimated betas in the summary data frame.
fdr_info, fig, ax = fg.utils.summary.summaries_fdr_control(lm_summary)

# %%
# Out of curiousity, how many are below critical *p* for this FDR control?
pvals = lm_summary.query("key=='P-val'").to_numpy().flatten()  # fetch p values
assert fdr_info['n_pvals'] == len(pvals)  # these must agree

n_crit_p = len(np.where(pvals < fdr_info["crit_p"])[0])
print(
    f"There are {n_crit_p}/{fdr_info['n_pvals']} = {n_crit_p/fdr_info['n_pvals']:.3f} "
    f"below critical p = {fdr_info['crit_p']} for FDR control"
)

# %%
# Out of curiousity, how many are below uncorrected *p* = 0.05?
n_crit_05 = len(np.where(pvals < 0.05)[0])
print(
    f"There are {n_crit_05}/{fdr_info['n_pvals']} = {n_crit_05/fdr_info['n_pvals']:.3f} "
    "below unadjusted p=0.05"
)


# %%
# The fitgrid utilities can display FDR results along with the
# time series of coefficient estimates, the :math:`\hat\beta_i`.
# Four midline scalp channels are shown here. The black dots indicate
# a non-zero experimental effect of stimulus type
# according to this FDR control procedure. Their latency and scalp
# distribution around 300 ms post-stimulus over
# central and posterior scalp shows good agreement with the P300 average ERP
# effect typically observed in this kind of auditory oddball paradigm.

figs = fg.utils.summary.plot_betas(
    lm_summary,
    fdr_kw={"method": "BY", "rate": 0.05},
    betas=["stim[T.target]"],
    scatter_size=50,
)
for fig in figs:
    fig.get_axes()[0].set(ylim=(-15, 15))
