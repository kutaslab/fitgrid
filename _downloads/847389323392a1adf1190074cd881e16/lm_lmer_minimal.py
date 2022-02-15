"""Ordinary least squares and linear mixed-effects: minimal examples
====================================================================

OLS and mixed-effect models are specified by model formulas. The
results returned by ``statstmodels`` for OLS and ``lme4::lmer`` for
mixed-effcts models populate the :py:class:`FitGrid
<fitgrid.fitgrid.FitGrid>` object. The :py:class:`FitGrid[times, channels]
<fitgrid.fitgrid.FitGrid>` can be sliced by
times or channels with ``pandas`` index slicing. The results are
accessed via the fit object attributes and returned as a
``pandas.DataFrame`` or another :py:class:`FitGrid[times, channels]
<fitgrid.fitgrid.FitGrid>`.

"""


# %%
# Generate simulated data and load :py:class:`Epochs <fitgrid.epochs.Epochs>`
# ---------------------------------------------------------------------------

import fitgrid

epochs_df = fitgrid.generate(
    n_samples=6, n_channels=4, return_type="dataframe"
)
epochs_df.set_index(["epoch_id", "time"], inplace=True)
epochs_fg = fitgrid.epochs_from_dataframe(
    epochs_df,
    epoch_id="epoch_id",
    time="time",
    channels=["channel0", "channel1"],
)

# %%
# Ordinary least squares (OLS)
# ----------------------------
#
# These models are specified with :std:doc:`patsy <patsy:index>` Python formulas like ``lm`` in ``R``. The
# results come back via ``statsmodels`` as :py:class:`FitGrid[times, channels]
# <fitgrid.fitgrid.FitGrid>` objects populated with :py:class:`linear_model.RegressionResults
# <statsmodels.regression.linear_model.RegressionResults>`.

lm_grid = fitgrid.lm(epochs_fg, RHS='1 + categorical + continuous', quiet=True)

# %%
# Query and display OLS parameters
lm_grid.params

# %%
# Query and display parameter standard errors
lm_grid.bse

# %% label index and quick plot with pandas
params = lm_grid.params
params.index = params.index.set_names(["time", "params"])
for param, vals in params.groupby("params"):
    ax = vals.reset_index("params", drop=True).plot()
    ax.set_title(param)


# %%
# Linear mixed effects (LMER)
# ---------------------------
#
# These models are specified with ``lme4::lmer`` R formulas and the
# results come back via ``pymer4`` as :py:class:`FitGrid[times,
# channels] <fitgrid.fitgrid.FitGrid>` objects populated with
# :py:class:`Lmer <pymer4.models.Lmer>` objects from the
# ``lme4::lmer`` and ``lmerTest`` results.

# %%
# Fit a mixed-effects model with `lme4::lmer` via `pymer4`
lmer_grid = fitgrid.lmer(
    epochs_fg, RHS='1 + continuous + (continuous | categorical)', quiet=True
)

# %%
# Query and display some lme4::lmer fit results
lmer_grid.coefs
