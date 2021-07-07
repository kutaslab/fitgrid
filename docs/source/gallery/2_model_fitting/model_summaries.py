"""Model summary dataframes
========================


Whereas ``fitgrid.lm()`` and ``fitgrid.lmer()`` return all the
available fit information as LMFitGrid or LMERFitGrid object,
respectively, the ``fitgrid.utils.summary.summarize()`` function
gathers a useful subset into a tidy indexed pandas.Dataframe.

The summary dataframe row and column indexing is standardized, so
summaries for different models and model sets can be conveniently
split and stacked with ordinary pandas index slicing and dataframe
concatenation.

A list of model formulas can be summarized, in
which case their summaries are stacked and returned in a single
dataframe.

"""

import fitgrid

# a small random data set for illustration
epochs_fg = fitgrid.generate(n_samples=8, n_channels=4, seed=32)

# %%
# Fit the OLS model ... get an LMFitGrid object
fitgrid.lm(epochs_fg, LHS=epochs_fg.channels, RHS="1 + continuous", quiet=True)

# %%
# Summarize the OLS model ... get a pandas.DataFrame
fitgrid.utils.summary.summarize(
    epochs_fg,
    modeler="lm",
    LHS=epochs_fg.channels,
    RHS=["1 + continuous"],
    quiet=True,
)

# %%
# Summarize a **stack** of OLS models ... get a pandas.DataFrame with at **stack** of summaries
fitgrid.utils.summary.summarize(
    epochs_fg,
    modeler="lm",
    LHS=epochs_fg.channels,
    RHS=["1 + continuous", "1 + categorical"],
    quiet=True,
)

# %%
# Same goes for LMER models and model stacks ...
fitgrid.utils.summary.summarize(
    epochs_fg,
    modeler="lmer",
    LHS=epochs_fg.channels,
    RHS=[
        "1 + categorical + (continuous | categorical)",
        "1 + continuous + (1 | categorical)",
    ],
    parallel=True,
    n_cores=2,
    quiet=True,
)
