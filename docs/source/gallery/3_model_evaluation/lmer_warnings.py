"""LMER warning grids
#####################

The modeling at each cell in a fitted LMERFitGrid may generate
various numbers and types of lme4::lmer warnings.

The examples here illustrate how to collect and visualize them with
fitgrid utilities.

"""
import fitgrid

# %%
# To illustrate, we use a small random data set and silly model to
# generate lots of different warnings.
epochs_fg = fitgrid.generate(n_samples=8, n_channels=4, seed=32)

lmer_grid = fitgrid.lmer(
    epochs_fg,
    LHS=epochs_fg.channels,
    RHS="categorical + (continuous | categorical)",
    parallel=True,
    n_cores=2,
    quiet=True,
)

# %%
# :py:func:`fitgrid.utils.lmer.get_lmer_warnings`
# ===============================================
#
# The ``get_lmer_warnings()`` utility collects the warnings and
# returns them as an ordinary Python dictionary.  Each warning message
# is a key and its value is a time x channel indicator grid of 0s, and
# 1s: the 1s show which grid cells have the warning.
lmer_warnings = fitgrid.utils.lmer.get_lmer_warnings(lmer_grid)
for key, val in lmer_warnings.items():
    print(key, "\n", val)

# %%
# :py:func:`fitgrid.utils.lmer.get_lmer_warnings`
# ===============================================
#
# The ``plot_lmer_warnings()`` utility visualizes the warning grids.
#
# The warnings can be displayed in different ways with ``which=...`` keyword argument.
#
# The default (``which="each"``) plots each type of warning in a separate figure.
fitgrid.utils.lmer.plot_lmer_warnings(lmer_grid)

# %%
# Stacking all the warning grids into one summary grid (``which="all"``)
# shows immediately which grid cells have warnings and which do not.
fitgrid.utils.lmer.plot_lmer_warnings(
    lmer_grid,
    which="all",
)

# %%
# Specific warnings can be selected by matching a portion of the
# warning message text.
fitgrid.utils.lmer.plot_lmer_warnings(lmer_grid, which=["Hessian"])

# %%
# .. warning::
#
#    Watch out for typos when selecting LMER warnings to plot, the text must match some
#    part of the warning message exactly.
#
# This selection finds no convergence warnings.
fitgrid.utils.lmer.plot_lmer_warnings(lmer_grid, which=["converges"])

# %%
# They were missed because "converges" doesn't match "converge:" or
# "converge " in the warning messages.
fitgrid.utils.lmer.plot_lmer_warnings(lmer_grid, which=["converge"])
