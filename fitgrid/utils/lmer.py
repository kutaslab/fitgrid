"""utility functions for LMERFitGrid objects"""

import functools
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import fitgrid
from fitgrid.fitgrid import LMERFitGrid


def get_lmer_dfbetas(epochs, factor, **kwargs):
    r"""Fit lmers leaving out factor levels one by one, compute DBETAS.

    Parameters
    ----------
    epochs : Epochs
        Epochs object
    factor : str
        column name of the factor of interest
    **kwargs
        keyword arguments to pass on to ``fitgrid.lmer``, like ``RHS``

    Returns
    -------
    dfbetas : pandas.DataFrame
        dataframe containing DFBETAS values

    Examples
    --------
    Example calculation showing how to pass in model fitting parameters::

        dfbetas = fitgrid.utils.lmer.get_lmer_dfbetas(
            epochs=epochs,
            factor='subject_id',
            RHS='x + (x|a)
        )

    Notes
    -----
    DFBETAS is computed according to the following formula [NieGroPel2012]_:

    .. math::

       DFBETAS_{ij} = \frac{\hat{\gamma}_i - \hat{\gamma}_{i(-j)}}{se\left(\hat{\gamma}_{i(-j)}\right)}

    for parameter :math:`i` and level :math:`j` of ``factor``.


    """

    # get the factor levels
    table = epochs.table.reset_index().set_index(
        [epochs.epoch_id, epochs.time]
    )
    levels = table[factor].unique()

    # produce epochs tables with each level left out
    looo_epochs = (
        fitgrid.epochs_from_dataframe(
            table[table[factor] != level],
            time=epochs.time,
            epoch_id=epochs.epoch_id,
            channels=epochs.channels,
        )
        for level in levels
    )

    # fit lmer on these epochs
    fitter = functools.partial(fitgrid.lmer, **kwargs)
    grids = map(fitter, looo_epochs)
    coefs = (grid.coefs for grid in grids)

    # get coefficient estimates and se from leave one out fits
    looo_coefs = pd.concat(coefs, keys=levels, axis=1)
    looo_estimates = looo_coefs.loc[pd.IndexSlice[:, :, 'Estimate'], :]
    looo_se = looo_coefs.loc[pd.IndexSlice[:, :, 'SE'], :]

    # get coefficient estimates from regular fit (all levels included)
    all_levels_coefs = fitgrid.lmer(epochs, **kwargs).coefs
    all_levels_estimates = all_levels_coefs.loc[
        pd.IndexSlice[:, :, 'Estimate'], :
    ]

    # drop outer level of index for convenience
    for df in (looo_estimates, looo_se, all_levels_estimates):
        df.index = df.index.droplevel(level=-1)

    # (all_levels_estimate - level_excluded_estimate) / level_excluded_se
    dfbetas = all_levels_estimates.sub(looo_estimates, level=1).div(
        looo_se, level=1
    )

    return dfbetas.stack(level=0)


def get_lmer_warnings(lmer_grid):
    """grid the LMERFitGrid lme4::lmer4 warnings by type

    lmer warnings are a mishmash of characters, punctuation, and digits, some with
    numerical values specific to the message, for instance,

        | Model failed to converge with max|grad| = 0.00222262 (tol = 0.002, component 1)
        | unable to evaluate scaled gradient
        | boundary (singular) fit: see ?isSingular
        | np.nan

    The warning strings are returned as-is except for stripping
    leading and trailing whitespace and the "= N.NNNNNNNN" portion of the
    max \|grad\| convergence failure.

    Parameters
    ----------
    lmer_grid : fitgrid.LMERFitGrid
        as returned by ``fitgrid.lmer()``, shape = time x channel

    Returns
    -------
    warning_grids : dict
        A dictionary, the keys are lmer warning strings, each value
        is a `pandas.DataFrame` indicator grid where grid.loc[time, channel] == 1 if the
        lmer warning == key, otherwise 0.
    """

    if not isinstance(lmer_grid, LMERFitGrid):
        msg = (
            "get_lmer_warnings() must be called on an "
            f"LMERFitGrid not {type(lmer_grid)}"
        )
        raise ValueError(msg)

    # In pymer4 0.7.1+ and lme4::lmer 0.22+ warnings come back from
    # lme4::lmer via pymer4 as list of strings and each LMERFitgrid
    # cell may have a list of 0, 1, 2, ... ?  warnings. This means
    # LMERFitGrid.warnings time index may have missing time stamps (= no
    # warnings), a single time stamp (one warning), or duplicate time
    # stamps (> 1 warning) and np.nan at channels where there is no
    # warning at that timestamp.

    # strip reported decimal values so max|grad| convergence failures are one kind
    tidy_strings = lmer_grid.warnings.applymap(
        lambda x: re.sub(
            r"max\|grad\|\s+=\s+\d+\.\d+\s+", "max|grad| ", x
        ).strip()
        if isinstance(x, str)
        else x  # no warning == np.nan
    ).rename_axis([lmer_grid.time, "wdx", "_empty"], axis=0)

    # the number and types of warning generally vary by time and/or channel
    warning_kinds = (
        pd.Series(tidy_strings.to_numpy().flatten()).dropna().unique()
    )

    # collect messy gappy, multiple warnings as a dict of key==warning,
    # value==tidy time x channel indicator grid (0, 1)
    warning_grids = {}
    assert lmer_grid._grid.shape == lmer_grid.has_warning.shape
    for warning_kind in warning_kinds:

        # empty grid w/ correct shape, row index and columns
        warning_grid = pd.DataFrame(
            np.zeros(lmer_grid._grid.shape, dtype=int),
            index=lmer_grid._grid.index.copy(),
            columns=lmer_grid._grid.columns.copy(),
        )

        # select rows w/ at least one non-na
        warning_rows = tidy_strings[tidy_strings == warning_kind].dropna(
            axis=0, how="all"
        )

        assert warning_rows.index.names[0] == lmer_grid._grid.index.name
        assert all(
            warning_rows.index.get_level_values(0)
            == warning_rows.index.get_level_values(0).unique()
        )

        for rdx, row in warning_rows.iterrows():
            warning_grid.loc[rdx[0], :] = (row == warning_kind).astype(int)

        assert all(warning_grid.index == lmer_grid._grid.index)
        assert all(warning_grid.columns == lmer_grid._grid.columns)

        warning_grids[warning_kind] = warning_grid

    return warning_grids


def plot_lmer_warnings(lmer_grid, which="each", verbose=True):
    """Raster plot lme4::lmer warning grids

    Parameters
    ----------
    lmer_grid : fitgrid.LMERFitGrid
        as returned by ``fitgrid.lmer()``, shape = time x channel

    which :  {"each", "all", or list of str}
       select the types of warnings to plot. `each` (default) plots
       each type of warning separately. `all` plots one grid showing
       where any type of warning occured. A list of strings searches
       the lmer warnings and plots those that match.

    verbose : bool, default=True
       If `True` warn of failed matches for warnings keywords.


    Examples
    --------

    default, plot each warning grid separately

    >>> plot_lmer_warnings(lmer_grid)

    one plot shows everywhere there is a warning

    >>> plot_lmer_warnings(lmer_grid, which="all")

    plot just warnings that match these strings

    >>> plot_lmer_warnings(lmer_grid, which=["converge", "singular"])
    """

    def _plot_warnings(warning, warning_grid):
        # masked array non-values are transparent in pcolormesh
        _, axi = plt.subplots(figsize=(12, len(warning_grid.columns) / 2))
        axi.set_title(warning)

        ylabels = warning_grid.columns
        axi.yaxis.set_major_locator(
            mpl.ticker.FixedLocator(np.arange(len(ylabels)))
        )
        axi.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(ylabels))
        axi.pcolormesh(
            warning_grid.index,
            np.arange(len(ylabels)),
            np.ma.masked_not_equal(warning_grid.T.to_numpy(), 1),
            shading="nearest",
            cmap=mpl.colors.ListedColormap(['red']),
        )

    # validate kwarg
    if not (
        isinstance(which, str)
        or (
            isinstance(which, list)
            and all((isinstance(wrn, str) for wrn in which))
        )
    ):
        raise ValueError(
            "The value for which=value must be 'any', 'each', a warning "
            f"string pattern to match or list of them, not this: {which}"
        )

    warning_grids = get_lmer_warnings(lmer_grid)
    warning_grids["all"] = lmer_grid.has_warning.astype(int)

    keys = None
    if which == "all":
        keys = ["all"]
    elif which == "each":
        keys = list(warning_grids.keys())
    else:
        # lookup matching patterns var so as to not step on original kwarg
        patterns = [which] if isinstance(which, str) else which

        keys = []
        for pattern in patterns:
            matches = [key for key in warning_grids if pattern in key]
            keys += matches  # may be []
            if verbose and not matches:
                warnings.warn(f"warning pattern '{pattern}' not found")

    assert isinstance(keys, list), f"this should be type list: {type(keys)}"
    for key in keys:
        if verbose:
            print(f"{key}")
        _plot_warnings(key, warning_grids[key])

    if verbose and not keys:
        warnings.warn(f"no model warnings match {which}")
