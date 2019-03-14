import functools
import pandas as pd
import fitgrid


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
