from os import environ
from math import ceil
from functools import partial
from multiprocessing import Pool
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from pymer4 import Lmer  # moved up from lmer_single() for Multiprocessing
from tqdm import tqdm

from .errors import FitGridError
from . import tools
from .fitgrid import FitGrid, LMFitGrid, LMERFitGrid


def validate_LHS(epochs, LHS):

    # must be a list of strings
    if not (
        isinstance(LHS, list) and all(isinstance(item, str) for item in LHS)
    ):
        raise FitGridError('LHS must be a list of strings.')

    # all LHS items must be present in the epochs_table
    missing = set(LHS) - set(epochs.table.columns)
    if missing:
        raise FitGridError(
            'Items in LHS should all be present in the epochs table, '
            f'the following are missing: {missing}'
        )


def validate_RHS(RHS):

    # validate RHS
    if RHS is None:
        raise FitGridError('Specify the RHS argument.')
    if not isinstance(RHS, str):
        raise FitGridError('RHS has to be a string.')


def process_key_and_group(key_and_group, function, channels):
    key, group = key_and_group
    results = {channel: function(group, channel) for channel in channels}
    return pd.Series(results, name=key)


def run_model(
    epochs, function, channels=None, parallel=False, n_cores=4, quiet=False
):
    """Run an arbitrary model on the epochs.

    Parameters
    ----------
    epochs : Epochs
        the epochs object on which the model is to be run
    function : Python function
        function that runs a model, see Notes below for details
    channels : list of str
        list of channels to serve as dependent variables
    parallel : bool, defaults to False
        set to True in order to run in parallel
    n_cores : int, defaults to 4
        number of processes to run in parallel
    quiet : bool, defaults to False
        set to True to disable progress bar display

    Returns
    -------
    grid : FitGrid
        a FitGrid object containing the results

    Notes
    -----
    The function should take two parameters, ``data`` and ``channel``, run
    some model on the data, and return an object containing the results.
    ``data`` will be a snapshot across epochs at a single timepoint,
    containing all channels of interest. ``channel`` is the name of the
    target variable that the function runs the model against (uses it as
    the dependent variable).

    Examples
    --------
    Here's an example of a function that can be passed to ``run_model``::

        def regression(data, channel):
            formula = channel + ' ~ continuous + categorical'
            return ols(formula, data).fit()

    """

    _grid = _run_model(
        epochs,
        function,
        channels=channels,
        parallel=parallel,
        n_cores=n_cores,
        quiet=quiet,
    )
    return FitGrid(_grid, epochs.epoch_index, epochs.time)


def _run_model(
    epochs, function, channels=None, parallel=False, n_cores=4, quiet=False
):

    if channels is None:
        channels = epochs.channels

    validate_LHS(epochs, channels)

    groups = tqdm(epochs._snapshots, disable=quiet)
    processor = partial(
        process_key_and_group, function=function, channels=channels
    )

    if parallel:
        chunksize = ceil(len(groups) / n_cores)
        with tools.single_threaded(np):
            with Pool(n_cores) as pool:
                results = pool.map(processor, groups, chunksize=chunksize)

    else:
        results = map(processor, groups)

    grid = pd.concat(results, axis=1).T
    grid.index.name = epochs.time

    return grid  # dataframe, not FitGrid


def lm_single(data, channel, RHS, eval_env):
    formula = channel + ' ~ ' + RHS
    return ols(formula, data, eval_env=eval_env).fit()


def lm(
    epochs,
    LHS=None,
    RHS=None,
    parallel=False,
    n_cores=4,
    quiet=False,
    eval_env=4,
):
    """Run ordinary least squares linear regression on the epochs.

    Parameters
    ----------
    epochs : Epochs
        epochs object on which regression is to be run
    LHS : list of str, optional, defaults to all channels
        list of channels for the left hand side of the regression formula
    RHS : str
        right hand side of the regression formula
    parallel : bool, defaults to False
        change to True to run in parallel
    n_cores : int, defaults to 4
        number of processes to use for computation
    quiet : bool, defaults to False
        set to True to disable fitting progress bar
    eval_env : int or patsy.EvalEnvironment, defaults to 4
        environment to use for evaluating patsy formulas, see patsy docs

    Returns
    -------
    grid : LMFitGrid
        LMFitGrid object containing the results of the regression

    """

    if LHS is None:
        LHS = epochs.channels

    validate_LHS(epochs, LHS)
    validate_RHS(RHS)

    function = partial(lm_single, RHS=RHS, eval_env=eval_env)

    _grid = _run_model(
        epochs,
        function=function,
        channels=LHS,
        parallel=parallel,
        n_cores=n_cores,
        quiet=quiet,
    )

    return LMFitGrid(_grid, epochs.epoch_index, epochs.time)


def lmer_single(
    data, channel, RHS, family, conf_int, factors, permute, ordered, REML
):
    import re

    model = Lmer(channel + ' ~ ' + RHS, data=data, family=family)

    with redirect_stdout(StringIO()) as captured_stdout:
        model.fit(
            summarize=False,
            conf_int=conf_int,
            factors=factors,
            permute=permute,
            ordered=ordered,
            REML=REML,
        )

    # lmer prints warnings, capture them
    warning = captured_stdout.getvalue()

    # in pymer4 <= 0.6 lmer warnings were not attached to the model object
    # model.has_warning = True if warning else False
    # model.warning = warning

    # as of pymer4 0.7+  model.warning -> model.warnings
    model.has_warning = True if len(model.warnings) > 0 else False

    # captured_stdout.close()

    del model.data
    del model.design_matrix
    del model.model_obj

    # return model.AIC
    # return model._REML
    # return model.__class__
    return model


def lmer(
    epochs,
    LHS=None,
    RHS=None,
    family='gaussian',
    conf_int='Wald',
    factors=None,
    permute=None,
    ordered=False,
    REML=True,
    parallel=False,
    n_cores=4,
    quiet=False,
):
    """Fit lme4 linear mixed model by interfacing with R.

    Parameters
    ----------
    epochs : Epochs
        epochs object on which lmer is to be run
    LHS : list of str, optional, defaults to all channels
        list of channels for the left hand side of the lmer formula
    RHS : str
        right hand side of the lmer formula
    family : str, defaults to 'gaussian'
        distribution link function to use
    conf_int : str, defaults to 'Wald'
    factors : dict, optional
        Keys should be column names in data to treat as factors. Values
        should either be a list containing unique variable levels if
        dummy-coding or polynomial coding is desired. Otherwise values
        should themselves be dictionaries with unique variable levels as
        keys and desired contrast values (as specified in R!) as keys.
    permute : int, defaults to None
        if non-zero, computes parameter significance tests by permuting
        test stastics rather than parametrically. Permutation is done by
        shuffling observations within clusters to respect random effects
        structure of data.
    ordered : bool, defaults to False
        whether factors should be treated as ordered polynomial contrasts;
        this will parameterize a model with K-1 orthogonal polynomial
        regressors beginning with a linear contrast based on the factor
        order provided
    REML : bool, defaults to True
        change to False to use ML estimation
    parallel : bool, defaults to False
        change to True to run in parallel
    n_cores : int, defaults to 4
        number of processes to use for computation
    quiet : bool, defaults to False
        set to True to disable fitting progress bar

    Returns
    -------
    grid : LMERFitGrid
        LMERFitGrid object containing the results of lmer fitting
    """

    if LHS is None:
        LHS = epochs.channels

    validate_LHS(epochs, LHS)
    validate_RHS(RHS)

    function = partial(
        lmer_single,
        RHS=RHS,
        family=family,
        conf_int=conf_int,
        factors=factors,
        permute=permute,
        ordered=ordered,
        REML=REML,
    )
    _grid = _run_model(
        epochs,
        function,
        channels=LHS,
        parallel=parallel,
        n_cores=n_cores,
        quiet=quiet,
    )

    return LMERFitGrid(_grid, epochs.epoch_index, epochs.time)
