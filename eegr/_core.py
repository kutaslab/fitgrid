import numpy as np
import patsy
from statsmodels.formula.api import ols

from ._fitgrid import FitGrid


def build_bucket_dt(n_betas, n_epochs):
    """Build bucket dtype combining fit and diagnostic data.

    TODO: select diagnostic parameters carefully.
    """

    fit_dt = np.dtype([
        ('betas', np.float64, n_betas),
        ('se', np.float64, n_betas),
        ('ci', np.float64, (n_betas, 2))
    ])

    diag_dt = np.dtype([
        ('cooks_d', np.float64,  (2, n_epochs)),
        ('ess_press', np.float64),  # np.dot(resid_press, resid_press).sum()
        ('resid_press', np.float64, n_epochs),
        ('resid_std', np.float64, n_epochs),
        ('resid_var', np.float64, n_epochs),
        ('resid_studentized_internal', np.float64, n_epochs)
    ])

    bucket_dt = np.dtype([
        ('fit', fit_dt),
        ('diag', diag_dt)
    ])

    return bucket_dt


def fill_bucket(fit_obj, bucket_dt):
    """Given bucket dtype and fit object, fill bucket."""

    influence = fit_obj.get_influence()

    fit = np.array((
        fit_obj.params,
        fit_obj.bse,
        fit_obj.conf_int()
        ),
        bucket_dt['fit']
    )

    diag = np.array((
        np.array(influence.cooks_distance),  # have to wrap tuple of two arrays
        influence.ess_press,
        influence.resid_press,
        influence.resid_std,
        influence.resid_var,
        influence.resid_studentized_internal
        ),
        bucket_dt['diag']
    )

    bucket = np.array((fit, diag), bucket_dt)

    return bucket


def build_grid(epochs, LHS, RHS):
    """Given an epochs table, LHS, and RHS, build a grid with fit info.

    Parameters
    ----------

    epochs : pandas DataFrame
        must have 'epoch_id' and 'time' index columns
    LHS : list of str
        list of channels to be modeled as response variables
    RHS : str
        patsy formula specification of predictors

    Returns
    -------

    fitgrid : FitGrid object

    Notes
    -----

    Assumptions:
        - every epoch has equal number of samples
        - every epoch has same time index
    """

    EPOCH_ID = 'epoch_id'
    TIME = 'time'

    assert TIME in epochs.index.names and EPOCH_ID in epochs.index.names

    # build bucket datatype
    n_betas = len(patsy.ModelDesc.from_formula(RHS).rhs_termlist)
    n_epochs = len(epochs.index.unique(EPOCH_ID))
    bucket_dt = build_bucket_dt(n_betas, n_epochs)

    # run regressions
    formulas = (channel + ' ~ ' + RHS for channel in LHS)
    fits = (
        ols(formula, snapshot).fit()
        for formula in formulas
        for _, snapshot in epochs.groupby(TIME)
    )

    # fill buckets with results
    buckets = [
        fill_bucket(fit, bucket_dt)
        for fit in fits
    ]

    # build grid
    n_channels = len(LHS)
    n_samples = len(epochs.groupby(TIME))
    grid = np.array(buckets, bucket_dt).reshape(n_channels, n_samples)

    epoch_index = epochs.groupby(TIME).first().index
    return FitGrid(grid, LHS, epoch_index)
