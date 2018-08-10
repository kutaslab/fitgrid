import numpy as np
import patsy
from statsmodels.formula.api import ols

from ._fitgrid import FitGrid
from ._errors import EegrError

# index variables in epoch table
EPOCH_ID = 'epoch_id'
TIME = 'time'


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


def _check_group_indices(group_by, index_level):
    """Check groups have same index using transitivity."""

    prev_group = None
    for idx, cur_group in group_by:
        if prev_group is not None:
            prev_indices = prev_group.index.get_level_values(index_level)
            cur_indices = cur_group.index.get_level_values(index_level)
            if not prev_indices.equals(cur_indices):
                return False, idx
        prev_group = cur_group

    return True, None


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

    # these index columns are required for groupby's
    assert TIME in epochs.index.names and EPOCH_ID in epochs.index.names

    group_by_epoch = epochs.groupby(EPOCH_ID)
    group_by_time = epochs.groupby(TIME)

    # verify all epochs have same time index
    same_time_index, epoch_idx = _check_group_indices(group_by_epoch, TIME)
    if not same_time_index:
        raise EegrError(f'Epoch {epoch_idx} differs from previous epoch '
                        f'in {TIME} index.')

    # check snapshots are across same epochs
    same_epoch_index, snap_idx = _check_group_indices(group_by_time, EPOCH_ID)
    if not same_epoch_index:
        raise EegrError(f'Snapshot {snap_idx} differs from previous '
                        f'snapshot in {EPOCH_ID} index.')

    # build bucket datatype
    # we don't know in advance how many betas patsy will end up encoding, so we
    # run a dmatrix builder on a single snapshot and note the number of columns
    single_snapshot = group_by_time.get_group(0)
    n_betas = len(patsy.dmatrix(RHS, single_snapshot).design_info.column_names)
    n_epochs = len(single_snapshot)
    bucket_dt = build_bucket_dt(n_betas, n_epochs)

    # run regressions
    formulas = (channel + ' ~ ' + RHS for channel in LHS)
    fits = (
        ols(formula, snapshot).fit()
        for formula in formulas
        for _, snapshot in group_by_time
    )

    # fill buckets with results
    buckets = [
        fill_bucket(fit, bucket_dt)
        for fit in fits
    ]

    # build grid
    n_channels = len(LHS)
    n_samples = len(group_by_time)
    grid = np.array(buckets, bucket_dt).reshape(n_channels, n_samples)

    epoch_index = group_by_epoch.get_group(0).index
    return FitGrid(grid, LHS, epoch_index)
