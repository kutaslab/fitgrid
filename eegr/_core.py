import pandas as pd
from statsmodels.formula.api import ols
from tqdm import tqdm_notebook as tqdm

from ._fitgrid import FitGrid
from ._errors import EegrError

from . import EPOCH_ID, TIME


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

    if not isinstance(epochs, pd.DataFrame):
        raise EegrError('epochs must be a Pandas DataFrame.')

    if not (isinstance(LHS, list) and
            all(isinstance(item, str) for item in LHS)):
        raise EegrError('LHS must be a list of strings.')

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

    def regression(data, formula):
        # NOTE time is the groupby variable, we're dropping it here since
        # otherwise we get two index columns in statsmodels results, which
        # interferes with the creation of a dataframe in FitGrid
        data.reset_index(level=TIME, drop=True, inplace=True)
        return ols(formula, data).fit()

    # run regressions
    results = {
        channel: group_by_time.apply(regression, channel + ' ~ ' + RHS)
        for channel in tqdm(LHS)
    }

    grid = pd.DataFrame(results)

    return FitGrid(grid)
