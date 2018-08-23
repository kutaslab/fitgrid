import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

from ._fitgrid import FitGrid
from ._errors import EegrError
from . import EPOCH_ID, TIME


class Scout:

    def __init__(self):
        self.columns = set()

    def __getitem__(self, name):
        self.columns.add(name)
        return np.empty(0)




def linear_regression(epochs, LHS, RHS):
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





    return FitGrid(grid)


def mixed_linear_model(epochs, LHS, RHS, groups, re_formula=None):
    """Given an epochs table, LHS, and RHS, build a grid with fit info.

    Parameters
    ----------

    epochs : pandas DataFrame
        must have 'epoch_id' and 'time' index columns
    LHS : list of str
        list of channels to be modeled as response variables
    RHS : str
        patsy formula specification of fixed effects
    groups : str or list of str
        specification of groups for random intercepts
    re_formula : str
        formula specification of terms with random slopes

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

    group_by_time = epochs.groupby(TIME)

    # check snapshots are across same epochs
    same_epoch_index, snap_idx = _check_group_indices(group_by_time, EPOCH_ID)
    if not same_epoch_index:
        raise EegrError(f'Snapshot {snap_idx} differs from previous '
                        f'snapshot in {EPOCH_ID} index.')

    def mlm(data, formula):
        return mixedlm(
            formula=formula,
            data=data,
            groups=groups,
            re_formula=re_formula,
        ).fit()

    grid = pd.DataFrame({
        channel: group_by_time.apply(mlm, channel + ' ~ ' + RHS)
        for channel in tqdm(LHS, desc='Channels: ')
    })

    return FitGrid(grid)
