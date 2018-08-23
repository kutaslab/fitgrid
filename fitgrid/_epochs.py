import pandas as pd
from statsmodels.formula.api import ols
from tqdm import tqdm_notebook as tqdm

from . import EPOCH_ID, TIME
from ._errors import EegrError
from ._fitgrid import FitGrid

CHANNELS = [
    'lle', 'lhz', 'MiPf', 'LLPf', 'RLPf', 'LMPf', 'RMPf', 'LDFr',
    'RDFr', 'LLFr', 'RLFr', 'LMFr', 'RMFr', 'LMCe', 'RMCe', 'MiCe',
    'MiPa', 'LDCe', 'RDCe', 'LDPa', 'RDPa', 'LMOc', 'RMOc', 'LLTe',
    'RLTe', 'LLOc', 'RLOc', 'MiOc', 'A2', 'HEOG', 'rle', 'rhz'
]


def _check_group_indices(group_by, index_level):
    """Check groups have same index using transitivity."""

    return True, None


class Epochs:
    """Container class used for storing epochs tables and exposing statsmodels.

    Parameters
    ----------

    epochs_table : pandas DataFrame
    """

    def __init__(self, epochs_table):

        if not isinstance(epochs_table, pd.DataFrame):
            raise EegrError('epochs_table must be a Pandas DataFrame.')

        # these index columns are required for groupby's
        assert (TIME in epochs_table.index.names
                and EPOCH_ID in epochs_table.index.names)

        self.epochs_table = epochs_table
        snapshots = epochs_table.groupby(TIME)

        # check that snapshots across epochs have equal index by transitivity
        prev_group = None
        for idx, cur_group in snapshots:
            if prev_group is not None:
                prev_indices = prev_group.index.get_level_values(EPOCH_ID)
                cur_indices = cur_group.index.get_level_values(EPOCH_ID)
                if not prev_indices.equals(cur_indices):
                    raise EegrError(f'Snapshot {idx} differs from '
                                    f'previous snapshot in {EPOCH_ID} index:\n'
                                    f'Current snapshot\'s indices:\n'
                                    f'{prev_indices}\n'
                                    f'Previous snapshot\'s indices:\n'
                                    f'{cur_indices}')
            prev_group = cur_group

        # we're good, set instance variable
        self.snapshots = snapshots

    @classmethod
    def from_hdf_file(cls, hdf_filename):
        df = (pd.read_hdf(hdf_filename)
                .set_index([EPOCH_ID, TIME])
                .sort_index())
        return cls(df)

    def lm(self, LHS=CHANNELS, RHS=None):

        # validate LHS
        if not (isinstance(LHS, list) and
                all(isinstance(item, str) for item in LHS)):
            raise EegrError('LHS must be a list of strings.')

        # TODO check LHS subset of columns

        # validate RHS
        if RHS is None:
            raise EegrError('Specify the RHS argument.')
        if not isinstance(RHS, str):
            raise EegrError('RHS has to be a string.')

        def regression(data, formula):
            return ols(formula, data).fit()

        grid = pd.DataFrame({
            channel: self.snapshots.apply(regression, channel + ' ~ ' + RHS)
            for channel in tqdm(LHS, desc='Channels: ')
        })

        return FitGrid(grid)

    def mlm():
        pass

    def glm():
        pass
