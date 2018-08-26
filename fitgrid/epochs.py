import pandas as pd
from statsmodels.formula.api import ols
from tqdm._tqdm_notebook import tqdm_notebook as tqdm

from . import EPOCH_ID, TIME
from .errors import FitGridError
from .fitgrid import FitGrid
from . import plots

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
            raise FitGridError('epochs_table must be a Pandas DataFrame.')

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
                    raise FitGridError(
                        f'Snapshot {idx} differs from '
                        f'previous snapshot in {EPOCH_ID} index:\n'
                        f'Current snapshot\'s indices:\n'
                        f'{prev_indices}\n'
                        f'Previous snapshot\'s indices:\n'
                        f'{cur_indices}'
                    )
            prev_group = cur_group

        if not prev_group.index.is_unique:
            raise FitGridError(
                f'Duplicate values in {EPOCH_ID} index not allowed:'
                f'\n{prev_group.index}'
            )

        # we're good, set instance variable
        self.snapshots = snapshots

    def lm(self, LHS=CHANNELS, RHS=None):

        # validate LHS
        if not (isinstance(LHS, list) and
                all(isinstance(item, str) for item in LHS)):
            raise FitGridError('LHS must be a list of strings.')

        assert set(LHS).issubset(set(self.epochs_table.columns))

        # validate RHS
        if RHS is None:
            raise FitGridError('Specify the RHS argument.')
        if not isinstance(RHS, str):
            raise FitGridError('RHS has to be a string.')

        def regression(data, formula):
            return ols(formula, data).fit()

        results = {}
        for channel in tqdm(LHS, desc='Overall: '):
            tqdm.pandas(desc=channel)
            results[channel] = self.snapshots.progress_apply(
                    regression,
                    formula=channel + ' ~ ' + RHS
            )
        grid = pd.DataFrame(results)

        return FitGrid(grid)

    def mlm():
        pass

    def glm():
        pass

    def plot_average(self, channels=None):
        if channels is None:
            if set(CHANNELS).issubset(set(self.epochs_table.columns)):
                channels = CHANNELS
            else:
                raise FitGridError('Default channels missing in epochs table,'
                                   ' please pass list of channels.')
        else:
            data = self.snapshots.mean()
            plots.stripchart(data[channels])
