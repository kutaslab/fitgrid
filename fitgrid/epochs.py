import pandas as pd
from statsmodels.formula.api import ols
from tqdm import tqdm

from .errors import FitGridError
from .fitgrid import FitGrid
from . import tools


class Epochs:
    """Container class used for storing epochs tables and exposing statsmodels.

    Parameters
    ----------
    epochs_table : pandas DataFrame
        long form dataframe containing epochs with equal indices

    Returns
    -------
    epochs : Epochs
        epochs object

    """

    def __init__(self, epochs_table):

        from . import EPOCH_ID, TIME

        if not isinstance(epochs_table, pd.DataFrame):
            raise FitGridError('epochs_table must be a Pandas DataFrame.')

        # these index columns are required for consistency checks
        assert (
            TIME in epochs_table.index.names
            and EPOCH_ID in epochs_table.index.names
        )

        # now need to only keep EPOCH_ID in index
        # this is done so that any series that we get from fits are indexed on
        # EPOCH_ID only
        levels_to_remove = set(epochs_table.index.names)
        levels_to_remove.discard(EPOCH_ID)

        # copy since we are about to modify
        self.table = epochs_table.copy()
        # remove all levels from index except EPOCH_ID
        self.table.reset_index(list(levels_to_remove), inplace=True)
        assert self.table.index.names == [EPOCH_ID]

        snapshots = self.table.groupby(TIME)

        # check that snapshots across epochs have equal index by transitivity
        prev_group = None
        for idx, cur_group in snapshots:
            if prev_group is not None:
                if not prev_group.index.equals(cur_group.index):
                    raise FitGridError(
                        f'Snapshot {idx} differs from '
                        f'previous snapshot in {EPOCH_ID} index:\n'
                        f'Current snapshot\'s indices:\n'
                        f'{cur_group.index}\n'
                        f'Previous snapshot\'s indices:\n'
                        f'{prev_group.index}'
                    )
            prev_group = cur_group

        if not prev_group.index.is_unique:
            raise FitGridError(
                f'Duplicate values in {EPOCH_ID} index not allowed:',
                tools.get_index_duplicates_table(self.table, EPOCH_ID),
            )

        self.table.reset_index(inplace=True)
        self.table.set_index([EPOCH_ID, TIME], inplace=True)
        assert self.table.index.names == [EPOCH_ID, TIME]

        self.snapshots = snapshots

    def lm(self, LHS='default', RHS=None):
        """Run ordinary least squares linear regression on the epochs.

        Parameters
        ----------
        LHS : list of str, optional
            list of channels for the left hand side of the regression formula
        RHS : str
            right hand side of the regression formula

        Returns
        -------

        grid : FitGrid
            FitGrid object containing results of the regression

        """

        if LHS == 'default':
            from . import CHANNELS

            LHS = CHANNELS

        # validate LHS
        if not (
            isinstance(LHS, list)
            and all(isinstance(item, str) for item in LHS)
        ):
            raise FitGridError('LHS must be a list of strings.')

        assert set(LHS).issubset(set(self.table.columns))

        # validate RHS
        if RHS is None:
            raise FitGridError('Specify the RHS argument.')
        if not isinstance(RHS, str):
            raise FitGridError('RHS has to be a string.')

        def regression(data, formula):
            return ols(formula, data).fit()

        results = {
            channel: self.snapshots.apply(
                regression, formula=channel + ' ~ ' + RHS
            )
            for channel in tqdm(LHS, desc='Channels: ')
        }

        return FitGrid(pd.DataFrame(results))

    def plot_averages(self, channels=None, negative_up=True):
        """Plot grand mean averages for each channel, negative up by default.
        
        Parameters
        ----------
        channels : list of str, optional, default CHANNELS
            list of channel names to plot the averages
        negative_up : bool, optional, default True
            by convention, ERPs are plotted negative voltage up
        """

        from . import CHANNELS

        if channels is None:
            if set(CHANNELS).issubset(set(self.table.columns)):
                channels = CHANNELS
            else:
                raise FitGridError(
                    f'Default channels {CHANNELS} missing in epochs table,'
                    ' please pass list of channels or set CHANNELS.'
                )

        from . import plots

        data = self.snapshots.mean()
        plots.stripchart(data[channels], negative_up=negative_up)
