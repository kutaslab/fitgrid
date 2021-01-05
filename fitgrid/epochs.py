import numpy as np
import pandas as pd

from .errors import FitGridError
from . import tools


class Epochs:
    """Container class used for storing epochs tables and exposing statsmodels.

    Parameters
    ----------
    epochs_table : pandas DataFrame
        long form dataframe containing epochs with equal indices
    time : str
        time column name
    epoch_id : str
        epoch identifier column name
    channels : list of str
        list of channel names to serve as dependent variables

    Returns
    -------
    epochs : Epochs
        epochs object

    """

    def __init__(self, epochs_table, time, epoch_id, channels):

        # channels must be a list of strings
        if not isinstance(channels, list) or not all(
            isinstance(item, str) for item in channels
        ):
            raise FitGridError('channels should be a list of strings.')

        # all channels must be present as epochs table columns
        missing_channels = set(channels) - set(epochs_table.columns)
        if missing_channels:
            raise FitGridError(
                'channels should all be present in the epochs table, '
                f'the following are missing: {missing_channels}'
            )

        if not isinstance(epochs_table, pd.DataFrame):
            raise FitGridError('epochs_table must be a Pandas DataFrame.')

        # these index columns are required for consistency checks
        for item in (epoch_id, time):
            if item not in epochs_table.index.names:
                raise FitGridError(
                    f'{item} must be a column in the epochs table index.'
                )

        # check no duplicate column names in index and regular columns
        names = list(epochs_table.index.names) + list(epochs_table.columns)
        deduped_names = tools.deduplicate_list(names)
        if deduped_names != names:
            raise FitGridError('Duplicate column names not allowed.')

        # make our own copy so we are immune to modification to original table
        table = (
            epochs_table.copy().reset_index().set_index(epoch_id).sort_index()
        )
        assert table.index.names == [epoch_id]

        snapshots = table.groupby(time)

        # check that snapshots across epochs have equal index by transitivity
        prev_group = None
        for idx, cur_group in snapshots:
            if prev_group is not None:
                if not prev_group.index.equals(cur_group.index):
                    raise FitGridError(
                        f'Snapshot {idx} differs from '
                        f'previous snapshot in {epoch_id} index:\n'
                        f'Current snapshot\'s indices:\n'
                        f'{cur_group.index}\n'
                        f'Previous snapshot\'s indices:\n'
                        f'{prev_group.index}'
                    )
            prev_group = cur_group

        if not prev_group.index.is_unique:
            dupes = tools.get_index_duplicates_table(table, epoch_id)
            raise FitGridError(
                f'Duplicate values in {epoch_id} index not allowed:\n{dupes}'
            )

        # checks passed, set instance variables
        self.time = time
        self.epoch_id = epoch_id
        self.channels = channels
        self.table = table
        self._snapshots = snapshots
        self.epoch_index = tools.get_first_group(snapshots).index.copy()
        self.time_index = pd.Index([time for time, _ in snapshots], name=time)

    def distances(self):
        """Return scaled Euclidean distances of epochs from the "mean" epoch.

        Returns
        -------
        distances : pandas Series or DataFrame
            Series or DataFrame with epoch distances

        Notes
        -----
        Distances are scaled by dividing by the max.
        """

        table = self.table.reset_index().set_index([self.epoch_id, self.time])[
            self.channels
        ]

        n_channels = len(table.columns)
        n_epochs = len(table.index.unique(level=self.epoch_id))
        n_samples = len(table.index.unique(level=self.time))

        assert table.values.size == n_channels * n_epochs * n_samples
        values = table.values.reshape(n_epochs, n_samples, n_channels)

        mean = values.mean(axis=0)
        diff = values - mean

        def l2_norm(data, axis=1):
            return np.sqrt((data * data).sum(axis=axis))

        # first n_samples is axis 1, then n_channels, leaving epochs
        distances_arr = l2_norm(l2_norm(diff))
        distances_arr_scaled = distances_arr / distances_arr.max()
        distances = pd.Series(distances_arr_scaled, index=self.epoch_index)

        return distances

    def plot_averages(self, channels=None, negative_up=True):
        """Plot grand mean averages for each channel, negative up by default.

        Parameters
        ----------
        channels : list of str, optional, defaults to all channels
            list of channel names to plot the averages
        negative_up : bool, optional, default True
            by convention, ERPs are plotted negative voltage up

        Returns
        -------
        fig : matplotlib.figure.Figure
            figure containing plots
        axes : numpy.ndarray of matplotlib.axes.Axes
            axes objects
        """

        if channels is None:
            channels = self.channels

        from . import plots

        data = self._snapshots.mean()
        fig, axes = plots.stripchart(data[channels], negative_up=negative_up)
        return fig, axes
