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
    channels : list of str
        list of channel names to serve as dependent variables

    Returns
    -------
    epochs : Epochs
        epochs object

    """

    def __init__(self, epochs_table, channels='default'):

        from . import EPOCH_ID, TIME, CHANNELS

        if channels == 'default':
            channels = CHANNELS

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
        for item in (EPOCH_ID, TIME):
            if item not in epochs_table.index.names:
                raise FitGridError(
                    f'{item} must be a column in the epochs table index.'
                )

        # make our own copy so we are immune to modification to original table
        table = epochs_table.copy().reset_index().set_index(EPOCH_ID)
        assert table.index.names == [EPOCH_ID]

        snapshots = table.groupby(TIME)

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
                tools.get_index_duplicates_table(table, EPOCH_ID),
            )

        # checks passed, set instance variables
        self.channels = channels
        self.table = table
        self.snapshots = snapshots
        self.epoch_index = snapshots.get_group(0).index.copy()

    def run_model(self, function, channels):
        """Run an arbitrary model on the epochs.

        Parameters
        ----------
        function : Python function
            function that runs a model, see Notes below for details
        channels : list of str
            list of channels to serve as dependent variables

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

        results = {
            channel: self.snapshots.apply(function, channel=channel)
            for channel in tqdm(channels, desc='Channels: ')
        }

        return FitGrid(pd.DataFrame(results), self.epoch_index)

    def lm(self, LHS=None, RHS=None):
        """Run ordinary least squares linear regression on the epochs.

        Parameters
        ----------
        LHS : list of str, optional, defaults to all channels
            list of channels for the left hand side of the regression formula
        RHS : str
            right hand side of the regression formula

        Returns
        -------

        grid : FitGrid
            FitGrid object containing results of the regression

        """

        if LHS is None:
            LHS = self.channels

        # validate LHS
        if not (
            isinstance(LHS, list)
            and all(isinstance(item, str) for item in LHS)
        ):
            raise FitGridError('LHS must be a list of strings.')

        # all LHS items must be present in the epochs_table
        missing = set(LHS) - set(self.table.columns)
        if missing:
            raise FitGridError(
                'Items in LHS should all be present in the epochs table, '
                f'the following are missing: {missing}'
            )

        # validate RHS
        if RHS is None:
            raise FitGridError('Specify the RHS argument.')
        if not isinstance(RHS, str):
            raise FitGridError('RHS has to be a string.')

        def regression(data, channel):
            formula = channel + ' ~ ' + RHS
            return ols(formula, data).fit()

        return self.run_model(regression, LHS)

    def plot_averages(self, channels=None, negative_up=True):
        """Plot grand mean averages for each channel, negative up by default.

        Parameters
        ----------
        channels : list of str, optional, default CHANNELS
            list of channel names to plot the averages
        negative_up : bool, optional, default True
            by convention, ERPs are plotted negative voltage up
        """

        if channels is None:
            channels = self.channels

        from . import plots

        data = self.snapshots.mean()
        plots.stripchart(data[channels], negative_up=negative_up)
