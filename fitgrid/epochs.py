import numpy as np
import pandas as pd
from statsmodels.formula.api import ols, mixedlm
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
        self._snapshots = snapshots
        self._epoch_index = tools.get_first_group(snapshots).index.copy()

    def _validate_LHS(self, LHS):

        # must be a list of strings
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

    def _validate_RHS(self, RHS):

        # validate RHS
        if RHS is None:
            raise FitGridError('Specify the RHS argument.')
        if not isinstance(RHS, str):
            raise FitGridError('RHS has to be a string.')

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

        from . import EPOCH_ID, TIME

        table = self.table.reset_index().set_index([EPOCH_ID, TIME])[
            self.channels
        ]

        n_channels = len(table.columns)
        n_epochs = len(table.index.unique(level=EPOCH_ID))
        n_samples = len(table.index.unique(level=TIME))

        assert table.values.size == n_channels * n_epochs * n_samples
        values = table.values.reshape(n_epochs, n_samples, n_channels)

        mean = values.mean(axis=0)
        diff = values - mean

        def l2_norm(data, axis=1):
            return np.sqrt((data * data).sum(axis=axis))

        # first n_samples is axis 1, then n_channels, leaving epochs
        distances_arr = l2_norm(l2_norm(diff))
        distances_arr_scaled = distances_arr / distances_arr.max()
        distances = pd.Series(distances_arr_scaled, index=self._epoch_index)

        return distances

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
            channel: self._snapshots.apply(function, channel=channel)
            for channel in tqdm(channels, desc='Channels: ')
        }

        return FitGrid(pd.DataFrame(results), self._epoch_index)

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

        self._validate_LHS(LHS)
        self._validate_RHS(RHS)

        def regression(data, channel):
            formula = channel + ' ~ ' + RHS
            return ols(formula, data).fit()

        return self.run_model(regression, LHS)

    def mlm(
        self, LHS=None, RHS=None, re_formula=None, vc_formula=None, groups=None
    ):
        """Run a linear mixed effects model on the epochs.

        Parameters
        ----------
        LHS : list of str, optional, defaults to all channels
            list of channels for the left hand side of the formula
        RHS : str
            fixed effects part of the mixed effect model formula
        re_formula : str
            A one-sided formula defining the variance structure of the model.
            The default gives a random intercept for each group.
        vc_formula : dict
            Formulas describing variance components.  ``vc_formula[vc]`` is the
            formula for the component with variance parameter named vc. The
            formula is processed into a matrix, and the columns of this matrix
            are linearly combined with independent random coefficients having
            mean zero and a common variance.
        groups : str
            name of the grouper column

        Returns
        -------
        grid : FitGrid
            FitGrid object containing results of the mixed effect model fitting

        Notes
        -----
        Note that ``statsmodels`` (and thus ``fitgrid``) does not support
        arbitrary crossed models. For more information see `Statsmodels
        documentation on mixed models
        <http://www.statsmodels.org/stable/mixed_linear.html>`_.

        """

        if LHS is None:
            LHS = self.channels

        self._validate_LHS(LHS)
        self._validate_RHS(RHS)

        def mixed_linear_model(data, channel):
            formula = channel + ' ~ ' + RHS
            return mixedlm(
                formula,
                data,
                re_formula=re_formula,
                vc_formula=vc_formula,
                groups=groups,
            )

        return self.run_model(mixed_linear_model, LHS)

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

        data = self._snapshots.mean()
        plots.stripchart(data[channels], negative_up=negative_up)
