import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from contextlib import redirect_stdout
from io import StringIO
from math import ceil

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

        # check no duplicate column names in index and regular columns
        names = list(epochs_table.index.names) + list(epochs_table.columns)
        deduped_names = tools.deduplicate_list(names)
        if deduped_names != names:
            raise FitGridError('Duplicate column names not allowed.')

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
            dupes = tools.get_index_duplicates_table(table, EPOCH_ID)
            raise FitGridError(
                f'Duplicate values in {EPOCH_ID} index not allowed:\n{dupes}'
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

    def process_key_and_group(self, key_and_group, function, channels):
        key, group = key_and_group
        results = {channel: function(group, channel) for channel in channels}
        return pd.Series(results, name=key)

    def run_model(self, function, channels=None, parallel=False, n_cores=4):
        """Run an arbitrary model on the epochs.

        Parameters
        ----------
        function : Python function
            function that runs a model, see Notes below for details
        channels : list of str
            list of channels to serve as dependent variables
        parallel : bool, defaults to False
            set to True in order to run in parallel
        n_cores : int, defaults to 4
            number of processes to run in parallel

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

        from . import TIME

        if channels is None:
            channels = self.channels

        self._validate_LHS(channels)

        gb = self.table.groupby(TIME)

        process_key_and_group = partial(
            self.process_key_and_group, function=function, channels=channels
        )

        if parallel:
            chunksize = ceil(len(gb) / n_cores)
            with tools.single_threaded(np):
                with Pool(n_cores) as pool:
                    results = pool.map(
                        process_key_and_group, tqdm(gb), chunksize=chunksize
                    )
        else:
            results = map(process_key_and_group, tqdm(gb))

        grid = pd.concat(results, axis=1).T
        grid.index.name = TIME
        return FitGrid(grid, self._epoch_index)

    def _lm(self, data, channel, RHS, eval_env):
        formula = channel + ' ~ ' + RHS
        return ols(formula, data, eval_env=eval_env).fit()

    def lm(self, LHS=None, RHS=None, parallel=False, n_cores=4, eval_env=4):
        """Run ordinary least squares linear regression on the epochs.

        Parameters
        ----------
        LHS : list of str, optional, defaults to all channels
            list of channels for the left hand side of the regression formula
        RHS : str
            right hand side of the regression formula
        parallel : bool, defaults to False
            change to True to run in parallel
        n_cores : int, defaults to 4
            number of processes to use for computation
        eval_env : int or patsy.EvalEnvironment, defaults to 4
            environment to use for evaluating patsy formulas, see patsy docs

        Returns
        -------

        grid : FitGrid
            FitGrid object containing results of the regression

        """

        if LHS is None:
            LHS = self.channels

        self._validate_LHS(LHS)
        self._validate_RHS(RHS)

        _lm = partial(self._lm, RHS=RHS, eval_env=eval_env)

        return self.run_model(
            _lm, channels=LHS, parallel=parallel, n_cores=n_cores
        )

    # family, factors, REML
    def _lmer(
        self,
        data,
        channel,
        RHS,
        family,
        conf_int,
        factors,
        permute,
        ordered,
        REML,
    ):
        from pymer4 import Lmer

        model = Lmer(channel + ' ~ ' + RHS, data=data, family=family)
        with redirect_stdout(StringIO()) as captured_stdout:
            model.fit(
                summarize=False,
                conf_int=conf_int,
                factors=factors,
                permute=permute,
                ordered=ordered,
                REML=REML,
            )

        # lmer prints warnings, capture them and attach to the model object
        warning = captured_stdout.getvalue()

        model.has_warning = True if warning else False
        model.warning = warning

        captured_stdout.close()

        del model.data
        del model.design_matrix
        del model.model_obj

        return model

    def lmer(
        self,
        LHS=None,
        RHS=None,
        family='gaussian',
        conf_int='Wald',
        factors=None,
        permute=None,
        ordered=False,
        REML=True,
        parallel=False,
        n_cores=4,
    ):
        """Fit lme4 linear mixed model by interfacing with R.

        Parameters
        ----------
        LHS : list of str, optional, defaults to all channels
            list of channels for the left hand side of the lmer formula
        RHS : str
            right hand side of the lmer formula
        family : str, defaults to 'gaussian'
            distribution link function to use
        conf_int : str, defaults to 'Wald'
        factors : dict, optional
            Keys should be column names in data to treat as factors. Values
            should either be a list containing unique variable levels if
            dummy-coding or polynomial coding is desired. Otherwise values
            should themselves be dictionaries with unique variable levels as
            keys and desired contrast values (as specified in R!) as keys.
        permute : int, defaults to None
            if non-zero, computes parameter significance tests by permuting
            test stastics rather than parametrically. Permutation is done by
            shuffling observations within clusters to respect random effects
            structure of data.
        ordered : bool, defaults to False
            whether factors should be treated as ordered polynomial contrasts;
            this will parameterize a model with K-1 orthogonal polynomial
            regressors beginning with a linear contrast based on the factor
            order provided
        REML : bool, defaults to True
            change to False to use ML estimation
        parallel : bool, defaults to False
            change to True to run in parallel
        n_cores : int, defaults to 4
            number of processes to use for computation

        Returns
        -------

        grid : FitGrid
            FitGrid object containing results of lmer fitting
        """

        if LHS is None:
            LHS = self.channels

        if RHS is None or not isinstance(RHS, str):
            raise ValueError('Please enter a valid lmer RHS as a string.')

        self._validate_LHS(LHS)

        lmer_runner = partial(
            self._lmer,
            RHS=RHS,
            family=family,
            conf_int=conf_int,
            factors=factors,
            permute=permute,
            ordered=ordered,
            REML=REML,
        )
        return self.run_model(
            lmer_runner, channels=LHS, parallel=parallel, n_cores=n_cores
        )

    def plot_averages(self, channels=None, negative_up=True):
        """Plot grand mean averages for each channel, negative up by default.

        Parameters
        ----------
        channels : list of str, optional, default CHANNELS
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
