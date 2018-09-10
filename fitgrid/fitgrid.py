import numpy as np
import pandas as pd
from functools import lru_cache
import matplotlib.pyplot as plt
import seaborn as sns

from .errors import FitGridError
from . import plots


def expand_series_or_df(temp):
    """Expand a DataFrame that has Series or DataFrames for values."""

    columns = (
        pd.concat(temp[channel].tolist(), keys=temp.index) for channel in temp
    )
    # concatenate columns, channel names are top level columns indices
    result = pd.concat(columns, axis=1, keys=temp.columns)

    # stack to achieve long form if columns have multiple levels
    if isinstance(result.columns, pd.core.indexes.multi.MultiIndex):
        return result.stack()
    return result


def _expand(temp):
    """Expand the values in the grid if possible, return frame or grid."""

    tester = temp.iloc[0, 0]

    # no processing needed
    if np.isscalar(tester):
        return temp

    # familiar types, expand them
    if isinstance(tester, pd.Series) or isinstance(tester, pd.DataFrame):
        # want single index level
        # can get more if original DataFrame had a multiindex
        # in Epochs we ensure that only EPOCH_ID is in the index for groupby
        if tester.index.nlevels > 1:
            raise NotImplementedError(
                f'index should have one level, have {tester.index.nlevels} '
                f'instead: {tester.index.names}'
            )
        return expand_series_or_df(temp)

    # array-like, try converting to array and then Series/DataFrame
    if isinstance(tester, tuple) or isinstance(tester, list):
        array_form = np.array(tester)
        if array_form.ndim == 1:
            temp = temp.applymap(lambda x: pd.Series(np.array(x)))
            return expand_series_or_df(temp)
        elif array_form.ndim == 2:
            temp = temp.applymap(lambda x: pd.DataFrame(np.array(x)))
            return expand_series_or_df(temp)
        else:
            raise NotImplementedError(
                'Cannot use elements with dim > 2,'
                f'element has ndim = {array_form.ndim}.'
            )

    # array, try converting to Series/DataFrame
    if isinstance(tester, np.ndarray):
        if tester.ndim == 1:
            temp = temp.applymap(lambda x: pd.Series(x))
            return expand_series_or_df(temp)
        elif tester.ndim == 2:
            temp = temp.applymap(lambda x: pd.DataFrame(x))
            return expand_series_or_df(temp)
        else:
            raise NotImplementedError(
                'Cannot use elements with dim > 2,'
                f'element has ndim = {tester.ndim}.'
            )

    # catchall for all types we don't handle explicitly
    # statsmodels objects, dicts, methods all go here
    return FitGrid(temp)


class FitGrid:
    """Hold rERP fit objects.

    FitGrid should not be created directly, the right way to build it is to
    start with an Epochs object and call a method like `.lm`.

    Parameters
    ----------
    _grid : Pandas DataFrame
        Pandas DataFrame of fit objects

    Returns
    -------

    grid: FitGrid
        FitGrid object
    """

    def __init__(self, _grid):

        self._grid = _grid
        self.tester = _grid.iloc[0, 0]
        # TODO the grid should be aware of the betas names
        # TODO the grid should be aware of the channel names

    def __getitem__(self, slicer):
        """Slice grid on time and channels, return new grid with that shape.

        The intended way to slice a FitGrid is to always slice on both time and
        channels, allowing wildcard colons:

            `grid[:, ['channel1', 'channel2']]`

        or

            `grid[:, :]`

        or

            `grid[25:-25, 'channel2']
        """

        if isinstance(slicer, slice) or len(slicer) != 2:
            raise ValueError('Must slice on time and channels.')

        # now we can unpack
        time, channels = slicer

        def check_slicer_component(component):
            if isinstance(component, slice) or isinstance(component, list):
                return component
            else:
                # wrap in list to always get a DataFrame in return on slicing
                # otherwise we might get a scalar or a pandas Series,
                # which can't be used to create a FitGrid object
                return [component]

        time = check_slicer_component(time)
        channels = check_slicer_component(channels)
        subgrid = self._grid.loc[time, channels]
        return self.__class__(subgrid)

    @lru_cache()
    def __getattr__(self, name):
        """Broadcast attribute extraction in the grid.

        This is the method that gets called when an attribute is requested that
        FitGrid does not have. We then go and check if the tester has it, and
        broadcast if it does.
        """

        if not hasattr(self.tester, name):
            raise FitGridError(f'No such attribute: {name}.')

        temp = self._grid.applymap(lambda x: getattr(x, name))
        return _expand(temp)

    def __call__(self, *args, **kwargs):
        """Broadcast method calling in the grid.

        This is the method that gets called when the grid is called as if it
        were a method.
        """
        if not callable(self.tester):
            raise FitGridError(
                f'This grid is not callable, '
                f'current type is {type(self.tester)}'
            )

        # if we are not callable, we'll get an appropriate exception
        temp = self._grid.applymap(lambda x: x(*args, **kwargs))
        return _expand(temp)

    def __dir__(self):

        # this exposes the fit object method but hides dunder methods, since
        # these likely overlap with the grid's own dunder methods
        return [item for item in dir(self.tester) if not item.startswith('__')]

    def __repr__(self):

        samples, chans = self._grid.shape
        return f'{samples} by {chans} FitGrid of type {type(self.tester)}.'

    def plot_betas(self, channel=None, beta_name=None):
        """Plot betas of the model. Pass either beta or channel name.

        Parameters
        ----------
        channel : str
            channel for which to plot the betas, if none passed, use all
        beta_name : str
            name of the beta parameter which should be plotted
        """

        if not hasattr(self.tester, 'params'):
            raise FitGridError('This FitGrid does not contain fit results.')

        if (channel and beta_name) or (beta_name is None and channel is None):
            raise NotImplementedError('Pass either channel or beta name.')

        if channel:
            plots.stripchart(self.params[channel])

        if beta_name:
            assert beta_name in self.betas_names
            data = (
                self.params.swaplevel(axis=1)
                .sort_index(axis=1, level=0)
                .reindex(axis=1, level=1, labels=self._grid.columns)
            )
            plots.stripchart(data[beta_name])

    def plot_adj_rsquared(self, by=None):
        """Plot adjusted :math:`R^2` by channels or time.

        Parameters
        ----------
        by : str, value is 'channels' or 'time'
            optional string triggering plotting by channels or time

        Notes
        -----
        If by is not set, a heatmap with time on the x-axis and channels on the
        y-axis is show. If by='channels', a bar plot is shown. If by='time', a
        single timeseries line plot is shown.
        """

        if by is None:
            plt.figure(figsize=(16, 8))
            sns.heatmap(self.rsquared_adj.T)
        elif by == 'channels':
            self.rsquared_adj.mean(axis=0).plot(kind='bar', figsize=(16, 8))
        elif by == 'time':
            self.rsquared_adj.mean(axis=1).plot(figsize=(16, 8))

    def influential_epochs(self, top=20, within_channel=None):
        """Return dataframe with top influential epochs ranked by Cook's-D.

        Parameters
        ----------
        top : int
            how many top epochs to return
        within_channel : str
            name of channel to which to restrict search

        Returns
        -------
        top_epochs : pandas DataFrame
            dataframe with EPOCH_ID as index and aggregated Cook's-D as values

        Notes
        -----
        Cook's distance is aggregated by simple averaging within given scope.

        """

        influence = self.get_influence()

        if within_channel is not None:
            a = influence.cooks_distance[within_channel].drop(
                axis=0, index=1, level=1
            )
            a.index = a.index.droplevel(1)
            result = (
                a.mean(axis=0)
                .sort_values(ascending=False)
                .to_frame(name='average_Cooks_D')
            )
        else:
            a = influence.cooks_distance.drop(axis=0, index=1, level=1)
            a.index = a.index.droplevel(1)
            result = (
                a.mean(axis=1, level=1)
                .mean(axis=0)
                .sort_values(ascending=False)
                .to_frame(name='average_Cooks_D')
            )

        return result.iloc[:top]

    def plot_residuals(self, within_channel=None):
        """Plot averaged studentized residuals.

        Parameters
        ----------

        within_channel : str
            channel name
        """
        influence = self.get_influence()

        if within_channel is not None:
            ar = (
                influence.resid_studentized_internal[within_channel]
                .mean(axis=0)
                .to_frame()
            )
        else:
            ar = (
                influence.resid_studentized_internal.mean(axis=1, level=1)
                .mean(axis=0)
                .to_frame()
            )
        plt.figure(figsize=(16, 8))
        sns.distplot(ar, bins=10)

    def plot_absolute_residuals(self, within_channel=None):
        """Plot average absolute studentized residuals.

        Parameters
        ----------

        across : str
            possible values are `epochs`, `time`, `channels`
        """

        influence = self.get_influence()

        if within_channel is not None:
            ar = (
                influence.resid_studentized_internal[within_channel]
                .abs()
                .mean(axis=0)
                .to_frame()
            )
        else:
            ar = (
                influence.resid_studentized_internal.abs()
                .mean(axis=1, level=1)
                .mean(axis=0)
                .to_frame()
            )
        plt.figure(figsize=(16, 8))
        sns.distplot(ar, bins=10)
