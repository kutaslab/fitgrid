import numpy as np
import pandas as pd
from functools import lru_cache
import matplotlib.pyplot as plt
import seaborn as sns

from .errors import FitGridError
from . import plots


def expand_series_or_df(temp):

    columns = (
        pd.concat(temp[channel].tolist(), keys=temp.index)
        for channel in temp
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
            raise NotImplementedError('Cannot use elements with dim > 2,'
                                      f'element has ndim = {tester.ndim}.')

    # catchall for all types we don't handle explicitly
    # statsmodels objects, dicts, methods all go here
    return FitGrid(temp)


class FitGrid:
    """Hold rERP fit objects.

    Parameters
    ----------

    grid : Pandas DataFrame
        Pandas DataFrame of fit objects
    """

    def __init__(self, grid):

        self.grid = grid
        self.tester = grid.iloc[0, 0]
        # TODO the grid should be aware of the betas names
        # TODO the grid should be aware of the channel names

    def __getitem__(self, slicer):

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
        subgrid = self.grid.loc[time, channels]
        return self.__class__(subgrid)

    @lru_cache()
    def __getattr__(self, name):

        if not hasattr(self.tester, name):
            raise FitGridError(f'No such attribute: {name}.')

        temp = self.grid.applymap(lambda x: getattr(x, name))
        return _expand(temp)

    def __call__(self, *args, **kwargs):

        # if we are not callable, we'll get an appropriate exception
        temp = self.grid.applymap(lambda x: x(*args, **kwargs))
        return _expand(temp)

    def __dir__(self):

        return dir(self.tester)

    def __repr__(self):

        samples, chans = self.grid.shape
        return f'{samples} by {chans} FitGrid of type {type(self.tester)}.'

    def plot_betas(self, channel=None, beta_name=None):

        if (channel and beta_name) or (beta_name is None and channel is None):
            raise NotImplementedError('Pass either channel or beta name.')

        if channel:
            plots.stripchart(self.params[channel])

        if beta_name:
            assert beta_name in self.betas_names
            data = (self.params.swaplevel(axis=1)
                               .sort_index(axis=1, level=0)
                               .reindex(axis=1,
                                        level=1,
                                        labels=self.grid.columns))
            plots.stripchart(data[beta_name])

    def plot_adj_rsquared(self, by=None):

        if by is None:
            plt.figure(figsize=(16, 8))
            sns.heatmap(self.rsquared_adj.T)
        elif by == 'channels':
            self.rsquared_adj.mean(axis=0).plot(kind='bar', figsize=(16, 8))
        elif by == 'time':
            self.rsquared_adj.mean(axis=1).plot(figsize=(16, 8))

    def influential_epochs(self, top=20, within_channel=None):

        influence = self.get_influence()

        if within_channel is not None:
            a = influence.cooks_distance[within_channel].drop(axis=0,
                                                              index=1,
                                                              level=1)
            a.index = a.index.droplevel(1)
            result = (a.mean(axis=0)
                       .sort_values(ascending=False)
                       .to_frame(name='average_Cooks_D'))
        else:
            a = influence.cooks_distance.drop(axis=0, index=1, level=1)
            a.index = a.index.droplevel(1)
            result = (a.mean(axis=1, level=1)
                       .mean(axis=0)
                       .sort_values(ascending=False)
                       .to_frame(name='average_Cooks_D'))

        return result.iloc[:top]

    def plot_averaged_studentized_residuals(self, within_channel=None):
        """
        Parameters
        ----------

        within : str
            channel name
        """
        influence = self.get_influence()

        if within_channel is not None:
            ar = (influence.resid_studentized_internal[within_channel]
                           .mean(axis=0)
                           .to_frame())
        else:
            ar = (influence.resid_studentized_internal.mean(axis=1, level=1)
                                                      .mean(axis=0)
                                                      .to_frame())
        plt.figure(figsize=(16, 8))
        sns.distplot(ar, bins=10)

    def plot_averaged_absolute_standardized_residuals(self,
                                                      within_channel=None):
        """
        Parameters
        ----------

        across : str
            possible values are `epochs`, `time`, `channels`
        """

        influence = self.get_influence()

        if within_channel is not None:
            ar = (influence.resid_studentized_internal[within_channel]
                           .abs()
                           .mean(axis=0)
                           .to_frame())
        else:
            ar = (influence.resid_studentized_internal
                           .abs().mean(axis=1, level=1)
                           .mean(axis=0)
                           .to_frame())
        plt.figure(figsize=(16, 8))
        sns.distplot(ar, bins=10)
