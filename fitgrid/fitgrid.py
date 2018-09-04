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
    return pd.concat(columns, axis=1, keys=temp.columns)


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

        # We only get here if no attribute found, so user is either asking for
        # a grid attribute, or is mistaken.
        #
        # The strategy is as follows:
        #
        # First see whether our tester object has the attribute.
        # If yes, then we see what shape the attribute is. If no, raise error.
        #
        # If the attribute is a scalar, we're golden, simply apply elementwise.
        # If the attribute is a series/numpy array, we use multiindex.
        # If the attribute is multidimensional (covariances), I have no clue.
        # If the attribute is a method, e.g. get_influence, create a new
        # grid.
        #
        # So for now, implement scalars, series/np.ndarray, and objects, as
        # they cover most use cases.

        if not hasattr(self.tester, name):
            raise FitGridError(f'No such attribute: {name}.')

        # try with our tester
        attr = getattr(self.tester, name)

        #######################################################################
        #                               REJECT                                #
        #######################################################################

        # can't handle 3D and up
        if hasattr(attr, 'ndim') and attr.ndim > 2:
            raise NotImplementedError('Cannot use elements with dim > 2.')

        # want Series with single index level
        # can get more if original DataFrame had a multiindex
        if isinstance(attr, pd.Series) and attr.index.nlevels > 1:
            raise NotImplementedError('Series index should have one level')

        #######################################################################
        #                           CONVERSION                                #
        #######################################################################

        # TODO this section is really hacky and ugly
        if isinstance(attr, tuple) or isinstance(attr, list):
            if attr.ndim == 1:
                tmp = self.grid.applymap(
                    lambda x: pd.Series(np.array(getattr(x, name)))
                )
                return expand_series_or_df(tmp)
            elif attr.ndim == 2:
                tmp = self.grid.applymap(
                    lambda x: pd.DataFrame(np.array(getattr(x, name)))
                )
                return expand_series_or_df(tmp)
            else:
                raise NotImplementedError('Cannot use elements with dim > 2,'
                                          f'element has ndim = {attr.ndim}.')

        if isinstance(attr, np.ndarray):
            if attr.ndim == 1:
                temp = self.grid.applymap(
                    lambda x: pd.Series(getattr(x, name))
                )
                return expand_series_or_df(temp)
            elif attr.ndim == 2:
                temp = self.grid.applymap(
                    lambda x: pd.DataFrame(getattr(x, name))
                )
                return expand_series_or_df(temp)
            else:
                raise NotImplementedError('Cannot use elements with dim > 2,'
                                          f'element has ndim = {attr.ndim}.')

        #######################################################################
        #                           VALID TYPES                               #
        #######################################################################

        # NOTE we don't run apply unless we are certain we can handle the type,
        # otherwise running apply could be a waste of time if expensive but we

        # SCALARS
        if np.isscalar(attr):
            temp = self.grid.applymap(lambda x: getattr(x, name))
            return temp

        # VECTORS
        if isinstance(attr, pd.Series) or isinstance(attr, pd.DataFrame):
            temp = self.grid.applymap(lambda x: getattr(x, name))
            return expand_series_or_df(temp)

        # create a grid of callables, in case we are being called
        if callable(attr):
            temp = self.grid.applymap(lambda x: getattr(x, name))
            return self.__class__(temp)

        #######################################################################
        #                           FALL THROUGH                              #
        #######################################################################

        raise NotImplementedError(f'Type {type(attr)} not supported yet')

    def __call__(self, *args, **kwargs):

        # if we are not callable, we'll get an appropriate exception
        return self.__class__(self.grid.applymap(lambda x: x(*args, **kwargs)))

    def __dir__(self):

        return dir(self.tester)

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
