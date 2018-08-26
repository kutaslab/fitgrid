import numpy as np
import pandas as pd
from functools import lru_cache

from .errors import FitGridError


def values_are_series(temp):

    columns = (
        pd.DataFrame(temp[channel].tolist(), index=temp.index)
        for channel in temp
    )
    # concatenate columns, channel names are top level columns indices
    return pd.concat(columns, axis=1, keys=temp.columns)


def values_are_dataframes(temp):

    # temp has DataFrames as values, need to unpack
    # build dataframe for each channel
    # we assume all Series have same index
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
            attr = np.array(attr)
            if attr.ndim == 1:
                tmp = self.grid.applymap(lambda x: pd.Series(getattr(x, name)))
                return values_are_series(tmp)
            elif attr.ndim == 2:
                tmp = self.grid.applymap(
                        lambda x: pd.DataFrame(np.array(getattr(x, name))))
                return values_are_dataframes(tmp)
            else:
                raise NotImplementedError('Cannot use elements with dim > 2,'
                                          f'element has ndim = {attr.ndim}.')

        if isinstance(attr, np.ndarray):
            if attr.ndim == 1:
                temp = self.grid.applymap(lambda x: getattr(x, name))
                return values_are_series(temp)
            elif attr.ndim == 2:
                temp = self.grid.applymap(lambda x: getattr(x, name))
                return values_are_dataframes(temp)
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
        if isinstance(attr, pd.Series):

            temp = self.grid.applymap(lambda x: getattr(x, name))
            return values_are_series(temp)

        # GRIDS
        if isinstance(attr, pd.DataFrame):

            temp = self.grid.applymap(lambda x: getattr(x, name))
            return values_are_dataframes(temp)

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
