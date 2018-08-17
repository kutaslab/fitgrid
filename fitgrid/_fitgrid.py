import numpy as np
import pandas as pd
from functools import lru_cache

from ._errors import EegrError


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
            raise EegrError(f'No such attribute: {name}.')

        # try with our tester
        attr = getattr(self.tester, name)

        # TODO I don't know how to handle multidimensional data, maybe reshape
        # and use MultiIndex
        if hasattr(attr, 'ndim') and attr.ndim > 1:
            raise NotImplementedError('Cannot use elements with dim > 1.')

        # scalars, easy
        if np.isscalar(attr):
            temp = self.grid.applymap(lambda x: getattr(x, name))
            return temp

        if isinstance(attr, pd.Series) or isinstance(attr, np.ndarray):
            if isinstance(attr, pd.Series) and attr.index.nlevels > 1:
                raise NotImplementedError('Series index should have one level')

            temp = self.grid.applymap(lambda x: getattr(x, name))
            # temp has Series as values, need to unpack
            # build dataframe for each channel, columns are Series indices
            # we assume all Series have same index
            dataframes = (
                pd.DataFrame(temp[channel].tolist(), index=temp.index)
                for channel in temp
            )

            # concatenate columns, channel names are top level columns indices
            return pd.concat(dataframes, axis=1, keys=temp.columns)

        # create a grid of callables, in case we are being called
        if callable(attr):
            temp = self.grid.applymap(lambda x: getattr(x, name))
            return self.__class__(temp)

        raise NotImplementedError(f'Type {type(attr)} not supported yet')

    def __call__(self, *args, **kwargs):

        # if we are not callable, we'll get an appropriate exception
        return self.__class__(self.grid.applymap(lambda x: x(*args, **kwargs)))

    def info(self):

        message = ''

        channels = ', '.join(self.grid.columns)
        message += f'Channels: {channels}\n'

        print(message)
