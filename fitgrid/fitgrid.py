import numpy as np
import pandas as pd
from functools import lru_cache

from .errors import FitGridError


def add_epoch_index(temp, epoch_index):
    """We assume that temp is in long form, the columns are channels, and the
    first index level is TIME."""

    from . import TIME

    # first index level must be TIME
    assert temp.index.names[0] == TIME

    # temp should be long form, columns have single level (channels hopefully)
    assert not isinstance(temp.columns, pd.core.index.MultiIndex)

    # we can only handle 2- or 3-dimensional
    assert temp.index.nlevels in (2, 3)

    for i in range(1, temp.index.nlevels):
        level = temp.index.levels[i]
        # if a level looks like it was automatically created by Pandas,
        # we replace it with the epoch_index
        if (
            isinstance(level, pd.RangeIndex)
            and len(level) == len(epoch_index)
            and level._start == 0
            and level._step == 1
            and level._stop == len(epoch_index)
        ):
            temp.index.set_levels(epoch_index, level=i, inplace=True)
            temp.index.rename(epoch_index.name, level=i, inplace=True)

    return temp


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


def _expand(temp, epoch_index):
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
        elif array_form.ndim == 2:
            temp = temp.applymap(lambda x: pd.DataFrame(np.array(x)))
        else:
            raise NotImplementedError(
                'Cannot use elements with dim > 2,'
                f'element has ndim = {array_form.ndim}.'
            )
        temp_expanded = expand_series_or_df(temp)
        temp_epoch_index = add_epoch_index(temp_expanded, epoch_index)
        return temp_epoch_index

    # array, try converting to Series/DataFrame
    if isinstance(tester, np.ndarray):
        if tester.ndim == 1:
            temp = temp.applymap(lambda x: pd.Series(x))
        elif tester.ndim == 2:
            temp = temp.applymap(lambda x: pd.DataFrame(x))
        else:
            raise NotImplementedError(
                'Cannot use elements with dim > 2,'
                f'element has ndim = {tester.ndim}.'
            )
        temp_expanded = expand_series_or_df(temp)
        temp_epoch_index = add_epoch_index(temp_expanded, epoch_index)
        return temp_epoch_index

    # catchall for all types we don't handle explicitly
    # statsmodels objects, dicts, methods all go here
    return FitGrid(temp, epoch_index)


class FitGrid:
    """Hold rERP fit objects.

    FitGrid should not be created directly, the right way to build it is to
    start with an Epochs object and call a method like `.lm`.

    Parameters
    ----------
    _grid : Pandas DataFrame
        Pandas DataFrame of fit objects
    epochs_index : pandas Index
        index containing epoch ids

    Returns
    -------

    grid: FitGrid
        FitGrid object
    """

    def __init__(self, _grid, epoch_index):

        self._grid = _grid
        self._epoch_index = epoch_index
        self.tester = _grid.iloc[0, 0]
        self.channels = list(_grid.columns)

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

        if (
            not isinstance(slicer, tuple)
            or not hasattr(slicer, '__len__')
            or len(slicer) != 2
        ):
            raise FitGridError('Must slice on time and channels.')

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
        subgrid = self._grid.loc[time, channels].copy()
        return self.__class__(subgrid, self._epoch_index)

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
        return _expand(temp, self._epoch_index)

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
        return _expand(temp, self._epoch_index)

    def __dir__(self):

        # this exposes the fit object method but hides dunder methods, since
        # these likely overlap with the grid's own dunder methods
        return [item for item in dir(self.tester) if not item.startswith('__')]

    def __repr__(self):

        samples, chans = self._grid.shape
        return f'{samples} by {chans} FitGrid of type {type(self.tester)}.'

    def plot_betas(self):
        """Plot betas of the model, one plot per channel, overplotting betas.
        """

        if not hasattr(self.tester, 'params'):
            raise FitGridError('This FitGrid does not contain fit results.')

        import matplotlib.pyplot as plt

        with plt.rc_context({'font.size': 14}):
            params = self.params.unstack()
            channels, betas = params.columns.levels
            figsize = (16, 4 * len(channels))

            fig, axes = plt.subplots(
                nrows=len(channels), figsize=figsize, sharey=True
            )

            for ax, chan in zip(axes, channels):
                for beta in params[chan]:
                    ax.plot(params[chan][beta], label=beta)
                ax.set(ylabel=chan, xlabel=params.index.name)
                ax.legend(
                    loc='upper center',
                    ncol=len(params[chan]),
                    bbox_to_anchor=(0.5, 1.2),
                    fancybox=True,
                    shadow=False,
                )

            fig.tight_layout()

    def plot_adj_rsquared(self):
        """Plot adjusted :math:`R^2` as a heatmap with marginal bar and line.
        """

        import matplotlib.pyplot as plt

        if not hasattr(self.tester, 'rsquared_adj'):
            raise FitGridError('This FitGrid does not contain fit results.')

        rsq_adj = self.rsquared_adj

        with plt.rc_context({'font.size': 14}):
            plt.figure(figsize=(16, 12))
            gs = plt.GridSpec(2, 2, width_ratios=[13, 3], height_ratios=[7, 2])

            bar = plt.subplot(gs[1])
            bar.barh(self._grid.columns, rsq_adj.mean(axis=0))

            heatmap = plt.subplot(gs[0], sharey=bar)
            heatmap.imshow(rsq_adj.T, aspect='auto')
            heatmap.get_xaxis().set_visible(False)

            line = plt.subplot(gs[2])
            line.plot(rsq_adj.mean(axis=1))

            plt.tight_layout()
            plt.margins(x=0)

    def influential_epochs(self, top=None):
        """Return dataframe with top influential epochs ranked by Cook's-D.

        Parameters
        ----------
        top : int, optional, default None
            how many top epochs to return, all epochs by default

        Returns
        -------
        top_epochs : pandas DataFrame
            dataframe with EPOCH_ID as index and aggregated Cook's-D as values

        Notes
        -----
        Cook's distance is aggregated by simple averaging across time and
        channels.
        """

        if not hasattr(self.tester, 'get_influence'):
            raise FitGridError('This FitGrid does not contain influence data.')

        return (
            self.get_influence()
            .cooks_distance.drop(axis=0, index=1, level=1)
            .reset_index(level=1, drop=True)
            .mean(axis=0, level=1)
            .mean(axis=1)
            .sort_values(ascending=False)
            .to_frame(name='average_Cooks_D')
            .iloc[:top]
        )
