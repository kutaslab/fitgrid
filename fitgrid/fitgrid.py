import numpy as np
import pandas as pd
import pickle
from functools import lru_cache
import warnings

from .errors import FitGridError
from . import tools


class FitGrid:
    """Hold rERP fit objects.

    FitGrid should not be created directly, the right way to build it is to
    start with an Epochs object and pass it to a function like `fitgrid.lm`.

    Parameters
    ----------
    _grid : Pandas DataFrame
        Pandas DataFrame of fit objects
    epochs_index : pandas Index
        index containing epoch ids
    time : str
        time column name

    Returns
    -------
    grid: FitGrid
        FitGrid object

    Notes
    -----
    Slicing FitGrids is a little different than slicing Pandas DataFrames. For
    instance, we require that the keys in a list used to slice a FitGrid on
    time or channels be unique. The following Pandas quirk is inherited by
    FitGrids: slicing using a list where some keys are present but some
    are missing from a grid succeeds silently and creates columns with the
    missing keys as names. For example, if you have a grid with columns

        'channel1', 'channel2'

    and you slice:

        grid[:, ['channel1', 'blah']]

    this returns a new grid with columns

        'channel1', 'blah'

    where column 'blah' consists of NaN's. Since version 21.0 of Pandas this
    throws a warning and should in future be replaced with a KeyError.
    """

    def __init__(self, _grid, epoch_index, time):

        # check no duplicate column names
        names = list(_grid.columns)
        deduped_names = tools.deduplicate_list(names)
        if deduped_names != names:
            raise FitGridError('Duplicate column names not allowed.')
        self._grid = _grid
        self.epoch_index = epoch_index
        self.time = time
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
            if isinstance(component, slice):
                return component
            elif isinstance(component, list):
                # deduplicate and warn if duplicates found
                deduped_component = tools.deduplicate_list(component)
                if deduped_component != component:
                    msg = f'Slicer {component} contained duplicates, '
                    msg += f'slicing instead on deduped: {deduped_component}.'
                    warnings.warn(UserWarning(msg))
                return deduped_component
            else:
                # wrap in list to always get a DataFrame in return on slicing
                # otherwise we might get a scalar or a pandas Series,
                # which can't be used to create a FitGrid object
                return [component]

        time = check_slicer_component(time)
        channels = check_slicer_component(channels)
        subgrid = self._grid.loc[time, channels].copy()
        return self.__class__(subgrid, self.epoch_index, self.time)

    @lru_cache()
    def __getattr__(self, name):
        """Broadcast attribute extraction in the grid.

        This is the method that gets called when an attribute is requested that
        FitGrid does not have. We then go and check if the tester has it, and
        broadcast if it does.
        """

        if not hasattr(self.tester, name):
            raise AttributeError(f'No such attribute: {name}.')

        temp = self._grid.applymap(lambda x: getattr(x, name))
        return self._expand(temp)

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
        return self._expand(temp)

    def __dir__(self):

        # the result of this call is what shows up for tab completion

        # so we add the cell attributes:
        cell_attrs = [
            item for item in dir(self.tester) if not item.startswith('__')
        ]

        # and the attributes of the grid itself:
        grid_attrs = [self.save.__name__]

        return cell_attrs + grid_attrs

    def __repr__(self):

        samples, chans = self._grid.shape
        classname = self.__class__.__name__
        return f'{samples} by {chans} {classname} of type {type(self.tester)}.'

    def save(self, filename):
        """Save FitGrid object to file (reload with ``fitgrid.load_grid``).

        Parameters
        ----------
        filename : str
            file name to use

        """

        with open(filename, 'wb') as file:
            kernel = self._grid, self.epoch_index, self.time
            pickle.dump(kernel, file, protocol=pickle.HIGHEST_PROTOCOL)

    def expand_series_or_df(self, temp):
        """Expand a DataFrame that has Series or DataFrames for values."""

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

    def _expand(self, temp):
        """Expand the values in the grid if possible, return frame or grid."""

        tester = temp.iloc[0, 0]

        # no processing needed
        if np.isscalar(tester):
            return temp

        # familiar types, expand them
        if isinstance(tester, pd.Series) or isinstance(tester, pd.DataFrame):
            # want single index level
            # can get more if original DataFrame had a multiindex
            # in Epochs we ensure that only epoch_id is in the index for
            # groupby
            if tester.index.nlevels > 1:
                raise NotImplementedError(
                    f'index should have one level, have {tester.index.nlevels}'
                    f' instead: {tester.index.names}'
                )
            return self.expand_series_or_df(temp)

        # array-like, try converting to array and then Series/DataFrame
        if isinstance(tester, tuple) or isinstance(tester, list):
            array_form = np.array(
                tester, dtype="O"
            )  # deprecated without dtype="O"

            if array_form.ndim == 1:
                # Tidy 1-D are Series. Untidy 1-D may be broadcastable
                # into a tidy DataFrame

                def series_fun(x):
                    return pd.Series(np.array(x))  # default

                if array_form.dtype != np.dtype("object"):
                    pd_fun = series_fun  # tidy
                else:
                    try:
                        # broadcastable to DataFrame?
                        def df_fun(x):
                            return pd.DataFrame(np.broadcast(*x)).T

                        pd_fun = df_fun
                        pd_fun(tester)
                    except Exception:
                        # oh well, fall back to Series
                        pd_fun = series_fun

                temp = temp.applymap(lambda x: pd_fun(x))

            elif array_form.ndim == 2:
                temp = temp.applymap(lambda x: pd.DataFrame(np.array(x)))
            else:
                raise NotImplementedError(
                    'Cannot use elements with dim > 2,'
                    f'element has ndim = {array_form.ndim}.'
                )
            temp_expanded = self.expand_series_or_df(temp)
            temp_epoch_index = self.add_epoch_index(temp_expanded)
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
            temp_expanded = self.expand_series_or_df(temp)
            temp_epoch_index = self.add_epoch_index(temp_expanded)
            return temp_epoch_index

        # catchall for all types we don't handle explicitly
        # statsmodels objects, dicts, methods all go here
        return FitGrid(temp, self.epoch_index, self.time)

    def add_epoch_index(self, temp):
        """We assume that temp is in long form, the columns are channels, and the
        first index level is time."""

        # first index level is time
        assert temp.index.names[0] == self.time

        # temp should be long form, columns have single level (channels
        # hopefully)
        assert not isinstance(temp.columns, pd.core.indexes.multi.MultiIndex)

        # we can only handle 2- or 3-dimensional
        assert temp.index.nlevels in (2, 3)

        for i in range(1, temp.index.nlevels):
            level = temp.index.levels[i]
            # if a level looks like it was automatically created by Pandas,
            # we replace it with the epoch_index
            if (
                isinstance(level, pd.RangeIndex)
                and len(level) == len(self.epoch_index)
                and level.start == 0
                and level.step == 1
                and level.stop == len(self.epoch_index)
            ):
                # inplace is deprecated pandas 1.2+
                # temp.index.set_levels(self.epoch_index, level=i, inplace=True)
                temp.index = temp.index.set_levels(self.epoch_index, level=i)
                temp.index.rename(self.epoch_index.name, level=i, inplace=True)

        return temp


class LMFitGrid(FitGrid):
    def __dir__(self):

        lmfitgrid_attrs = [
            self.plot_betas.__name__,
            self.plot_adj_rsquared.__name__,
            self.influential_epochs.__name__,
        ]
        return super().__dir__() + lmfitgrid_attrs

    def plot_betas(self, legend_on_bottom=False):
        """Plot betas of the model, one plot per channel, overplotting betas.

        Parameters
        ----------
        legend_on_bottom : bool, defaults to False
            set to True to plot single legend below all channel plots

        Returns
        -------
        fig : matplotlib.figure.Figure
            figure containing plots
        axes : numpy.ndarray of matplotlib.axes.Axes
            axes objects
        """

        import matplotlib.pyplot as plt

        with plt.rc_context({'font.size': 14}):
            params = self.params.unstack()
            channels, betas = params.columns.levels
            figsize = (16, 4 * len(channels))

            n_plots = len(channels) + 1 if legend_on_bottom else len(channels)

            fig, axes = plt.subplots(
                nrows=n_plots, figsize=figsize, sharey=True
            )

            # wrap in list if single axis to allow for zipping later
            if not isinstance(axes, np.ndarray):
                axes = [axes]

            if legend_on_bottom:
                legend_ax = axes[-1]
                channel_axes = axes[:-1]
            else:
                channel_axes = axes

            for ax, chan in zip(channel_axes, channels):
                for beta in params[chan]:
                    ax.plot(params[chan][beta], label=beta)
                ax.set(ylabel=chan, xlabel=params.index.name)
                if not legend_on_bottom:
                    ax.legend(
                        loc='upper center',
                        ncol=len(params[chan]),
                        bbox_to_anchor=(0.5, 1.2),
                        fancybox=True,
                        shadow=False,
                    )

            if legend_on_bottom:
                handles, labels = ax.get_legend_handles_labels()
                legend_ax.set_axis_off()
                legend_ax.legend(handles, labels, mode='expand', ncol=3)

            fig.tight_layout()

            return fig, axes

    def plot_adj_rsquared(self):
        """Plot adjusted :math:`R^2` as a heatmap with marginal bar and line.

        Returns
        -------
        fig : matplotlib.figure.Figure
            figure containing plots
        gs : matplotlib.gridspec.GridSpec
            grid specification that determines locations and sizes of subplots
        bar, heatmap, colorbar, line : matplotlib.axes._subplots.AxesSubplot
            subplot objects
        """

        import matplotlib.pyplot as plt

        rsq_adj = self.rsquared_adj

        with plt.rc_context({'font.size': 14}):
            fig = plt.figure(figsize=(16, 12))
            gs = plt.GridSpec(2, 2, width_ratios=[13, 3], height_ratios=[7, 2])

            bar = plt.subplot(gs[1])
            bar.barh(self._grid.columns, rsq_adj.mean(axis=0))

            heatmap = plt.subplot(gs[0], sharey=bar)
            heatmap_image = heatmap.imshow(rsq_adj.T, aspect='auto')
            heatmap.get_xaxis().set_visible(False)

            colorbar = plt.subplot(gs[3])
            plt.colorbar(mappable=heatmap_image, cax=colorbar, aspect=0.75)

            line = plt.subplot(gs[2])
            line.plot(rsq_adj.mean(axis=1))

            plt.tight_layout()
            plt.margins(x=0)

        return fig, gs, bar, heatmap, colorbar, line

    def influential_epochs(self, top=None):
        """Return dataframe with top influential epochs ranked by Cook's-D.

        Parameters
        ----------
        top : int, optional, default None
            how many top epochs to return, all epochs by default

        Returns
        -------
        top_epochs : pandas DataFrame
            dataframe with epoch_id as index and aggregated Cook's-D as values

        Notes
        -----
        Cook's distance is aggregated by simple averaging across time and
        channels.
        """

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


class LMERFitGrid(FitGrid):
    def __or__(self, other):

        if not isinstance(other, self.__class__):
            raise FitGridError(
                'Can only compare LMERFitGrid to other LMERFitGrid.'
            )
        # figure out the situation
        this_fixef = self.tester.fixef.columns
        other_fixef = other.tester.fixef.columns
        same_fixef = this_fixef.equals(other_fixef)
        at_least_one_fit_with_REML = self.tester._REML or other.tester._REML

        if (not same_fixef) and at_least_one_fit_with_REML:
            raise FitGridError(
                'Cannot compare models with different fixed effects when '
                'REML is used. For more context, see '
                'https://stats.stackexchange.com/a/116796'
            )
        diff = self.AIC - other.AIC

        import matplotlib.pyplot as plt

        with plt.rc_context({'font.size': 14}):
            fig = plt.figure(figsize=(16, 12))
            gs = plt.GridSpec(1, 2, width_ratios=[15, 1])

            heatmap = plt.subplot(gs[0])
            heatmap_image = heatmap.imshow(diff.T, aspect='auto')

            colorbar = plt.subplot(gs[1])
            plt.colorbar(mappable=heatmap_image, cax=colorbar)

        return fig, gs, heatmap, colorbar
