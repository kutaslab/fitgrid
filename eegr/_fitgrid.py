import numpy as np
import pandas as pd


class FitGrid:
    """Hold rERP fit and diagnostic data.

    Parameters
    ----------

    LHS : list of str
        list of response variables in same order as used for regression
    sample_index : pandas Index
        Index of a single epoch
    grid : NumPy array
        NumPy structured array of dtype bucket_dt with 2 dimensions

    Notes
    -----

    The idea is to wrap the numpy grid and provide convenient slicing
    along channels (LHS) and samples (epoch_index).
    """
    def __init__(self, grid, LHS, epoch_index):

        assert grid.ndim == 2
        assert len(LHS) == grid.shape[0]
        assert len(epoch_index) == grid.shape[1]

        self._grid = grid
        self.LHS = LHS
        self.channel_index = pd.Series(np.arange(len(LHS)), index=LHS)

        # this is never used, can't get sample indexing to work so far
        self.sample_index = pd.Series(np.arange(len(epoch_index)),
                                      index=epoch_index)

    # TODO get sample indexing to work
    def __getitem__(self, slicer):
        is_str = isinstance(slicer, str)
        is_list_of_str = (isinstance(slicer, list) and
                          all(isinstance(item, str) for item in slicer))
        if is_str or is_list_of_str:
            indexer = self.channel_index.loc[slicer]
            return self._grid[indexer]
        else:
            raise TypeError('Use channel names or lists'
                            'of channel names for indexing.')

    def info(self):
        channels = ', '.join(self.LHS)
        fit_data = ', '.join(self._grid.dtype['fit'].names)
        diagnostic_data = ', '.join(self._grid.dtype['diag'].names)

        message = ''
        message += f'Channels: {channels}\n'
        message += f'Fit data: {fit_data}\n'
        message += f'Diagnostic data: {diagnostic_data}'
        print(message)
