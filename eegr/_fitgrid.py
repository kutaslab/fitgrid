import numpy as np
import pandas as pd

from eegr._errors import EegrError


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
        is_slice = isinstance(slicer, slice)
        is_wildcard_slice = (slicer == slice(None, None, None))

        # single response variable
        if is_str:
            if slicer not in self.LHS:
                raise EegrError(f'{slicer} not in the list of response '
                                f'variables: {self.LHS}')
            else:
                indexer = self.channel_index.loc[slicer]
                return self._grid[indexer]

        # a list of response variables
        if is_list_of_str:
            asked_for = set(slicer)
            have = set(self.LHS)
            if not asked_for.issubset(have):
                missing = asked_for - have
                raise EegrError(f'The following requested response variables '
                                f'were not in the model: {missing}')
            else:
                indexer = self.channel_index.loc[slicer]
                return self._grid[indexer]

        if is_slice:
            if is_wildcard_slice:
                return self._grid
            else:
                raise EegrError('Only wildcard slicing is supported for colon '
                                'slicing')

        # the slicer is neither a string, list of strings nor a wildcard slice
        raise EegrError(f'Expected a channel name, list of channel names or '
                        f'a colon, got {slicer} of type {type(slicer)}.')

    def info(self):

        message = ''

        channels = ', '.join(self.LHS)
        message += f'Channels: {channels}\n'

        fit_data = ', '.join(self._grid.dtype['fit'].names)
        message += f'Fit data: {fit_data}\n'

        diagnostic_data = ', '.join(self._grid.dtype['diag'].names)
        message += f'Diagnostic data: {diagnostic_data}'

        print(message)
