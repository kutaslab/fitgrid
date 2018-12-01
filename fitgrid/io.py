import pandas as pd
import pickle
from .epochs import Epochs
from .fitgrid import FitGrid
from .errors import FitGridError


def epochs_from_hdf(hdf_filename, key=None, channels='default'):
    """Construct Epochs object from an HDF5 file containing an epochs table.

    The HDF5 file should contain columns with names defined by EPOCH_ID and
    TIME either as index columns or as regular columns. This is added as a
    convenience, in general, input epochs tables should contain these columns
    in the index.

    Parameters
    ----------
    hdf_filename : str
        HDF5 file name
    key : str
        group identifier for the dataset when HDF5 file contains more than one
    channels : list of str, optional, defaults to CHANNELS
        list of string channel names

    Returns
    -------
    epochs : Epochs
        an Epochs object with the data
    """

    from . import EPOCH_ID, TIME

    df = pd.read_hdf(hdf_filename, key=key)

    # time and epoch id already present in index
    if EPOCH_ID in df.index.names and TIME in df.index.names:
        return Epochs(df, channels)

    # time and epoch id present in columns, set index
    if EPOCH_ID in df.columns and TIME in df.columns:
        df.set_index([EPOCH_ID, TIME], inplace=True)
        return Epochs(df, channels)

    raise FitGridError(
        f'Dataset has to contain {EPOCH_ID} and {TIME} as columns or indices.'
    )


def epochs_from_dataframe(dataframe, channels='default'):
    """Construct Epochs object from a Pandas DataFrame epochs table.

    The DataFrame should contain columns with names defined by EPOCH_ID and
    TIME as index columns.

    Parameters
    ----------
    dataframe : pandas DataFrame
        a pandas DataFrame object
    channels : list of str, optional, defaults to CHANNELS
        list of string channel names

    Returns
    -------
    epochs : Epochs
        an Epochs object with the data
    """
    return Epochs(dataframe, channels)


def load_grid(filename):
    """Load a FitGrid object from file (created by running grid.save).

    Parameters
    ----------
    filename : str
        indicates file to load from

    Returns
    -------
    grid : FitGrid
        loaded FitGrid object
    """

    with open(filename, 'rb') as file:
        _grid, _epoch_index = pickle.load(file)

    return FitGrid(_grid, _epoch_index)
