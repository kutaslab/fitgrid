import pandas as pd
import pickle
import statsmodels
from statsmodels.regression.linear_model import (
    RegressionResults,
    RegressionResultsWrapper,
)
from .epochs import Epochs
from .fitgrid import FitGrid, LMFitGrid, LMERFitGrid
from .errors import FitGridError
from . import defaults


def epochs_from_hdf(filename, key, time, epoch_id, channels):
    """Construct Epochs object from an HDF5 file containing an epochs table.

    The HDF5 file should contain columns with names defined by `epoch_id` and
    `time` either as index columns or as regular columns. This is added as a
    convenience, in general, input epochs tables should contain these columns
    in the index.

    Parameters
    ----------
    filename : str
        HDF5 file name
    key : str
        group identifier for the dataset when HDF5 file contains more than one
    time : str
        time column name
    epoch_id : str
        epoch identifier column name
    channels : list of str
        list of string channel names

    Returns
    -------
    epochs : Epochs
        an Epochs object with the data
    """

    df = pd.read_hdf(filename, key=key)

    # time and epoch id already present in index
    if epoch_id in df.index.names and time in df.index.names:
        return Epochs(df, time=time, epoch_id=epoch_id, channels=channels)

    # time and epoch id present in columns, set index
    if epoch_id in df.columns and time in df.columns:
        df.set_index([epoch_id, time], inplace=True)
        return Epochs(df, time=time, epoch_id=epoch_id, channels=channels)

    raise FitGridError(
        f'Dataset has to contain {epoch_id} and {time} as columns or indices.'
    )


def epochs_from_dataframe(dataframe, time, epoch_id, channels):
    """Construct Epochs object from a Pandas DataFrame epochs table.

    The DataFrame should contain columns with names defined by epoch_id and
    time as index columns.

    Parameters
    ----------
    dataframe : pandas DataFrame
        a pandas DataFrame object
    time : str
        time column name
    epoch_id : str
        epoch identifier column name
    channels : list of str
        list of string channel names

    Returns
    -------
    epochs : Epochs
        an Epochs object with the data
    """
    return Epochs(dataframe, time=time, epoch_id=epoch_id, channels=channels)


def epochs_from_feather(filename, time, epoch_id, channels):
    """Construct Epochs object from a Feather file containing an epochs table.

    The file should contain columns with names defined by epoch_id and time.

    Parameters
    ----------
    filename : str
        Feather file name
    time : str
        time column name
    epoch_id : str
        epoch identifier column name
    channels : list of str
        list of string channel names

    Returns
    -------
    epochs : Epochs
        an Epochs object with the data
    """

    df = pd.read_feather(filename)

    # time and epoch id present in columns, set index
    if epoch_id in df.columns and time in df.columns:
        df.set_index([epoch_id, time], inplace=True)
        return Epochs(df, time=time, epoch_id=epoch_id, channels=channels)

    raise FitGridError(
        f'Dataset has to contain {epoch_id} and {time} as columns or indices.'
    )


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

    from pymer4 import Lmer

    with open(filename, 'rb') as file:
        _grid, epoch_index, time = pickle.load(file)

    tester = _grid.iloc[0, 0]

    if isinstance(tester, (RegressionResults, RegressionResultsWrapper)):
        return LMFitGrid(_grid, epoch_index, time)
    elif isinstance(tester, Lmer):
        return LMERFitGrid(_grid, epoch_index, time)
    else:
        return FitGrid(_grid, epoch_index, time)
