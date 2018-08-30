import pandas as pd
from .epochs import Epochs
from .errors import FitGridError
from . import EPOCH_ID, TIME


def epochs_from_hdf(hdf_filename):
    """Construct Epochs object from an HDF5 file containing an epochs table."""

    df = pd.read_hdf(hdf_filename)

    # time and epoch id already present in index
    if EPOCH_ID in df.index.names and TIME in df.index.names:
        return Epochs(df)

    # time and epoch id present in columns, set index
    if EPOCH_ID in df.columns and TIME in df.columns:
        df.set_index([EPOCH_ID, TIME], inplace=True)
        return Epochs(df)

    raise FitGridError(
        f'Dataset has to contain {EPOCH_ID} and {TIME} as columns'
        ' or indices.'
    )


def epochs_from_dataframe(dataframe):
    """Construct Epochs object from a Pandas DataFrame epochs table."""
    return Epochs(dataframe)
