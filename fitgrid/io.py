import pandas as pd
from .epochs import Epochs
from . import EPOCH_ID, TIME


def epochs_from_hdf(hdf_filename):
    """Construct Epochs object from an HDF5 file containing an epochs table."""
    df = (pd.read_hdf(hdf_filename)
            .set_index([EPOCH_ID, TIME])
            .sort_index())
    return Epochs(df)


def epochs_from_dataframe(dataframe):
    """Construct Epochs object from a Pandas DataFrame epochs table."""
    return Epochs(dataframe)
