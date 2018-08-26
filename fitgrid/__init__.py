# package level constants, at the top to avoid circular imports
EPOCH_ID = 'Epoch_idx'
TIME = 'Time'

from .tools import generate
from .io import epochs_from_hdf, epochs_from_dataframe


def __dir__():
    return [
        'generate',
        'epochs_from_hdf', 'epochs_from_dataframe',
        'EPOCH_ID', 'TIME',
    ]
