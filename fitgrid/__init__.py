from .fake_data import generate
from .io import epochs_from_hdf, epochs_from_dataframe

EPOCH_ID = 'Epoch_idx'
TIME = 'Time'
CHANNELS = [  #: Sphinx autodoc
    'lle',
    'lhz',
    'MiPf',
    'LLPf',
    'RLPf',
    'LMPf',
    'RMPf',
    'LDFr',
    'RDFr',
    'LLFr',
    'RLFr',
    'LMFr',
    'RMFr',
    'LMCe',
    'RMCe',
    'MiCe',
    'MiPa',
    'LDCe',
    'RDCe',
    'LDPa',
    'RDPa',
    'LMOc',
    'RMOc',
    'LLTe',
    'RLTe',
    'LLOc',
    'RLOc',
    'MiOc',
    'A2',
    'HEOG',
    'rle',
    'rhz',
]


__all__ = [
    'generate',
    'epochs_from_hdf',
    'epochs_from_dataframe',
    'EPOCH_ID',
    'TIME',
    'CHANNELS',
]


def __dir__():
    return __all__
