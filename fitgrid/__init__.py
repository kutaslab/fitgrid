from .fake_data import generate
from .io import epochs_from_hdf, epochs_from_dataframe

#: default epoch identifier
EPOCH_ID = 'Epoch_idx'

#: default time identifier
TIME = 'Time'

#: default channel list
CHANNELS = [
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
