from .fake_data import generate
from .io import epochs_from_hdf, epochs_from_dataframe

#: ``mkpy`` convention
EPOCH_ID = 'Epoch_idx'

#: ``mkpy`` convention
TIME = 'Time'

#: a reasonable default for Kutas Lab
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
