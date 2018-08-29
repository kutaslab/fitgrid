# package level constants, at the top to avoid circular imports
EPOCH_ID = 'Epoch_idx'
TIME = 'Time'
CHANNELS = [
    'lle', 'lhz', 'MiPf', 'LLPf', 'RLPf', 'LMPf', 'RMPf', 'LDFr',
    'RDFr', 'LLFr', 'RLFr', 'LMFr', 'RMFr', 'LMCe', 'RMCe', 'MiCe',
    'MiPa', 'LDCe', 'RDCe', 'LDPa', 'RDPa', 'LMOc', 'RMOc', 'LLTe',
    'RLTe', 'LLOc', 'RLOc', 'MiOc', 'A2', 'HEOG', 'rle', 'rhz'
]


from .tools import generate
from .io import epochs_from_hdf, epochs_from_dataframe


def __dir__():
    return [
        'generate',
        'epochs_from_hdf', 'epochs_from_dataframe',
        'EPOCH_ID', 'TIME', 'CHANNELS'
    ]
