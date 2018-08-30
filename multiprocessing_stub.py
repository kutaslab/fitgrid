import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from tqdm import tqdm_notebook as tqdm
import patsy
from multiprocessing import Pool
from random import shuffle

df = pd.read_hdf('../../arquant_raw_articles.h5')

input('Finished reading epochs, press enter to continue\n')

# name suggested by TPU
class Scout:
    
    def __init__(self):
        self.columns = set()
    
    def __getitem__(self, name):
        self.columns.add(name)
        return np.empty(0)
    
channels = ['lle','lhz', 'MiPf', 'LLPf', 'RLPf', 'LMPf', 'RMPf', 'LDFr', 
            'RDFr', 'LLFr', 'RLFr', 'LMFr', 'RMFr', 'LMCe', 'RMCe', 'MiCe',
            'MiPa', 'LDCe', 'RDCe', 'LDPa', 'RDPa', 'LMOc', 'RMOc', 'LLTe', 
            'RLTe', 'LLOc', 'RLOc', 'MiOc', 'A2', 'HEOG', 'rle', 'rhz']

LHS = channels

RHS = 'noun_cloze'

scout = Scout()

patsy.dmatrix(RHS, scout)

df = df.set_index('Time')

parcels = [
    (
        df[list(scout.columns | set([channel]))].copy(),
        channel + ' ~ ' + RHS
    )
    for channel in LHS
]

input('Finished preparing data for regression, press enter to continue\n')

def regression(data, formula):
    return ols(formula, data).fit()

def processor(parcel):
    data, formula = parcel
    print(f'Processing parcel {formula}')
    return data.groupby('Time').apply(regression, formula)

#results = list(map(processor, parcels))
with Pool(processes=4) as pool:
    results = pool.map(processor, parcels)
