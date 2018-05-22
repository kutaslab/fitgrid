from eegr import fit_bucket
from eegr import sim_data as sd
from random import randint
import numpy as np 

def fit_grid_gen():
    rand = randint(1,9)
    epoch_p = (0,6)
    time_p = (0,rand,1)
    cat_p = [(0,rand)]
    cont_p = [(0,1)]
    noise = (0,0)
    df = sd.df_gen(epoch_p=epoch_p, time_p=time_p, 
                   cat_p=cat_p, cont_p=cont_p, noise_p=noise)
    LHS = ["data"]
    RHS = "cat"
    fit_grids = fit_bucket.make_fit_table(df,LHS,RHS)
    
    return rand, df, fit_grids

def test_coef_shape():
    rand, df, fit_grids = fit_grid_gen()
    assert fit_grids[0]['fit']['coef'].shape[0] == rand


def test_ci():
    rand, epochs, fit_grids = fit_grid_gen()
    
    times = epochs.index.levels[epochs.index.names.index('Time')].tolist()
    values = (np.tile(times,1) + 0) * rand
    for i, val in enumerate(values): 
        assert(fit_grids[0]['fit']['ci'][i][0][0] == [val,val]).all()
