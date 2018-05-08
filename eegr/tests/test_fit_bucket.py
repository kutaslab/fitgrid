from eegr import fit_bucket
from eegr import sim_data as sd
from random import randint

def fit_grid_gen():
    rand = randint(1,9)
    epoch_p = (0,6)
    time_p = (0,rand,1)
    cat_p = [(0,2)]
    cont_p = [(0,1)]
    noise = (0,0)
    df = sd.df_gen(epoch_p=epoch_p, time_p=time_p, 
                   cat_p=cat_p, cont_p=cont_p, noise_p=noise)
    LHS = ["data"]
    RHS = "cat"
    fit_grids = fit_bucket.make_fit_table(df,LHS,RHS)
    
    return rand, fit_grids

def test_coef_shape():
    rand, fit_grids = fit_grid_gen()
    assert fit_grids[0]['fit']['coef'].shape[0] == rand

def test_ci_val():
    rand, fit_grids = fit_grid_gen()
    assert(fit_grids[0]['fit']['ci'][0][0][0] == [0,0]).all()