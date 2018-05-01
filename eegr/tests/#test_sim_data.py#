# imports
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.formula.api import OLS
from patsy import dmatrix
from eegr import sim_data as sd
import pytest

def df_test_set(complexity):
    if complexity == 'simple':
        epoch_p = (1,2)
        time_p = (1,3,1)
        cat_p = [(1,1)]
        cont_p = [(1,1)]
        noise = (0,0)
        simple_df = sd.df_gen(epoch_p=epoch_p, time_p=time_p, 
                       cat_p=cat_p, cont_p=cont_p, noise_p=noise)
        return simple_df
    elif complexity == 'complex1':
        epoch_p = (1,5)
        time_p = (0,30,5)
        cat_p = [(1,1),(2,2),(3,3)]
        cont_p = [(1,3),(2,3)]
        noise = (2,3)
        cmplx1_df = sd.df_gen(epoch_p=epoch_p, time_p=time_p, 
                       cat_p=cat_p, cont_p=cont_p, noise_p=noise)
        return cmplx1_df
    elif complexity == 'complex2':
        epoch_p = (1,10)
        time_p = (0,100,10)
        cat_p = [(1,3),(2,5),(3,4)]
        cont_p = [(10,15),(20,30)]
        noise = (2.5,39)
        cmplx2_df = sd.df_gen(epoch_p=epoch_p, time_p=time_p, 
                       cat_p=cat_p, cont_p=cont_p, noise_p=noise)
        return cmplx2_df

def test_df_shape():
    simple_df = df_test_set('simple')
    assert simple_df.shape == (2, 4)
    cmplx1_df = df_test_set('complex1')
    assert cmplx1_df.shape == (72, 5)
    cmplx2_df = df_test_set('complex2')
    assert cmplx2_df.shape == (270, 5)
    
def test_df_dtype():
    simple_df = df_test_set('simple')
    assert type(simple_df.cat[0]) == str
    assert type(simple_df.cat[1]) == str
    cmplx1_df = df_test_set('complex1')
    assert type(cmplx1_df.cat[0]) == str
    assert type(cmplx1_df.cat[len(cmplx1_df)-1]) == str
    cmplx2_df = df_test_set('complex2')
    assert type(cmplx2_df.cat[0]) == str
    assert type(cmplx2_df.cat[len(cmplx1_df)-1]) == str

def test_empty_epoch00():
    with pytest.raises(Exception):
        epoch_p = (0,0)
        time_p = (0,0,1)
        cat_p = [(0,0)]
        cont_p = [(0,0)]
        noise = (0,0)
        df = sd.df_gen(epoch_p=epoch_p, time_p=time_p, 
                   cat_p=cat_p, cont_p=cont_p, noise_p=noise)
                   
def test_empty_time001():
    with pytest.raises(Exception):
        epoch_p = (0,1)
        time_p = (0,0,0)
        cat_p = [(0,0)]
        cont_p = [(0,0)]
        noise = (0,0)
        df = sd.df_gen(epoch_p=epoch_p, time_p=time_p, 
                   cat_p=cat_p, cont_p=cont_p, noise_p=noise)
