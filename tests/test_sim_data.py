# imports
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.formula.api import OLS
from patsy import dmatrix
from eegr import sim_data as sd
import pytest
from random import randint

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
    assert simple_df.shape == (2, 3)
    cmplx1_df = df_test_set('complex1')
    assert cmplx1_df.shape == (72, 4)
    cmplx2_df = df_test_set('complex2')
    assert cmplx2_df.shape == (270, 4)
    
def test_df_dtype():
    simple_df = df_test_set('simple')
    assert type(simple_df.cat.values[0]) == str
    assert type(simple_df.cat.values[1]) == str
    cmplx1_df = df_test_set('complex1')
    assert type(cmplx1_df.cat.values[0]) == str
    assert type(cmplx1_df.cat.values[len(cmplx1_df)-1]) == str
    cmplx2_df = df_test_set('complex2')
    assert type(cmplx2_df.cat.values[0]) == str
    assert type(cmplx2_df.cat.values[len(cmplx1_df)-1]) == str

def test_empty_epoch00():
    with pytest.raises(Exception):
        epoch_p = (0,0)
        time_p = (0,1,1)
        cat_p = [(0,0)]
        cont_p = [(0,0)]
        noise = (0,0)
        df = sd.df_gen(epoch_p=epoch_p, time_p=time_p, 
                   cat_p=cat_p, cont_p=cont_p, noise_p=noise)
                   
def test_empty_time001():
    with pytest.raises(Exception):
        epoch_p = (0,1)
        time_p = (0,1,0)
        cat_p = [(0,0)]
        cont_p = [(0,0)]
        noise = (0,0)
        df = sd.df_gen(epoch_p=epoch_p, time_p=time_p, 
                   cat_p=cat_p, cont_p=cont_p, noise_p=noise)

def test_time_same():
    with pytest.raises(Exception):
        epoch_p = (0,1)
        time_p = (0,0,0)
        cat_p = [(0,0)]
        cont_p = [(0,0)]
        noise = (0,0)
        df = sd.df_gen(epoch_p=epoch_p, time_p=time_p, 
                   cat_p=cat_p, cont_p=cont_p, noise_p=noise)

def test_time_stop_small():
    with pytest.raises(Exception):
        epoch_p = (0,1)
        time_p = (6,0,0)
        cat_p = [(0,0)]
        cont_p = [(0,0)]
        noise = (0,0)
        df = sd.df_gen(epoch_p=epoch_p, time_p=time_p, 
                   cat_p=cat_p, cont_p=cont_p, noise_p=noise)

# Nose to tail test        
# generate sample df for testing
def sample_df():
    rand = randint(1,4)
    epoch_p = (rand,rand * rand+1)
    time_p = (rand,2*rand,1)
    cat_p = [(randint(1,4),rand)]
    cont_p = [(rand,rand+5)]
    noise = (0,0)
    df = sd.df_gen(epoch_p=epoch_p, time_p=time_p, 
                   cat_p=cat_p, cont_p=cont_p, noise_p=noise)
    return epoch_p, time_p, cat_p, cont_p, df

# test epoch idx
def test_epoch_idx():
    epoch_p, time_p, cat_p, cont_p, epochs = sample_df()
    epoch_idx = epochs.index.levels[epochs.index.names.index('Epoch_idx')].tolist()
    start, end = epoch_p
    e = np.arange(start, end)
    assert (epoch_idx == e).all

# test time set
def test_time_set():
    epoch_p, time_p, cat_p, cont_p, epochs = sample_df()
    times = epochs.index.levels[epochs.index.names.index('Time')].tolist()
    start, end, step = time_p
    t = np.arange(start, end, step)
    assert (times == t).all

# test if column matches cont_p range
def test_cont_p():
    epoch_p, time_p, cat_p, cont_p, epochs = sample_df()
    for i,range_set in enumerate(cont_p):
        start = range_set[0]
        end = range_set[1]
        col_label = 'cont_' + str(i+1) + '_range' + str(start) + '_' + str(end)
        assert (((epochs[col_label].values > start) | (epochs[col_label].values == start))
                & (epochs[col_label].values < end | (epochs[col_label].values == end))).all

# test if data matches cat_p range
def test_cat_p_match_data():
    epoch_p, time_p, cat_p, cont_p, epochs = sample_df()
    start, end = epoch_p
    times = epochs.index.levels[epochs.index.names.index('Time')].tolist()
    for cat in cat_p:
        b0, b1 = cat[0], cat[1]
        want_arr = (np.tile(times, end - start) + b0) * b1
        label = 'L_' + str(b0) + '_' + str(b1)
        act_arr = epochs[epochs['cat'] == label]['data'].values
        assert (want_arr == act_arr).all
