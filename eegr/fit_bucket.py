from mkpy import mkh5 as mkh5

from matplotlib import pyplot as plt
from matplotlib import cm as cmap

import numpy as np
import pandas as pd
import pdb
from patsy import *
import pandas as pd
import pprint
from statsmodels.formula.api import ols
from statsmodels import stats as stats
import copy

slicer = pd.IndexSlice

def make_fit_table(data, LHS, RHS):
    '''Make a fit table with fit_grids for all subjects
       each element is a fit object with mixed data types
    '''
    
    # accessing times and subject ids of the experiment
    times = data.index.levels[data.index.names.index('Time')]
    subids = data.index.levels[data.index.names.index('data_group')]
    
    # creating empty fit_grids 
    fit_grids = []

    # every subject has a fit_grid
    for subid in subids:
        print('subid: ' + subid)
        fit_grid = np.empty((len(times),len(LHS)),dtype=object)
        subid_slice = None
        subid_slice = data.loc[slicer[:,:, subid,:],:]
        subid_slice = subid_slice.sort_index()
        for it, time in enumerate(times):
            time_slice = None
            time_slice = subid_slice.loc[slicer[:,:,:,time],:]
            for ic, chan in enumerate(LHS):
                this_fit = None
                formula = chan + " ~ " + RHS
                this_fit = ols(formula, data = time_slice).fit()
                fit_grid[it,ic] = get_fit(this_fit)
        fit_grids.append(fit_grid)
    
    return fit_grids

def get_fit(fit_obj):
    '''Grabbing necessary information into a single fit grid
    '''
    # influence object for diagnosis
    infl = fit_obj.get_influence()
        
    n = len(fit_obj.params)
    k = len(infl.resid_press)

    # fit and diagnostic data types
    dt_fit_names = ['coef','se','ci']
    dt_diag_names = ['cooks_d','ess_press','resid_press',
                     'resid_std','resid_var','resid_studentized_internal']
    
    dt_fit_formats = [('float32',n), ('float32',n), ('float32',(n,2))]
    dt_diag_formats = [('int16',1),
                       ('float32',k), 
                       ('float32',k), 
                       ('float32',k), 
                       ('float32',k), 
                       ('float32',k)]
    
    dt_fit = np.dtype({'names' : dt_fit_names,
                       'formats' : dt_fit_formats})
    dt_diag = np.dtype({'names' : dt_diag_names,
                       'formats' : dt_diag_formats})

    dt = np.dtype(([('fit', dt_fit), ('diag', dt_diag)]))

    # creating single fit object
    fit = np.empty((1,), dtype = dt)
    
    # fit 
    fit['fit']['coef']= fit_obj.params
    fit['fit']['se'] = fit_obj.bse
    fit['fit']['ci'][0] = fit_obj.conf_int()
    
    # diagnositic 
    fit['diag']['cooks_d'] = len([x for x in infl.cooks_distance[1] if x < 0.05])
    fit['diag']['ess_press'] = infl.ess_press    
    fit['diag']['resid_press'] = infl.resid_press
    fit['diag']['resid_var'] = infl.resid_var
    fit['diag']['resid_studentized_internal'] = infl.resid_studentized_internal

    return fit
