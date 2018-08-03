# import os
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eegr import modeler as modl
import numpy as np
import pdb
import logging 

# update these as more junk is added to the fit buckets
test_fields = ['time', 'chan', 'diagnostics']

def make_data(cat, ncont, neach, ntimes, nchans):
    """builds and returns numpy ndarray with specified 
    fully crossed experimental design

    Parameters
    ----------
    cat: tuple of (S1, uint)
      each tuple gives label, number of levels, e.g., (('S',20), ('I',32)),
      labels must be unique.
    ncont: uint
      number of continuous factors
    nreps : uint
      number of repetions of each design cell, 1, 4, 16, 32
    ntimes: uint
      number of data samples per epoch
    nchans: uint
      number of data channels
    """
    # build up the data type
    dts = [('{0}'.format(n),'S8') for n,v in cat] # categorical 
    dts += [('V{0}'.format(n),'f4') for n in range(ncont)] # continuous
    dts += [('Epoch_idx', np.uint)]
    dts += [('Time', int)]
    dts += [('chan_{0}'.format(n),'f4') for n in range(nchans)] # continuous
    
    ns = np.array([n for (label,n) in cat])
    prod_ns = np.prod(ns) # length
    design = np.zeros(shape=(prod_ns,), dtype=dts)
    # load factor levels into design template
    for i,c in enumerate(cat):
        for l in range(prod_ns):
            design[l][i] = '{0}_{1}_{2}'.format(c[0],i,l%c[1])

    data = np.zeros(shape=(design.shape[0] * neach * ntimes,),
                    dtype=dts)
    epoch_idx = 0
    for i,d in enumerate(design):
        for e in range(neach):
            for t in range(ntimes):
                idx = (i*neach*ntimes) + (e*ntimes) + t
                data[idx] = d
                data['Epoch_idx'][idx] = epoch_idx
                data['Time'][idx] = t
                print('Epoch_idx: {0} time: {1}'.format(epoch_idx,t))
            epoch_idx = epoch_idx + 1
    for c in [col for col in data.dtype.names \
              if 'chan_' in col]:
        data[c] = np.random.random(data.shape[0])
    return(data)

def make_FitGrid(ntimes, nchans):
    ''' returns a fit grid for the tests to use '''

    times = [-1 * round((ntimes)/2) + t for t in range(ntimes)]
    chans = ['ch_{0}'.format(ch) for ch in range(nchans)]
    times_chans = modl.FitGrid(ntimes, nchans)
    for c,chan in enumerate(chans):
        for t,time in enumerate(times):
            times_chans[t,c].time = time
            times_chans[t,c].chan = chan
        
            # these are stubs ... set with actual values 
            times_chans[t,c].reg_fit['params'] =  ['b_0', 'b_1', 'b_2']
            times_chans[t,c].reg_fit['hat_matrix'] =  ['stub']
            times_chans[t,c].diagnostics['cooks_stuff'] = 'stub'
            times_chans[t,c].diagnostics['other_stuff'] = 'stub'

    return(times_chans)

def test_small_FitGrid(ntimes=17, nchans=4):
    times_chans = make_FitGrid(ntimes, nchans)

def test_large_FitGrid(ntimes=10000, nchans=128):
    ''' 10 seconds at 1000 samples/seconds x 128 channels '''
    times_chans = make_FitGrid(ntimes, nchans)

def test_get_bucket_at(ntimes=17,nchans=4):
    times_chans = make_FitGrid(ntimes, nchans)
    for time in range(ntimes):
        for chan in range(nchans):
            print(( 'bucket[{0},{1}]: time_idx {2}: {3}'
                    ', chan_jdx {4}: {5}' ).format(
                        time,chan,
                        times_chans[time,chan].time_idx, 
                        times_chans[time,chan].time,
                        times_chans[time,chan].chan_jdx,
                        times_chans[time,chan].chan,
                    ) )
            
def test_bucket_grid_getters(ntimes=17,nchans=4):
    b_grid = make_FitGrid(ntimes, nchans)
    for field in test_fields:
        got_em = getattr(b_grid, field)

def test_slicing_fail(ntimes=17, nchans=4):
    ''' check that FitGrid blocks slicing '''
    b_grid = make_FitGrid(ntimes, nchans)
    for field in test_fields:
        
        # try to access grid column slice
        try:
            got_em = getattr(b_grid[:,0], field)
        except:
            pass
            # print('caught column slicing error')
        else:
            msg = 'uh oh, failed to catch grid column slicing error'
            raise RuntimeError(msg)

        # try to access grid column slice
        try:
            got_em = getattr(b_grid[0,:], field)
        except:
            pass
            # print('caught column slicing error')
        else:
            msg = 'uh oh, failed to catch grid row slicing error'
            raise RuntimeError(msg)


def test_make_data(cat=(('S',4),('I',8),('A',2),('B',2)),
                   ncont=2, neach=1, ntimes=5, nchans=3):
    data = make_data(cat, ncont, neach, ntimes, nchans)
    print(data.shape)

def test_ols():
    # cat=('S',32),('I',30),('A',2),('B',2)
    # ncont=2
    # neach=1 
    # ntimes=500
    # nchans=32
    cat=('S',4),('I',8),('A',2),('B',2)
    ncont=2
    neach=1 
    ntimes=5
    nchans=3
    print('making data')
    data = make_data(cat, ncont, neach, ntimes, nchans)

    epoch_idx = np.unique(data['Epoch_idx'])
    for e in epoch_idx:
        epoch = data[data['Epoch_idx']==e]
        for c in [name for name in epoch.dtype.names if 'ch' in name]:
            pass


