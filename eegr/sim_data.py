'''Generates simulated data

'''

import numpy as np
import pandas as pd

def df_gen(epoch_p, time_p, cat_p, cont_p, noise_p=(0, 0)):
    ''' Parameters:
        epoch_p: tuple,int
            (start,stop) epoch index range
        time_p: tuple,int
            (start,stop,step) time point range
        cat_p: list(s) of tuples [(b0,b1),...,(b0,b1)]
            categorical variables (ints)
            cat_p[i] gives time dependent of i-th level of categorical data
        cont_p: list(s) of tuples [(start,stop),...,(start,stop)]
            continuous variables (floats)
            cont_p[i] gives a range of numerical values for each continuous predictor
        noise_p: (i,j) tuple,float
            additive random noise with normal mean i and std j
    '''
    # parsing out the information from given
    e_start, e_stop = epoch_p
    num_epoch = e_stop-e_start

    t_start, t_stop, t_step = time_p
    if t_stop < t_start or t_stop == t_start:
        raise ValueError('stop time must be greater than start time!')

    mean, std = noise_p
    cat_length = len(cat_p)

    data_frame = pd.DataFrame()

    # generating the time index
    time_length = len([x for x in range(t_start, t_stop, t_step)])
    time = cat_length*(num_epoch*[x for x in range(t_start, t_stop, t_step)])
    data_frame['Time'] = time

    data_frame_size = (num_epoch*time_length*cat_length)

    # generating epoch index
    epoch_idx = np.asarray([np.tile(x, time_length*cat_length) 
                            for x in range(num_epoch)])
    epoch_idx = np.concatenate(epoch_idx)
    data_frame['Epoch_idx'] = epoch_idx

    # generating id index
    data_frame['Index'] = np.tile('id_10001', num_epoch*time_length*cat_length)
    # generating data group index
    data_frame['data_group'] = np.tile('testing1', num_epoch*time_length*cat_length)

    # generating cat_index
    cat = None
    cat = np.asarray(num_epoch*[np.tile('L_'+str(i)+'_'+str(j), time_length) 
                     for i, j in cat_p])
    cat = np.concatenate(cat)
    data_frame['cat']= cat

    # generating actual data
    data = []
    for epoch in range(num_epoch):
        for cat in cat_p:
            b_0 = cat[0]
            b_1 = cat[1]
            data_slice = None
            data_slice = [b_0+b_1*x+np.random.normal(loc=mean,scale=std)
                          for x in range(t_start, t_stop, t_step)]
            data.append(data_slice)
    data = np.concatenate((data))

    for i, cont in enumerate(cont_p):
        cont_val = np.random.uniform(low=cont[0], high=cont[1], size=data_frame_size)
        col_name = 'cont_{counter}_range{start}_{stop}'.format(counter=i+1,
                                                               start=cont[0],
                                                               stop=cont[1])
        data_frame[col_name] = cont_val

    data_frame['data'] = data
    data_frame.set_index(['Index', 'Epoch_idx', 'data_group', 'Time'], inplace=True)
    data_frame.sort_index(inplace=True)
    return data_frame
