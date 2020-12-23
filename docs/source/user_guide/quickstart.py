"""
.. _quickstart:

Quickstart
==========
"""

#!/usr/bin/env python
# coding: utf-8

# # Workflow

# The `fitgrid` data modeling workflow consists of 4 steps:
# 
# 1. Prepare your dataset as a `pandas.DataFrame`
# 2. Load your dataset into fitgrid, which creates an `Epochs` object.
# 3. Run a model on the `Epochs`, which creates a `FitGrid`.
# 4. Query the `FitGrid` for the model fit information estimates and diagnostic information at each sample and stream.
# 
# The commands below are in Python and are executed in Jupyter, but any Python environment will work. We recommend Jupyter Lab, Jupyter Notebook, or IPython.

# In[2]:


import fitgrid


# ## Prepare your dataset
# 
# `fitgrid` assumes you are modeling "epochs", i.e., fixed-length segments of time-stamped data streams ("channels").
# 
# Prepare your dataset as a `pandas.DataFrame` layed out in a long rows x narrow columns format with the response and predictor variables as the named columns and the data values ("observations") in rows.
# 
# Besides the response and predictor columns, two additional index columns that give the epoch identifier and time stamp of each observation (see the examples below).
# 
# 
# Dataset Format
# 
# - column names can be chosen freely
# 
# - all epochs must have exactly the same time stamps
# 
# - each epoch must have a unique identifier in the dataset, no duplicates are allowed
# 
# 
# Example
# 
# 
# Here is a toy dataset, `my_data`.
# 
# There are two predictor variable columns and two response variable columns.
# 
# There are 4 epochs:
#     * each has a unique identifier, one of `1, 2, 3, 4`
#     * all have the same 3 time stamps `0, 100, 200`
# 

# In[1]:


import numpy as np
import pandas as pd
n_timestamps = 3

categorical_factor = ["a", "b"]
n_each = 2  
n_epochs = len(categorical_factor) * n_each

my_data = pd.DataFrame(
    {
        'epochs': 1 + np.repeat(range(n_epochs), n_timestamps),
        'times': 100 * np.tile(range(n_timestamps), n_epochs),
        'predictor_x1': np.random.random(n_timestamps * n_epochs),
        'predictor_x2': np.tile(
            np.repeat(categorical_factor, n_timestamps), n_each
        ),
        'data_y1': np.random.random(n_timestamps * n_epochs),
        'data_y2': np.random.random(n_timestamps * n_epochs),
    }
)

# this index is required!
my_data.set_index(['epochs', 'times'], inplace=True)
my_data


# ## 2. Load your dataset into fitgrid
# 
# `fitgrid` can load your dataframe directly from the Python workspace or from a file.
# 
# Either way, you need to tell `fitgrid`:
# 
# 1. which index column is the epoch identifier
# 2. which index column is the time identifier
# 3. which columns are the response variable(s) to model

# ### 2.1 Example: Load a dataset from the Python workspace

# In[ ]:


# feed the toy dataset to fitgrid
epochs_fg = fitgrid.epochs_from_dataframe(
    my_data,
    time='times',
    epoch_id='epochs',
    channels=['data_y1', 'data_y2']
)


# Now you have a fitgrid epochs table

# In[ ]:


epochs_fg


# ### 2.2 Load a dataset from an HDF5 file
# 
# The file `example.h5` was saved with `pandas.to_hdf()`. 
# 
# It contains a toy dataset with 20 epochs and 100 timestamps that looks like this:
# 
# |   Epoch_idx  | Time  | continuous | categorical   | channel0   | channel1   |
# |--------------|-------|------------|---------------|------------|------------|
# |          0   |  0    | 0.439425   |     cat0      | -13.332813 |  24.074655 |
# |          0   |  1    | 0.028030   |     cat0      | -16.005318 |  23.879106 |
# |          0   |  2    | 0.484779   |     cat0      |  21.309482 |  13.479029 |
# |          0   |  3    | 0.008352   |     cat0      | -39.315872 |  46.974077 |
# |          0   |  4    | 0.597296   |     cat0      |  34.399671 |  -3.740801 |
# |          .   |  .    |    .       |      .        |      .     |      .     |
# |          .   |  .    |    .       |      .        |      .     |      .     |
# |          .   |  .    |    .       |      .        |      .     |      .     |
# |         19   | 95    | 0.611419   |     cat1      |   6.877276 |  -3.882082 |
# |         19   | 96    | 0.728147   |     cat1      | -38.291487 |   1.024060 |
# |         19   | 97    | 0.605416   |     cat1      |   4.123766 |  56.674669 |
# |         19   | 98    | 0.199554   |     cat1      | -45.001713 | -18.420173 |
# |         19   | 99    | 0.008011   |     cat1      | -30.901878 |  35.481089 |
# 

# To load this dataset directly, run:

# In[ ]:


epochs_fg = fitgrid.epochs_from_hdf(
    filename='../data/example.h5',
    key=None,
    time='Time',
    epoch_id='Epoch_idx',
    channels=['channel0', 'channel1']
)


# This also creates an epochs object

# In[ ]:


epochs_fg


# ### 2.3 Load a dataset from a feather format file
# 
# See `fitgrid` Reference
# 

# ## 3. Run a model
# 
# Once the epochs are loaded, `fitgrid` fits a model (formula) at each time point and channel and captures the results in 2-D grid (time x channels).
# 
# Each cell in the grid has the model fit information for that time point and channel, such as estimated betas and diagnostic information like $R^2$ and Cook's $D$ in the case of linear regression.
# 
# Model formulas are the same ones used by `lme4:lmer` in `R` and `patsy` for `statsmodels` in `Python`
# 
# As of now, linear regression (via ``statsmodels``' ``ols``) and linear mixed
# models (via ``lme4``'s ``lmer``) are available. 
# 
# Running a model on the epochs creates a `FitGrid` object, 
# 
# ### 3.1 Ordinary least squares
# 
# To run linear regression on the epochs, use the `lm` function. This calls `statsmodels.ols` under the hood.

# In[ ]:


lm_grid = fitgrid.lm(
    epochs_fg,
    RHS='continuous + categorical'
)


# `fitgrid.lm` runs linear regression for each channel, with a single channel
# data as the left hand side, and the right hand side given by the Patsy/R style
# formula passed in using the `RHS` parameter:
# 
#     channel0 ~ continuous + categorical
#     channel1 ~ continuous + categorical
#     ...
#     channel31 ~ continuous + categorical

# If you want to model only a specific subset of channels, pass the list of channels to the `LHS` parameter.
# 
# ```python
# lm_grid = fitgrid.lm(
#     epochs_fg,
#     LHS=['channel1', 'channel3', 'channel5'],
#     RHS='continuous + categorical'
# )
# ```

# ### 3.2 linear mixed effects

# Similarly, to model linear mixed effects use the `lmer` function. This calls `lme4::lmer()` under the hood.

# In[ ]:


# get_ipython().run_cell_magic('time', '', "\nlmer_grid = fitgrid.lmer(\n    epochs_fg,\n    RHS='continuous + (continuous | categorical)'\n)")


# ### 3.3 Multicore parallel processing
# 
# With lmer especially, it may be useful to run your model with multiple
# processes to speed it up.
# 
# This can be achieved by setting ``parallel`` to
# ``True`` and ``n_cores`` to the desired value (defaults to 4) as follows.
# 
# The performance benefits depend on your hardware. A laptop usually has 4 cores, a good workstation may have 8, a high performance server may have dozens. If working on a shared system it is rude (and unproductive) to hog cores.

# In[ ]:


# get_ipython().run_cell_magic('time', '', "\nlmer_grid = fitgrid.lmer(\n    epochs_fg,\n    RHS='continuous + (continuous | categorical)', \n    parallel=True,\n    n_cores=4)")


# ## 4. Working with the grid
# 
# 
# ``FitGrid`` objects, like `lm_grid` or `lmer_grid` above, can be queried for attributes just like a
# ``fit`` object returned by ``statsmodels`` or ``lmer`` (see Overview Doing Statistics in Python for more
# background), for example:
# 
# **Hint**: If you are using an interactive environment like Jupyter Notebook or IPython,
#   you can use tab completion to see what attributes are available:
# 
# ```python
# # type 'lm_grid.' and press Tab
# lm_grid.<TAB>
# ```

# ### 4.1 Examples
# 
# #### Beta estimates

# In[ ]:


betas = lm_grid.params
betas.head(6)


# ####  Adjusted $R^2$

# In[ ]:


rsquared_adj = lm_grid.rsquared_adj
rsquared_adj.head(6)


# #### Cook's distance

# In[ ]:


influence = lm_grid.get_influence()
cooks_distance = influence.cooks_distance
cooks_distance.head()


# ### 4.2 Queries return a `DataFrame` or another `FitGrid`
# 
# Calling an attribute of a `FitGrid` objects returns either a pandas `DataFrame` of the
# appropriate shape or another `FitGrid` object:

# In[ ]:


# this is a dataframe
lm_grid.params.head()


# In[ ]:


# this is a FitGrid
lm_grid.get_influence()


# If a dataframe is returned, it is always presented in long form with the same
# indices and columns on the outer side as a single epoch: channels as columns
# and time as indices.

# ### 4.3 Slicing the grid
# 
# In addition, slicing on a `FitGrid` can be performed to produce a smaller grid
# of the shape you want. Suppose you want to only look at a certain channel
# within a given time interval. You can slice as follows:

# In[ ]:


smaller_grid = lm_grid[25:75, 'channel0']
smaller_grid


# Or multiple channels:

# In[ ]:


smaller_grid = lm_grid[25:75, ['channel0', 'channel1']]
smaller_grid


# To include all timepoints or all channels, use a colon:

# In[ ]:


# all channels within certain timeframe
lm_grid[25:75, :]


# In[ ]:


# all timepoints, two channels
lm_grid[:, ['channel0', 'channel1']]

