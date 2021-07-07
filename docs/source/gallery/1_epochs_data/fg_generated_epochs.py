"""Simulated epochs data
========================

Use :py:func:`fitgrid.generate <fitgrid.fake_data.generate>` to quickly
generate small or large random data sets with categorical and random
continuous predictor variables. The random values can be seeded for
replicability.

The data can be returned in ``fitgrid`` :py:class:`Epochs
<fitgrid.epochs.Epochs>` format for immediate modeling or as a
:py:class:`pandas.DataFrame`. The latter is useful for mocking up the
conversion from a dataframe to fitgrid.Epochs when developing an
analysis pipeline.

Small data sets are useful for trying out features and
functions. Larger sets are useful for testing system performance and
limitations.

"""

# %%
import fitgrid

# %%
# Small random data set as fitgrid.Epochs
fitgrid.generate(n_samples=8, n_channels=4, seed=32)

#%%
# The same data as a pandas.DataFrame
fitgrid.generate(n_samples=8, n_channels=4, seed=32, return_type="dataframe")
