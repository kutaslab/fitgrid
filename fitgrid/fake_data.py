import warnings
import numpy as np
import pandas as pd
from .epochs import Epochs
from . import defaults

_TIME = defaults.TIME
_EPOCH_ID = defaults.EPOCH_ID


def generate(
    n_epochs=10,
    n_samples=100,
    n_categories=2,
    n_channels=32,
    time=_TIME,
    epoch_id=_EPOCH_ID,
    seed=None,
    return_type="epochs",
):
    """Return Epochs object or pandas.DataFrame with fake EEG data.

    Parameters
    ----------
    n_epochs : int
        number of epochs per category to be generated
    n_samples : int
        number of samples in a single epochs
    n_categories : int
        number of levels of the categorical variable
    n_channels : int
        number of time series representing EEG channels
    time : str, defaults to defaults.TIME
        time column name
    epoch_id : str, defaults to defaults.EPOCH_ID
        epoch identifier column name
    seed=None : {None, int, array_like}, optional
        Random number generation seed. Default=None lets data
        vary from run to run. Set `seed` to a 32-bit unsigned
        integer to generate the same fake data run to run.
        See numpy.random.RandomState for details.
    return_type : str {epochs, dataframe}
        return fitgrid.Epochs or the fitgrid.Epochs.table dataframe


    Returns
    -------
    epochs : fitgrid.Epochs or pandas.DataFrame
        Epochs object or just the data

    Notes
    -----
    ``n_epochs`` and ``n_categories`` interact in the sense that ``n_epochs``
    epochs are generated for each level of the categorical variable. In other
    words, the true number of epochs in the generated data is equal to
    ``n_epochs`` * ``n_categories``.

    For example, the default ``n_epochs = 10`` and ``n_categories
    = 2`` produces 20 epochs, 10 per category.
    """
    df, channels = _generate(
        n_epochs, n_samples, n_categories, n_channels, time, epoch_id, seed
    )
    epochs_fg = Epochs(df, time=time, epoch_id=epoch_id, channels=channels)
    if return_type == "epochs":
        return epochs_fg
    elif return_type == "dataframe":
        return epochs_fg.table.reset_index()
    else:
        raise ValueError("return_type must be 'epochs' or 'dataframe'")


def _generate(
    n_epochs, n_samples, n_categories, n_channels, time, epoch_id, seed=None
):
    """Return Pandas DataFrame with fake EEG data, and a list of channels."""

    if seed is not None:
        np.random.seed(seed)

    total = n_epochs * n_samples * n_categories

    categories = np.array([f'cat{i}' for i in range(n_categories)])

    indices = {
        epoch_id: np.repeat(np.arange(n_epochs * n_categories), n_samples),
        time: np.tile(np.arange(n_samples), n_epochs * n_categories),
    }

    predictors = {
        'categorical': np.tile(np.repeat(categories, n_samples), n_epochs),
        'continuous': np.random.uniform(size=total),
    }

    channels = [f'channel{i}' for i in range(n_channels)]
    eeg = {
        channel: np.random.normal(loc=0, scale=30, size=total)
        for channel in channels
    }

    data = {**indices, **predictors, **eeg}

    df = pd.DataFrame(data).set_index([epoch_id, time]).sort_index()

    return df, channels
