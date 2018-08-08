import numpy as np
import pandas as pd


def generate(n_epochs=10, n_samples=100, n_categories=2, n_channels=32):
    """Generate fake EEG data."""

    total = n_epochs * n_samples * n_categories

    categories = np.array([f'cat{i}' for i in range(n_categories)])

    indices_and_predictors = {
        'epoch_id': np.repeat(np.arange(n_epochs * n_categories), n_samples),
        'time': np.tile(np.arange(n_samples), n_epochs * n_categories),
        'categorical': np.tile(np.repeat(categories, n_samples), n_epochs),
        'continuous': np.random.uniform(size=total),
    }

    eeg = {f'channel{i}': np.random.normal(loc=0, scale=30, size=total)
           for i in range(n_channels)}

    data = {**indices_and_predictors, **eeg}

    df = (pd.DataFrame(data)
            .set_index(['epoch_id', 'time'])
            .sort_index())

    return df
