import pytest
import numpy as np
from .context import fitgrid
from fitgrid import fake_data, errors
from fitgrid.epochs import Epochs

import matplotlib

matplotlib.use('Agg')


def test_epochs_lm_bad_inputs():

    epochs = fake_data.generate()
    with pytest.raises(errors.FitGridError):
        epochs.lm(LHS=['bad_channel'], RHS='categorical')


def test_epochs_unequal_snapshots():

    epochs_table, channels = fake_data._generate(
        n_epochs=10, n_samples=100, n_categories=2, n_channels=32
    )

    epochs_table.drop(epochs_table.index[42], inplace=True)
    with pytest.raises(errors.FitGridError) as error:
        Epochs(epochs_table, channels)
    assert 'differs from previous snapshot' in str(error.value)


def test__raises_error_on_epoch_index_mismatch():
    """Bad: all epochs have the same shape, but indices differ."""

    from fitgrid import TIME

    # strategy: generate epochs, but insert meaningless time index
    epochs_table, channels = fake_data._generate(
        n_epochs=10, n_samples=100, n_categories=2, n_channels=32
    )

    # blow up index to misalign epochs and time
    bad_index = np.arange(len(epochs_table))
    epochs_table.index.set_levels(levels=bad_index, level=TIME, inplace=True)
    epochs_table.index.set_labels(labels=bad_index, level=TIME, inplace=True)

    # now time index is equal to row number in the table overall
    with pytest.raises(errors.FitGridError) as error:
        Epochs(epochs_table, channels)

    assert 'differs from previous snapshot' in str(error.value)


def test_multiple_indices_end_up_EPOCH_ID():

    from fitgrid import EPOCH_ID, TIME

    epochs_table, channels = fake_data._generate(
        n_epochs=10, n_samples=100, n_categories=2, n_channels=32
    )
    epochs_table.reset_index(inplace=True)
    epochs_table.set_index([EPOCH_ID, TIME, 'categorical'], inplace=True)

    epochs = Epochs(epochs_table, channels)
    # internal table has EPOCH_ID in index
    assert epochs.table.index.names == [EPOCH_ID]
    # input table is not altered
    assert epochs_table.index.names == [EPOCH_ID, TIME, 'categorical']


def test_smoke_plot_averages():

    epochs = fake_data.generate()
    epochs.plot_averages(channels=['channel0', 'channel1'])
