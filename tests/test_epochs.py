import pytest
import numpy as np
import pandas as pd
from .context import fitgrid
from fitgrid import fake_data, defaults
from fitgrid.errors import FitGridError
from fitgrid.epochs import Epochs


def test_epochs_unequal_snapshots():

    epochs_table, channels = fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time=defaults.TIME,
        epoch_id=defaults.EPOCH_ID,
    )

    epochs_table.drop(epochs_table.index[42], inplace=True)
    with pytest.raises(FitGridError) as error:
        Epochs(
            epochs_table,
            time=defaults.TIME,
            epoch_id=defaults.EPOCH_ID,
            channels=channels,
        )
    assert 'differs from previous snapshot' in str(error.value)


def test_raises_error_on_duplicate_channels():

    epochs_table, channels = fitgrid.fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time=defaults.TIME,
        epoch_id=defaults.EPOCH_ID,
    )
    dupe_channel = channels[0]
    dupe_column = epochs_table[dupe_channel]
    bad_epochs_table = pd.concat([epochs_table, dupe_column], axis=1)

    with pytest.raises(FitGridError) as error:
        fitgrid.epochs_from_dataframe(
            bad_epochs_table,
            time=defaults.TIME,
            epoch_id=defaults.EPOCH_ID,
            channels=channels,
        )

    assert "Duplicate column names" in str(error.value)


def test__raises_error_on_epoch_index_mismatch():
    """Bad: all epochs have the same shape, but indices differ."""

    # strategy: generate epochs, but insert meaningless time index
    epochs_table, channels = fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time=defaults.TIME,
        epoch_id=defaults.EPOCH_ID,
    )

    # blow up index to misalign epochs and time
    bad_index = np.arange(len(epochs_table))
    epochs_table.index.set_levels(
        levels=bad_index, level=defaults.TIME, inplace=True
    )
    epochs_table.index.set_labels(
        labels=bad_index, level=defaults.TIME, inplace=True
    )

    # now time index is equal to row number in the table overall
    with pytest.raises(FitGridError) as error:
        Epochs(
            epochs_table,
            time=defaults.TIME,
            epoch_id=defaults.EPOCH_ID,
            channels=channels,
        )

    assert 'differs from previous snapshot' in str(error.value)


def test_multiple_indices_end_up_EPOCH_ID():

    epochs_table, channels = fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time=defaults.TIME,
        epoch_id=defaults.EPOCH_ID,
    )
    epochs_table.reset_index(inplace=True)
    epochs_table.set_index(
        [defaults.EPOCH_ID, defaults.TIME, 'categorical'], inplace=True
    )

    epochs = Epochs(
        epochs_table,
        time=defaults.TIME,
        epoch_id=defaults.EPOCH_ID,
        channels=channels,
    )
    # internal table has epoch_id in index
    assert epochs.table.index.names == [defaults.EPOCH_ID]
    # input table is not altered
    assert epochs_table.index.names == [
        defaults.EPOCH_ID,
        defaults.TIME,
        'categorical',
    ]


def test_smoke_plot_averages():

    epochs = fake_data.generate()
    epochs.plot_averages(channels=['channel0', 'channel1'])


def test_smoke_epochs_distances():

    epochs = fake_data.generate()
    epochs.distances()
