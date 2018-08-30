import pytest

from fitgrid import fake_data, epochs, errors


def test_epochs_unequal_snapshots():

    epochs_table = fake_data._generate(
        n_epochs=10, n_samples=100, n_categories=2, n_channels=32
    )

    epochs_table.drop(epochs_table.index[42], inplace=True)
    with pytest.raises(errors.FitGridError) as error:
        epochs.Epochs(epochs_table)
    assert 'differs from previous snapshot' in str(error.value)
