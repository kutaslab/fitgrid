import pytest

from fitgrid._demo_data import generate
from fitgrid._epochs import Epochs
from fitgrid._errors import EegrError


def test_epochs():
    """Test creation of Epochs object."""

    epochs_table = generate(n_epochs=10, n_samples=100,
                            n_categories=2, n_channels=32)
    epochs = Epochs(epochs_table)


def test_epochs_unequal_snapshots():

    epochs_table = generate(n_epochs=10, n_samples=100,
                            n_categories=2, n_channels=32)

    epochs_table.drop(epochs_table.index[42], inplace=True)
    epochs = Epochs(epochs_table)

