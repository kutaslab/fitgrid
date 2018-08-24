import pytest
import numpy as np
import random

from fitgrid import generate, Epochs
from fitgrid._errors import EegrError
from fitgrid import TIME


# TODO add tests on real data

def test__shapes():

    # default parameters but listed here for explicitness
    epochs_table = generate(n_epochs=10, n_samples=100,
                            n_categories=2, n_channels=32)

    build_grid(epochs_table,
               LHS=['channel0', 'channel12', 'channel23'],
               RHS='continuous + categorical')

    raise NotImplementedError


def test__raises_error_on_epoch_shape_mismatch():
    """Bad: epochs have different shapes."""

    # strategy: generate epochs, but kill a couple rows
    epochs_table = generate()
    row_to_drop = random.choice(epochs_table.index)
    epochs_table.drop(row_to_drop, inplace=True)

    with pytest.raises(EegrError) as error:
        build_grid(epochs_table,
                   LHS=['channel0', 'channel1'],
                   RHS='continuous + categorical')

    assert 'differs from previous epoch' in str(error.value)


def test__raises_error_on_epoch_index_mismatch():
    """Bad: all epochs have the same shape, but indices differ."""

    # strategy: generate epochs, but insert meaningless time index
    epochs_table = generate(n_epochs=10, n_samples=100,
                            n_categories=2, n_channels=32)

    # blow up index to misalign epochs and time
    bad_index = np.arange(len(epochs_table))
    epochs_table.index.set_levels(levels=bad_index, level=TIME, inplace=True)
    epochs_table.index.set_labels(labels=bad_index, level=TIME, inplace=True)

    # now time index is equal to row number in the table overall
    with pytest.raises(EegrError) as error:
        build_grid(epochs_table,
                   LHS=['channel0', 'channel1'],
                   RHS='continuous + categorical')
    assert 'differs from previous epoch' in str(error.value)
