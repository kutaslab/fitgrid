import pytest
import numpy as np
import random

from eegr import generate, build_grid
from eegr.errors import EegrError
from eegr._core import EPOCH_ID, TIME


# TODO add tests on real data

def test__shapes():

    # default parameters but listed here for explicitness
    epochs_table = generate(n_epochs=10, n_samples=100,
                            n_categories=2, n_channels=32)

    epoch = epochs_table.groupby(EPOCH_ID).get_group(0)
    snapshot = epochs_table.groupby(TIME).get_group(0)

    fitgrid = build_grid(epochs_table,
                         LHS=['channel0', 'channel12', 'channel23'],
                         RHS='continuous + categorical')

    # shape of betas should be (# samples per epoch, # predictors)
    # the intercept is implicit in patsy formulas, so 2 + 1 = 3, quick mafs
    assert fitgrid['channel0']['fit']['betas'].shape == (len(epoch), 3)

    # shape of residuals should be (# samples per epoch, # epochs)
    assert fitgrid['channel0']['diag']['resid_press'].shape \
        == (len(epoch), len(snapshot))


def test__raises_error_on_epoch_shape_mismatch():
    """Bad: epochs have different shapes."""

    # strategy: generate epochs, but kill a couple rows
    epochs_table = generate()
    row_to_drop = random.choice(epochs_table.index)
    epochs_table.drop(row_to_drop, inplace=True)

    with pytest.raises(EegrError):
        build_grid(epochs_table,
                   LHS=['channel0', 'channel1'],
                   RHS='continuous + categorical')


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


def test__raises_error_on_bad_slicer_type():
    """Bad: slicer not a string or list of strings."""

    epochs_table = generate()
    fitgrid = build_grid(epochs_table,
                         LHS=['channel0', 'channel1'],
                         RHS='continuous + categorical')
    with pytest.raises(TypeError):
        fitgrid[2]


def test__raises_error_on_missing_channel():
    """Bad: user indexes using unknown channel."""

    epochs_table = generate(n_channels=3)
    fitgrid = build_grid(epochs_table,
                         LHS=['channel0', 'channel1'],
                         RHS='continuous + categorical')

    # channel present in epochs_table, but not included in LHS
    with pytest.raises(KeyError):
        fitgrid['channel2']

    # completely unknown channel
    with pytest.raises(KeyError):
        fitgrid['channel_blah']
