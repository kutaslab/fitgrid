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

    # shape of betas should be (# samples per epoch, # predictors) in this case
    # the intercept is implicit in patsy formulas, so 2 + 1 = 3, quick mafs
    assert fitgrid['channel0']['fit']['betas'].shape == (len(epoch), 3)

    # shape of residuals should be (# samples per epoch, # epochs)
    assert fitgrid['channel0']['diag']['resid_press'].shape \
        == (len(epoch), len(snapshot))

def test__many_categories():
    """Test bucket datatype fits the betas.

    Depending on the coding scheme, the number of betas might not be equal to
    the number of predictors, so we cannot rely on the formula to determine the
    shape of betas for the bucket datatype.
    """
    epochs_table = generate(n_categories=4)
    fitgrid = build_grid(epochs_table,
                         LHS=['channel0', 'channel12', 'channel23'],
                         RHS='continuous + categorical')


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


def test__raises_error_on_bad_slicer_type():
    """Bad: slicer not a string or list of strings."""

    epochs_table = generate()
    fitgrid = build_grid(epochs_table,
                         LHS=['channel0', 'channel1'],
                         RHS='continuous + categorical')
    with pytest.raises(EegrError) as bad_slicer_error:
        fitgrid[2]

    assert 'Expected a channel name' in str(bad_slicer_error.value)


def test__raises_error_on_missing_channel():
    """Bad: user indexes using unknown channel."""

    epochs_table = generate(n_channels=3)
    fitgrid = build_grid(epochs_table,
                         LHS=['channel0', 'channel1'],
                         RHS='continuous + categorical')

    # channel present in epochs_table, but not included in LHS
    with pytest.raises(EegrError) as first_unknown_channel:
        fitgrid['channel2']

    assert 'channel2 not in the list' in str(first_unknown_channel.value)

    # completely unknown channel
    with pytest.raises(EegrError) as second_unknown_channel:
        fitgrid['channel_blah']

    assert 'channel_blah not in the list' in str(second_unknown_channel.value)


def test__colon_wildcard_slicer():
    """Good: passing a colon slicer (fitgrid[:]) should return all channels."""

    n_samples = 100
    n_channels = 4
    LHS_all_four = ['channel0', 'channel1', 'channel2', 'channel3']
    RHS = 'continuous + categorical'

    epochs_table = generate(n_samples=n_samples, n_channels=n_channels)
    fitgrid_all_four = build_grid(epochs_table, LHS=LHS_all_four, RHS=RHS)

    # generate and fit with 4 channels, so expect 4 by 100 fitgrid
    assert fitgrid_all_four[:].shape == (4, 100)

    LHS_only_two = ['channel0', 'channel1']
    fitgrid_only_two = build_grid(epochs_table, LHS=LHS_only_two, RHS=RHS)

    # fit with 2 channels, expect 2 by 100 fitgrid
    assert fitgrid_only_two[:].shape == (2, 100)


def test__colon_slicing_nonwildcard():
    """Bad: nonwildcard indexing makes no sense, raise KeyError."""

    epochs_table = generate()
    fitgrid = build_grid(epochs_table,
                         LHS=['channel0', 'channel1'],
                         RHS='continuous + categorical')

    with pytest.raises(EegrError) as bad_colon_1:
        fitgrid[:1]
    assert 'Only wildcard slicing is supported' in str(bad_colon_1.value)

    with pytest.raises(EegrError) as bad_colon_2:
        fitgrid[:'channel1']
    assert 'Only wildcard slicing is supported' in str(bad_colon_2.value)

    # meaningless right now, might change (all channels, all samples)
    with pytest.raises(EegrError) as bad_colon_3:
        fitgrid[:, :]
    assert 'Expected a channel name' in str(bad_colon_3.value)
