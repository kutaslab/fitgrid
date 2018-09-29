from .context import fitgrid


def test__epochs_from_hdf():
    TEST_FILE = 'tests/data/fake_epochs.h5'
    channels = [f'channel{i}' for i in range(32)]
    fitgrid.epochs_from_hdf(TEST_FILE, channels)


def test__epochs_from_dataframe_good_data():

    # should silently create an Epochs object from normally generated data
    # this is a helper function, all further testing of Epochs creation is
    # done in test_epochs.py
    table, channels = fitgrid.fake_data._generate(
        n_epochs=10, n_samples=100, n_categories=2, n_channels=32
    )
    fitgrid.epochs_from_dataframe(table, channels)
