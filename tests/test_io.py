from .context import fitgrid
from fitgrid import defaults, DATA_DIR


def test__epochs_from_hdf():
    TEST_FILE = DATA_DIR / 'fake_epochs.h5'
    channels = [f'channel{i}' for i in range(3)]
    fitgrid.epochs_from_hdf(
        TEST_FILE,
        key=None,
        time=defaults.TIME,
        epoch_id=defaults.EPOCH_ID,
        channels=channels,
    )


def test__epochs_from_dataframe_good_data():

    # should silently create an Epochs object from normally generated data
    # this is a helper function, all further testing of Epochs creation is
    # done in test_epochs.py
    table, channels = fitgrid.fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time=defaults.TIME,
        epoch_id=defaults.EPOCH_ID,
    )
    fitgrid.epochs_from_dataframe(
        table,
        time=defaults.TIME,
        epoch_id=defaults.EPOCH_ID,
        channels=channels,
    )


def test__epochs_from_feather():
    TEST_FILE = DATA_DIR / "fake_epochs.feather"
    channels = [f'channel{i}' for i in range(32)]
    fitgrid.epochs_from_feather(
        TEST_FILE,
        time=defaults.TIME,
        epoch_id=defaults.EPOCH_ID,
        channels=channels,
    )


# fitgrid.load_grid is tested in test_fitgrid.py
