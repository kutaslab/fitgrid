import pytest
from fitgrid import sample_data, DATA_DIR


def test_get_file():

    test_f = "sub000p3.h5"

    # remove if found
    if (DATA_DIR / test_f).exists():
        (DATA_DIR / test_f).unlink()

    # this should download
    sample_data.get_file(test_f)
    assert (DATA_DIR / test_f).exists()

    # this should skip quietly
    sample_data.get_file(test_f)


@pytest.mark.xfail(strict=True, raises=ConnectionError)
def test_get_file_xfail():
    xtest_f = "xsub00p3.h5"
    assert not (DATA_DIR / xtest_f).exists()  # shouldn't but ensure
    sample_data.get_file(xtest_f)
