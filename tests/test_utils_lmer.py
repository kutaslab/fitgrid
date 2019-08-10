from pathlib import Path
import numpy as np
import pandas as pd
from .context import fitgrid, tpath

# pytest evaluates tpath to the local tests directory


def get_epochs(tpath=tpath, epochs_f='expt1_epochs.h5', interval=(-8, 12)):
    """hard coded file of single trial epochs, entire experiment

    Parameters
    ----------
    tpath : pytest.Fixture autoset on import
      points to local test dir

    epochs_f : str
        path to HDF5 single trial epochs file

    interval : (start, stop) int
        where start and stop are Time index labels, short default for testing
    """

    EPOCHS_F = Path.joinpath(tpath, 'data', 'expt1_epochs.h5')
    epochs = (
        pd.read_hdf(EPOCHS_F, 'epochs')
        .reset_index()
        .set_index(['Epoch_idx', 'Time'])
        .loc[pd.IndexSlice[:, interval[0] : interval[1]], :]
    )
    return epochs


def get_lmer_coefs(tpath):
    """hard coded HDF5 test file with pd.DataFrame of lmer_coefs

    Parameters
    ----------
    tpath : pytest.Fixture
       evaluates to the local test directory
    """

    LMER_COEFS_F = Path.joinpath(tpath, 'data', 'lmer_coefs_1c_4m.h5')
    lmer_coefs = pd.read_hdf(LMER_COEFS_F, 'lmer_coefs')
    return lmer_coefs


def test_get_lmer_dfbetas(tpath):

    # the expected DFBETAS dataset was computed using the following code:
    """
    library(influence.ME)
    dat <- read.csv('epochs_to_test_dfbetas.csv')
    model <- lmer(channel0 ~ continuous + (continuous | categorical), data=dat)
    estex <- influence(model, 'categorical')
    write.csv(dfbetas(estex), 'dfbetas_test_values.csv')
    """
    TEST_EPOCHS = Path.joinpath(tpath, 'data', 'epochs_to_test_dfbetas.csv')
    TEST_DFBETAS = Path.joinpath(tpath, 'data', 'dfbetas_test_values.csv')

    expected = pd.read_csv(TEST_DFBETAS, index_col=0).T

    table = pd.read_csv(TEST_EPOCHS).set_index(['Epoch_idx', 'Time'])
    epochs = fitgrid.epochs_from_dataframe(
        table, channels=['channel0'], time='Time', epoch_id='Epoch_idx'
    )
    dfbetas = fitgrid.utils.lmer.get_lmer_dfbetas(
        epochs, 'categorical', RHS='continuous + (continuous | categorical)'
    )
    actual = dfbetas.loc[0, 'channel0'].unstack().astype(float)

    assert np.allclose(actual, expected, atol=0)
