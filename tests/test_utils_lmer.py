import pytest
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from .context import fitgrid, tpath
from fitgrid.utils import lmer as fgu

# pytest evaluates tpath to the local tests directory

EEG_STREAMS = [
    'lle',
    'lhz',
    'MiPf',
    'LLPf',
    'RLPf',
    'LMPf',
    'RMPf',
    'LDFr',
    'RDFr',
    'LLFr',
    'RLFr',
    'LMFr',
    'RMFr',
    'LMCe',
    'RMCe',
    'MiCe',
    'MiPa',
    'LDCe',
    'RDCe',
    'LDPa',
    'RDPa',
    'LMOc',
    'RMOc',
    'LLTe',
    'RLTe',
    'LLOc',
    'RLOc',
    'MiOc',
    'A2',
    'HEOG',
    'rle',
    'rhz',
]


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


@pytest.mark.parametrize("epochs_f", ["data/expt1_epochs.h5"])
@pytest.mark.parametrize("LHS", [['MiPa'], ['MiCe', 'MiPa']])  # n=1, >1
@pytest.mark.parametrize(
    "RHS",
    [
        ["var_a + (1 | sub_id)"],  # singleton model
        ["var_a + (1 | sub_id)", "var_a + (1 | item_id)"],  # n=1, >1
    ],
)
@pytest.mark.parametrize("interval", [(-50, 50)])
def test_fit_lmers(tpath, epochs_f, interval, LHS, RHS):
    """test multiple model fitting loop

    Parameters
    ----------
    tpath, epochs_f, interval : see get_epochs()

    LHS : fitgrid.lmer LHS specification

    RHS : list of fitgrid.lmer RHS specifications

    """
    epochs = get_epochs(tpath=tpath, epochs_f=epochs_f, interval=interval)
    fg_epochs = fitgrid.epochs_from_dataframe(epochs)
    lmer_coefs = fgu.fit_lmers(fg_epochs, LHS, RHS, parallel=True, n_cores=2)


def test_get_lmer_AICs(tpath):
    """scrape AICs and AIC_min deltas from lmer_coefs in previously saved pd.DataFrame"""

    lmer_coefs = get_lmer_coefs(tpath)
    aics = fgu.get_lmer_AICs(lmer_coefs)
    return aics


def test_plot_lmer_AICs(tpath):
    """plot lmer_coefs AICs, min deltas"""

    aics = test_get_lmer_AICs(tpath)
    paic = fgu.plot_lmer_AICs(aics)


@pytest.mark.parametrize("LHS", ["cproi"])
def test_plot_lmer_rERPs(tpath, LHS):
    """rERP plotter for a model set"""

    lmer_coefs = get_lmer_coefs(tpath)
    LHS = list(lmer_coefs.columns)

    for modl in lmer_coefs.index.get_level_values('model').unique():
        fs = fgu.plot_lmer_rERPs(
            LHS, lmer_coefs.loc[pd.IndexSlice[:, modl, :], :]
        )
        for f in fs:
            assert isinstance(f, plt.Figure)
            # plt.show(f)
