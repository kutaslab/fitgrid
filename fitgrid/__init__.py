from os import environ
from pathlib import Path
from .fake_data import generate
from .io import (
    epochs_from_hdf,
    epochs_from_dataframe,
    load_grid,
    epochs_from_feather,
)
from .models import run_model, lm, lmer
from . import utils, defaults

__version__ = "0.5.0"

# for use by pytests and docs
DATA_DIR = Path(__file__).parent / "data"
