from .fake_data import generate
from .io import (
    epochs_from_hdf,
    epochs_from_dataframe,
    load_grid,
    epochs_from_feather,
)
from .models import run_model, lm, lmer
from . import utils, defaults

__version__ = "0.4.6"
