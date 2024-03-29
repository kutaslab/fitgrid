import warnings
import pytest
import numpy as np
import pandas as pd
from .context import fitgrid, FIT_ATOL, FIT_RTOL
from fitgrid import DATA_DIR

_TIME = fitgrid.defaults.TIME
_EPOCH_ID = fitgrid.defaults.EPOCH_ID


def test_get_lmer_dfbetas():

    # the expected DFBETAS dataset was computed using the following code:
    """
    library(influence.ME)
    dat <- read.csv('epochs_to_test_dfbetas.csv')
    model <- lmer(channel0 ~ continuous + (continuous | categorical), data=dat)
    estex <- influence(model, 'categorical')
    write.csv(dfbetas(estex), 'dfbetas_test_values.csv')
    """

    TEST_EPOCHS = DATA_DIR / "epochs_to_test_dfbetas.csv"
    TEST_DFBETAS = DATA_DIR / "dfbetas_test_values.csv"

    expected = pd.read_csv(TEST_DFBETAS, index_col=0).T

    table = pd.read_csv(TEST_EPOCHS).set_index([_EPOCH_ID, _TIME])
    epochs = fitgrid.epochs_from_dataframe(
        table, channels=['channel0'], time=_TIME, epoch_id=_EPOCH_ID
    )
    dfbetas = fitgrid.utils.lmer.get_lmer_dfbetas(
        epochs, 'categorical', RHS='continuous + (continuous | categorical)'
    )
    actual = dfbetas.loc[0, 'channel0'].unstack().astype(float)

    # assert np.allclose(actual, expected, atol=0)
    in_tol = np.isclose(actual, expected, atol=FIT_ATOL, rtol=FIT_RTOL)
    if not in_tol.all():
        actual['val'] = 'actual'
        expected['val'] = 'expected'
        for_display = (
            pd.concat([actual, expected])
            .set_index("val", append=True)
            .T.stack(0)
        )
        warnings.warn(
            f'\n------------------------------------------------------------\n'
            f'calculated lmer_dfbetas out of tolerance: {FIT_ATOL} + {FIT_RTOL} * expected\n'
            f'{in_tol}\n'
            f'{for_display}\n'
            f'------------------------------------------------------------\n'
        )


def test_smoke_get_plot_lmer_warnings():

    # this seed and model generates a nice assortment of errors
    epochs_fg = fitgrid.generate(n_samples=8, n_channels=4, seed=32)
    channels = [
        column for column in epochs_fg.table.columns if "channel" in column
    ]

    lmer_grid = fitgrid.lmer(
        epochs_fg,
        RHS="1 + categorical + (continuous | categorical)",
        LHS=channels,
        parallel=True,
        n_cores=2,
    )

    for key, val in fitgrid.utils.lmer.get_lmer_warnings(lmer_grid).items():
        print(key + "\n", val)

    fitgrid.utils.lmer.plot_lmer_warnings(lmer_grid)
    for show_warning in [
        "each",
        "all",
        "converge",
        ["converge"],
        ["singular", "converge"],
    ]:
        fitgrid.utils.lmer.plot_lmer_warnings(lmer_grid, which=show_warning)

    with pytest.raises(ValueError) as fail:
        fitgrid.utils.lmer.plot_lmer_warnings(lmer_grid, which=1.0)
    assert (
        "ValueError: "
        "The value for which=value must be 'any', 'each', a warning "
        "string pattern to match or list of them, not this:"
    ) in fail.exconly()
