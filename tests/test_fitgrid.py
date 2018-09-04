import numpy as np
import fitgrid


def test__method_returning_dataframe_expands_correctly():

    epochs = fitgrid.generate()
    LHS = ['channel0', 'channel1', 'channel2']
    grid = epochs.lm(
        LHS=LHS,
        RHS='categorical + continuous',
    )

    conf_int = grid.conf_int()
    assert (conf_int.columns == LHS).all()
    assert (
        conf_int.index.levels[1]
        == ['Intercept', 'categorical[T.cat1]', 'continuous']
    ).all()
    assert (conf_int.index.levels[2] == [0, 1]).all()
    assert (conf_int.dtypes == float).all()


def test__residuals_have_long_form_and_correct_index():

    n_epochs = 10
    n_categories = 2
    n_samples = 100

    epochs = fitgrid.generate(
        n_epochs=n_epochs, n_categories=n_categories, n_samples=n_samples
    )

    new_index = np.repeat(
        np.random.permutation(np.arange(n_categories * n_epochs)), n_samples
    )

    epochs.table.index = new_index

    grid = epochs.lm(
        LHS=['channel0', 'channel1', 'channel2'],
        RHS='categorical + continuous',
    )

    single_epoch = epochs.snapshots.get_group(0)
    assert single_epoch.index.equals(grid.resid.index.levels[1])
