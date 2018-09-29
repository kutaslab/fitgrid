from .context import fitgrid as fg


def test__generate():

    n_samples = 100
    n_epochs = 10
    n_categories = 2
    n_channels = 32
    table, channels = fg.fake_data._generate(
        n_samples=n_samples,
        n_epochs=n_epochs,
        n_categories=n_categories,
        n_channels=n_channels,
    )

    # check index columns
    assert fg.EPOCH_ID, fg.TIME == table.index.names

    # test general shape
    assert len(table) == n_samples * n_categories * n_epochs

    # test uniqueness within columns
    assert len(table.index.unique(fg.TIME)) == n_samples
    assert len(table.index.unique(fg.EPOCH_ID)) == n_epochs * n_categories
    assert len(table['categorical'].unique()) == n_categories

    # test uniqueness of groups (multiple columns)
    epochs_and_cat = table.groupby([fg.EPOCH_ID, 'categorical']).size()
    assert len(epochs_and_cat) == n_epochs * n_categories
    assert (epochs_and_cat == n_samples).all()

    # want n_epochs epochs per category
    categories = table.groupby('categorical')
    assert len(categories) == n_categories
    for i, category in categories:
        assert len(category.index.unique(fg.EPOCH_ID)) == n_epochs

    # test number of columns (2 predictors + n_channels)
    assert len(channels) == n_channels
    assert len(table.columns) == 2 + n_channels
