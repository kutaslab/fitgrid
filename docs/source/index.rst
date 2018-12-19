#########
`fitgrid`
#########

`fitgrid` is a Python library for running linear models on stacks of epochs.

Here's a quick demo:

1. Import `fitgrid` and read an epochs table from an HDF5 file::

    import fitgrid
    epochs = fitgrid.epochs_from_hdf('epochs_table.h5')

2. Run a regression model, which creates a ``FitGrid``::

    grid = fitgrid.lm(epochs, RHS='stimulus_magnitude + stimulus_type')

3. Now all diagnostic and fit information is available as attributes. For
   example, the betas::

    betas = grid.params
   
   or adjusted :math:`R^2`::

    rsquared_adj = grid.rsquared_adj

.. toctree::
    :hidden:

    research_context
    tutorial
    reference
