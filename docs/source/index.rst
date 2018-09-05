.. fitgrid documentation master file, created by
   sphinx-quickstart on Fri Aug 17 15:10:35 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#########
`fitgrid`
#########


`fitgrid` attempts to be a convenient tool for running linear models on stacks
of epochs. Here's a quick example:

1. Import `fitgrid` and read an epochs table from an HDF5 file::

    import fitgrid
    epochs = fitgrid.epochs_from_hdf('epochs_table.h5')

2. Run a regression model, which creates a fitgrid::

    grid = epochs.lm(RHS='noun_cloze + stimulus_type')

3. Now all diagnostic and fit information is available as attributes. For
   example, the betas of the model::

    betas = grid.params
   
   Or adjusted :math:`R^2`::

    rsquared_adj = grid.rsquared_adj

.. toctree::
   :hidden:

   quickstart
