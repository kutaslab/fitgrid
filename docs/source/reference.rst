#########
Reference
#########

This is a list of tools available in ``fitgrid``.

.. module:: fitgrid


===============
Data generation
===============

``fitgrid`` has a built-in function that generates data and creates ``Epochs``:

.. autofunction:: generate

==============
Data ingestion
==============

Functions that read epochs tables and create ``Epochs``.

.. autofunction:: epochs_from_dataframe

.. autofunction:: epochs_from_hdf

==================
``Epochs`` methods
==================

Models and plotting.

.. autofunction:: fitgrid.epochs.Epochs.run_model

.. autofunction:: fitgrid.epochs.Epochs.lm

.. autofunction:: fitgrid.epochs.Epochs.plot_averages

===================
``FitGrid`` methods
===================

Plotting and statistics.

.. autofunction:: fitgrid.fitgrid.FitGrid.influential_epochs

.. autofunction:: fitgrid.fitgrid.FitGrid.plot_betas

.. autofunction:: fitgrid.fitgrid.FitGrid.plot_adj_rsquared
