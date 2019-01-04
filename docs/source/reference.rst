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

Functions that read epochs tables and create ``Epochs`` and load ``FitGrid``
objects.

.. autofunction:: epochs_from_dataframe

.. autofunction:: epochs_from_hdf

.. autofunction:: load_grid

==================
``Epochs`` methods
==================

Models and plotting.

.. autofunction:: fitgrid.epochs.Epochs.plot_averages

=============
Model running
=============

.. autofunction:: fitgrid.lm

.. autofunction:: fitgrid.lmer

.. autofunction:: fitgrid.run_model

===================
``FitGrid`` methods
===================

.. autofunction:: fitgrid.fitgrid.FitGrid.save

=====================
``LMFitGrid`` methods
=====================

Plotting and statistics.

.. autofunction:: fitgrid.fitgrid.LMFitGrid.influential_epochs

.. autofunction:: fitgrid.fitgrid.LMFitGrid.plot_betas

.. autofunction:: fitgrid.fitgrid.LMFitGrid.plot_adj_rsquared


.. _model-diagnostic-utilities:

==========================
Model diagnostic utilities
==========================

.. autofunction:: fitgrid.utils.lmer.fit_lmers

.. autofunction:: fitgrid.utils.lmer.get_lmer_AICs

.. autofunction:: fitgrid.utils.lmer.plot_lmer_AICs

.. autofunction:: fitgrid.utils.lmer.plot_lmer_rERPs

.. autofunction:: fitgrid.utils.lmer.get_lmer_dfbetas
