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

=========
Utilities
=========

-------------------
model fit summaries
-------------------

.. autofunction:: fitgrid.utils.summary.summarize

.. autofunction:: fitgrid.utils.summary.plot_betas

.. autofunction:: fitgrid.utils.summary.plot_AICmin_deltas


--------------
lm diagnostics
--------------

.. autofunction:: fitgrid.utils.lm.get_vifs

.. autofunction:: fitgrid.utils.lm.list_diagnostics

.. autofunction:: fitgrid.utils.lm.get_diagnostic

.. autofunction:: fitgrid.utils.lm.filter_diagnostic


----------------
lmer diagnostics
----------------

.. autofunction:: fitgrid.utils.lmer.get_lmer_dfbetas

