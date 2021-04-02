
This is a list of tools available in ``fitgrid``.

.. module:: fitgrid
   :noindex:



.. _data_ingestion:

==============
Data Ingestion
==============

Functions that read epochs tables and create ``Epochs`` and load ``FitGrid``
objects.

.. autofunction:: epochs_from_dataframe
   :noindex:

.. autofunction:: epochs_from_hdf
   :noindex:

.. autofunction:: load_grid
   :noindex:



.. _data_simulation:

===============
Data Simulation
===============

``fitgrid`` has a built-in function that generates data and creates ``Epochs``:

.. autofunction:: generate
   :noindex:


==================
``Epochs`` methods
==================

Models and plotting.

.. autofunction:: fitgrid.epochs.Epochs.plot_averages
   :noindex:

=============
Model running
=============

.. autofunction:: fitgrid.lm
   :noindex:

.. autofunction:: fitgrid.lmer
   :noindex:

.. autofunction:: fitgrid.run_model
   :noindex:

===================
``FitGrid`` methods
===================


.. autofunction:: fitgrid.fitgrid.FitGrid.save
   :noindex:


=====================
``LMFitGrid`` methods
=====================

Plotting and statistics.

.. autofunction:: fitgrid.fitgrid.LMFitGrid.influential_epochs
   :noindex:

.. autofunction:: fitgrid.fitgrid.LMFitGrid.plot_betas
   :noindex:

.. autofunction:: fitgrid.fitgrid.LMFitGrid.plot_adj_rsquared
   :noindex:


.. _model-diagnostic-utilities:

=========
Utilities
=========

-------------------
model fit summaries
-------------------

.. autofunction:: fitgrid.utils.summary.summarize
   :noindex:

.. autofunction:: fitgrid.utils.summary.plot_betas
   :noindex:

.. autofunction:: fitgrid.utils.summary.plot_AICmin_deltas
   :noindex:


--------------
lm diagnostics
--------------

.. autofunction:: fitgrid.utils.lm.get_vifs
   :noindex:

.. autofunction:: fitgrid.utils.lm.list_diagnostics
   :noindex:

.. autofunction:: fitgrid.utils.lm.get_diagnostic
   :noindex:

.. autofunction:: fitgrid.utils.lm.filter_diagnostic
   :noindex:


----------------
lmer diagnostics
----------------

.. autofunction:: fitgrid.utils.lmer.get_lmer_dfbetas
   :noindex:

