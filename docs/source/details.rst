*******
Details
*******

================================
How mixed effects models are run
================================

Mixed effects models do not have a complete implementation in Python, so we
interface with R from Python and use `lme4` in R. The results that you get when
fitting mixed effects models in `fitgrid` are the same as if you used `lme4`
directly, because we use `lme4` (indirectly).

========================
Saving and loading grids
========================

Running models like `lmer` on large datasets can take a long time. `fitgrid`
lets you save your grid to disk so you can restore them later without having to
refit the models. Suppose you run `lmer` like so::

    grid = fitgrid.lmer(epochs, RHS='x + (x|a)')

Save the ``grid``::

    grid.save('lmer_results')

Later you can reload the ``grid``::

    grid = fitgrid.load_grid('lmer_results')


=======================
Multicore model fitting
=======================

On a multicore machine, model fitting can be parallelized to achieve a
significant speedup. ``fitgrid.lm`` uses ``statsmodels`` under the hood to fit
a linear least squares model, which in turn employs ``numpy`` for calculations.
``numpy`` itself depends on linear algebra libraries that might be configured
to use multiple threads by default. This means that on a 48 core machine,
common linear algebra calculations might use 24 cores automatically, without
any explicit parallelization. So when you explicitly parallelize your
calculations using Python processes (say 4 of them), each process might start
24 threads. In this situation, 96 CPU bound threads are wrestling each other
for time on the 48 core CPU. This is called oversubscription and results in
*slower* computations.

To deal with this when running ``fitgrid.lm``, we try to instruct the linear
algebra libraries your ``numpy`` distribution depends on to only use a single
thread in every computation. This then lets you control the number of CPU cores
being used by setting the ``n_cores`` parameter in ``fitgrid.lm``.

If you are using your own 8-core laptop, you might want to use all cores, so
set something like ``n_cores=7``. On a shared machine, it's a good idea to run
on half or 3/4 of the cores if no one else is running heavy computations.

===============
Useful routines
===============

Several diagnostic routines are available in ``fitgrid.utils``. See
:ref:`model-diagnostic-utilities` for details.
