################
Research context
################

In this section, we describe the problem `fitgrid` tries to solve and explain
the design. 

===================
Smith and Kutas I
===================

``fitgrid`` is an implementation of the rERP framework proposed in `Smith and Kutas I <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5308234/>`_.

.. todo::

    Finish this section.

==========================
Doing statistics in Python
==========================

The modeling described above can be done using ``statsmodels``, the major
statistics package in Python. Suppose you have a pandas DataFrame ``data`` with
independent variables ``x`` and ``a``, where ``x`` is continuous and ``a`` is
categorical. Suppose also ``channel`` is your continuous dependent variable.
Here's how you would run linear regression of ``y`` on ``x + a`` using
`statsmodels <http://www.statsmodels.org>`_::

    from statsmodels.formula.api import ols

    fit = ols('channel ~ x + a', data).fit()

Now this ``fit`` object contains all the fit and diagnostic information,
mirroring what is provided by ``lm`` in R. This information can be retrieved by
accessing various attributes of ``fit``. For example, the betas::

    betas = fit.params

or the t-values::
    
    tvalues = fit.tvalues

or :math:`Pr(>|t|)`::

    pvalues = fit.pvalues

Compare to R, where this is usually done by calling functions like ``summary``
or ``coef``. 

Now the issue with using that interface for single trial rERP analyses is of
course the dimensionality: instead of fitting a single model, we need to fit
:math:`m \times n` models, where :math:`m` is the number of samples and
:math:`n` is the number of channels.

This can be handled using ``for`` loops of the form::

    for channel in channels:
        for timepoint in timepoints:
            # run regression 'channel ~ x + a', save fit object somewhere

And to access some particular kind of fit information, the exact same two
nested ``for`` loops are required::

    for channel in channels:
        for timepoint in timepoints:
            # extract diagnostic or fit measure, save it somewhere

======
Design
======

``fitgrid`` abstracts this complexity away and handles the iteration and
storage of the data behind the scenes. The first loop above is now replaced
with::

    grid = epochs.lm(RHS='x + a')

and the second loop with::

    betas = grid.params

or::

    tvalues = grid.tvalues

or::

    pvalues = grid.pvalues

The crux of the design approach is that ``grid``, a ``FitGrid`` object, can
be queried for the exact same attributes as a regular ``statsmodels`` ``fit``
object (see section above). The result is most often a pandas DataFrame,
sometimes another ``FitGrid``. In other words, if you are running linear
regression, any attribute of a fit object `documented
<http://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html>`_
by ``statsmodels`` as part of their API, can be used to query a ``FitGrid``.

``statsmodels``::

    fit.rsquared

``fitgrid``::

    grid.rsquared

Some of the attributes are methods. For example, influence diagnostics in
``statsmodels`` are stored in a separate object that is created by calling the
``get_influence`` method. So Cook's distance measures can be retrieved as follows::

    influence = fit.get_influence()
    cooks_d = influence.cooks_distance

The exact same approach works in ``fitgrid``::

    influence = grid.get_influence()
    cooks_d = influence.cooks_distance
