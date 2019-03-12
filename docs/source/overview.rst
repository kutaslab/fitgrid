########
Overview
########

======================================
Background: signal + noise across time
======================================

``fitgrid`` began as an implementation of the regression ERP (rERP)
framework described in [SmiKut2015a]_. Since the 1960's, the dominant
(but not only) experimental paradigm in human EEG research has been
the time-domain averge ERP. This approach uses the familiar trick of
averaging a set of noisy measurements to get a better (more precise)
estimate of the "true" value. Under the assumption that there is a
small but consistent brain response to the event, averaging multiple
measurements should tend to cancel out the (typically much larger)
variation in the background EEG. From a signal processing perspective,
this assumes the time-series of measurements :math:`y_{i}` are a
combination of event-related signal that varies systematically with
time, :math:`x(t)`, and random noise that does not.

.. math:: y_{i}(t) = x(t) + \mathit{noise}_{i}

--------------------------------------
Generalization #1: regression modeling
--------------------------------------

As Smith pointed out, the signal + noise model is a special case of
linear regression. The familiar time-domain average ERP is
mathematically identical to the estimated intercept coefficient
:math:`\hat{\beta}_{0}` ("beta-hat nought") in an intercept-only
linear regression model fit to the :math:`n` data samples :math:`i =
0, ... n` at each time point :math:`t`,

.. math:: y_{i}(t) = \beta_{0}(t) + \epsilon_{i}

The time series of these :math:`\hat{\beta}_{0}` estimates is exactly
the time-domain average ERP viewed from a regression modeling
perspective. More generally, regression models characterize
relationships between multiple predictor variables :math:`X_{1} ... X_{k}` and
the response variable :math:`y`.

.. math:: y_{i}(t) = \beta_{0}(t) + \beta_{1}X_{1}(t) +
          ... \beta_{k}X_{k}(t) + \epsilon_{i}

Fitting such models at each time point estimates all the :math:`\beta`
s together. Smith dubbed the time series of these beta-hats
:math:`\hat{\beta}_{0}, \hat{\beta}_{1}, \hat{\beta}_{k}` regression
ERPs, of which the familiar time-domain average is one special case.

The rERP framework encourages a shift in perspective from the
engineering view in which "the" ERP signal is a needle in the haystack
of background EEG to be discovered by averaging, to a data modelling
perspective that asks questions like, "which beta-hat regression ERPs
:math:`\hat{\beta}_{0} ... \hat{\beta}_{n}` best model the data?" and
relies on tried and true methods of applied regression analysis for
answers.


--------------------------------
Generalization #2: sensor arrays
--------------------------------

Regression ERPs sweep a regression model across EEG data recordings
time-point by time-point, channel by channel, to track how the model
estimates and goodness of fits evolve over time at different scalp
locations.

EEG recordings are a special case of regularly sampled, discrete
time-series, synchronized across a sensor array. Synchronized
multi-sensor recording arrays abound in neuroimaing and across the
physical sciences and engineering: magneto-encephalogray,
event-related fMRI, weather stations, buoy arrays, Internet of Things.

Stepping back from EEG data analysis, we see that versions of the
"event-related mass time-series regression modeling" approach could be
be applied to event-related sensor array data in a wide range of other
domains where regression modeling is appropriate

However.

==============================================================
Fitting regression models is easy, regression modeling is hard
==============================================================

Guidance on regression modelling best practices emphasizes the fluid,
iterative nature of the project: data screening for pathological
(non-)data points; preliminary model fitting; diagnostic testing for
outliers and influential data points, examination of residuals for
signs of violated assumptions; evaluation of signs of over-fitting,
multicollinearity. Evaluate, remediate, refit, adjust parameters,
refit. Repeat as necessary. [GelHil200

This is for a single model. Regression analyses that model arrays of
multi-channel time series such as EEG data time point by time point,
channel by channel, multiply these iterative investigation by the
number of points x the number of channels in a combinatorial
explosion. There are 750 time points in 3 second epochs of EEG data
sampled 250 times per second. For 32 EEG channels, this makes 750
timepoints x 32 channels = 24,000 data sets. The pairwise comparison
of three candidate models requires 72,000 model fits.

Nothing can prevent this combinatorial explosion, FitGrid is a
resource for managing it. 


=======
FitGrid 
=======

For an overview of the workflow see :doc:`Quickstart`

With one function call, FitGrid sweeps a model formula cross each of
the timepoints x channels (in parallel on multiple CPU cores if
supported by hardware) and collects the fits. When the fitting is
complete the results of the fitting and diagnostic measures can be
queried in a format ready for visualization and analysis.

To make things easier, models are defined using the formula
language notations familiar from R.

Ordinary least squares models are fit in Python using the
`statsmodels`_ statstics package and the `patsy
<https://patsy.readthedocs.io/en/latest/>`_ formula language. Linear
mixed effects models are shipped out of Python and into R via the
`pymer <https://github.com/kmerkmer/pymer>`_ interface and fit with
`lme4::lmer
<https://cran.r-project.org/web/packages/lme4/index.html>`_ (see
[BatMaeBolWal2015]_).

--------------------------------
Doing statistics in Python and R
--------------------------------

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

--------------
FitGrid Design
--------------


``fitgrid`` abstracts this complexity away and handles the iteration and
storage of the data behind the scenes. The first loop above is now replaced
with::

    grid = fitgrid.lm(epochs, RHS='x + a')

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
