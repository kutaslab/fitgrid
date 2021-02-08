.. _why_fitgrid:

##############
Why `fitgrid`?
##############


============================
EEG and signal-averaged ERPs
============================

In the late 1920's Berger demonstrated that some of the human brain
activity related to external stimulation and internal mental events
could be measured at the surface of the scalp as tiny time-varying
electrical potential waveforms on the order of tens of microvolts
peak-to-peak, the human electroencephalogram (EEG). In the early 1950s
Dawson presented a demonstration that even tinier brain responses to
external stimulation that were too small to be seen by the naked eye
in the EEG could, however, be observed by repeating the stimulation
multiple times, aligning fixed-length segments ("epochs") of the EEG
recordings to the onset of the stimulus and summing the recordings
together at each time-point ([Dawson1951]_, [Dawson1954]_). The idea
of aggregating several noisy measurements so the pluses and minuses of
random variation cancel to yield a better estimate the "true" value of
the sum or mean was already well-known. The conceptual breakthrough
for EEG data analysis that Dawson credited to Hunt was to sweep the
familiar noise-reduction trick along the time-aligned EEG epochs to
supress larger variation in the EEG (noise) and reveal the time course
of the much smaller brain response (signal) to the stimulation, a
waveform on the order of microvolts peak-to-peak. Laboratory
experiments in subsequent decades found that the Hunt-Dawson
aggregation procedure could reveal a variety systematic brain
responses before, during, and after sensory stimulation, motor
responses, and internal mental events. With the advance of computer
hardware and software, oscilloscopic sweep averagers were replaced by
analog-to-digital conversion and sum-and-divide averaging in software
on general purpose computers. Since the 1970s, this discrete time
series average event-related brain potential (ERP) has been a
cornerstone of experimental EEG research on human sensation,
perception, and cognition. For a compendium see the Oxford Handbook of
Event-related Potentials, [LucKap2011]_.


===============
regression ERPs
===============

In 2015 Smith and Kutas published a seminal paper ([SmiKut15]_) noting
that the average of a set of values, :math:`y`, is identical to
the estimated constant, :math:`\hat{\beta}_{0}` for the linear model

.. math::

  y = \beta_{0} + e

fit by minimizing squared error (ordinary least squares). They pointed
out that this makes the average ERP a special case of sweeping a
linear regression model along the EEG and generalized this to
more complex multiple regression models,

.. math::

   y = \beta_{0} + \beta_{1}X_{1} \ldots \beta_{n}X_{i} + e

Sweeping any such model along the EEG time point by time point
likewise produces time series of estimates for the intercept and
regressor coefficients, the :math:`\hat{\beta}_{i}` they dubbed the
"regression ERP" (rERP) waveforms.

This insight extends sum-and-divide Hunt-Dawson aggregation and embeds
event-related EEG data analysis in a general framework for
discovering, evaluating, and comparing a wide range of models to
account for systematic variation in the time course of EEG responses
using well-established methods of applied regression. With
this shift, however, comes a new problem.

==========================================================
Modeling: fit, diagnose, compare, evaluate, revise, repeat
==========================================================

These days specifying and fitting a linear regression model is a
matter of organizing the data into a table of rows (observations) and
columns (variables), typing a model specification formula like
:math:`1 + a + b + a:b` and pressing Enter. While **fitting** a model is
relatively easy and mechanical, **modeling**, by contrast, is a laborious
process that iterates cycles of data quality control, fitting,
data diagnosis, model evaluation, comparison, and selection with numerous
decision points that require thought and judgment along the way.

Modeling EEG data as regression ERPs at each time point and data
channel multiplies the iterative cycles in a combinatorial explosion
of time points :math:`\times` channels :math:`\times` models
:math:`\times` comparisons. For example, at a digital sampling rate of
250 samples per second, there are 750 time points in 3 seconds of EEG
data. For 32 EEG channels, this makes 750 timepoints x 32 channels =
24,000 data sets. To fit three candidate models requires 72,000
separate model fits where the size of the data set might range
anywhere from a few dozens of observations for a single subject to
tens of thousands of observations for a large scale experiment.

Nothing can prevent the combinatorial explosion; `fitgrid`
is designed to contain it.


=======================================
`fitgrid`: Modeling :math:`\times` 10e4
=======================================

The `fitgrid` package allows researchers generally familiar with
regression modeling and model specification formulas in Python
(`statsmodels.formula.api` via `patsy`) or R (`lm`, `lme4`,
`lmerTest`) to use these tools to readily and reproducibly fit
ordinary least squares and linear mixed-effects regression models of 
multichannel event-related time series recordings, at scale, with
a few lines of scripted Python. 

With one function call, `fitgrid` sweeps a model formula across the
data at each of the timepoints x channels (in parallel on multiple CPU
cores if supported by hardware) and collects the resulting fit objects
returned by `statsmodels.ols` and `lme4::lmer` via `pymer4` in a
single time x Channel `FitGrid` Python object. 

The grid can be sliced by time and channel like a dataframe,
`FitGrid[times, channels]` and fit results for the grid are accessed
with the same familiar syntax as a single fit object. These results
include the time-series of coefficient estimates comprising the
regression ERPs, including, but not restricted to, the Hunt-Dawson
ERP.  Equally important for modeling, the results also include
everything else in the bundle of information comprising the fit object
such as coefficient standard errors, model log likelihood, Akiake's
information criterion, model and error mean squares, and so
forth. The results are returned as tidy Time x Channel dataframes
for easy visualization and analysis in Python and data interchange
across scientific computing platforms as illustrated in
:ref:`getting_started` and the :ref:`examples_gallery`.


==============================
`fitgrid` Design: How it works
==============================

Ordinary least squares models are fit in Python using the
`statsmodels`_ statstics package and the `patsy
<https://patsy.readthedocs.io/en/latest/>`_ formula language. Linear
mixed effects models are shipped out of Python and into R via Eshin Jolly's
`pymer4 <https://github.com/kmerkmer/pymer>`_ interface [Jolly19]_ and fit with
`lme4::lmer
<https://cran.r-project.org/web/packages/lme4/index.html>`_ (see
[BatMaeBolWal2015]_).

For illustration with `patsy` and `statsmodels`, suppose you have a
pandas DataFrame ``data`` with independent variables ``x`` and ``a``,
where ``x`` is continuous and ``a`` is categorical. Suppose also
``channel`` is your continuous dependent variable.  Here's how you
would run an ordinary least squares linear regression of ``y`` on
``x + a`` using `statsmodels <http://www.statsmodels.org>`_::

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

Now the issue with using that interface for single trial rERP analyses
is of course the dimensionality: instead of fitting a single model, we
need to fit :math:`m \times n` models, where :math:`m` is the number
of discrete time points and :math:`n` is the number of channels.

This can be handled using ``for`` loops of the form::

    for channel in channels:
        for timepoint in timepoints:
            # run regression 'channel ~ x + a', save fit object somewhere

And to access some particular kind of fit information, the exact same two
nested ``for`` loops are required::

    for channel in channels:
        for timepoint in timepoints:
            # extract diagnostic or fit measure, save it somewhere


``fitgrid`` abstracts this complexity away and handles the iteration and
storage of the data behind the scenes. The first loop above is now replaced
with::

    lm_grid = fitgrid.lm(epochs, RHS='x + a')

and the second loop with::

    betas = lm_grid.params

or::

    tvalues = lm_grid.tvalues

or::

    pvalues = lm_grid.pvalues

The crux of the approach conceived and implemented by Andrey Portnoy
is that ``lm_grid``, a ``FitGrid`` object, can be queried for the
exact same attributes as a regular ``statsmodels`` ``fit`` object as
above.

The result is most often a pandas DataFrame, sometimes another
``FitGrid``. In other words, if you are running linear regression, any
attribute of a fit object `documented
<http://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html>`_
by ``statsmodels`` as part of their API, can be used to query a
``FitGrid``.

``statsmodels``::

    fit.rsquared

``fitgrid``::

    lm_grid.rsquared

Some of the attributes are methods. For example, influence diagnostics in
``statsmodels`` are stored in a separate object that is created by calling the
``get_influence`` method. So Cook's distance measures can be retrieved as follows::

    influence = fit.get_influence()
    cooks_d = influence.cooks_distance

The exact same approach works in ``fitgrid``::

    influence = lm_grid.get_influence()
    cooks_d = influence.cooks_distance


==========================
`fitgrid` in other domains
==========================

Although the origins of `fitgrid` are in EEG data analysis, `fitgrid`
can also be used with sensor array time-series data from other domains
where event-related signal averaging and and regression modeling is
appropriate. The :ref:`Examples Gallery` uses hourly NOAA tide and
atmospheric data to illustrate an outdated but instructive example
model for detecting lunar tides in the atmosphere that Dawson
attributes to Laplace.

