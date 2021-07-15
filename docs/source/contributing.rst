.. _how_to_contribute:

############
Contributing
############


Field reports and ideas large and small for how to improve ``fitgrid``
are welcome. Please post what you have in mind on GitHub in
``fitgrid`` `Issues <https://github.com/kutaslab/fitgrid/issues>`_ in
accordance with the ``fitgrid`` `Code of Conduct
<https://github.com/kutaslab/fitgrid/blob/main/CODE_OF_CONDUCT.md>`_
to start a discussion of next steps and plan the approach.


================
Package overview
================

The ``fitgrid`` code is written in Python, and requires many
open-source scientific computing packages including :std:doc:`numpy
<numpy:index>`, :std:doc:`pandas <pandas:index>`, :std:doc:`matplotlib
<matplotlib:index>`, and :std:doc:`statsmodels
<statsmodels:index>`. The linear mixed-effects modeling further
requires the :std:doc:`pymer4 <pymer4:index>` and :std:doc:`rpy2
<rpy2:index>` Python packages as well as the `R
<https://www.r-project.org/other-docs.html>`_ language, and many R
packages including `lme4
<https://cran.r-project.org/web/packages/lme4/index.html>`_, and
`lmerTest
<https://cran.r-project.org/web/packages/lmerTest/index.html>`_. Version
control is managed with `git <https://git-scm.com/doc>`_ and the
primary source repository is hosted on GitHub in
https://github.com/kutaslab.

The stable and pre-release :std:doc:`fitgrid <fitgrid:index>` packages
are available on https://anaconda.org/kutaslab/fitgrid for easy
installation into conda virtual environments with :std:doc:`conda
<conda:index>` or :std:doc:`mamba <mamba:index>`. The Python package
installer :std:doc:`pip <pip:index>` is not supported because of the R
dependencies.

The documentation is generated with :std:doc:`sphinx
<sphinx:contents>` and :std:doc:`sphinx-gallery
<sphinx-gallery:index>`. 


Versions
  ``fitgrid`` is semantically versioned following a simplified 
  Python `PEP 440 <https://www.python.org/dev/peps/pep-0440>`_ scheme.

  * **Syntax.** Legal ``fitgrid`` version strings have three required
    numeric segments, ``M.N.P`` for major, minor, patch versions and
    an optional ``.devX`` suffix with numeric ``X``. As
    usual, the numeric values increment monotonically except that subordinate
    segments reset to 0 when a superordinate segment
    increments. Version strings match the regular expression,
    ``\d+\.\d+\.\d+(\.dev\d+){0,1}``.

  * **Semantics.** ``M.N.P`` designates a stable version and ``M.N.P.devX``
    designates a development version.
    
  .. note::
     In this scheme, the stable package version ``M.N.P`` sorts higher
     than the development version ``M.N.P.devX``, for example,
     ``0.5.1.`` > ``0.5.1.dev1`` > ``0.5.1.dev0`` > ``0.5.0``

GitHub reserved branch names and version strings: https://github.com/kutsalab/fitgrid
  * The branch `main` = the latest stable release, version ``M.N.P`` with tag ``vM.N.P``
  * The branch `dev` = the latest development version ``M.N.P.devX``.
  * Names and version strings for working branches other than `main`
    and `dev` are not strictly defined in this scheme. However, the
    natural version sequence increments the ``devX`` segment to
    ``M.N.P.devX+1`` following the pre-release package upload of
    ``M.N.P.devX`` and increments the patch and resets the ``devX``
    segment to ``M.N.P+1.dev0`` following the stable release of
    ``vM.N.P``.

Stable releases: ``vM.N.P``
  Stable versions are tagged ``vM.N.P`` and released manually on GitHub
  (``fitgrid`` `Releases
  <https://github.com/kutaslab/fitgrid/releases>`_). The stable
  version ``vM.N.P`` source code is frozen and subsequent
  modifications require the version to increment at least the patch
  segment. Development versions ``M.N.P.devX`` are not released on
  GitHub but they are deployed as conda packages (see next).

Conda packages and channels: stable ``M.N.P`` and pre-release ``M.N.P.devX``
  Conda packages are deployed for stable releases (``vM.N.P`` on branch
  `main`) and for development versions (``M.N.P.devX`` on branch
  `dev`). The stable release deploys to conda channel `kutaslab/label/main
  <https://anaconda.org/kutaslab/fitgrid/files>`_ and is the
  default for ``fitgrid`` conda installation. The development
  package deploys to channel `kutaslab/label/pre-release
  <https://anaconda.org/kutaslab/fitgrid/files>`_
  so the latest features and bug-fixes can be installed in conda
  environments with conda package dependency resolution.

Sphinx and sphinx-gallery documentation
  Documentation for the latest stable conda package ``vM.N.P`` is
  deployed to `gh-pages
  <https://github.com/kutaslab/fitgrid/tree/gh-pages>`_ and available
  online at https://kutaslab.github.io/fitgrid. Documentation for the
  latest development version ``M.N.P.devX`` is deployed to
  `gh-pages-dev
  <https://github.com/kutaslab/fitgrid-dev/tree/gh-pages-dev>`_ and
  available online at https://kutaslab.github.io/fitgrid-dev-docs.
  

The ``fitgrid`` CID is implemented in a single-pass GitHub Action
workflow, `figrid-cid.yml
<https://github.com/kutaslab/fitgrid/blob/main/.github/workflows/fitgrid-cid.yml>`_.
The continuous integration workflow is triggered by push, pull request
and manual release events on GitHub. The deploy phase selectively
uploads the conda packages and documentation for development version
pre-releases and stable releases. This scheme allows conda or mamba
installation of both stable and development versions and automatically
synchronizes the stable release version string and source code across
the GitHub repository at `github.com/kutaslab/fitgrid
<https://github.com/kutaslab/fitgrid>`_, the conda packages at
`anaconda.org/kutaslab/fitgrid <https://anaconda.org>`_ , the online
`sphinx documentation <https:kutaslab.github.io/fitgrid>`_, and the
Zenodo source code archive at `DOI 10.5281/zenodo.3581496
<https://doi.org/10.5281/zenodo.3581496>`_.


.. _cid-figure:

.. figure:: _static/fitgrid_cid_scheme.png

   Continuous Integration and Deployment Scheme

	    
Continuous Integration
  The conda package is built from the
  source on the triggering branch and installed into a newly created
  conda test environment.  The pytests in `fitgrid/tests/test_*.py`
  are run and the Sphinx html documentation is generated, including the
  sphinx-gallery `*.py` examples, in the test environment with
  the just-built package as installed.

Deployment
  If the CI passes, workflows triggered on branch `dev`
  with version string of the form ``M.N.P.devX`` or triggered by a GitHub manual
  releases tagged ``vM.N.P`` on branch main auto-upload the just-built conda
  package and Sphinx documentation to the appropriate destination
  repositories.

  * Pre-release: ``M.N.P.devX``

    * Conda packages: `--channel kutaslab/label/pre-release <https://anaconda.org/kutaslab/fitgrid/files>`_
    * Sphinx documentation: `kutaslab.github.io/fitgrid-dev-docs <https://kutaslab.github.io/fitgrid-dev-docs>`_
      
  * Stable release: ``vM.N.P``

    * Conda packages: `--channel kutaslab <https://anaconda.org/kutaslab/fitgrid/files>`_
    * Sphinx documentation: `kutaslab.github.io/fitgrid <https://kutaslab.github.io/fitgrid>`_
    * Zenodo archive DOI: `10.5281/zenodo.3581496 <https://doi.org/10.5281/zenodo.3581496>`_


New work
  Updates to ``fitgrid`` source and docs are committed to working
  branches derived from the `kutaslab/fitgrid/dev` branch and not
  directly to the `main` or `dev` branches which are reserved for
  deploying conda packages and documentation. As development on the
  working branches progesses (magenta in the :ref:`cid-figure`),
  maintainers periodically pull the changes to the `dev` branch in
  order to deploy a pre-release package for installation into conda
  environments. When development is ready for a stable release,
  maintainers pull `dev` to the `main` branch and manually issue a
  stable release on GitHub tagged ``vM.N.P``. The tagged release
  uploads the ``M.N.P`` conda packages and sphinx documentation and
  archives the ``M.N.P`` source on Zenodo.


=======================
Development environment
=======================

The following illustrates how to set up and develop new work
that begins with a working branch in the source repository called
`kutaslab/fitgrid/new-feature`. It assumes the ``git``, ``conda``, and
``mamba`` executables are already installed on the local computer and
the commands are executed in a bash shell.

--------
One time
--------

* Sign in to GitHub and create a fork of `github.com/kutaslab/fitgrid
  <https://github.com/kutaslab/fitgrid>`_ in your GitHub account.

* On the local computer where you plan to work, ``git clone`` the fork: ::

    git clone https://github.com/<your_github_username>/fitgrid

  You may wish to clone just the working branch: ::

    git clone https://github.com/<your_github_username>/fitgrid --single-branch --branch new-feature


----------------------
Each time for new work
----------------------

#. Ensure the commit history of the `new-feature` branch in your
   GitHub fork and local ``fitgrid`` repositories are both up to date with
   the upstream branch `kutaslab/fitgrid/new-feature`. This helps
   reduce risk of merge conflicts later when changes are pulled back
   into the upstream repository.

#. It is generally advisable to develop-and-test on the local computer
   in an active conda environment populated with the latest compatible
   ``fitgrid`` dependencies along with :std:doc:`pytest
   <pytest:index>`, the :std:doc:`black <black:index>` code formatter, and
   sphinx documentation generation packages because that's what the
   continuous integration workflow does. For example, a fresh conda
   environment for developing a new fitgrid feature or fix can be
   created and activated like so: ::
  
    mamba create --name fg-new-feature fitgrid \
        -c kutaslab/label/pre-release -c ejolly -c conda-forge \
        --strict-channel-priority
    mamba install --name fg-new-feature \
        black pytest \
	sphinx sphinx-gallery sphinx_rtd_theme \
	-c conda-forge --strict-channel-priority
    conda activate fg-new-feature

#. Install fitgrid from source in editable (a.k.a "develop") mode **in
   the active conda environment** from the top-level directory of the
   local fitgrid repository, like so: ::

     git checkout new-feature
     pip install --no-deps -e .

   This way the just-created `fg-new-feature` conda environment
   provides the required fitgrid dependencies and the modified source
   files in your local repo are loaded when ``fitgrid`` modules are
   imported.

   .. note::

      Experience indicates this is the **only** time ``pip install``
      should be used while developing ``fitgrid`` on pain of
      corrupting the conda environment.


**Develop-and-test**

#. Activate the `fg-new-feature` development environment.

#. Make the changes to the source code and write pytests
   to cover any new functionality in the local source repo
   `fitgrid/tests/test_*.py`.

#. Navigate to the top level of the ``fitgrid`` repository, run the code formatter and
   pytests. ::

     black -S --line-length 79 .
     pytest

#. When pytests pass, generate the documentation and verify that the
   sphinx-gallery ``*.py`` Python examples in the subdirectories
   of `fitgrid/docs/source/gallery` in the local source repository all
   run with the expected results and figures::

    make -C docs clean; make -C docs html

#. When pytests pass and documentation builds locally, commit the
   changes on branch `fg-feature` and push the working branch to your
   forked ``fitgrid`` repository on GitHub.

#. Sign in to GitHub, navigate to the fork's Action tab and verify
   that the push on branch `fg-feature` triggered a github CI Action
   workflow that runs without error.

#. If the workflow fails, inspect the Action log, diagnose the
   problem, fix it in the local repo, commit the changes, and push them
   to the forked repository.

#. When the CI workflow for branch `fg-feature` passes in the forked
   repository, make a pull request back to the upstream working
   branch `kutaslab/fitgrid/fg-feature`.


====   
Tips
====

* Run ``conda list fitgrid`` to confirm it is correctly installed from
  source in development mode in the active environment. The result
  should look something like this:
  
  .. code-block:: bash

     (fg-new-feature) userid@machine$ conda list fitgrid
     # packages in environment at /home/userid/miniconda3/envs/fg-new-feature:
     #
     # Name                    Version                   Build  Channel
     fitgrid                   0.5.1.dev5                dev_0    <develop>


  Check that Version matches the version string in
  `fitgrid/__init__.py` in your local source and the conda Channel
  is `<develop>`.

* If you plan to use :std:doc:`Jupyter <jupyter:index>` or
  :std:doc:`JupyterLab <jupyterlab:index>` to develop code or
  documentation examples things may go more smoothly if you conda or
  mamba install the package directly into the conda enviroment where
  you are developing fitgrid.

* If working in a Jupyter notebook, you can use
  :py:func:`importlib.reload` to load modified source code between
  kernel restarts.

* You can rebuild the .rst documentation quickly without running the sphinx-gallery 
  Python examples by running this command in the top-level repository directory: ::

    make -C docs html-noexec 
