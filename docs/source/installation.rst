.. _installation:

############
Installation
############

**TL;DR** Use `mamba` or `conda` to install the fitgrid conda package
along with your other packages into a fresh conda environment on a
fast multicore x64_86 computer with gobs of RAM.


==================
conda environments
==================

`fitgrid` is packaged on `anaconda.org
<https://anaconda.org/kutaslab/fitgrid>`_ for installation into conda
"virtual" environments using the (good) `conda <https://conda.io>`_ or
(better) `mamba <https://mamba.readthedocs.io/en/latest/>`_ package
manager. The virtual environment isolates the `fitgrid` installation
to prevent clashes with what is already installed on your system.  The
`conda` package installer automatically populates the environment with
compatible versions of the hundreds of Python and R packages `fitgrid`
requires to run. The `mamba` package installer does the same thing but
much faster and, in some cases, more reliably.

The steps for creating conda environments and installing fitgrid are
simple but it is prudent to first have a general understanding of
conda virtual environments and at least these commands: ``conda create
...``, ``conda activate ...``, ``conda deactivate``, ``conda
install -c ....``. See the `Conda Cheat Sheet
<https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html>`_
for a summary. For fine tuning and troubleshooting environments, it is
useful to have a grasp of conda channels and channel priority and the
difference between the default conda channels `main` and `r`
commercially maintained by Anaconda and the parallel universe
of the `conda-forge` channel maintained by the open-source community.


.. _conda_install_fitgrid:


================================
`mamba` install (best practices)
================================

These examples show how to install `fitgrid` from a bash shell command
prompt in a linux or Mac terminal window.  They assume the `conda`
executable is already installed (we use `miniconda3
<https://docs.conda.io/en/latest/miniconda.html>`_ for development and
testing) and `mamba` has been installed into the base conda
environment with ``conda install --name base mamba``. The
environment names may be chosen freely, those shown are for
illustration only.

The `--strict-channel-priority` option is the recommended
configuration for working in the `conda-forge
<https://conda-forge.org>`_ ecosystem.  Command line options can be
also be configured in the .condarc `configuration file
<https://docs.conda.io/projects/conda/en/master/user-guide/configuration/use-condarc.html#using-the-condarc-conda-configuration-file>`_,
these examples assume the default .condarc file has not been modified.

.. note::

   The commands shown here are broken into separate lines here for
   readability. If you do this, make sure the \\ is the last character on each line.
   Alternatively you can enter the command as a single line without any \\.

Stable release
--------------

This is a typical installation of the latest stable release of fitgrid
into a fresh conda environment. This pattern
is most likely to be compatible with recent versions of other conda
packages on linux and (Intel) Mac OSX.

.. code-block:: bash

   $ mamba create --name fg_012021 \
       fitgrid \
       -c kutaslab -c ejolly -c conda-forge \
       --strict-channel-priority


Development version
-------------------

At times, the development version of fitgrid runs ahead of the latest
stable release and includes bug fixes and new features. The
development version may be installed by overriding the default
`kutaslab` conda channel with `kutaslab/label/pre-release` like so:

.. code-block:: bash

   $ mamba create --name fg_012021 \
       fitgrid \
       -c kutaslab/label/pre-release -c ejolly -c conda-forge \
       --strict-channel-priority



Selecting a Python version
--------------------------

Specific versions of Python and other packages can be selected for
installation with the conda package specification syntax. This example
installs fitgrid with the most recent version of Python 3.8.

.. code-block:: bash

   $ mamba create --name fg_012021 \
       fitgrid python=3.8 \
       -c kutaslab -c ejolly -c conda-forge \
       --strict-channel-priority

       

Selecting MKL vs. openblas
--------------------------

On Intel CPUs, the Math Kernel Library (MKL) builds of optimized math
libraries like the Basic Linear Algebra Subprograms (BLAS) may offer a
substantial performance advantage over OpenBLAS. For AMD CPUs OpenBLAS
may outperform MKL. This example shows how to enforce installation of
the MKL build and use `conda list` to inspect the installed packages.
It is readily adapted for OpenBLAS by replacing `mkl` with
`openblas`.

.. code-block:: bash

   $ mamba create --name fg_012021 \
       fitgrid "blas=*=mkl*" \
       -c kutaslab -c ejolly -c conda-forge \
       --strict-channel-priority
   $ activate fg_012021
   $ conda list | egrep "(mkl|blas|liblapack)"
   # packages in environment at /home/userid/miniconda3/envs/fg_012021:
   blas                      2.109                       mkl    conda-forge
   blas-devel                3.9.0                     9_mkl    conda-forge
   libblas                   3.9.0                     9_mkl    conda-forge
   libcblas                  3.9.0                     9_mkl    conda-forge
   liblapack                 3.9.0                     9_mkl    conda-forge
   liblapacke                3.9.0                     9_mkl    conda-forge
   mkl                       2021.2.0           h06a4308_296  
   mkl-devel                 2021.2.0           h66538d2_296  
   mkl-include               2021.2.0           h06a4308_296  



Install fitgrid and run Example Gallery notebooks
-------------------------------------------------
   
To run the notebooks in the :ref:`gallery` install fitgrid and jupyter
lab or jupyter and launch like so:

.. code-block:: bash

   $ mamba create --name fg_012021 \
       fitgrid jupyterlab \
       -c kutaslab -c ejolly -c conda-forge \
       --strict-channel-priority
   $ conda activate fg_012021
   $ jupyter lab


Prioritize conda default channels over conda-forge
--------------------------------------------------

This example shows how to install fitgrid into an environment
populated primarily with the stale-but-stable packages from the
Anaconda default channels. The explicit conda-forge channel is
necessary here because not all dependencies are available on these
default conda channels. Strict channel priority may cause problems and
is omitted by design.

.. code-block:: bash

   $ mamba create --name fg_012021 \
       fitgrid \
       -c kutaslab -c ejolly -c defaults -c conda-forge

       
==============================
`conda` install (if necessary)
==============================

If mamba is not available, replace `mamba` in the examples above with
`conda`. The conda dependency solver is much slower than
mamba and may take anywhere from a few to tens of minutes to create
the environment. In rare cases the `conda` installer fails where the
`mamba` installer succeeds.

.. code-block:: bash

   $ conda create --name fg_012021 \
       fitgrid "blas=*=mkl*" \
       -c kutaslab -c ejolly -c conda-forge \
       --strict-channel-priority


   
======================================
`pip` and source install (for experts)
======================================

Installing fitgrid with pip is asking for trouble because Python
packaging doesn't know or care about the many R dependencies. We
upload stable releases of the Python package to PyPI (`here
<https://pypi.org/project/fitgrid/>`_) as a courtesy, it is not
intended for general use. If you are working without conda
environments and thinking about `pip install` you might consider
cloning the github repository https://github.com/kutaslab/fitgrid and
pip installing from source. The cloned repo will include the pytests
to run for checking that the installed package behaves as expected.


===================
System requirements
===================

The platform of choice is linux. Minimum system requirements are not
known but obviously large scale regression modeling with millions of
data points is computationally demanding. Current versions of fitgrid
are developed and used in Ubuntu 20.04 running on a high-performance
multicore server with Intel CPUs (72 cores/144 threads, 1TB RAM);
continuous integrations tests run on ubuntu-latest and macos-10.15 on
github Actions `hosted runners
<https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources>`_.
Previous versions of `fitgrid` were developed and used in CentOS 7
with Intel CPUs (24 cores/48 threads, 256-512 GB RAM). We are unable
to test the Windows 64-bit conda package, field reports are welcome.

===============
Troubleshooting
===============

* Use `conda list` to inspect package versions and the channels they come
  from when constructing conda enviroments.

* `mamba create` and `mamba install` are not exact drop in
  replacements for `conda create` and `conda install` because the
  conda installer has an affinity for packages on default conda
  channels and mamba has an affinity for packages on conda-forge and
  they may resolve dependencies differently.

* What works and what doesn't when creating conda environments and
  installing packages depends greatly on the *combinations* of
  packages you wish to install. Not all combinations of platforms,
  Python versions, installers, channel priority, and packages are
  compatible.

* To avoid package version conflicts and speed up the dependency solver
  it can be useful to specify the Python version and install `fitgrid`
  along with the other conda packages you want into a fresh
  environment in one fell swoop. The package installers cannot see
  into the future. If packages are installed one by one, the next
  package version you want may not be compatible with what is already
  in the environment.

* Depending on your computer hardware, you may see a significant
  performance difference between the Intel MKL and OpenBLAS builds of
  the Basic Linear Algebra Support (BLAS) and Linear Algebra Package
  (LAPACK) libraries, particularly for fitting mixed-effects models.


