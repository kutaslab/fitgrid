.. _installation:

############
Installation
############

**TL;DR** Use :std:doc:`mamba <mamba:index>` or :std:doc:`conda
<conda:index>` to install the ``fitgrid`` conda package along with your
other packages into a fresh conda environment on a fast multicore
x64_86 computer with gobs of RAM.


================================
About conda virtual environments
================================

``fitgrid`` is packaged on `anaconda.org/kutaslab/fitgrid
<https://anaconda.org/kutaslab/fitgrid>`_ for installation into conda
"virtual" environments using the `conda <https://conda.io>`_ or
:std:doc:`mamba <mamba:index>` package manager. A virtual environment
isolates the ``fitgrid`` installation to prevent clashes with what is
already installed elsewhere in your system and other virtual
environments. When the package manager installs ``fitgrid`` in a
virtual environment it also automatically installs compatible versions
of the hundreds of Python and R packages ``fitgrid`` requires to run
including :std:doc:`numpy <numpy:index>`, :std:doc:`pandas
<pandas:index>`, :std:doc:`matplotlib <matplotlib:index>`,
:std:doc:`statsmodels <statsmodels:index>`, and :std:doc:`pymer4
<pymer4:index>`, :std:doc:`rpy2 <rpy2:index>`, `R
<https://www.r-project.org/other-docs.html>`_, `lme4
<https://cran.r-project.org/web/packages/lme4/index.html>`_, and
`lmerTest
<https://cran.r-project.org/web/packages/lmerTest/index.html>`_ to
name a few. You can also install other conda packages in addition to
``fitgrid`` as needed for the task at hand.

The steps for creating conda environments and installing ``fitgrid``
are straightforward but it is prudent to have a general understanding
of conda virtual environments and at least these commands: ``conda
create ...``, ``conda install -c ...``, ``conda activate ...``, and
``conda deactivate``. See the `Conda Cheat Sheet
<https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html>`_
for a summary. For fine-tuning conda environments and working around
incompatible package versions refer to the core `conda tasks
<https://conda.io/projects/conda/en/latest/user-guide/tasks/index.html>`_
especially `managing conda channels and channel priority
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html>`_
and `installing packages
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-packages>`_.
The :std:doc:`mamba <mamba:index>` package installer is an alternative
to ``conda``. At present, the ``mamba create ...`` and ``mamba install
...`` commands resolve the complex ``fitgrid`` package dependencies
substantially faster than ``conda``.

For working with ``fitgrid`` and mixed Python and R conda environments
generally, it is important to attend to the difference between the
``conda`` default channels `anaconda.org/main
<https://anaconda.org.main>`_ and `anaconda.org/r
<https://anaconda.org/r>`_ where packages are maintained by the
Anaconda, Inc. team and the not-always-compatible parallel universe of
the `conda-forge <https://conda-forge.org/>`_ channel where many of
the same-named conda packages are maintained by the open-source
community. Choosing suitable channels for installing conda packages
can be tricky because the specific versions of packages required for
compatibility and best performance depend on the computing hardware,
operating systems, compilers, and the version requirements of packages
that are already in the environment or will be. The conda-forge
maintainers recommend setting the .condarc `configuration file
<https://docs.conda.io/projects/conda/en/master/user-guide/configuration/use-condarc.html#using-the-condarc-conda-configuration-file>`_
to strict conda-forge channel priority.  However, revising the default
channel priority may not be appropriate for all users, in which case
the command-line options ``--channel conda-forge
--strict-channel-priority`` may be used. The examples below illustrate
command line options for a few common installation scenarios
encountered in practice.

Our current recommended best practice for working with conda
environments is to install the lightweight `miniconda3
<https://docs.conda.io/en/latest/miniconda.html>`_ and then avoid
polluting the "base" conda environment with data analysis and
application packages like ``fitgrid``.  Instead, create separate new
working environments, each populated with the packages needed for a
given project. The :std:doc:`mamba <mamba:index>` package is an
exception to this rule. If you elect to use ``mamba`` follow the
`installation instructions
<https://mamba.readthedocs.io/en/latest/installation.html>`_
carefully.


.. _conda_install_fitgrid:

==========================
How to install ``fitgrid``
==========================

These examples show how to install ``fitgrid`` into a new conda
working environment from the conda base environment with a shell
command in a linux or Mac terminal window.  They assume the ``conda``
and ``mamba`` executables are already installed in the base
environment and the users's channel configuration is the minconda3
default shown here:

.. code-block:: bash

   (base) $ which conda mamba
   /home/your_userid/miniconda3/bin/conda
   /home/your_userid/miniconda3/bin/mamba
   (base) $ conda config --show channels default_channels channel_priority
   channels:
     - defaults
   default_channels:
     - https://repo.anaconda.com/pkgs/main
     - https://repo.anaconda.com/pkgs/r
   channel_priority: flexible

.. note::

   The example installation commands are broken into separate lines for
   readability. If you do this, make sure the \\ is the last character on each line.
   Alternatively you can enter the command as a single line without any \\.

~~~~~~~~~~~~~~
with ``mamba``
~~~~~~~~~~~~~~

``fitgrid`` stable release
--------------------------

This is a typical installation of the latest stable release of
``fitgrid`` into a fresh conda environment named ``fg_012021``. This
pattern is likely to be compatible with recent versions of other conda
packages for x86_64 linux platforms and recent Intel Mac OSX.

.. code-block:: bash

   (base) $ mamba create --name fg_012021 \
       -c kutaslab -c ejolly -c conda-forge \
       fitgrid

.. note::

   This installation currently defaults to OpenBLAS builds of matrix
   math and linear algebra libraries so execution time on some Intel
   CPUs may be substantially longer than for the Intel Math
   Kernel (MKL) builds of the libraries. For a workaround see
   :ref:`mkl_v_openblas` below.


``fitgrid`` development version
-------------------------------

At times, the development version of ``fitgrid`` runs ahead of the latest
stable release and includes bug fixes and new features. The
latest development version may be installed by overriding the default
`kutaslab` conda channel with `kutaslab/label/pre-release` like so:

.. code-block:: bash

   (base) $ mamba create --name fg_012021 \
       -c kutaslab/label/pre-release -c ejolly -c conda-forge \
       fitgrid



Selecting a Python version
--------------------------

Specific versions of Python and other packages can be selected for
installation with the conda package specification syntax. This example
installs ``fitgrid`` with the most recent version of Python 3.8.

.. code-block:: bash

   (base) $ mamba create --name fg_012021 \
       -c kutaslab -c ejolly -c conda-forge \
       fitgrid python=3.8



.. _mkl_v_openblas:


       
Selecting MKL or OpenBLAS
-------------------------

On Intel CPUs, the `Intel Math Kernel Library (MKL)
<https://en.wikipedia.org/wiki/Math_Kernel_Library>`_ builds of
optimized math libraries like the Basic Linear Algebra Subprograms
(BLAS) may offer a substantial performance advantage over `OpenBLAS
<https://en.wikipedia.org/wiki/OpenBLAS>`_. For AMD CPUs OpenBLAS may
outperform MKL. This example shows how to enforce installation of the
MKL build and use ``conda list`` to inspect the installed packages.  To
select OpenBLAS builds, replace ``mkl`` with ``openblas`` in the first
command.

.. code-block:: bash

   (base) $ mamba create --name fg_012021 \
       -c kutaslab -c ejolly -c conda-forge \
       fitgrid "blas=*=mkl*"
   (base) $ activate fg_012021
   (fg_012021) $ conda list | egrep "(mkl|blas|liblapack)"
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



Install fitgrid and run Examples Gallery notebooks
--------------------------------------------------
   
To run the notebooks in the :ref:`gallery` install `JupyterLab or
Jupyter <https://jupyter.org/>`_ in the same conda environment as
``fitgrid`` and it launch like so:

.. code-block:: bash

   (base) $ mamba create --name fg_012021 \
       -c kutaslab -c ejolly -c conda-forge \
       fitgrid jupyterlab
   (base) $ conda activate fg_012021
   (fg_012021) $ jupyter lab


Prioritize anaconda.org default channels over conda-forge
---------------------------------------------------------

This example shows how to install fitgrid into an environment
populated primarily with the stale-but-stable packages from the
Anaconda default channels. The explicit ``-c conda-forge`` channel is
necessary here because not all dependencies are available on the
default conda channels. Strict channel priority may cause problems and
is omitted by design.

.. code-block:: bash

   (base) $ mamba create --name fg_012021 \
       -c kutaslab -c ejolly -c defaults -c conda-forge \
       fitgrid


~~~~~~~~~~~~~~
with ``conda``
~~~~~~~~~~~~~~

If mamba is not available, replace ``mamba`` in the examples above
with ``conda``. The ``conda`` dependency solver tends to be slower
than mamba and may take anywhere from a few to tens of minutes to
create the environment. The `conda` and `mamba` dependency resolution
algorithms are not identical and may arrive at different solutions.

.. code-block:: bash

   (base) $ conda create --name fg_012021 \
       -c kutaslab -c ejolly -c conda-forge \
       --strict-channel-priority \
       fitgrid "blas=*=mkl*"


~~~~~~~~~~~~~~~~~~~~~~~~
``pip`` is not supported
~~~~~~~~~~~~~~~~~~~~~~~~

Since ``fitgrid`` requires numerous R packages, installing with the
Python package installer, :std:doc:`pip <pip:index>` is no longer
supported and is not recommended for general use.


===================
System requirements
===================

The platform of choice is linux. Minimum system requirements are not
known but obviously large scale regression modeling with millions of
data points is computationally demanding. Current versions of fitgrid
are developed and used in Ubuntu 20.04 running on a high-performance
multicore server with Intel CPUs (72 cores/144 threads, 1TB RAM);
continuous integrations tests run on ubuntu-latest and macos-10.15 on
GitHub Actions `hosted runners
<https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources>`_.
Previous versions of ``fitgrid`` were developed and used in CentOS 7
with Intel CPUs (24 cores/48 threads, 256-512 GB RAM). We are unable
to test the Windows 64-bit conda package, field reports are welcome,
see :ref:`Contributing <how_to_contribute>` for more information.

====
Tips
====

* Use ``conda list`` to inspect package versions and the channels they come
  from when constructing conda enviroments.

* To help avoid package version conflicts and speed up the dependency
  solver it can be useful to specify the Python version and install
  ``fitgrid`` along with the other conda packages you want into a
  fresh environment in one fell swoop. The package installers cannot
  see into the future. If packages are installed one by one, the next
  package version you want may not be compatible with what is already
  in the environment.

* ``mamba create`` and ``mamba install`` are not exact drop in
  replacements for ``conda create`` and ``conda install`` because
  ``conda`` has an affinity for packages on default conda channels and
  ``mamba`` has an affinity for packages on conda-forge and they may
  resolve dependencies differently.

* What works and what doesn't when creating conda environments and
  installing packages depends greatly on the *combinations* of
  packages you wish to install. Not all combinations of platforms,
  Python versions, installers, channel priority, and packages are
  compatible.

* Depending on your computer hardware, you may see a significant
  performance difference between the Intel MKL and OpenBLAS builds of
  the Basic Linear Algebra Support (BLAS) and Linear Algebra Package
  (LAPACK) libraries, particularly for fitting mixed-effects models.



  
