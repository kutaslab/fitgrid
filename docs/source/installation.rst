.. _installation:

============
Installation
============

**TL;DR** Use `conda install` not `pip` and for real work, run
`fitgrid` on a fast multicore x64_86 linux server with gobs of RAM.


-----------------------------
`conda install` (recommended)
-----------------------------

`fitgrid` is packaged and distributed on `Anaconda Cloud
<https://anaconda.org/kutaslab/fitgrid>`_ for installation into conda
virtual environments using the `conda package manager
<https://conda.io>`_. The virtual environment isolates the `fitgrid`
installation to prevent clashes with what is already installed on your
system and the `conda` installer automatically searches for compatible
versions of the hundreds of Python and R packages `fitgrid` requires
to run. The steps are simple but before proceeding it is prudent to
have a general understanding of conda virtual environments and at
least these commands: ``conda create ...``, ``conda activate ...``,
``conda deactivate``, ``conda install -c ....``. See the `Conda Cheat
Sheet
<https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html>`_
for a summary.

This example shows how to install `fitgrid` from a bash shell command
prompt in a linux terminal window.  It assumes the `conda` executable
is already installed (we use `miniconda3
<https://docs.conda.io/en/latest/miniconda.html>`_). It bakes a date
into the name and provisions the new environment with `fitgrid` and a
compatible `jupyterlab`. The order of the `-c` conda channels matters,
the channels are searched for compatible packages from left to right.

.. _conda_install_fitgrid:

**Example installation**

.. code-block:: bash

    $ conda create --name fg_012021 -c kutaslab -c defaults -c conda-forge fitgrid jupyterlab
    $ conda activate fg_012021

.. note::

   To avoid Python and R package version conflicts, experience teaches
   it is helpful to start with a fresh conda environment if possible,
   and install `fitgrid` along with any other conda packages you want
   to run in one fell swoop. The conda installer cannot see into the
   future. If you install packages one by one, the versions you get
   will be compatible with what you have already installed but may not 
   be compatible with what you want to install next.
 

-------------
`pip install`
-------------

This is asking for trouble because Python packaging doesn't know or
care about the R dependencies. The `fitgrid` source may be downloaded
or git cloned from https://github.com/kutaslab/fitgrid and we upload
stable releases of the Python package to PyPI (`here
<https://pypi.org/project/fitgrid/>`_) as a courtesy, it is not
intended for general use. 

-------------------
System requirements
-------------------

The platform of choice is linux. Minimum system requirements are not
known but obviously large scale regression modeling with millions of
data points is computationally demanding. We develop, test, and use
`fitgrid` on a high-performance linux server (CentOS 7, Intel x86_64,
48 cores, 500GB RAM) and our continuous integration runs all but the
parallel processing pytests on Ubuntu 18.04, 7GB RAM. The 64-bit OSX
conda package is spot-checked from time to time on an
Intel MacBook Pro (8 cores, 32GB RAM). The pytests pass when conda is
installed as above, useability for modeling at scale is unknown. We
don't test the 64-bit Windows conda package, field reports from
contributors are welcome.
