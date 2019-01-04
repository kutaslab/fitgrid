************
Installation
************

The recommended way to install `fitgrid` is by using the `Conda package
manager <https://conda.io>`_. We assume you already have it installed.

A good idea is to start in a fresh environment. `fitgrid` depends on several
Python and R libraries. You don't want them to clash with your existing
installation.

To create a new environment, in your shell run:

.. code-block:: console

    conda create -n fitgrid

Now activate the environment:

.. code-block:: console

    conda activate fitgrid

This creates an empty environment called `fitgrid`. To install `fitgrid`, run:

.. code-block:: console

    conda install -c aportnoy fitgrid

The installation might take several minutes, but when it's done you are ready
to go.
