This fitgrid/binder folder isolates the binder config files from the
rest of the repo, in particular environment.yml

The environment.yml must have what is necessary to run fitgrid but not
be so specific that it doesn't run on the binder platform.

To get binder working ahead of a stable M.N.P release, make sure that
the dev branch with M.N.P pushed fitgrid M.N.P (no devX) to
kutaslab/fitgrid/pre-release.

Then ...

1. To capture a usable environment.yml this has worked

Install a stable or pre-release version of fitgrid with conda and defaults priority into a
Python 3.7 conda environment, activate the environment and dump it like so

  conda create -n fg_binder python=3.7 fitgrid -c kutaslab -c defaults -c conda-forge -c ejolly -y
  activate fg_binder
  conda env export -f environment.yml 

Alternatively, in a Python 3.7 environment with mambma and conda-forge priority 

  conda create -n fg_binder python=3.7 "r-base<4.0" fitgrid -c kutaslab/label/pre-release -c ejolly -c conda-forge -y
  activate fg_binder
  conda env export -f environment.yml

If the current stable release dependencies are too far behind to run the binder notebook replace 

   -c kutaslab

with 

   -c kutaslab/label/pre-release


2. In fitgrid/README.md set the binder repo branch to whatever you
plan to push

     * a working branch for testing
     * branch dev to pre-flight a stable release
     * branch main when ready for a github tagged stable release




