# to install in development mode, from this directory run
#
#     python ./setup.py build_ext --inplace
#     python ./setup.py develop -d ~/.local/lib/python3.6/site-packages/
#
#  to install stable package
#
#    pip install .
# 
# http://python-packaging.readthedocs.io/en/latest/minimal.html

# from distutils.core import setup
# from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
from setuptools import  find_packages, setup, Extension

setup(
    name='eegr',
    version = '0.1',
    description='regression ERP',
    author='Tom Urbach, Lauren Liao',
    author_email='turbach@ucsd.edu',
    url='http://kutaslab.ucsd.edu/people/urbach',
    packages=find_packages(), 
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension("_eegr",
                  ["_eegr.pyx"],
                  include_dirs=[np.get_include()],
              ), ],
)

