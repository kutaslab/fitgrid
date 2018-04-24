# to install in development mode, from this directory run
# 
#     pip uninstall eegr
#     python ./setup.py build_ext --inplace
#     python ./setup.py develop -d ~/.local/lib/python3.6/site-packages/
#
#  to install stable package, as root run 
#
#    pip install .
# 
# http://python-packaging.readthedocs.io/en/latest/minimal.html
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from setuptools import  find_packages, setup, Extension
import numpy as np

# from Cython.Distutils import build_ext
# import numpy as np
# from setuptools import  find_packages, setup, Extension

extensions =  [
    Extension("eegr._eegr",
              ["eegr/_eegr.pyx"],
              include_dirs=[np.get_include()],
          )
]

setup(
    name='eegr',
    version = '0.1',
    description='regression ERP',
    author='Tom Urbach, Lauren Liao',
    author_email='turbach@ucsd.edu',
    url='http://kutaslab.ucsd.edu/people/urbach',
    packages=find_packages(), 
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(extensions)
,
)

