# to install in development mode, from this directory run
# 
#     pip install --user -e .
#
# to install stable package systemwide, as root run
#
#     pip install .
# 
# http://python-packaging.readthedocs.io/en/latest/minimal.html

from setuptools import find_packages, setup

setup(
    name='eegr',
    version = '0.1',
    description='regression ERP',
    author='Tom Urbach, Lauren Liao, Andrey Portnoy',
    author_email='turbach@ucsd.edu',
    url='http://kutaslab.ucsd.edu/people/urbach',
    packages=find_packages()
)

