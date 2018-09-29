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
    name='fitgrid',
    version='0.1.1',
    description='Mass multiple regression manager',
    author='Thomas P. Urbach, Andrey Portnoy',
    author_email='turbach@ucsd.edu, aportnoy@ucsd.edu',
    packages=find_packages(),
    install_requires=['patsy', 'statsmodels', 'tqdm', 'tables'],
)
