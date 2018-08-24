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
    version='0.1',
    description='rERP runner',
    author='Andrey Portnoy',
    author_email='aportnoy@ucsd.edu',
    packages=find_packages(),
    install_requires=[
        'patsy',
        'statsmodels',
        'tqdm',
        'tables'
    ]
)
