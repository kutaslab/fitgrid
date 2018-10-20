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

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='fitgrid',
    version='0.1.2',
    description='Mass multiple regression manager',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='Thomas P. Urbach, Andrey Portnoy',
    author_email='turbach@ucsd.edu, aportnoy@ucsd.edu',
    url='https://github.com/kutaslab/fitgrid',
    license='BSD-3-Clause',
    packages=find_packages(exclude=['tests']),
    install_requires=['patsy', 'statsmodels', 'tqdm', 'tables'],
)
