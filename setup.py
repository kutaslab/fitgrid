# to install in development mode, from this directory run
#
#     pip install --user -e .
#
# to install stable package systemwide, as root run
#
#     pip install .
#
# http://python-packaging.readthedocs.io/en/latest/minimal.html

import re
from setuptools import find_packages, setup


# from fitgrid.version import __version__
def get_ver():
    with open("./fitgrid/__init__.py", "r") as stream:
        fg_ver = re.search(
            r".*__version__.*=.*[\"\'](?P<ver_str>\d+\.\d+\.\d+\S*)[\'\"].*",
            stream.read(),
        )

    if fg_ver is None:
        msg = f"""
        fitgrid.__init__.py must have an X.Y.Z semantic version, e.g.,

        __version__ = '0.0.0'
        __version__ = '0.0.0-dev.0.0'

        """
        raise ValueError(msg)
    else:
        return fg_ver['ver_str']


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='fitgrid',
    version=get_ver(),
    description='Mass multiple regression manager',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='Thomas P. Urbach, Andrey Portnoy',
    author_email='turbach@ucsd.edu, aportnoy@ucsd.edu',
    url='https://github.com/kutaslab/fitgrid',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
    ],
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'patsy',
        'statsmodels',
        'matplotlib',
        'scipy',
        'tqdm',
        'tables',
    ],
)
