"""To install fitgrid in development mode see
kutaslab.github.io/fitgrid-dev-docs/contributing.html
"""

import re
from setuptools import find_packages, setup


# from fitgrid.version import __version__
def get_ver():
    """format check"""

    with open("./fitgrid/__init__.py", "r") as stream:
        fg_ver = re.search(
            r".*__version__.*=.*[\"\'](?P<ver_str>\d+\.\d+\.\d+\S*)[\'\"].*",
            stream.read(),
        )

    if fg_ver is None:
        msg = """
        fitgrid.__init__.py must have an X.Y.Z semantic version, e.g.,
        __version__ = '0.0.0'
        __version__ = '0.0.0-dev.0.0'
        """
        raise ValueError(msg)

    return fg_ver['ver_str']


def readme():
    """slurp text"""

    with open('README.md') as strm:
        return strm.read()


setup(
    name='fitgrid',
    version=get_ver(),
    description='Mass multiple regression manager',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='Andrey Portnoy, Thomas P. Urbach',
    author_email='aportnoy@ucsd.edu, turbach@ucsd.edu',
    url='https://github.com/kutaslab/fitgrid',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
    ],
    packages=find_packages(exclude=['tests']),
)
