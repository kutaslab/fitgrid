# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('.'))

# after sys.path.insert to import local __version__
from fitgrid import __version__

# -- Project information -----------------------------------------------------

project = 'fitgrid'
copyright = '2018-2021, Andrey S. Portnoy, Thomas P. Urbach'
author = 'Andrey S. Portnoy, Thomas P. Urbach'
today = 'January 29, 2021'

# The short X.Y version. TPU: no, use full version for M.N.P(.devX) in sidebar
version = __version__

# The full version, including alpha/beta/rc tags
# release = ""

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.viewcode',
    'sphinx.ext.extlinks',
    # 'nbsphinx',
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.intersphinx",
]

# nbsphinx_timeout = -1 # don't timeout
# nbsphinx_timeout = 5 * 60  # lmer needs time to run in Tutorial

napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

# ------------------------------------------------------------
# intersphinx config
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("https://matplotlib.org", None),
    "statsmodels": ("https://www.statsmodels.org/stable", None),
    "pymer4": ("https://eshinjolly.com/pymer4", None),
    "rpy2": ("https://rpy2.github.io/doc/latest/html", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "sphinx-gallery": ("https://sphinx-gallery.github.io/stable", None),
    "conda": ("https://docs.conda.io/projects/conda/en/latest/", None),
    "pip": ("https://pip.pypa.io/en/stable", None),
    "mamba": ("https://mamba.readthedocs.io/en/latest", None),
    "black": ("https://black.readthedocs.io/en/stable", None),
    "pytest": ("https://docs.pytest.org/en/stable", None),
    "patsy": ("https://patsy.readthedocs.io/en/latest", None),
    "jupyter": ("https://jupyter.readthedocs.io/en/latest/", None),
    "jupyterlab": ("https://jupyterlab.readthedocs.io/en/stable/", None),
    "fitgrid": ("https://kutaslab.github.io/fitgrid", None),
    "fitgrid-pre-release": (
        "https://kutaslab.github.io/fitgrid-dev-docs",
        None,
    ),
}

# ------------------------------------------------------------
# sphinx gallery config TPU
from sphinx_gallery.scrapers import matplotlib_scraper


class fg_matplotlib_scraper(object):
    def __repr__(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        # bbox_inches="tight" or .png gallery images have excessive margins
        return matplotlib_scraper(*args, bbox_inches="tight", **kwargs)


sphinx_gallery_conf = {
    # source dirs
    "examples_dirs": [
        # "user_guide",
        "gallery",
        "gallery/1_epochs_data",
        "gallery/2_model_fitting",
        "gallery/3_model_evaluation",
    ],
    # generated output dirs
    "gallery_dirs": [
        # "auto_user_guide",
        "auto_gallery",
        "auto_gallery/1_epochs_data",
        "auto_gallery/2_model_fitting",
        "auto_gallery/3_model_evaluation",
    ],
    # execute all *.py (not default plot_*.py)
    "filename_pattern": "[a-zA-Z]+.py",
    # or ... select files individally
    # "filename_pattern": ".*/workflow.py",
    "image_scrapers": (fg_matplotlib_scraper(),),
    # or set to https://kutaslab.github.io/fitgrid/ to reduce notebook size
    "notebook_images": True,
}
# ------------------------------------------------------------


# alias long urls to keep line length under control
extlinks = {"sm_docs": ("https://www.statsmodels.org/stable/generated/%s", "")}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ['build', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_copy_source = False
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = False

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'fitgriddoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        'fitgrid.tex',
        'fitgrid Documentation',
        'Andrey Portnoy and Thomas P. Urbach',
        'manual',
    )
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, 'fitgrid', 'fitgrid Documentation', [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        'fitgrid',
        'fitgrid Documentation',
        author,
        'fitgrid',
        'One line description of project.',
        'Miscellaneous',
    )
]


# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
