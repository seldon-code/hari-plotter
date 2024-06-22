# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from sphinx.ext.apidoc import main as sphinx_apidoc_main

# -- Project information -----------------------------------------------------
project = 'Hari Plotter'
copyright = '2024, Amrita Goswami, Rohit Goswami, Moritz Sallermann, Ivan Tambovtsev'
author = 'Amrita Goswami, Rohit Goswami, Moritz Sallermann, Ivan Tambovtsev'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.coverage',
    'recommonmark',  # Handles markdown files
    'nbsphinx',  # Handles Jupyter notebooks
    'sphinx.ext.mathjax',
]

autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../hari_plotter'))

# -- Autogenerate API documentation ------------------------------------------


# If you want to allow errors in notebooks to not stop the build
nbsphinx_allow_errors = True


def run_apidoc(_):
    source_dir = os.path.abspath('../hari_plotter')
    output_dir = os.path.join(os.path.abspath('.'), 'source')
    apidoc_args = [
        '--force',
        '--separate',
        '--output-dir', output_dir,
        source_dir
    ]
    sphinx_apidoc_main(apidoc_args)


def setup(app):
    app.connect('builder-inited', run_apidoc)
