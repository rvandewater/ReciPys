import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))  # Add your project root to Python path
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ReciPies'
copyright = '2025, Robin P. van de Water'
author = 'Robin P. van de Water'
release = ''

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Theme configuration
html_theme = 'sphinx_immaterial'
# Theme options
html_theme_options = {
    # Toc options
}


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
    'sphinx_immaterial',
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Generate autosummary even if no references
autosummary_generate = True

# Optionally, switch to a more stable theme if issues persist:
# html_theme = 'sphinx_rtd_theme'
