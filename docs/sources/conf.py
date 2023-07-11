# *****************************************************************************
# Copyright (c) 2022, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

# coding: utf-8
# Configuration file for the Sphinx documentation builder.

import numba_dpex

# -- Project information -----------------------------------------------------

project = "numba-dpex"
copyright = "2020-2023, Intel Corporation"
author = "Intel Corporation"

# The full version, including alpha/beta/rc tags
# release = "main"

# -- General configuration ----------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.programoutput",
    "sphinxcontrib.googleanalytics",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/IntelPython/numba-dpex",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Gitter",
            "url": "https://app.gitter.im/#/room/#Data-Parallel-Python_community:gitter.im",
            "icon": "fab fa-brands fa-gitter",
        },
    ],
    "logo_only": True,
}

googleanalytics_id = "G-LGGL0NJK6P"
googleanalytics_enabled = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

html_sidebars = {
    # "**": [
    #     "globaltoc.html",
    #     "sourcelink.html",
    #     "searchbox.html",
    #     "relations.html",
    # ],
}

html_show_sourcelink = False

# -- Todo extension configuration  ----------------------------------------------
todo_include_todos = True
todo_link_only = True

# -- InterSphinx configuration: looks for objects in external projects -----
# Add here external classes you want to link from Intel SDC documentation
# Each entry of the dictionary has the following format:
#      'class name': ('link to object.inv file for that class', None)
# intersphinx_mapping = {
#    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
#    'python': ('http://docs.python.org/2', None),
#    'numpy': ('http://docs.scipy.org/doc/numpy', None)
# }
intersphinx_mapping = {}

# -- Napoleon extension configuration (Numpy and Google docstring options) -------
# napoleon_google_docstring = True
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = True
# napoleon_include_private_with_doc = True
# napoleon_include_special_with_doc = True
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = True

# -- Prepend module name to an object name or not -----------------------------------
add_module_names = False
