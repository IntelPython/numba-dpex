# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

# coding: utf-8
# Configuration file for the Sphinx documentation builder.

import numba_dpex

# -- Project information -----------------------------------------------------

project = "numba-dpex"
copyright = "2020-2024, Intel Corporation"
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
    "sphinxcontrib.programoutput",
    "sphinxcontrib.googleanalytics",
    "myst_parser",
    "autoapi.extension",
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
# html_theme = "pydata_sphinx_theme"
html_theme = "furo"

html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/IntelPython/numba-dpex",
            "icon": "fab fa-github-square",
        },
    ],
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

# -- Auto API configurations ---------------------------------------------------

autoapi_dirs = [
    "../../numba_dpex/kernel_api",
    "../../numba_dpex/core",
]
autoapi_type = "python"

autoapi_template_dir = "_templates/autoapi"

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

autoapi_keep_files = True
autodoc_typehints = "signature"

rst_prolog = """
.. role:: summarylabel
"""

html_css_files = [
    "css/custom.css",
]


def contains(seq, item):
    return item in seq


def prepare_jinja_env(jinja_env) -> None:
    jinja_env.tests["contains"] = contains


autoapi_prepare_jinja_env = prepare_jinja_env


def skip_member(app, what, name, obj, skip, options):
    # skip submodules
    if what == "module":
        skip = True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_member)
