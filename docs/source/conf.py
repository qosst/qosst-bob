# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

from qosst_bob import __version__

# -- Project information -----------------------------------------------------

project = "qosst_bob"
copyright = "2021-2024, Yoann Piétri, Valentina Marulanda Acosta, Ilektra Karakosta-Amarantidou, Matteo Schiavon"
author = "Yoann Piétri, Valentina Marulanda Acosta, Ilektra Karakosta-Amarantidou, Matteo Schiavon"

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx-prompt",
    "sphinx_argparse_cli",
    "sphinxcontrib.programoutput",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

root_doc = "index"
autoclass_content = "both"
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "qosst": ("https://qosst.readthedocs.io/en/latest/", None),
    "qosst-core": ("https://qosst-core.readthedocs.io/en/latest/", None),
    "qosst-hal": ("https://qosst-hal.readthedocs.io/en/latest/", None),
    "qosst-alice": ("https://qosst-alice.readthedocs.io/en/latest/", None),
    "qosst-skr": ("https://qosst-skr.readthedocs.io/en/latest/", None),
    "qosst-sim": ("https://qosst-sim.readthedocs.io/en/latest/", None),
}

html_logo = "_static/qosst_logo_square_white.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}

latex_documents = [
    (
        root_doc,
        "qosst-bob.tex",
        "qosst-bob",
        author.replace(", ", "\\and ").replace(" and ", "\\and and "),
        "manual",
    ),
]

latex_elements = {
    "preamble": r"""
\DeclareUnicodeCharacter{2588}{~}
\DeclareUnicodeCharacter{2557}{~}
\DeclareUnicodeCharacter{2554}{~}
\DeclareUnicodeCharacter{2550}{~}
\DeclareUnicodeCharacter{255D}{~}
\DeclareUnicodeCharacter{255A}{~}
\DeclareUnicodeCharacter{2551}{~}
\DeclareUnicodeCharacter{2584}{~}
\DeclareUnicodeCharacter{2580}{~}
"""
}
