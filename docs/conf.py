"""Sphinx configuration for the TBR documentation."""

from __future__ import annotations

import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# Project information
project = "TBR"
author = "Ido Hirsh"
copyright = f"{datetime.now().year}, {author}"

try:
    release = package_version("tbr")
except PackageNotFoundError:
    release = "0.1.6.dev0"

version = ".".join(release.split(".")[:2])


# General configuration
extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# MyST Markdown support
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]
myst_heading_anchors = 3


# Bibliography
bibtex_bibfiles = ["references.bib"]


# HTML output
html_theme = "pydata_sphinx_theme"
html_title = "TBR Documentation"

html_theme_options = {
    "github_url": "https://github.com/idohi/tbr",
    "show_toc_level": 2,
}


# API documentation
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

autodoc_typehints = "description"
autodoc_typehints_format = "short"

# Keep `# doctest: +SKIP` markers out of the rendered examples (Sphinx default,
# pinned explicitly because the published examples rely on it).
trim_doctest_flags = True

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True


# Cross-project references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
}
