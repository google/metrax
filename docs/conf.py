# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration file for the Sphinx documentation builder."""

import os
import sys

# Import local version of metrax.
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information

project = 'metrax'
copyright = '2025, The metrax Authors'
author = 'The metrax Authors'

release = ''
version = ''


# -- General configuration

extensions = [
    'myst_nb',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/google/metrax",
    "use_repository_button": True,
    "navigation_with_keys": False,
    "show_navbar_depth": 2,
}

# -- Options for EPUB output
epub_show_urls = 'footnote'


# -- Extension configuration

autodoc_member_order = 'bysource'

autodoc_default_options = {
    'members': None,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__call__, __init__',
}

autosummary_generate = True