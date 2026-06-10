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
    'show_toc_level': 2,
    'repository_url': 'https://github.com/google/metrax',
    'use_repository_button': True,
    'navigation_with_keys': False,
    'show_navbar_depth': 2,
}
html_static_path = ['static']
html_logo = 'static/metrax_logo.png'
html_css_files = [
    'custom.css',
]

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


def _generate_rst_files() -> None:
  """Automates creation of Sphinx RST files for Metrax metrics."""
  import importlib  # pylint: disable=import-outside-toplevel
  import metrax  # pylint: disable=import-outside-toplevel

  mods = {}
  for name in metrax.__all__:
    mod_name = getattr(metrax, name).__module__
    mods.setdefault(mod_name, []).append(name)

  docs_dir = os.path.dirname(os.path.abspath(__file__))
  cats = []

  for mod_name, metrics in mods.items():
    cat = mod_name.split('.')[-1].replace('_metrics', '')
    cats.append(cat)
    title = cat.upper() if len(cat) <= 3 else cat.capitalize()
    mod = sys.modules.get(mod_name)
    if not mod:
      mod = importlib.import_module(mod_name)
    doc = getattr(mod, '__doc__', '') or ''
    desc = doc.strip().split('\n\n')[0] if doc else ''
    items = '\n'.join(f'   ~{m}' for m in sorted(metrics))

    content = (
        f'{title}\n{"=" * len(title)}\n\n'
        '.. currentmodule:: metrax\n\n'
        f'{desc}\n\n'
        '.. autosummary::\n'
        '   :toctree: api/\n'
        '   :template: autosummary/class.rst\n\n'
        f'{items}\n'
    )
    with open(
        os.path.join(docs_dir, f'{cat}.rst'), 'w', encoding='utf-8'
    ) as f:
      f.write(content)

  if cats:
    toc = '\n'.join(
        f'   {c}'
        for c in sorted(cats, key=lambda x: (0 if x == 'base' else 1, x))
    )
    intro = (
        'Metrax provides high-performance, JAX-native evaluation metrics'
        ' organized into specialized categories. Select a category below to'
        ' explore the available metrics and their API reference:'
    )
    metrax_rst = (
        f'Metrax Metrics\n==============\n\n{intro}\n\n'
        f'.. toctree::\n   :maxdepth: 2\n\n{toc}\n'
    )
    with open(
        os.path.join(docs_dir, 'metrax.rst'), 'w', encoding='utf-8'
    ) as f:
      f.write(metrax_rst)


_generate_rst_files()
