# Copyright 2026 Google LLC
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

"""Tests that Metrax doc Python snippets execute with expected results."""

from collections.abc import Mapping, Sequence
import dataclasses
import os
import pathlib
import re
import textwrap
from typing import Any

from absl.testing import absltest
import numpy as np


@dataclasses.dataclass(frozen=True)
class Snippet:
  """A Python code snippet extracted from a documentation file."""

  file_path: pathlib.Path
  line_number: int
  code: str


def _resolve_doc_files() -> list[tuple[str, pathlib.Path]]:
  """Resolves doc file paths across Blaze, local google3, and OSS layouts."""
  test_srcdir = os.environ.get("TEST_SRCDIR", "")
  if test_srcdir:
    base_dir = (
        pathlib.Path(test_srcdir) / "google3" / "third_party" / "py" / "metrax"
    )
    return [
        ("README.md", base_dir / "oss" / "README.md"),
    ]

  this_dir = pathlib.Path(__file__).resolve().parent
  if (this_dir / "oss" / "README.md").exists():
    return [
        ("README.md", this_dir / "oss" / "README.md"),
    ]

  oss_root = this_dir.parents[1]
  return [
      ("README.md", oss_root / "README.md"),
  ]


def _extract_markdown_snippets(file_path: pathlib.Path) -> list[Snippet]:
  """Extracts fenced Python blocks from a Markdown file."""
  snippets = []
  lines = file_path.read_text(encoding="utf-8").splitlines()
  in_block = False
  block_lines: list[str] = []
  start_line = 0

  for idx, line in enumerate(lines, 1):
    stripped = line.strip()
    if not in_block:
      if stripped.startswith("```python") or stripped.startswith("```py"):
        in_block = True
        block_lines = []
        start_line = idx + 1
    elif stripped == "```":
      in_block = False
      snippets.append(
          Snippet(
              file_path=file_path,
              line_number=start_line,
              code=textwrap.dedent("\n".join(block_lines)),
          )
      )
    else:
      block_lines.append(line)
  return snippets


_EXPECTED_SCALARS: Mapping[tuple[str, str], float] = {
    ("README.md", "accuracy"): 0.625,
    ("README.md", "nnx_accuracy"): 0.625,
}

_EXPECTED_DICT_SCALARS: Mapping[tuple[str, str], Mapping[str, float]] = {}

_EXPECTED_SNIPPET_COUNTS: Mapping[str, int] = {
    "README.md": 2,
}


class SnippetTest(absltest.TestCase):
  """Verifies doc snippets execute on CPU and produce expected values."""

  def _assert_snippet_outcomes(
      self, rel_name: str, snippet: Snippet, env: dict[str, Any]
  ) -> None:
    """Asserts expected output variables created by `snippet`."""
    for (doc_name, var_name), expected in _EXPECTED_SCALARS.items():
      pattern = rf"(?m)^{re.escape(var_name)}\s*="
      if doc_name == rel_name and re.search(pattern, snippet.code):
        self.assertIn(var_name, env)
        np.testing.assert_allclose(env[var_name], expected, rtol=1e-5)

    for (doc_name, var_name), expected_map in _EXPECTED_DICT_SCALARS.items():
      pattern = rf"(?m)^{re.escape(var_name)}\s*="
      if doc_name == rel_name and re.search(pattern, snippet.code):
        self.assertIn(var_name, env)
        for key, expected in expected_map.items():
          self.assertIn(key, env[var_name])
          np.testing.assert_allclose(env[var_name][key], expected, rtol=1e-5)

  def test_all_doc_snippets(self):
    doc_files = _resolve_doc_files()
    counts: dict[str, int] = {}

    for rel_name, file_path in doc_files:
      self.assertTrue(file_path.exists(), f"Missing doc file: {file_path}")
      snippets: Sequence[Snippet] = _extract_markdown_snippets(file_path)
      counts[rel_name] = len(snippets)

      for snippet in snippets:
        with self.subTest(snippet=f"{rel_name}:{snippet.line_number}"):
          env: dict[str, Any] = {}
          exec(snippet.code, env)  # pylint: disable=exec-used
          self._assert_snippet_outcomes(rel_name, snippet, env)

    self.assertEqual(counts, _EXPECTED_SNIPPET_COUNTS)


if __name__ == "__main__":
  absltest.main()
