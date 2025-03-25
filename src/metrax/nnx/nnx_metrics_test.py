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

"""Tests for metrax NNX metrics."""

import importlib
import inspect
import pkgutil
from absl.testing import absltest
from absl.testing import parameterized
from clu import metrics as clu_metrics
import metrax
import metrax.nnx


class NnxMetricsTest(parameterized.TestCase):

  def test_nnx_metrics_exists(self):
    """Tests that every metrax CLU metric has an NNX counterpart."""
    metrax_metrics = metrax.nnx.__all__
    for _, module_name, _ in pkgutil.iter_modules(metrax.__path__):
      full_module_name = f"{metrax.__name__}.{module_name}"
      module = importlib.import_module(full_module_name)
      for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, clu_metrics.Metric):
          self.assertIn(name, metrax_metrics)


if __name__ == '__main__':
  absltest.main()
