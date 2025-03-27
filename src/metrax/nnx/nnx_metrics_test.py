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

import dataclasses
import inspect

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import metrax
import metrax.nnx


class NnxMetricsTest(parameterized.TestCase):

  def test_nnx_metrics_exists(self):
    """Tests that every metrax CLU metric has an NNX counterpart."""
    metrax_metric_keys = [
        key for key, metric in inspect.getmembers(metrax)
        if dataclasses.is_dataclass(metric)
    ]
    metrax_nnx_metric_keys = [
        key for key, metric in inspect.getmembers(metrax.nnx)
        if inspect.isclass(metric) and issubclass(metric, nnx.Metric)
    ]
    self.assertGreater(len(metrax_metric_keys), 0)
    self.assertSameElements(metrax_metric_keys, metrax_nnx_metric_keys)


if __name__ == '__main__':
  absltest.main()
