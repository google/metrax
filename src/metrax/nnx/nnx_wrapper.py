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

"""A wrapper for metrax metrics to be used with NNX."""

from flax import nnx


class NnxWrapper(nnx.metrics.Metric):
  """A wrapper class for clu metrics to be used with NNX."""

  def __init__(self, cls):
    self.clu_metric = cls.empty()

  def reset(self) -> None:
    self.clu_metric = self.clu_metric.empty()

  def update(self, **kwargs) -> None:
    other_clu_metric = self.clu_metric.from_model_output(**kwargs)
    self.clu_metric = self.clu_metric.merge(other_clu_metric)

  def compute(self):
    return self.clu_metric.compute()
