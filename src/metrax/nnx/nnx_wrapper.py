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
import inspect


class NnxWrapper(nnx.metrics.Metric):
  """A wrapper class for clu metrics to be used with NNX."""

  def __init__(self, cls):
    self.clu_metric = cls.empty()

  def reset(self) -> None:
    self.clu_metric = self.clu_metric.empty()

  def update(self, **kwargs) -> None:
    # Filter kwargs to only those accepted by from_model_output
    sig = inspect.signature(self.clu_metric.from_model_output)
    params = sig.parameters
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_keyword:
      # Method accepts **kwargs, pass everything
      filtered_kwargs = kwargs
    else:
      accepted = {
        name
        for name, p in params.items()
        if p.kind
        in (
          inspect.Parameter.POSITIONAL_OR_KEYWORD,
          inspect.Parameter.KEYWORD_ONLY,
        )
      }
      filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    other_clu_metric = self.clu_metric.from_model_output(**filtered_kwargs)
    self.clu_metric = self.clu_metric.merge(other_clu_metric)

  def compute(self):
    return self.clu_metric.compute()

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(pytree=False, **kwargs)
