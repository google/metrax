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

"""A collection of base metrics for metrax."""

from clu import metrics as clu_metrics
import flax
import jax
import jax.numpy as jnp


def divide_no_nan(x: jax.Array, y: jax.Array) -> jax.Array:
  """Computes a safe divide which returns 0 if the y is zero."""
  return jnp.where(y != 0, jnp.divide(x, y), 0.0)


@flax.struct.dataclass
class Average(clu_metrics.Average):
  r"""Average Metric inherits clu.metrics.Average and performs safe division."""

  @classmethod
  def from_model_output(
      cls,
      values: jax.Array,
      sample_weights: jax.Array | None = None,
  ) -> 'Average':
    """Updates the metric.

    Args:
      values: A floating point 1D vector representing the values. The shape
      should be (batch_size,).
      sample_weights: An optional floating point 1D vector representing the
        weight of each sample. The shape should be (batch_size,).

    Returns:
      Updated Average metric.
    """
    total = values
    count = jnp.ones_like(values, dtype=values.dtype)
    if sample_weights is not None:
      total = values * sample_weights
      count = count * sample_weights
    return cls(
        total=total.sum(),
        count=count.sum(),
    )

  def compute(self) -> jax.Array:
    return divide_no_nan(self.total, self.count)
