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

"""Tests for metrax base metrics."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import keras
import metrax
from metrax import base_metrics
import numpy as np

np.random.seed(42)
BATCHES = 4
BATCH_SIZE = 8
OUTPUT = np.random.uniform(size=(BATCHES, BATCH_SIZE))
OUTPUT_F16 = OUTPUT.astype(jnp.float16)
OUTPUT_F32 = OUTPUT.astype(jnp.float32)
OUTPUT_BF16 = OUTPUT.astype(jnp.bfloat16)
OUTPUT_BS1 = np.random.uniform(size=(BATCHES, 1)).astype(jnp.float32)
SAMPLE_WEIGHTS = np.tile(
    [0.5, 1, 0, 0, 0, 0, 0, 0],
    (BATCHES, 1),
)


class BaseMetricsTest(parameterized.TestCase):

  def test_basic_division(self):
    x = jnp.array([10.0, 20.0, 30.0])
    y = jnp.array([2.0, 4.0, 5.0])
    expected = jnp.array([5.0, 5.0, 6.0])
    result = base_metrics.divide_no_nan(x, y)
    self.assertTrue(jnp.array_equal(result, expected))

  def test_division_by_zero(self):
    x = jnp.array([10.0, 20.0, 30.0])
    y = jnp.array([2.0, 0.0, 5.0])
    expected = jnp.array([5.0, 0.0, 6.0])
    result = base_metrics.divide_no_nan(x, y)
    self.assertTrue(jnp.array_equal(result, expected))

  def test_all_zeros_denominator(self):
    x = jnp.array([10.0, 20.0, 30.0])
    y = jnp.array([0.0, 0.0, 0.0])
    expected = jnp.array([0.0, 0.0, 0.0])
    result = base_metrics.divide_no_nan(x, y)
    self.assertTrue(jnp.array_equal(result, expected))

  def test_all_zeros_numerator(self):
    x = jnp.array([0.0, 0.0, 0.0])
    y = jnp.array([2.0, 4.0, 5.0])
    expected = jnp.array([0.0, 0.0, 0.0])
    result = base_metrics.divide_no_nan(x, y)
    self.assertTrue(jnp.array_equal(result, expected))

  def test_mixed_zeros(self):
    x = jnp.array([10.0, 0.0, 30.0, 0.0])
    y = jnp.array([2.0, 0.0, 5.0, 4.0])
    expected = jnp.array([5.0, 0.0, 6.0, 0.0])
    result = base_metrics.divide_no_nan(x, y)
    self.assertTrue(jnp.array_equal(result, expected))

  def test_scalar_inputs(self):
    x = jnp.array(10.0)
    y = jnp.array(2.0)
    expected = jnp.array(5.0)
    result = base_metrics.divide_no_nan(x, y)
    self.assertTrue(jnp.array_equal(result, expected))

  def test_scalar_denominator_zero(self):
    x = jnp.array(10.0)
    y = jnp.array(0.0)
    expected = jnp.array(0.0)
    result = base_metrics.divide_no_nan(x, y)
    self.assertTrue(jnp.array_equal(result, expected))

  def test_negative_values(self):
    x = jnp.array([-10.0, 20.0, -30.0])
    y = jnp.array([2.0, -4.0, 5.0])
    expected = jnp.array([-5.0, -5.0, -6.0])
    result = base_metrics.divide_no_nan(x, y)
    self.assertTrue(jnp.array_equal(result, expected))

  def test_negative_and_zero_values(self):
    x = jnp.array([-10.0, 20.0, -30.0, 10.0])
    y = jnp.array([2.0, -4.0, 0.0, 0.0])
    expected = jnp.array([-5.0, -5.0, 0.0, 0.0])
    result = base_metrics.divide_no_nan(x, y)
    self.assertTrue(jnp.array_equal(result, expected))

  @parameterized.named_parameters(
      ('basic_f16', OUTPUT_F16, None),
      ('basic_f32', OUTPUT_F32, None),
      ('basic_bf16', OUTPUT_BF16, None),
      ('batch_size_one', OUTPUT_BS1, None),
      ('weighted_f16', OUTPUT_F16, SAMPLE_WEIGHTS),
      ('weighted_f32', OUTPUT_F32, SAMPLE_WEIGHTS),
      ('weighted_bf16', OUTPUT_BF16, SAMPLE_WEIGHTS),
  )
  def test_average(self, values, sample_weights):
    """Test that `Average` metric computes correct values."""
    if sample_weights is None:
      sample_weights = jnp.ones_like(values)
    sample_weights = jnp.array(sample_weights, dtype=values.dtype)
    metric = metrax.Average.from_model_output(
        values=values,
        sample_weights=sample_weights,
    )

    keras_mean = keras.metrics.Mean(dtype=values.dtype)
    keras_mean.update_state(values, sample_weights)
    keras_metrics = keras_mean.result()
    keras_metrics = jnp.array(keras_metrics, dtype=values.dtype)

    # Use lower tolerance for lower precision dtypes.
    rtol = 1e-2 if values.dtype in (jnp.float16, jnp.bfloat16) else 1e-05
    atol = 1e-2 if values.dtype in (jnp.float16, jnp.bfloat16) else 1e-05
    np.testing.assert_allclose(
        metric.compute(),
        keras_metrics,
        rtol=rtol,
        atol=atol,
    )


if __name__ == '__main__':
  absltest.main()
