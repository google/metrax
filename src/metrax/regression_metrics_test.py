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

"""Tests for metrax regression metrics."""

import os
os.environ['KERAS_BACKEND'] = 'jax'

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import keras
import metrax
import numpy as np
from sklearn import metrics as sklearn_metrics

np.random.seed(42)
BATCHES = 4
BATCH_SIZE = 8
OUTPUT_LABELS = np.random.randint(
    0,
    2,
    size=(BATCHES, BATCH_SIZE),
).astype(np.float32)
OUTPUT_PREDS = np.random.uniform(size=(BATCHES, BATCH_SIZE))
OUTPUT_PREDS_F16 = OUTPUT_PREDS.astype(jnp.float16)
OUTPUT_PREDS_F32 = OUTPUT_PREDS.astype(jnp.float32)
OUTPUT_PREDS_BF16 = OUTPUT_PREDS.astype(jnp.bfloat16)
OUTPUT_LABELS_BS1 = np.random.randint(
    0,
    2,
    size=(BATCHES, 1),
).astype(np.float32)
OUTPUT_PREDS_BS1 = np.random.uniform(size=(BATCHES, 1)).astype(np.float32)
SAMPLE_WEIGHTS = np.tile(
    [0.5, 1, 0, 0, 0, 0, 0, 0],
    (BATCHES, 1),
).astype(np.float32)


class RegressionMetricsTest(parameterized.TestCase):

  def test_multiple_devices(self):
    """Test that metrax metrics work across multiple devices using R2 as an example."""

    def create_r2(logits, labels):
      """Creates a metrax RSQUARED metric given logits and labels."""
      return metrax.RSQUARED.from_model_output(logits, labels)

    def sharded_r2(logits, labels):
      """Calculates sharded R2 across devices."""
      num_devices = jax.device_count()

      shard_size = logits.shape[0] // num_devices
      sharded_logits = logits.reshape(num_devices, shard_size, logits.shape[-1])
      sharded_labels = labels.reshape(num_devices, shard_size, labels.shape[-1])

      r2_for_devices = jax.pmap(create_r2)(sharded_logits, sharded_labels)
      return r2_for_devices

    y_pred = OUTPUT_PREDS
    y_true = OUTPUT_LABELS
    metric = jax.jit(sharded_r2)(y_pred, y_true)
    metric = metric.reduce()

    keras_r2 = keras.metrics.R2Score()
    for labels, logits in zip(y_true, y_pred):
      keras_r2.update_state(
          labels[:, jnp.newaxis],
          logits[:, jnp.newaxis],
      )
    expected = keras_r2.result()
    np.testing.assert_allclose(
        metric.compute(),
        expected,
        rtol=1e-05,
        atol=1e-05,
    )

  def test_mse_empty(self):
    """Tests the `empty` method of the `MSE` class."""
    m = metrax.MSE.empty()
    self.assertEqual(m.total, jnp.array(0, jnp.float32))
    self.assertEqual(m.count, jnp.array(0, jnp.int32))

  def test_rmse_empty(self):
    """Tests the `empty` method of the `RMSE` class."""
    m = metrax.RMSE.empty()
    self.assertEqual(m.total, jnp.array(0, jnp.float32))
    self.assertEqual(m.count, jnp.array(0, jnp.int32))

  def test_rsquared_empty(self):
    """Tests the `empty` method of the `RSQUARED` class."""
    m = metrax.RSQUARED.empty()
    self.assertEqual(m.total, jnp.array(0, jnp.float32))
    self.assertEqual(m.count, jnp.array(0, jnp.float32))
    self.assertEqual(m.sum_of_squared_error, jnp.array(0, jnp.float32))
    self.assertEqual(m.sum_of_squared_label, jnp.array(0, jnp.float32))

  @parameterized.named_parameters(
      ('basic_f16', OUTPUT_LABELS, OUTPUT_PREDS_F16, None),
      ('basic_f32', OUTPUT_LABELS, OUTPUT_PREDS_F32, None),
      ('basic_bf16', OUTPUT_LABELS, OUTPUT_PREDS_BF16, None),
      ('batch_size_one', OUTPUT_LABELS_BS1, OUTPUT_PREDS_BS1, None),
      ('weighted_f16', OUTPUT_LABELS, OUTPUT_PREDS_F16, SAMPLE_WEIGHTS),
      ('weighted_f32', OUTPUT_LABELS, OUTPUT_PREDS_F32, SAMPLE_WEIGHTS),
      ('weighted_bf16', OUTPUT_LABELS, OUTPUT_PREDS_BF16, SAMPLE_WEIGHTS),
  )
  def test_mse(self, y_true, y_pred, sample_weights):
    """Test that `MSE` Metric computes correct values."""
    y_true = y_true.astype(y_pred.dtype)
    y_pred = y_pred.astype(y_true.dtype)
    if sample_weights is None:
      sample_weights = np.ones_like(y_true)

    metric = None
    for labels, logits, weights in zip(y_true, y_pred, sample_weights):
      update = metrax.MSE.from_model_output(
          predictions=logits,
          labels=labels,
          sample_weights=weights,
      )
      metric = update if metric is None else metric.merge(update)

    # TODO(jiwonshin): Use `keras.metrics.MeanSquaredError` once it supports
    # sample weights.
    expected = sklearn_metrics.mean_squared_error(
        y_true.flatten(),
        y_pred.flatten(),
        sample_weight=sample_weights.flatten(),
    )
    # Use lower tolerance for lower precision dtypes.
    rtol = 1e-2 if y_true.dtype in (jnp.float16, jnp.bfloat16) else 1e-05
    atol = 1e-2 if y_true.dtype in (jnp.float16, jnp.bfloat16) else 1e-05
    np.testing.assert_allclose(
        metric.compute(),
        expected,
        rtol=rtol,
        atol=atol,
    )

  @parameterized.named_parameters(
      ('basic_f16', OUTPUT_LABELS, OUTPUT_PREDS_F16, None),
      ('basic_f32', OUTPUT_LABELS, OUTPUT_PREDS_F32, None),
      ('basic_bf16', OUTPUT_LABELS, OUTPUT_PREDS_BF16, None),
      ('batch_size_one', OUTPUT_LABELS_BS1, OUTPUT_PREDS_BS1, None),
      ('weighted_f16', OUTPUT_LABELS, OUTPUT_PREDS_F16, SAMPLE_WEIGHTS),
      ('weighted_f32', OUTPUT_LABELS, OUTPUT_PREDS_F32, SAMPLE_WEIGHTS),
      ('weighted_bf16', OUTPUT_LABELS, OUTPUT_PREDS_BF16, SAMPLE_WEIGHTS),
  )
  def test_rmse(self, y_true, y_pred, sample_weights):
    """Test that `RMSE` Metric computes correct values."""
    y_true = y_true.astype(y_pred.dtype)
    y_pred = y_pred.astype(y_true.dtype)
    if sample_weights is None:
      sample_weights = np.ones_like(y_true)

    metric = None
    keras_rmse = keras.metrics.RootMeanSquaredError()
    for labels, logits, weights in zip(y_true, y_pred, sample_weights):
      update = metrax.RMSE.from_model_output(
          predictions=logits,
          labels=labels,
          sample_weights=weights,
      )
      metric = update if metric is None else metric.merge(update)
      keras_rmse.update_state(labels, logits, sample_weight=weights)

    # Use lower tolerance for lower precision dtypes.
    rtol = 1e-2 if y_true.dtype in (jnp.float16, jnp.bfloat16) else 1e-05
    atol = 1e-2 if y_true.dtype in (jnp.float16, jnp.bfloat16) else 1e-05
    np.testing.assert_allclose(
        metric.compute(),
        keras_rmse.result(),
        rtol=rtol,
        atol=atol,
    )

  @parameterized.named_parameters(
      ('basic_f16', OUTPUT_LABELS, OUTPUT_PREDS_F16, None),
      ('basic_f32', OUTPUT_LABELS, OUTPUT_PREDS_F32, None),
      ('basic_bf16', OUTPUT_LABELS, OUTPUT_PREDS_BF16, None),
      ('batch_size_one', OUTPUT_LABELS_BS1, OUTPUT_PREDS_BS1, None),
      ('weighted_f16', OUTPUT_LABELS, OUTPUT_PREDS_F16, SAMPLE_WEIGHTS),
      ('weighted_f32', OUTPUT_LABELS, OUTPUT_PREDS_F32, SAMPLE_WEIGHTS),
      ('weighted_bf16', OUTPUT_LABELS, OUTPUT_PREDS_BF16, SAMPLE_WEIGHTS),
  )
  def test_rsquared(self, y_true, y_pred, sample_weights):
    """Test that `RSQUARED` Metric computes correct values."""
    y_true = y_true.astype(y_pred.dtype)
    y_pred = y_pred.astype(y_true.dtype)
    if sample_weights is None:
      sample_weights = np.ones_like(y_true)

    metric = None
    keras_r2 = keras.metrics.R2Score()
    for labels, logits, weights in zip(y_true, y_pred, sample_weights):
      update = metrax.RSQUARED.from_model_output(
          predictions=logits,
          labels=labels,
          sample_weights=weights,
      )
      metric = update if metric is None else metric.merge(update)

      keras_r2.update_state(
          labels[:, jnp.newaxis],
          logits[:, jnp.newaxis],
          sample_weight=weights[:, jnp.newaxis],
      )

    # Use lower tolerance for lower precision dtypes.
    rtol = 1e-2 if y_true.dtype in (jnp.float16, jnp.bfloat16) else 1e-05
    atol = 1e-2 if y_true.dtype in (jnp.float16, jnp.bfloat16) else 1e-05
    np.testing.assert_allclose(
        metric.compute(),
        keras_r2.result(),
        rtol=rtol,
        atol=atol,
    )


if __name__ == '__main__':
  os.environ['XLA_FLAGS'] = (
      '--xla_force_host_platform_device_count=4'  # Use 4 CPU devices
  )
  absltest.main()
