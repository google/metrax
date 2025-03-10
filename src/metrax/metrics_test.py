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

"""Tests for metrax metrics."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import keras
import keras_hub
import metrax
import numpy as np
import os
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
    0, 2, size=(BATCHES, 1),
).astype(np.float32)
OUTPUT_PREDS_BS1 = np.random.uniform(size=(BATCHES, 1)).astype(np.float32)
SAMPLE_WEIGHTS = np.tile(
    [0.5, 1, 0, 0, 0, 0, 0, 0],
    (BATCHES, 1),
).astype(np.float32)


class MetricsTest(parameterized.TestCase):

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

  def test_precision_empty(self):
    """Tests the `empty` method of the `Precision` class."""
    m = metrax.Precision.empty()
    self.assertEqual(m.true_positives, jnp.array(0, jnp.float32))
    self.assertEqual(m.false_positives, jnp.array(0, jnp.float32))

  def test_recall_empty(self):
    """Tests the `empty` method of the `Recall` class."""
    m = metrax.Recall.empty()
    self.assertEqual(m.true_positives, jnp.array(0, jnp.float32))
    self.assertEqual(m.false_negatives, jnp.array(0, jnp.float32))

  def test_aucpr_empty(self):
    """Tests the `empty` method of the `AUCPR` class."""
    m = metrax.AUCPR.empty()
    self.assertEqual(m.true_positives, jnp.array(0, jnp.float32))
    self.assertEqual(m.false_positives, jnp.array(0, jnp.float32))
    self.assertEqual(m.false_negatives, jnp.array(0, jnp.float32))
    self.assertEqual(m.num_thresholds, 0)

  def test_aucroc_empty(self):
    """Tests the `empty` method of the `AUCROC` class."""
    m = metrax.AUCROC.empty()
    self.assertEqual(m.true_positives, jnp.array(0, jnp.float32))
    self.assertEqual(m.true_negatives, jnp.array(0, jnp.float32))
    self.assertEqual(m.false_positives, jnp.array(0, jnp.float32))
    self.assertEqual(m.false_negatives, jnp.array(0, jnp.float32))
    self.assertEqual(m.num_thresholds, 0)

  def test_perplexity_empty(self):
    """Tests the `empty` method of the `Perplexity` class."""
    m = metrax.Perplexity.empty()
    self.assertEqual(m.aggregate_crossentropy, jnp.array(0, jnp.float32))
    self.assertEqual(m.num_samples, jnp.array(0, jnp.float32))

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

  @parameterized.named_parameters(
      ('basic_f16', OUTPUT_LABELS, OUTPUT_PREDS_F16, 0.5),
      ('high_threshold_f16', OUTPUT_LABELS, OUTPUT_PREDS_F16, 0.7),
      ('low_threshold_f16', OUTPUT_LABELS, OUTPUT_PREDS_F16, 0.1),
      ('basic_f32', OUTPUT_LABELS, OUTPUT_PREDS_F32, 0.5),
      ('high_threshold_f32', OUTPUT_LABELS, OUTPUT_PREDS_F32, 0.7),
      ('low_threshold_f32', OUTPUT_LABELS, OUTPUT_PREDS_F32, 0.1),
      ('basic_bf16', OUTPUT_LABELS, OUTPUT_PREDS_BF16, 0.5),
      ('high_threshold_bf16', OUTPUT_LABELS, OUTPUT_PREDS_BF16, 0.7),
      ('low_threshold_bf16', OUTPUT_LABELS, OUTPUT_PREDS_BF16, 0.1),
      ('batch_size_one', OUTPUT_LABELS_BS1, OUTPUT_PREDS_BS1, 0.5),
  )
  def test_precision(self, y_true, y_pred, threshold):
    """Test that `Precision` metric computes correct values."""
    y_true = y_true.reshape((-1,))
    y_pred = jnp.where(y_pred.reshape((-1,)) >= threshold, 1, 0)
    keras_precision = keras.metrics.Precision(thresholds=threshold)
    keras_precision.update_state(y_true, y_pred)
    expected = keras_precision.result()

    metric = None
    for logits, labels in zip(y_pred, y_true):
      update = metrax.Precision.from_model_output(
          predictions=logits,
          labels=labels,
          threshold=threshold,
      )
      metric = update if metric is None else metric.merge(update)

    # Use lower tolerance for lower precision dtypes.
    rtol = 1e-2 if y_true.dtype in (jnp.float16, jnp.bfloat16) else 1e-5
    atol = 1e-2 if y_true.dtype in (jnp.float16, jnp.bfloat16) else 1e-5
    np.testing.assert_allclose(
        metric.compute(),
        expected,
        rtol=rtol,
        atol=atol,
    )

  @parameterized.named_parameters(
      ('basic', OUTPUT_LABELS, OUTPUT_PREDS, 0.5),
      ('high_threshold', OUTPUT_LABELS, OUTPUT_PREDS, 0.7),
      ('low_threshold', OUTPUT_LABELS, OUTPUT_PREDS, 0.1),
      ('basic_f32', OUTPUT_LABELS, OUTPUT_PREDS_F32, 0.5),
      ('high_threshold_f32', OUTPUT_LABELS, OUTPUT_PREDS_F32, 0.7),
      ('low_threshold_f32', OUTPUT_LABELS, OUTPUT_PREDS_F32, 0.1),
      ('basic_bf16', OUTPUT_LABELS, OUTPUT_PREDS_BF16, 0.5),
      ('high_threshold_bf16', OUTPUT_LABELS, OUTPUT_PREDS_BF16, 0.7),
      ('low_threshold_bf16', OUTPUT_LABELS, OUTPUT_PREDS_BF16, 0.1),
      ('batch_size_one', OUTPUT_LABELS_BS1, OUTPUT_PREDS_BS1, 0.5),
  )
  def test_recall(self, y_true, y_pred, threshold):
    """Test that `Recall` metric computes correct values."""
    y_true = y_true.reshape((-1,))
    y_pred = jnp.where(y_pred.reshape((-1,)) >= threshold, 1, 0)
    keras_recall = keras.metrics.Recall(thresholds=threshold)
    keras_recall.update_state(y_true, y_pred)
    expected = keras_recall.result()

    metric = None
    for logits, labels in zip(y_pred, y_true):
      update = metrax.Recall.from_model_output(
          predictions=logits,
          labels=labels,
          threshold=threshold,
      )
      metric = update if metric is None else metric.merge(update)

    np.testing.assert_allclose(
        metric.compute(),
        expected,
    )

  @parameterized.product(
      inputs=(
          (OUTPUT_LABELS, OUTPUT_PREDS, None),
          (OUTPUT_LABELS_BS1, OUTPUT_PREDS_BS1, None),
          (OUTPUT_LABELS, OUTPUT_PREDS, SAMPLE_WEIGHTS),
      ),
      dtype=(
        jnp.float16,
        jnp.float32,
        jnp.bfloat16,
      ),
  )
  def test_aucpr(self, inputs, dtype):
    """Test that `AUC-PR` Metric computes correct values."""
    y_true, y_pred, sample_weights = inputs
    y_true = y_true.astype(dtype)
    y_pred = y_pred.astype(dtype)
    if sample_weights is None:
      sample_weights = np.ones_like(y_true)

    metric = None
    for labels, logits, weights in zip(y_true, y_pred, sample_weights):
      update = metrax.AUCPR.from_model_output(
          predictions=logits,
          labels=labels,
          sample_weights=weights,
      )
      metric = update if metric is None else metric.merge(update)

    keras_aucpr = keras.metrics.AUC(curve='PR')
    for labels, logits, weights in zip(y_true, y_pred, sample_weights):
      keras_aucpr.update_state(labels, logits, sample_weight=weights)
    expected = keras_aucpr.result()
    np.testing.assert_allclose(
        metric.compute(),
        expected,
        # Use lower tolerance for lower precision dtypes.
        rtol=1e-2 if y_true.dtype in (jnp.float16, jnp.bfloat16) else 1e-5,
        atol=1e-2 if y_true.dtype in (jnp.float16, jnp.bfloat16) else 1e-5,
    )

  @parameterized.product(
      inputs=(
          (OUTPUT_LABELS, OUTPUT_PREDS, None),
          (OUTPUT_LABELS_BS1, OUTPUT_PREDS_BS1, None),
          (OUTPUT_LABELS, OUTPUT_PREDS, SAMPLE_WEIGHTS),
      ),
      dtype=(
        jnp.float16,
        jnp.float32,
        jnp.bfloat16,
      ),
  )
  def test_aucroc(self, inputs, dtype):
    """Test that `AUC-ROC` Metric computes correct values."""
    y_true, y_pred, sample_weights = inputs
    y_true = y_true.astype(dtype)
    y_pred = y_pred.astype(dtype)
    if sample_weights is None:
      sample_weights = np.ones_like(y_true)

    metric = None
    for labels, logits, weights in zip(y_true, y_pred, sample_weights):
      update = metrax.AUCROC.from_model_output(
          predictions=logits,
          labels=labels,
          sample_weights=weights,
      )
      metric = update if metric is None else metric.merge(update)

    keras_aucroc = keras.metrics.AUC(curve='ROC')
    for labels, logits, weights in zip(y_true, y_pred, sample_weights):
      keras_aucroc.update_state(labels, logits, sample_weight=weights)
    expected = keras_aucroc.result()
    np.testing.assert_allclose(
        metric.compute(),
        expected,
        # Use lower tolerance for lower precision dtypes.
        rtol=1e-2 if y_true.dtype in (jnp.float16, jnp.bfloat16) else 1e-7,
        atol=1e-2 if y_true.dtype in (jnp.float16, jnp.bfloat16) else 1e-7,
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

  @parameterized.named_parameters(
      (
          'basic',
          np.random.randint(10, size=[2, 5, 10]),
          np.random.uniform(size=(2, 5, 10, 20)),
          None,
      ),
      (
          'weighted',
          np.random.randint(10, size=[2, 5, 10]),
          np.random.uniform(size=(2, 5, 10, 20)),
          np.random.randint(2, size=(2, 5, 10)).astype(np.float32),
      ),
  )
  def test_perplexity(self, y_true, y_pred, sample_weights):
    """Test that `Perplexity` Metric computes correct values."""
    keras_metric = keras_hub.metrics.Perplexity()
    metrax_metric = None
    for index, (labels, logits) in enumerate(zip(y_true, y_pred)):
      weights = sample_weights[index] if sample_weights is not None else None
      keras_metric.update_state(labels, logits, sample_weight=weights)
      update = metrax.Perplexity.from_model_output(
          predictions=logits,
          labels=labels,
          sample_weights=weights,
      )
      metrax_metric = update if metrax_metric is None else metrax_metric.merge(
          update
      )

    expected = keras_metric.result()
    np.testing.assert_allclose(
        metrax_metric.compute(),
        expected,
        rtol=1e-05,
        atol=1e-05,
    )


if __name__ == '__main__':
  os.environ['XLA_FLAGS'] = (
      '--xla_force_host_platform_device_count=4'  # Use 4 CPU devices
  )
  absltest.main()
