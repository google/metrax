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

"""Tests for metrax.metrax."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
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
OUTPUT_PREDS = np.random.uniform(size=(BATCHES, BATCH_SIZE)).astype(np.float32)
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


class MetricsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # TODO(jeffcarp): Merge these into generated fixtures.
    self.model_outputs = (
        dict(
            logits=jnp.array(
                [0.34, 0.89, 0.12, 0.67, 0.98, 0.23, 0.56, 0.71, 0.45, 0.08]
            ),
            labels=jnp.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1]),
        ),
        dict(
            logits=jnp.array(
                [0.23, 0.89, 0.57, 0.11, 0.99, 0.38, 0.76, 0.05, 0.62, 0.44]
            ),
            labels=jnp.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 0]),
        ),
        dict(
            logits=jnp.array(
                [0.67, 0.21, 0.95, 0.03, 0.88, 0.51, 0.34, 0.79, 0.15, 0.42]
            ),
            labels=jnp.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 1]),
        ),
        dict(
            logits=jnp.array(
                [0.91, 0.37, 0.18, 0.75, 0.59, 0.02, 0.83, 0.26, 0.64, 0.48]
            ),
            labels=jnp.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0]),
        ),
    )
    self.model_outputs_batch_size_one = (
        dict(
            logits=jnp.array([[0.32]]),
            labels=jnp.array([1]),
        ),
        dict(
            logits=jnp.array([[0.74]]),
            labels=jnp.array([1]),
        ),
        dict(
            logits=jnp.array([[0.86]]),
            labels=jnp.array([1]),
        ),
        dict(
            logits=jnp.array([[0.21]]),
            labels=jnp.array([1]),
        ),
    )
    self.sample_weights = jnp.array([0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0])

  def compute_aucpr(self, model_outputs, sample_weights=None):
    metric = None
    for model_output in model_outputs:
      update = metrax.AUCPR.from_model_output(
          predictions=model_output.get('logits'),
          labels=model_output.get('labels'),
          sample_weights=sample_weights,
      )
      metric = update if metric is None else metric.merge(update)
    return metric.compute()

  def compute_aucroc(self, model_outputs, sample_weights=None):
    metric = None
    for model_output in model_outputs:
      update = metrax.AUCROC.from_model_output(
          predictions=model_output.get('logits'),
          labels=model_output.get('labels'),
          sample_weights=sample_weights,
      )
      metric = update if metric is None else metric.merge(update)
    return metric.compute()

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

    expected = sklearn_metrics.r2_score(
        y_true.flatten(),
        y_pred.flatten(),
    )
    np.testing.assert_allclose(
        metric.compute(),
        expected,
        rtol=1e-05,
        atol=1e-05,
    )

  @parameterized.named_parameters(
      ('basic', OUTPUT_LABELS, OUTPUT_PREDS, 0.5),
      ('high_threshold', OUTPUT_LABELS, OUTPUT_PREDS, 0.7),
      ('low_threshold', OUTPUT_LABELS, OUTPUT_PREDS, 0.1),
      ('batch_size_one', OUTPUT_LABELS_BS1, OUTPUT_PREDS_BS1, 0.5),
  )
  def test_precision(self, y_true, y_pred, threshold):
    """Test that Precision metric computes correct values."""
    y_true = y_true.reshape((-1,))
    y_pred = jnp.where(y_pred.reshape((-1,)) >= threshold, 1, 0)
    expected = sklearn_metrics.precision_score(y_true, y_pred)

    metric = None
    for logits, labels in zip(y_pred, y_true):
      update = metrax.Precision.from_model_output(
          predictions=logits,
          labels=labels,
          threshold=threshold,
      )
      metric = update if metric is None else metric.merge(update)

    np.testing.assert_allclose(
        metric.compute(),
        expected,
    )

  @parameterized.named_parameters(
      ('basic', OUTPUT_LABELS, OUTPUT_PREDS, 0.5),
      ('high_threshold', OUTPUT_LABELS, OUTPUT_PREDS, 0.7),
      ('low_threshold', OUTPUT_LABELS, OUTPUT_PREDS, 0.1),
      ('batch_size_one', OUTPUT_LABELS_BS1, OUTPUT_PREDS_BS1, 0.5),
  )
  def test_recall(self, y_true, y_pred, threshold):
    """Test that Recall metric computes correct values."""
    y_true = y_true.reshape((-1,))
    y_pred = jnp.where(y_pred.reshape((-1,)) >= threshold, 1, 0)
    expected = sklearn_metrics.recall_score(y_true, y_pred)

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

  def test_aucpr(self):
    """Test that AUC-PR Metric computes correct values."""
    np.testing.assert_allclose(
        self.compute_aucpr(self.model_outputs),
        jnp.array(0.41513795, dtype=jnp.float32),
    )

  def test_aucpr_with_sample_weight(self):
    """Test that AUC-PR Metric computes correct values when using sample weights."""
    np.testing.assert_allclose(
        self.compute_aucpr(self.model_outputs, self.sample_weights),
        jnp.array(0.32785615, dtype=jnp.float32),
    )

  def test_aucpr_with_batch_size_one(self):
    """Test that AUC-PR Metric computes correct values with batch size one."""
    np.testing.assert_allclose(
        self.compute_aucpr(self.model_outputs_batch_size_one),
        jnp.array(1.0, dtype=jnp.float32),
    )

  def test_aucroc(self):
    """Test that AUC-ROC Metric computes correct values."""
    # Concatenate logits and labels
    all_logits = jnp.concatenate(
        [model_output['logits'] for model_output in self.model_outputs]
    )
    all_labels = jnp.concatenate(
        [model_output['labels'] for model_output in self.model_outputs]
    )
    np.testing.assert_allclose(
        self.compute_aucroc(self.model_outputs),
        sklearn_metrics.roc_auc_score(all_labels, all_logits),
    )

  def test_aucroc_with_sample_weight(self):
    """Test that AUC-ROC Metric computes correct values when using sample weights."""
    # Concatenate logits and labels
    all_logits = jnp.concatenate(
        [model_output['logits'] for model_output in self.model_outputs]
    )
    all_labels = jnp.concatenate(
        [model_output['labels'] for model_output in self.model_outputs]
    )
    sample_weights = jnp.concatenate(
        [self.sample_weights] * len(self.model_outputs)
    )
    np.testing.assert_allclose(
        jnp.array(
            self.compute_aucroc(self.model_outputs, self.sample_weights),
            dtype=jnp.float16,
        ),
        jnp.array(
            sklearn_metrics.roc_auc_score(
                all_labels, all_logits, sample_weight=sample_weights
            ),
            dtype=jnp.float16,
        ),
    )

  @parameterized.named_parameters(
      ('basic', OUTPUT_LABELS, OUTPUT_PREDS, None),
      ('batch_size_one', OUTPUT_LABELS_BS1, OUTPUT_PREDS_BS1, None),
      ('weighted', OUTPUT_LABELS, OUTPUT_PREDS, SAMPLE_WEIGHTS),
  )
  def test_mse(self, y_true, y_pred, sample_weights):
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

    expected = sklearn_metrics.mean_squared_error(
        y_true.flatten(),
        y_pred.flatten(),
        sample_weight=sample_weights.flatten(),
    )
    np.testing.assert_allclose(
        metric.compute(),
        expected,
        rtol=1e-07,
        atol=1e-07,
    )

  @parameterized.named_parameters(
      ('basic', OUTPUT_LABELS, OUTPUT_PREDS, None),
      ('batch_size_one', OUTPUT_LABELS_BS1, OUTPUT_PREDS_BS1, None),
      ('weighted', OUTPUT_LABELS, OUTPUT_PREDS, SAMPLE_WEIGHTS),
  )
  def test_rmse(self, y_true, y_pred, sample_weights):
    if sample_weights is None:
      sample_weights = np.ones_like(y_true)

    metric = None
    for labels, logits, weights in zip(y_true, y_pred, sample_weights):
      update = metrax.RMSE.from_model_output(
          predictions=logits,
          labels=labels,
          sample_weights=weights,
      )
      metric = update if metric is None else metric.merge(update)

    # `sklearn_metrics.root_mean_squared_error` is not available.
    expected = jnp.sqrt(
        jnp.average(
            jnp.square(y_pred.flatten() - y_true.flatten()),
            weights=sample_weights.flatten(),
        ),
    )
    np.testing.assert_allclose(
        metric.compute(),
        expected,
        rtol=1e-07,
        atol=1e-07,
    )

  @parameterized.named_parameters(
      ('basic', OUTPUT_LABELS, OUTPUT_PREDS, None),
      ('batch_size_one', OUTPUT_LABELS_BS1, OUTPUT_PREDS_BS1, None),
      ('weighted', OUTPUT_LABELS, OUTPUT_PREDS, SAMPLE_WEIGHTS),
  )
  def test_rsquared(self, y_true, y_pred, sample_weights):
    if sample_weights is None:
      sample_weights = np.ones_like(y_true)

    metric = None
    for labels, logits, weights in zip(y_true, y_pred, sample_weights):
      update = metrax.RSQUARED.from_model_output(
          predictions=logits,
          labels=labels,
          sample_weights=weights,
      )
      metric = update if metric is None else metric.merge(update)

    expected = sklearn_metrics.r2_score(
        y_true.flatten(),
        y_pred.flatten(),
        sample_weight=sample_weights.flatten(),
    )
    np.testing.assert_allclose(
        metric.compute(),
        expected,
        rtol=1e-05,
        atol=1e-05,
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