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

"""Tests for metrax classification metrics."""

import os
os.environ['KERAS_BACKEND'] = 'jax'

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import keras
import metrax
import numpy as np

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

class ClassificationMetricsTest(parameterized.TestCase):

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

  # Test the empty method for the FBetsScore class
  def test_fbeta_empty(self):
    """Tests the `empty` method of the `FBetaScore` class."""
    m = metrax.FBetaScore.empty()
    self.assertEqual(m.true_positives, jnp.array(0, jnp.float32))
    self.assertEqual(m.false_positives, jnp.array(0, jnp.float32))
    self.assertEqual(m.false_negatives, jnp.array(0, jnp.float32))
    self.assertEqual(m.beta, 1.0)

  @parameterized.named_parameters(
      ('basic_f16', OUTPUT_LABELS, OUTPUT_PREDS_F16, SAMPLE_WEIGHTS),
      ('basic_f32', OUTPUT_LABELS, OUTPUT_PREDS_F32, SAMPLE_WEIGHTS),
      ('basic_bf16', OUTPUT_LABELS, OUTPUT_PREDS_BF16, SAMPLE_WEIGHTS),
      ('batch_size_one', OUTPUT_LABELS_BS1, OUTPUT_PREDS_BS1, None),
  )
  def test_accuracy(self, y_true, y_pred, sample_weights):
    """Test that `Accuracy` metric computes correct values."""
    if sample_weights is None:
      sample_weights = np.ones_like(y_true)
    metrax_accuracy = metrax.Accuracy.empty()
    keras_accuracy = keras.metrics.Accuracy()
    for labels, logits, weights in zip(y_true, y_pred, sample_weights):
      update = metrax.Accuracy.from_model_output(
          predictions=logits,
          labels=labels,
          sample_weights=weights,
      )
      metrax_accuracy = metrax_accuracy.merge(update)
      keras_accuracy.update_state(labels, logits, weights)

    # Use lower tolerance for lower precision dtypes.
    rtol = 1e-2 if y_true.dtype in (jnp.float16, jnp.bfloat16) else 1e-5
    atol = 1e-2 if y_true.dtype in (jnp.float16, jnp.bfloat16) else 1e-5
    np.testing.assert_allclose(
        metrax_accuracy.compute(),
        keras_accuracy.result(),
        rtol=rtol,
        atol=atol,
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

# Testing function for F-Beta classification
  # name, output true, output prediction, threshold, beta
  @parameterized.named_parameters(
      ('basic_f16_beta_1.0', OUTPUT_LABELS, OUTPUT_PREDS_F16, 0.5, 1.0),
      ('basic_f32_beta_1.0', OUTPUT_LABELS, OUTPUT_PREDS_F32, 0.5, 1.0),
      ('low_threshold_f32_beta_1.0', OUTPUT_LABELS, OUTPUT_PREDS_F32, 0.1, 1.0),
      ('high_threshold_bf16_beta_1.0', OUTPUT_LABELS, OUTPUT_PREDS_BF16, 0.7, 1.0),
      ('batch_size_one_beta_1.0', OUTPUT_LABELS_BS1, OUTPUT_PREDS_BS1, 0.5, 1.0),
      ('high_threshold_f16_beta_2.0', OUTPUT_LABELS, OUTPUT_PREDS_F16, 0.7, 2.0),
      ('high_threshold_f32_beta_2.0', OUTPUT_LABELS, OUTPUT_PREDS_F32, 0.7, 2.0),
      ('low_threshold_bf16_beta_2.0', OUTPUT_LABELS, OUTPUT_PREDS_BF16, 0.1, 2.0),
      ('low_threshold_f16_beta_3.0', OUTPUT_LABELS, OUTPUT_PREDS_F16, 0.1, 3.0),
      ('basic_bf16_beta_3.0', OUTPUT_LABELS, OUTPUT_PREDS_BF16, 0.5, 3.0),
  )
  def test_fbetascore(self, y_true, y_pred, threshold, beta):
    # Define the Keras FBeta class to be tested against
    keras_fbeta = keras.metrics.FBetaScore(beta=beta, threshold=threshold)
    keras_fbeta.update_state(y_true, y_pred)
    expected = keras_fbeta.result()

    # Calculate the F-beta score using the metrax variant
    metric = metrax.FBetaScore
    metric = metric.from_model_output(y_pred, y_true, beta, threshold)

    # Use lower tolerance for lower precision dtypes.
    rtol = 1e-2 if y_true.dtype in (jnp.float16, jnp.bfloat16) else 1e-5
    atol = 1e-2 if y_true.dtype in (jnp.float16, jnp.bfloat16) else 1e-5
    np.testing.assert_allclose(
        metric.compute(),
        expected,
        rtol=rtol,
        atol=atol,
    )


if __name__ == '__main__':
  absltest.main()
