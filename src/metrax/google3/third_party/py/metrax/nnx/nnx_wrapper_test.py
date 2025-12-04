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

"""Tests for metrax NNX metric wrapper."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import metrax
import metrax.nnx
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


class NnxWrapperTest(parameterized.TestCase):

  def test_reset(self):
    """Tests the `reset` method of the `NnxWrapper` class."""
    nnx_metric = metrax.nnx.MSE()
    self.assertEqual(nnx_metric.clu_metric.total, jnp.array(0, jnp.float32))
    self.assertEqual(nnx_metric.clu_metric.count, jnp.array(0, jnp.int32))
    nnx_metric.update(
        predictions=jnp.array([1.0, 2.0, 3.0]),
        labels=jnp.array([1.0, 2.0, 3.0]),
        sample_weights=jnp.array([1.0, 1.0, 1.0]),
    )
    nnx_metric.reset()
    self.assertEqual(nnx_metric.clu_metric.total, jnp.array(0, jnp.float32))
    self.assertEqual(nnx_metric.clu_metric.count, jnp.array(0, jnp.int32))

  @parameterized.named_parameters(
      ('basic_f16', OUTPUT_LABELS, OUTPUT_PREDS_F16, None),
      ('basic_f32', OUTPUT_LABELS, OUTPUT_PREDS_F32, None),
      ('basic_bf16', OUTPUT_LABELS, OUTPUT_PREDS_BF16, None),
      ('batch_size_one', OUTPUT_LABELS_BS1, OUTPUT_PREDS_BS1, None),
      ('weighted_f16', OUTPUT_LABELS, OUTPUT_PREDS_F16, SAMPLE_WEIGHTS),
      ('weighted_f32', OUTPUT_LABELS, OUTPUT_PREDS_F32, SAMPLE_WEIGHTS),
      ('weighted_bf16', OUTPUT_LABELS, OUTPUT_PREDS_BF16, SAMPLE_WEIGHTS),
  )
  def test_metric_update_and_compute(self, y_true, y_pred, sample_weights):
    """Test that `MSE` Metric in `NnxWrapper` computes correct values."""
    y_true = y_true.astype(y_pred.dtype)
    y_pred = y_pred.astype(y_true.dtype)
    if sample_weights is None:
      sample_weights = np.ones_like(y_true)

    nnx_metric = metrax.nnx.MSE()
    for labels, logits, weights in zip(y_true, y_pred, sample_weights):
      nnx_metric.update(
          predictions=logits,
          labels=labels,
          sample_weights=weights,
      )

    # TODO(jiwonshin): Use `keras.metrics.MeanSquaredError` once it supports
    # sample weights.
    expected = sklearn_metrics.mean_squared_error(
        y_true.astype('float32').flatten(),
        y_pred.astype('float32').flatten(),
        sample_weight=sample_weights.astype('float32').flatten(),
    )
    # Use lower tolerance for lower precision dtypes.
    rtol = 1e-2 if y_true.dtype in (jnp.float16, jnp.bfloat16) else 1e-05
    atol = 1e-2 if y_true.dtype in (jnp.float16, jnp.bfloat16) else 1e-05
    np.testing.assert_allclose(
        nnx_metric.compute(),
        expected,
        rtol=rtol,
        atol=atol,
    )


if __name__ == '__main__':
  absltest.main()
