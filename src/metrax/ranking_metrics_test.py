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

"""Tests for metrax ranking metrics."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import metrax
import numpy as np

np.random.seed(42)
BATCH_SIZE = 4
VOCAB_SIZE = 8
OUTPUT_LABELS = np.random.randint(
    0,
    2,
    size=(BATCH_SIZE, VOCAB_SIZE),
).astype(np.float32)
OUTPUT_PREDS = np.random.uniform(size=(BATCH_SIZE, VOCAB_SIZE)).astype(
    np.float32
)
OUTPUT_RELEVANCES = np.random.randint(
    0,
    2,
    size=(BATCH_SIZE, VOCAB_SIZE),
).astype(np.float32)
OUTPUT_LABELS_VS1 = np.random.randint(
    0,
    2,
    size=(BATCH_SIZE, 1),
).astype(np.float32)
OUTPUT_PREDS_VS1 = np.random.uniform(size=(BATCH_SIZE, 1)).astype(np.float32)
OUTPUT_RELEVANCES_VS1 = np.random.randint(
    0,
    2,
    size=(BATCH_SIZE, 1),
).astype(np.float32)
# TODO(jiwonshin): Replace with keras metric once it is available in OSS.
MAP_FROM_KERAS = np.array([
    0.2083333432674408,
    0.4791666865348816,
    0.4791666865348816,
    0.5416666865348816,
    0.574999988079071,
    0.637499988079071,
])
MAP_FROM_KERAS_VS1 = np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75])
P_FROM_KERAS = np.array([0.75, 0.875, 0.58333337306976320, 0.5625, 0.5, 0.5])
P_FROM_KERAS_VS1 = np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75])
R_FROM_KERAS = np.array([
    0.2083333432674408,
    0.5416666865348816,
    0.5416666865348816,
    0.625,
    0.6666666865348816,
    0.75,
])
R_FROM_KERAS_VS1 = np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75])
DCG_FROM_KERAS = np.array([
    0.25,
    0.880929708480835,
    1.255929708480835,
    1.5789371728897095,
    1.8690768480300903,
    2.04718017578125,
])
DCG_FROM_KERAS_VS1 = np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75])


class RankingMetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('basic', OUTPUT_LABELS, OUTPUT_PREDS, MAP_FROM_KERAS, False),
      ('basic_jitted', OUTPUT_LABELS, OUTPUT_PREDS, MAP_FROM_KERAS, True),
      (
          'vocab_size_one',
          OUTPUT_LABELS_VS1,
          OUTPUT_PREDS_VS1,
          MAP_FROM_KERAS_VS1,
          False,
      ),
      (
          'vocab_size_one_jitted',
          OUTPUT_LABELS_VS1,
          OUTPUT_PREDS_VS1,
          MAP_FROM_KERAS_VS1,
          True,
      ),
  )
  def test_averageprecisionatk(self, y_true, y_pred, map_from_keras, jitted):
    """Test that `AveragePrecisionAtK` Metric computes correct values."""
    average_precision_at_k = metrax.AveragePrecisionAtK.from_model_output
    if jitted:
      average_precision_at_k = jax.jit(average_precision_at_k)
    ks = jnp.array([1, 2, 3, 4, 5, 6])
    metric = average_precision_at_k(
        predictions=y_pred,
        labels=y_true,
        ks=ks,
    )

    np.testing.assert_allclose(
        metric.compute(),
        map_from_keras,
        rtol=1e-05,
        atol=1e-05,
    )

  @parameterized.named_parameters(
      ('basic', OUTPUT_LABELS, OUTPUT_PREDS, P_FROM_KERAS),
      (
          'vocab_size_one',
          OUTPUT_LABELS_VS1,
          OUTPUT_PREDS_VS1,
          P_FROM_KERAS_VS1,
      ),
  )
  def test_precisionatk(self, y_true, y_pred, map_from_keras):
    """Test that `PrecisionAtK` Metric computes correct values."""
    ks = jnp.array([1, 2, 3, 4, 5, 6])
    metric = metrax.PrecisionAtK.from_model_output(
        predictions=y_pred,
        labels=y_true,
        ks=ks,
    )

    np.testing.assert_allclose(
        metric.compute(),
        map_from_keras,
        rtol=1e-05,
        atol=1e-05,
    )

  @parameterized.named_parameters(
      ('basic', OUTPUT_LABELS, OUTPUT_PREDS, R_FROM_KERAS),
      (
          'vocab_size_one',
          OUTPUT_LABELS_VS1,
          OUTPUT_PREDS_VS1,
          R_FROM_KERAS_VS1,
      ),
  )
  def test_recallatk(self, y_true, y_pred, map_from_keras):
    """Test that `RecallAtK` Metric computes correct values."""
    ks = jnp.array([1, 2, 3, 4, 5, 6])
    metric = metrax.RecallAtK.from_model_output(
        predictions=y_pred,
        labels=y_true,
        ks=ks,
    )

    np.testing.assert_allclose(
        metric.compute(),
        map_from_keras,
        rtol=1e-05,
        atol=1e-05,
    )

  @parameterized.named_parameters(
      ('basic', OUTPUT_RELEVANCES, OUTPUT_PREDS, DCG_FROM_KERAS),
      (
          'vocab_size_one',
          OUTPUT_RELEVANCES_VS1,
          OUTPUT_PREDS_VS1,
          DCG_FROM_KERAS_VS1,
      ),
  )
  def test_dcgatk(self, y_true, y_pred, map_from_keras):
    """Test that `DCGAtK` Metric computes correct values."""
    ks = jnp.array([1, 2, 3, 4, 5, 6])
    metric = metrax.DCGAtK.from_model_output(
        predictions=y_pred,
        labels=y_true,
        ks=ks,
    )

    np.testing.assert_allclose(
        metric.compute(),
        map_from_keras,
        rtol=1e-05,
        atol=1e-05,
    )


if __name__ == '__main__':
  absltest.main()
