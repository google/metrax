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
import jax.numpy as jnp
import keras_rs
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
OUTPUT_PREDS = np.random.uniform(size=(BATCH_SIZE, VOCAB_SIZE)).astype(np.float32)
OUTPUT_RELEVANCES = np.random.randint(
  0,
  4,
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
  4,
  size=(BATCH_SIZE, 1),
).astype(np.float32)


class RankingMetricsTest(parameterized.TestCase):
  @parameterized.named_parameters(
    (
      "averageprecisionatk_basic",
      metrax.AveragePrecisionAtK,
      keras_rs.metrics.MeanAveragePrecision,
      OUTPUT_LABELS,
      OUTPUT_PREDS,
    ),
    (
      "averageprecisionatk_vocab_size_one",
      metrax.AveragePrecisionAtK,
      keras_rs.metrics.MeanAveragePrecision,
      OUTPUT_LABELS_VS1,
      OUTPUT_PREDS_VS1,
    ),
    (
      "precisionatk_basic",
      metrax.PrecisionAtK,
      keras_rs.metrics.PrecisionAtK,
      OUTPUT_LABELS,
      OUTPUT_PREDS,
    ),
    (
      "precisionatk_vocab_size_one",
      metrax.PrecisionAtK,
      keras_rs.metrics.PrecisionAtK,
      OUTPUT_LABELS_VS1,
      OUTPUT_PREDS_VS1,
    ),
    (
      "recallatk_basic",
      metrax.RecallAtK,
      keras_rs.metrics.RecallAtK,
      OUTPUT_LABELS,
      OUTPUT_PREDS,
    ),
    (
      "recallatk_vocab_size_one",
      metrax.RecallAtK,
      keras_rs.metrics.RecallAtK,
      OUTPUT_LABELS_VS1,
      OUTPUT_PREDS_VS1,
    ),
    (
      "dcgatk_basic",
      metrax.DCGAtK,
      keras_rs.metrics.DCG,
      OUTPUT_RELEVANCES,
      OUTPUT_PREDS,
    ),
    (
      "dcgatk_vocab_size_one",
      metrax.DCGAtK,
      keras_rs.metrics.DCG,
      OUTPUT_RELEVANCES_VS1,
      OUTPUT_PREDS_VS1,
    ),
    (
      "ndcgatk_basic",
      metrax.NDCGAtK,
      keras_rs.metrics.NDCG,
      OUTPUT_RELEVANCES,
      OUTPUT_PREDS,
    ),
    (
      "ndcgatk_vocab_size_one",
      metrax.NDCGAtK,
      keras_rs.metrics.NDCG,
      OUTPUT_RELEVANCES_VS1,
      OUTPUT_PREDS_VS1,
    ),
    (
      "mrr_basic",
      metrax.MRR,
      keras_rs.metrics.MeanReciprocalRank,
      OUTPUT_LABELS,
      OUTPUT_PREDS,
    ),
    (
      "mrr_vocab_size_one",
      metrax.MRR,
      keras_rs.metrics.MeanReciprocalRank,
      OUTPUT_LABELS_VS1,
      OUTPUT_PREDS_VS1,
    ),
  )
  def test_ranking_metrics(self, metric, keras_metric, y_true, y_pred):
    """Test that a metrax ranking metric computes correct values."""
    ks = jnp.array([1, 2, 3, 4, 5, 6])
    metric = metric.from_model_output(
      predictions=y_pred,
      labels=y_true,
      ks=ks,
    )

    keras_metrics = [keras_metric(k=n + 1) for n in range(6)]
    results = []
    for keras_metric in keras_metrics:
      keras_metric.update_state(y_true, y_pred)
      results.append(keras_metric.result())

    np.testing.assert_allclose(
      metric.compute(),
      jnp.array(results),
      rtol=1e-05,
      atol=1e-05,
    )


if __name__ == "__main__":
  absltest.main()
