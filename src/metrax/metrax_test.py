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

"""Tests for metrax NNX metrics."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import metrax
import metrax.nnx
import numpy as np

np.random.seed(42)
BATCHES = 1
BATCH_SIZE = 8
OUTPUT_LABELS = np.random.randint(
    0,
    2,
    size=(BATCHES, BATCH_SIZE),
).astype(np.float32)
OUTPUT_PREDS = np.random.uniform(size=(BATCHES, BATCH_SIZE))

STRING_PREDS = [
    'the cat sat on the mat',
    'a quick brown fox jumps over the lazy dog',
    'hello world',
]
STRING_REFS = [
    'the cat sat on the hat',
    'the quick brown fox jumps over the lazy dog',
    'hello beautiful world',
]
TOKENIZED_PREDS = [sentence.split() for sentence in STRING_PREDS]
TOKENIZED_REFS = [sentence.split() for sentence in STRING_REFS]


class MetraxTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'aucpr',
          metrax.AUCPR,
          {'predictions': OUTPUT_LABELS, 'labels': OUTPUT_PREDS},
      ),
      (
          'aucroc',
          metrax.AUCROC,
          {'predictions': OUTPUT_LABELS, 'labels': OUTPUT_PREDS},
      ),
      (
          'average',
          metrax.Average,
          {'values': OUTPUT_PREDS},
      ),
      (
          'averageprecisionatk',
          metrax.AveragePrecisionAtK,
          {
              'predictions': OUTPUT_LABELS,
              'labels': OUTPUT_PREDS,
              'ks': np.array([3]),
          },
      ),
      (
          'mse',
          metrax.MSE,
          {'predictions': OUTPUT_LABELS, 'labels': OUTPUT_PREDS},
      ),
      (
          'perplexity',
          metrax.Perplexity,
          {'predictions': OUTPUT_LABELS, 'labels': OUTPUT_PREDS},
      ),
      (
          'precision',
          metrax.Precision,
          {'predictions': OUTPUT_LABELS, 'labels': OUTPUT_PREDS},
      ),
      (
          'rmse',
          metrax.RMSE,
          {'predictions': OUTPUT_LABELS, 'labels': OUTPUT_PREDS},
      ),
      (
          'rsquared',
          metrax.RSQUARED,
          {'predictions': OUTPUT_LABELS, 'labels': OUTPUT_PREDS},
      ),
      (
          'recall',
          metrax.Recall,
          {'predictions': OUTPUT_LABELS, 'labels': OUTPUT_PREDS},
      ),
  )
  def test_metrics_jittable(self, metric, kwargs):
    """Tests that jitted metrax metric yields the same result as non-jitted metric."""
    computed_metric = metric.from_model_output(**kwargs)
    jitted_metric = jax.jit(metric.from_model_output)(**kwargs)
    np.testing.assert_allclose(
        computed_metric.compute(), jitted_metric.compute()
    )

  @parameterized.named_parameters(
      (
          'wer',
          metrax.WER,
          {'predictions': TOKENIZED_PREDS, 'references': TOKENIZED_REFS},
      ),
  )
  def test_metrics_not_jittable(self, metric, kwargs):
    """Tests that attempting to jit and call a known non-jittable metric raises an error."""
    np.testing.assert_raises(
        TypeError, lambda: jax.jit(metric.from_model_output)(**kwargs)
    )


if __name__ == '__main__':
  absltest.main()
