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
KS = np.array([3])
# For nlp_metrics.
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
# For image_metrics.SSIM and image_metrics.PSNR.
IMG_SHAPE = (4, 32, 32, 3)
PRED_IMGS = np.random.rand(*IMG_SHAPE).astype(np.float32)
TARGET_IMGS = np.random.rand(*IMG_SHAPE).astype(np.float32)
MAX_IMG_VAL = 255.0
# For image_metrics.IoU.
IOU_NUM_CLASSES = 3
IOU_TARGETS = np.random.randint(0, IOU_NUM_CLASSES, size=(2, 8, 8)).astype(
    np.int32
)
IOU_PREDICTIONS = np.random.randint(0, IOU_NUM_CLASSES, size=(2, 8, 8)).astype(
    np.int32
)
IOU_TARGET_CLASS_IDS = np.array([0, 1])
# For audio_metrics.
AUDIO_SHAPE = (2, 16000)
AUDIO_PREDS = np.random.randn(*AUDIO_SHAPE).astype(np.float32)
AUDIO_TARGETS = np.random.randn(*AUDIO_SHAPE).astype(np.float32)

class MetraxTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'accuracy',
          metrax.Accuracy,
          {'predictions': OUTPUT_LABELS, 'labels': OUTPUT_PREDS},
      ),
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
              'ks': KS,
          },
      ),
      (
          'dcgAtK',
          metrax.DCGAtK,
          {
              'predictions': OUTPUT_LABELS,
              'labels': OUTPUT_PREDS,
              'ks': KS,
          },
      ),
      (
          'dice',
          metrax.Dice,
          {'predictions': OUTPUT_LABELS, 'labels': OUTPUT_PREDS},
      ),
      (
          'fbetascore',
          metrax.FBetaScore,
          {'predictions': OUTPUT_LABELS, 'labels': OUTPUT_PREDS},
      ),
      (
          'iou',
          metrax.IoU,
          {
              'predictions': IOU_PREDICTIONS,
              'targets': IOU_TARGETS,
              'num_classes': IOU_NUM_CLASSES,
              'target_class_ids': IOU_TARGET_CLASS_IDS,
          },
      ),
      (
          'mrr',
          metrax.MRR,
          {
              'predictions': OUTPUT_LABELS,
              'labels': OUTPUT_PREDS,
              'ks': KS,
          },
      ),
      (
          'mae',
          metrax.MAE,
          {'predictions': OUTPUT_LABELS, 'labels': OUTPUT_PREDS},
      ),
      (
          'mse',
          metrax.MSE,
          {'predictions': OUTPUT_LABELS, 'labels': OUTPUT_PREDS},
      ),
      (
          'ndcgAtK',
          metrax.NDCGAtK,
          {
              'predictions': OUTPUT_LABELS,
              'labels': OUTPUT_PREDS,
              'ks': KS,
          },
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
          'precisionAtK',
          metrax.PrecisionAtK,
          {
              'predictions': OUTPUT_LABELS,
              'labels': OUTPUT_PREDS,
              'ks': KS,
          },
      ),
      (
          'psnr',
          metrax.PSNR,
          {
              'predictions': PRED_IMGS,
              'targets': TARGET_IMGS,
              'max_val': MAX_IMG_VAL,
          },
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
      (
          'recallAtK',
          metrax.RecallAtK,
          {
              'predictions': OUTPUT_LABELS,
              'labels': OUTPUT_PREDS,
              'ks': KS,
          },
      ),
      (
          'ssim',
          metrax.SSIM,
          {
              'predictions': PRED_IMGS,
              'targets': TARGET_IMGS,
              'max_val': MAX_IMG_VAL,
          },
      ),
      (
          'snr',
          metrax.SNR,
          {
              'predictions': AUDIO_PREDS,
              'targets': AUDIO_TARGETS,
              'zero_mean': False,
          },
      ),
  )
  def test_metrics_jittable(self, metric, kwargs):
    """Tests that jitted metrax metric yields the same result as non-jitted metric."""
    computed_metric = metric.from_model_output(**kwargs)
    jitted_metric = jax.jit(metric.from_model_output)(**kwargs)
    np.testing.assert_allclose(
        computed_metric.compute(), jitted_metric.compute(), rtol=1e-2, atol=1e-2
    )

  @parameterized.named_parameters(
      (
          'wer',
          metrax.WER,
          {'predictions': STRING_PREDS, 'references': STRING_REFS},
      ),
      (
          'bleu',
          metrax.BLEU,
          {'predictions': STRING_PREDS, 'references': STRING_REFS},
      ),
      (
          'rougeL',
          metrax.RougeL,
          {'predictions': STRING_PREDS, 'references': STRING_REFS},
      ),
      (
          'rougeN',
          metrax.RougeN,
          {'predictions': STRING_PREDS, 'references': STRING_REFS},
      ),
  )
  def test_metrics_not_jittable(self, metric, kwargs):
    """Tests that attempting to jit and call a known non-jittable metric raises an error."""
    np.testing.assert_raises(
        TypeError, lambda: jax.jit(metric.from_model_output)(**kwargs)
    )


if __name__ == '__main__':
  absltest.main()
