# Copyright 2025 Google LLC
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

"""Tests for metrax image metrics."""

import os

os.environ['KERAS_BACKEND'] = 'jax'

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import metrax
import numpy as np
import torch
from torchmetrics.functional.audio import snr as tm_snr

np.random.seed(42)

# Simple 1D audio signal.
AUDIO_SHAPE_1D = (1000,)
AUDIO_TARGET_1D = np.sin(
    np.linspace(0, 2 * np.pi * 5, AUDIO_SHAPE_1D[0])
).astype(np.float32)
AUDIO_PREDS_1D_NOISY = (
    AUDIO_TARGET_1D + 0.1 * np.random.randn(*AUDIO_SHAPE_1D)
).astype(np.float32)
AUDIO_PREDS_1D_PERFECT = AUDIO_TARGET_1D
# Multi-dimensional batch of signals
AUDIO_SHAPE_2D = (4, 500)  # This is likely the source of the 4 elements.
AUDIO_TARGET_2D = (np.random.randn(*AUDIO_SHAPE_2D) * 5.0).astype(np.float32)
AUDIO_PREDS_2D_NOISY = (
    AUDIO_TARGET_2D + 0.5 * np.random.randn(*AUDIO_SHAPE_2D)
).astype(np.float32)
# Target and preds are all zeros.
AUDIO_SHAPE_ZEROS = (100,)
AUDIO_TARGET_ZEROS = np.zeros(AUDIO_SHAPE_ZEROS).astype(np.float32)
AUDIO_PREDS_ZEROS = np.zeros(AUDIO_SHAPE_ZEROS).astype(np.float32)


class AudioMetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'snr_1d_noisy_false_zero_mean',
          AUDIO_TARGET_1D,
          AUDIO_PREDS_1D_NOISY,
          False,
      ),
      (
          'snr_1d_noisy_true_zero_mean',
          AUDIO_TARGET_1D,
          AUDIO_PREDS_1D_NOISY,
          True,
      ),
      (
          'snr_1d_perfect_false_zero_mean',
          AUDIO_TARGET_1D,
          AUDIO_PREDS_1D_PERFECT,
          False,
      ),
      (
          'snr_1d_perfect_true_zero_mean',
          AUDIO_TARGET_1D,
          AUDIO_PREDS_1D_PERFECT,
          True,
      ),
      (
          'snr_2d_noisy_false_zero_mean',
          AUDIO_TARGET_2D,
          AUDIO_PREDS_2D_NOISY,
          False,
      ),
      (
          'snr_2d_noisy_true_zero_mean',
          AUDIO_TARGET_2D,
          AUDIO_PREDS_2D_NOISY,
          True,
      ),
      (
          'snr_zeros_false_zero_mean',
          AUDIO_TARGET_ZEROS,
          AUDIO_PREDS_ZEROS,
          False,
      ),
      ('snr_zeros_true_zero_mean', AUDIO_TARGET_ZEROS, AUDIO_PREDS_ZEROS, True),
  )
  def test_snr(
      self,
      target_np: np.ndarray,
      preds_np: np.ndarray,
      zero_mean: bool,
  ):
    """Tests metrax.SNR against torchmetrics.functional.audio.snr."""
    metrax_snr_metric = metrax.SNR.from_model_output(
        predictions=jnp.array(preds_np),
        targets=jnp.array(target_np),
        zero_mean=zero_mean,
    )
    metrax_snr_result = metrax_snr_metric.compute()

    torchmetrics_snr_result = (
        tm_snr.signal_noise_ratio(
            preds=torch.from_numpy(preds_np),
            target=torch.from_numpy(target_np),
            zero_mean=zero_mean,
        )
        .mean()
        .item()
    )

    np.testing.assert_allclose(
        metrax_snr_result,
        torchmetrics_snr_result,
        rtol=1e-5,
        atol=1e-5,
        err_msg=(
            f'SNR mismatch for zero_mean={zero_mean}.\n'
            f'Metrax SNR: {metrax_snr_result:.8f} dB, '
            f'Torchmetrics SNR: {torchmetrics_snr_result:.8f} dB'
        ),
    )


if __name__ == '__main__':
  absltest.main()
