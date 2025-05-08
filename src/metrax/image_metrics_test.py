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

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import metrax
import numpy as np
import tensorflow as tf

np.random.seed(42)

# Default SSIM parameters (matching tf.image.ssim and metrax defaults)
DEFAULT_FILTER_SIZE = 11
DEFAULT_FILTER_SIGMA = 1.5
DEFAULT_K1 = 0.01
DEFAULT_K2 = 0.03

# Test data for SSIM
# Case 1: Basic, float normalized [0,1], single channel
IMG_SHAPE_1 = (2, 16, 16, 1)  # batch, height, width, channels
# Ensure height/width >= filter_size
PREDS_1_NP = np.random.rand(*IMG_SHAPE_1).astype(np.float32)
TARGETS_1_NP = np.random.rand(*IMG_SHAPE_1).astype(np.float32)
MAX_VAL_1 = 1.0

# Case 2: Multi-channel (3), float normalized [0,1]
IMG_SHAPE_2 = (4, 32, 32, 3)
PREDS_2_NP = np.random.rand(*IMG_SHAPE_2).astype(np.float32)
TARGETS_2_NP = np.random.rand(*IMG_SHAPE_2).astype(np.float32)
MAX_VAL_2 = 1.0

# Case 3: Uint8 range representation (0-255), single channel
IMG_SHAPE_3 = (2, 20, 20, 1)  # height/width = 20 >= filter_size = 11
PREDS_3_NP = (np.random.rand(*IMG_SHAPE_3) * 255.0).astype(np.float32)
TARGETS_3_NP = (np.random.rand(*IMG_SHAPE_3) * 255.0).astype(np.float32)
MAX_VAL_3 = 255.0

# Case 4: Custom filter parameters (using data similar to Case 1)
IMG_SHAPE_4 = (2, 16, 16, 1)  # height/width = 16 >= custom_filter_size = 7
PREDS_4_NP = np.random.rand(*IMG_SHAPE_4).astype(np.float32)
TARGETS_4_NP = np.random.rand(*IMG_SHAPE_4).astype(np.float32)
MAX_VAL_4 = 1.0
FILTER_SIZE_CUSTOM = 7
FILTER_SIGMA_CUSTOM = 1.0
K1_CUSTOM = 0.02
K2_CUSTOM = 0.05

# Case 5: Identical images
IMG_SHAPE_5 = (2, 16, 16, 1)
PREDS_5_NP = np.random.rand(*IMG_SHAPE_5).astype(np.float32)
TARGETS_5_NP = np.copy(PREDS_5_NP)  # Identical images
MAX_VAL_5 = 1.0


class ImageMetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'ssim_basic_norm_single_channel',
          PREDS_1_NP,
          TARGETS_1_NP,
          MAX_VAL_1,
          DEFAULT_FILTER_SIZE,
          DEFAULT_FILTER_SIGMA,
          DEFAULT_K1,
          DEFAULT_K2,
      ),
      (
          'ssim_multichannel_norm',
          PREDS_2_NP,
          TARGETS_2_NP,
          MAX_VAL_2,
          DEFAULT_FILTER_SIZE,
          DEFAULT_FILTER_SIGMA,
          DEFAULT_K1,
          DEFAULT_K2,
      ),
      (
          'ssim_uint8_range_single_channel',
          PREDS_3_NP,
          TARGETS_3_NP,
          MAX_VAL_3,
          DEFAULT_FILTER_SIZE,
          DEFAULT_FILTER_SIGMA,
          DEFAULT_K1,
          DEFAULT_K2,
      ),
      (
          'ssim_custom_params_norm_single_channel',
          PREDS_4_NP,
          TARGETS_4_NP,
          MAX_VAL_4,
          FILTER_SIZE_CUSTOM,
          FILTER_SIGMA_CUSTOM,
          K1_CUSTOM,
          K2_CUSTOM,
      ),
      (
          'ssim_identical_images',
          PREDS_5_NP,
          TARGETS_5_NP,
          MAX_VAL_5,
          DEFAULT_FILTER_SIZE,
          DEFAULT_FILTER_SIGMA,
          DEFAULT_K1,
          DEFAULT_K2,
      ),
  )
  def test_ssim_against_tensorflow(
      self,
      predictions_np: np.ndarray,
      targets_np: np.ndarray,
      max_val: float,
      filter_size: int,
      filter_sigma: float,
      k1: float,
      k2: float,
  ):
    """Test that metrax.SSIM computes values close to tf.image.ssim."""
    # Calculate SSIM using Metrax
    predictions_jax = jnp.array(predictions_np)
    targets_jax = jnp.array(targets_np)
    metrax_metric = metrax.SSIM.from_model_output(
        predictions=predictions_jax,
        targets=targets_jax,
        max_val=max_val,
        filter_size=filter_size,
        filter_sigma=filter_sigma,
        k1=k1,
        k2=k2,
    )
    metrax_result = metrax_metric.compute()

    # Calculate SSIM using TensorFlow
    predictions_tf = tf.convert_to_tensor(predictions_np, dtype=tf.float32)
    targets_tf = tf.convert_to_tensor(targets_np, dtype=tf.float32)
    tf_ssim_per_image = tf.image.ssim(
        img1=predictions_tf,
        img2=targets_tf,
        max_val=max_val,
        filter_size=filter_size,
        filter_sigma=filter_sigma,
        k1=k1,
        k2=k2,
    )
    tf_result_mean = tf.reduce_mean(tf_ssim_per_image).numpy()

    np.testing.assert_allclose(
        metrax_result,
        tf_result_mean,
        rtol=1e-5,
        atol=1e-5,
        err_msg=(
            f'SSIM mismatch for params: max_val={max_val}, '
            f'filter_size={filter_size}, filter_sigma={filter_sigma}, '
            f'k1={k1}, k2={k2}'
        ),
    )
    # For identical images, we expect a value very close to 1.0
    if np.array_equal(predictions_np, targets_np):
      self.assertAlmostEqual(float(metrax_result), 1.0, delta=1e-6)
      self.assertAlmostEqual(float(tf_result_mean), 1.0, delta=1e-6)


if __name__ == '__main__':
  absltest.main()
