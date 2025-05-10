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
import keras
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
PREDS_1 = np.random.rand(*IMG_SHAPE_1).astype(np.float32)
TARGETS_1 = np.random.rand(*IMG_SHAPE_1).astype(np.float32)
MAX_VAL_1 = 1.0

# Case 2: Multi-channel (3), float normalized [0,1]
IMG_SHAPE_2 = (4, 32, 32, 3)
PREDS_2 = np.random.rand(*IMG_SHAPE_2).astype(np.float32)
TARGETS_2 = np.random.rand(*IMG_SHAPE_2).astype(np.float32)
MAX_VAL_2 = 1.0

# Case 3: Uint8 range representation (0-255), single channel
IMG_SHAPE_3 = (2, 20, 20, 1)  # height/width = 20 >= filter_size = 11
PREDS_3 = (np.random.rand(*IMG_SHAPE_3) * 255.0).astype(np.float32)
TARGETS_3 = (np.random.rand(*IMG_SHAPE_3) * 255.0).astype(np.float32)
MAX_VAL_3 = 255.0

# Case 4: Custom filter parameters (using data similar to Case 1)
IMG_SHAPE_4 = (2, 16, 16, 1)  # height/width = 16 >= custom_filter_size = 7
PREDS_4 = np.random.rand(*IMG_SHAPE_4).astype(np.float32)
TARGETS_4 = np.random.rand(*IMG_SHAPE_4).astype(np.float32)
MAX_VAL_4 = 1.0
FILTER_SIZE_CUSTOM = 7
FILTER_SIGMA_CUSTOM = 1.0
K1_CUSTOM = 0.02
K2_CUSTOM = 0.05

# Case 5: Identical images
IMG_SHAPE_5 = (2, 16, 16, 1)
PREDS_5 = np.random.rand(*IMG_SHAPE_5).astype(np.float32)
TARGETS_5 = np.copy(PREDS_5)  # Identical images
MAX_VAL_5 = 1.0

# Case 6: Large batch size 
IMG_SHAPE_6 = (8, 16, 16, 3)
PREDS_6 = np.random.rand(*IMG_SHAPE_6).astype(np.float32)
TARGETS_6 = np.random.rand(*IMG_SHAPE_6).astype(np.float32)
MAX_VAL_6 = 1.0

B_IOU, H_IOU, W_IOU = 2, 4, 4  # Common batch, height, width for IoU tests

# Case IoU 1: Binary segmentation (num_classes=2), target_class_ids=[1] (foreground)
# Targets: (Batch, Height, Width)
TARGETS_IOU_1 = np.array(
    [
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],  # Batch item 1
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],  # Batch item 2
    ],
    dtype=np.int32,
)
PREDS_IOU_1 = np.array(
    [
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],  # Batch item 1 (IoU for C1: 2/(4+4-2)=2/6)
        [
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ],  # Batch item 2 (IoU for C1: 2/(4+4-2)=2/6)
    ],
    dtype=np.int32,
)
NUM_CLASSES_IOU_1 = 2
TARGET_CLASS_IDS_IOU_1 = np.array(
    [1]
)  # Expected Keras/Metrax result: mean([2/6, 2/6]) = 1/3

# Case IoU 2: Multi-class (num_classes=3), target_class_ids=[0, 2] (mean over these two)
TARGETS_IOU_2 = np.array(
    [
        [[0, 0, 1, 1], [0, 1, 1, 2], [2, 2, 1, 0], [0, 0, 2, 2]],  # B1
        [[1, 1, 0, 0], [1, 2, 2, 0], [0, 0, 1, 1], [2, 2, 0, 0]],  # B2
    ],
    dtype=np.int32,
)
PREDS_IOU_2 = np.array(
    [
        [[0, 1, 1, 1], [0, 1, 2, 2], [2, 0, 1, 0], [0, 0, 2, 0]],  # B1
        [[1, 0, 0, 0], [1, 2, 1, 0], [0, 2, 1, 1], [2, 1, 0, 0]],  # B2
    ],
    dtype=np.int32,
)
NUM_CLASSES_IOU_2 = 3
TARGET_CLASS_IDS_IOU_2 = np.array([0, 2])

# Case IoU 3: Perfect overlap for target class [1] (using a smaller H, W for simplicity)
_H_IOU3, _W_IOU3 = 3, 3
TARGETS_IOU_3 = np.array(
    [
        [[1, 1, 0], [1, 1, 0], [0, 0, 0]],  # B1
        [[0, 0, 0], [0, 1, 1], [0, 1, 1]],  # B2
    ],
    dtype=np.int32,
).reshape((B_IOU, _H_IOU3, _W_IOU3))
PREDS_IOU_3 = np.copy(TARGETS_IOU_3)
NUM_CLASSES_IOU_3 = 2
TARGET_CLASS_IDS_IOU_3 = np.array([1])  # Expected Keras/Metrax result: 1.0

# Case IoU 4: No overlap for target class [1] (class present in union)
TARGETS_IOU_4 = np.array(
    [
        [[1, 1, 0], [0, 0, 0], [0, 0, 0]],  # B1: Target has class 1
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # B2: Target has no class 1
    ],
    dtype=np.int32,
).reshape((B_IOU, _H_IOU3, _W_IOU3))
PREDS_IOU_4 = np.array(
    [
        [[0, 0, 2], [2, 0, 0], [0, 0, 0]],  # B1: Pred has no class 1
        [[0, 2, 0], [2, 1, 1], [0, 1, 0]],  # B2: Pred has class 1
    ],
    dtype=np.int32,
).reshape((B_IOU, _H_IOU3, _W_IOU3))
NUM_CLASSES_IOU_4 = 3  # Max label is 2
TARGET_CLASS_IDS_IOU_4 = np.array([1])  # Expected Keras/Metrax result: 0.0

# Case IoU 5: Input from Logits (binary case, reuse targets from IoU 1)
TARGETS_IOU_5 = TARGETS_IOU_1  # (B, H, W)
_B5, _H5, _W5 = TARGETS_IOU_5.shape
_NC5 = NUM_CLASSES_IOU_1
PREDS_IOU_5_LOGITS = np.random.randn(_B5, _H5, _W5, _NC5).astype(np.float32)
# Create logits such that argmax yields PREDS_IOU_1
temp_preds_for_logits = PREDS_IOU_1
for b_idx in range(_B5):
  for h_idx in range(_H5):
    for w_idx in range(_W5):
      label = temp_preds_for_logits[b_idx, h_idx, w_idx]
      for c_idx in range(_NC5):
        PREDS_IOU_5_LOGITS[b_idx, h_idx, w_idx, c_idx] = -5.0
      PREDS_IOU_5_LOGITS[b_idx, h_idx, w_idx, label] = 5.0
NUM_CLASSES_IOU_5 = NUM_CLASSES_IOU_1
TARGET_CLASS_IDS_IOU_5 = TARGET_CLASS_IDS_IOU_1

# Case IoU 6: Target all classes (None for Metrax, list for Keras)
TARGETS_IOU_6 = TARGETS_IOU_2
PREDS_IOU_6 = PREDS_IOU_2
NUM_CLASSES_IOU_6 = NUM_CLASSES_IOU_2
TARGET_CLASS_IDS_IOU_6 = np.array(range(NUM_CLASSES_IOU_6))


class ImageMetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'ssim_basic_norm_single_channel',
          PREDS_1,
          TARGETS_1,
          MAX_VAL_1,
          DEFAULT_FILTER_SIZE,
          DEFAULT_FILTER_SIGMA,
          DEFAULT_K1,
          DEFAULT_K2,
      ),
      (
          'ssim_multichannel_norm',
          PREDS_2,
          TARGETS_2,
          MAX_VAL_2,
          DEFAULT_FILTER_SIZE,
          DEFAULT_FILTER_SIGMA,
          DEFAULT_K1,
          DEFAULT_K2,
      ),
      (
          'ssim_uint8_range_single_channel',
          PREDS_3,
          TARGETS_3,
          MAX_VAL_3,
          DEFAULT_FILTER_SIZE,
          DEFAULT_FILTER_SIGMA,
          DEFAULT_K1,
          DEFAULT_K2,
      ),
      (
          'ssim_custom_params_norm_single_channel',
          PREDS_4,
          TARGETS_4,
          MAX_VAL_4,
          FILTER_SIZE_CUSTOM,
          FILTER_SIGMA_CUSTOM,
          K1_CUSTOM,
          K2_CUSTOM,
      ),
      (
          'ssim_identical_images',
          PREDS_5,
          TARGETS_5,
          MAX_VAL_5,
          DEFAULT_FILTER_SIZE,
          DEFAULT_FILTER_SIGMA,
          DEFAULT_K1,
          DEFAULT_K2,
      ),
  )
  def test_ssim_against_tensorflow(
      self,
      predictions: np.ndarray,
      targets: np.ndarray,
      max_val: float,
      filter_size: int,
      filter_sigma: float,
      k1: float,
      k2: float,
  ):
    """Test that metrax.SSIM computes values close to tf.image.ssim."""
    # Calculate SSIM using Metrax
    predictions_jax = jnp.array(predictions)
    targets_jax = jnp.array(targets)
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
    predictions_tf = tf.convert_to_tensor(predictions, dtype=tf.float32)
    targets_tf = tf.convert_to_tensor(targets, dtype=tf.float32)
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
    if np.array_equal(predictions, targets):
      self.assertAlmostEqual(float(metrax_result), 1.0, delta=1e-6)
      self.assertAlmostEqual(float(tf_result_mean), 1.0, delta=1e-6)

  @parameterized.named_parameters(
      (
          'iou_binary_target_foreground',
          TARGETS_IOU_1,
          PREDS_IOU_1,
          NUM_CLASSES_IOU_1,
          TARGET_CLASS_IDS_IOU_1,
          False,
      ),
      (
          'iou_multiclass_target_subset',
          TARGETS_IOU_2,
          PREDS_IOU_2,
          NUM_CLASSES_IOU_2,
          TARGET_CLASS_IDS_IOU_2,
          False,
      ),
      (
          'iou_multiclass_target_single_from_set2',
          TARGETS_IOU_2,
          PREDS_IOU_2,
          NUM_CLASSES_IOU_2,
          [1],
          False,
      ),
      (
          'iou_perfect_overlap_binary',
          TARGETS_IOU_3,
          PREDS_IOU_3,
          NUM_CLASSES_IOU_3,
          TARGET_CLASS_IDS_IOU_3,
          False,
      ),
      (
          'iou_no_overlap_target_class',
          TARGETS_IOU_4,
          PREDS_IOU_4,
          NUM_CLASSES_IOU_4,
          TARGET_CLASS_IDS_IOU_4,
          False,
      ),
      (
          'iou_from_logits_binary',
          TARGETS_IOU_5,
          PREDS_IOU_5_LOGITS,
          NUM_CLASSES_IOU_5,
          TARGET_CLASS_IDS_IOU_5,
          True,
      ),
      (
          'iou_target_all_metrax_none_keras_list',
          TARGETS_IOU_6,
          PREDS_IOU_6,
          NUM_CLASSES_IOU_6,
          TARGET_CLASS_IDS_IOU_6,
          False,
      ),
  )
  def test_iou_against_keras(
      self,
      targets: np.ndarray,
      predictions: np.ndarray,
      num_classes: int,
      target_class_ids: np.ndarray,
      from_logits: bool,
  ):
    """Tests metrax.IoU against keras.metrics.IoU."""
    # Metrax IoU
    metrax_metric = metrax.IoU.from_model_output(
        predictions=jnp.array(predictions),
        targets=jnp.array(targets),
        num_classes=num_classes,
        target_class_ids=jnp.array(target_class_ids),
        from_logits=from_logits,
    )
    metrax_result = metrax_metric.compute()

    # Keras IoU
    keras_iou_metric = keras.metrics.IoU(
        num_classes=num_classes,
        target_class_ids=target_class_ids,
        name='keras_iou',
        sparse_y_pred=not from_logits,
    )
    keras_iou_metric.update_state(targets, predictions)
    keras_result = keras_iou_metric.result()

    np.testing.assert_allclose(
        metrax_result,
        keras_result,
        rtol=1e-5,
        atol=1e-5,
        err_msg=(
            f'IoU mismatch for num_classes={num_classes},'
            f' target_class_ids={target_class_ids} (TF was'
            f' {target_class_ids}),'
            f' from_logits={from_logits}.\nMetrax: {metrax_result}, Keras:'
            f' {keras_result}'
        ),
    )

    # Specific assertions for clearer test outcomes
    if 'perfect_overlap' in self.id():
      self.assertAlmostEqual(
          float(metrax_result),
          1.0,
          delta=1e-6,
          msg=f'Metrax IoU failed for {self.id()}',
      )
      if not np.isnan(keras_result):
        self.assertAlmostEqual(
            float(keras_result),
            1.0,
            delta=1e-6,
            msg=f'Keras IoU failed for {self.id()}',
        )

    if 'no_overlap' in self.id():
      self.assertAlmostEqual(
          float(metrax_result),
          0.0,
          delta=1e-6,
          msg=f'Metrax IoU failed for {self.id()}',
      )
      if not np.isnan(keras_result):
        self.assertAlmostEqual(
            float(keras_result),
            0.0,
            delta=1e-6,
            msg=f'Keras IoU failed for {self.id()}',
        )
  @parameterized.named_parameters(
        (
            "psnr_basic_norm_single_channel",
            PREDS_1,
            TARGETS_1,
            MAX_VAL_1,
        ),
        (
            "psnr_multichannel_norm",
            PREDS_2,
            TARGETS_2,
            MAX_VAL_2,
        ),
        (
            "psnr_uint8_range_single_channel",
            PREDS_3,
            TARGETS_3,
            MAX_VAL_3,
        ),
        (
            "psnr_identical_images",
            PREDS_4,
            TARGETS_4,
            MAX_VAL_4,
        ),
        (
            "psnr_large_batch",
            PREDS_6,
            TARGETS_6,
            MAX_VAL_6,
        ),
    )
  def test_psnr_against_tensorflow(
        self,
        predictions_np: np.ndarray,
        targets_np: np.ndarray,
        max_val: float,
    ) -> None:

        """Test that metrax.SSIM computes values close to tf.image.ssim.

        Note: TensorFlow returns `inf` for identical images (MSE=0).
        Metrax returns a very large finite value due to the eps guard. 
        """
        # Calculate PSNR using Metrax
        metrax_psnr = metrax.PSNR.from_model_output(
            predictions=jnp.array(predictions_np),
            targets=jnp.array(targets_np),
            max_val=max_val,
        ).compute()

        # Calculate PSNR using TensorFlow 
        tf_psnr = tf.image.psnr(
            predictions_np.astype(np.float32),
            targets_np.astype(np.float32),
            max_val=max_val,
        )
        tf_mean = tf.reduce_mean(tf_psnr).numpy()

        if np.isinf(tf_mean):
            self.assertTrue(np.isinf(metrax_psnr))
        else:
            np.testing.assert_allclose(
                metrax_psnr,
                tf_mean,
                rtol=1e-4,
                atol=1e-4,
                err_msg="PSNR mismatch",
    )
if __name__ == '__main__':
  absltest.main()
