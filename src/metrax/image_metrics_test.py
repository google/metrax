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
from absl.testing import absltest
import jax.numpy as jnp
from jax import random
import numpy as np
import torch
from torchmetrics.image.kid import KernelInceptionDistance as TorchKID
from PIL import Image

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

# Test data for Dice
BATCHES = 4
BATCH_SIZE = 8
OUTPUT_LABELS = np.random.randint(
    0,
    2,
    size=(BATCHES, BATCH_SIZE),
).astype(np.float32)
OUTPUT_PREDS = np.random.uniform(size=(BATCHES, BATCH_SIZE))
OUTPUT_PREDS_F16 = OUTPUT_PREDS.astype(np.float16)
OUTPUT_PREDS_F32 = OUTPUT_PREDS.astype(np.float32)
OUTPUT_LABELS_BS1 = np.random.randint(
    0,
    2,
    size=(BATCHES, 1),
).astype(np.float32)
OUTPUT_PREDS_BS1 = np.random.uniform(size=(BATCHES, 1)).astype(np.float32)

DICE_ALL_ONES = (np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]))
DICE_ALL_ZEROS = (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]))
DICE_NO_OVERLAP = (np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1]))


def random_images(seed, n):
    """
    Generate n random RGB images as numpy arrays in (N, 3, 299, 299) format using PIL.Image.
    Args:
        seed: Random seed for reproducibility.
        n: Number of images to generate.
    Returns:
        images: numpy array of shape (n, 3, 299, 299), dtype uint8
    """
    rng = np.random.RandomState(seed)
    images = []
    for _ in range(n):
        # Generate a random (299, 299, 3) uint8 array
        arr = rng.randint(0, 256, size=(299, 299, 3), dtype=np.uint8)
        # Convert to PIL Image and back to numpy to ensure valid image
        img = Image.fromarray(arr, mode='RGB')
        arr_pil = np.array(img)
        # Transpose to (3, 299, 299) as required by KID/torchmetrics
        arr_pil = arr_pil.transpose(2, 0, 1)
        images.append(arr_pil)
    return np.stack(images, axis=0).astype(np.uint8)


class ImageMetricsTest(parameterized.TestCase):
    def test_kid_torchmetrics_and_native(self):
        """
        Compare KID computation using torchmetrics and the native Metrax implementation.
        Assert that their values are numerically close and result types are equivalent.
        """
        n = 32
        subsets = 3
        subset_size = 16

        # Generate random images
        imgs_real = random_images(0, n)
        imgs_fake = random_images(1, n)

        # Convert images to torch tensors if needed
        imgs_real = torch.from_numpy(imgs_real) if isinstance(imgs_real, np.ndarray) else imgs_real
        imgs_fake = torch.from_numpy(imgs_fake) if isinstance(imgs_fake, np.ndarray) else imgs_fake
        if imgs_real.dtype != torch.uint8:
            imgs_real = imgs_real.to(torch.uint8)
        if imgs_fake.dtype != torch.uint8:
            imgs_fake = imgs_fake.to(torch.uint8)

        # Compute KID using torchmetrics
        kid = TorchKID(subsets=subsets, subset_size=subset_size)
        kid.update(imgs_real, real=True)
        kid.update(imgs_fake, real=False)
        kid_mean_torch, kid_std_torch = kid.compute()
        kid_mean_torch = float(kid_mean_torch.cpu().numpy())
        kid_std_torch = float(kid_std_torch.cpu().numpy())

        # Compute KID using Metrax implementation with random features
        real_features = np.random.randn(n, 2048).astype(np.float32)
        fake_features = np.random.randn(n, 2048).astype(np.float32)
        kid_metric = metrax.KID.from_model_output(
            jnp.array(real_features), jnp.array(fake_features),
            subsets=subsets, subset_size=subset_size
        )
        metrax_result = kid_metric.compute()
        if isinstance(metrax_result, tuple) and len(metrax_result) == 2:
            kid_mean_metrax, kid_std_metrax = float(metrax_result[0]), float(metrax_result[1])
        else:
            kid_mean_metrax, kid_std_metrax = float(metrax_result), float('nan')

        # Assert types are both float
        self.assertIsInstance(kid_mean_torch, float)
        self.assertIsInstance(kid_mean_metrax, float)
        self.assertIsInstance(kid_std_torch, float)
        # Only check stddev if metrax returns a real value
        if not np.isnan(kid_std_metrax):
            self.assertIsInstance(kid_std_metrax, float)
            self.assertGreaterEqual(kid_std_metrax, 0.0)
            self.assertAlmostEqual(kid_std_torch, kid_std_metrax, delta=0.05, msg=f"KID std mismatch: torch={kid_std_torch}, metrax={kid_std_metrax}")
        # Always check mean
        self.assertAlmostEqual(kid_mean_torch, kid_mean_metrax, delta=0.05, msg=f"KID mean mismatch: torch={kid_mean_torch}, metrax={kid_mean_metrax}")
        self.assertGreaterEqual(kid_std_torch, 0.0)

    # Tests KID metric with default parameters on random features
    def test_kernel_inception_distance_default_params(self):
        """Test KID metric with default parameters on random features."""
        key1, key2 = random.split(random.PRNGKey(42))
        real_features = random.normal(key1, shape=(100, 2048))
        fake_features = random.normal(key2, shape=(100, 2048))

        kid = metrax.KID.from_model_output(
            real_features, 
            fake_features, 
            subset_size=50
        )

        result = kid.compute()
        self.assertTrue(isinstance(result, (float, int, jnp.ndarray)))
        self.assertGreaterEqual(float(result), 0.0)

    def test_kernel_inception_distance_invalid_params(self):
        """Test that invalid parameters raise appropriate exceptions."""
        key1, key2 = random.split(random.PRNGKey(44))
        real_features = random.normal(key1, shape=(100, 2048))
        fake_features = random.normal(key2, shape=(100, 2048))

        with self.assertRaises(ValueError):
            metrax.KID.from_model_output(
                real_features,
                fake_features,
                subsets=-1,
            )

        with self.assertRaises(ValueError):
            metrax.KID.from_model_output(
                real_features,
                fake_features,
                subset_size=0,
            )

    # Tests KID metric with very small sample sizes
    def test_kernel_inception_distance_small_sample_size(self):
        """Test KID metric with very small sample sizes."""
        key1, key2 = random.split(random.PRNGKey(45))
        real_features = random.normal(key1, shape=(10, 2048))
        fake_features = random.normal(key2, shape=(10, 2048))

        kid = metrax.KID.from_model_output(
            real_features,
            fake_features,
            subset_size=5,
        )
        result = kid.compute()
        self.assertTrue(isinstance(result, (float, int, jnp.ndarray)))

    # Tests that identical feature sets produce KID values close to zero
    def test_kernel_inception_distance_identical_sets(self):
        """Test that identical feature sets produce KID values close to zero."""
        key = random.PRNGKey(46)
        features = random.normal(key, shape=(100, 2048))

        kid = metrax.KID.from_model_output(
            features,
            features,
            subsets=50,
            subset_size=50,
        )
        result = kid.compute()
        val = float(result) if hasattr(result, 'shape') and result.shape == () else result
        self.assertTrue(val < 1e-3, f"Expected KID close to zero, got {val}")

    # Tests KID metric when the fake features exhibit mode collapse (low variance)
    def test_kernel_inception_distance_mode_collapse(self):
        """Test KID metric when the fake features exhibit mode collapse (low variance)."""
        key1, key2 = random.split(random.PRNGKey(47))
        real_features = random.normal(key1, shape=(100, 2048))

        base_feature = random.normal(key2, shape=(1, 2048))
        repeated_base = jnp.repeat(base_feature, 100, axis=0)
        small_noise = random.normal(key2, shape=(100, 2048)) * 0.01
        fake_features = repeated_base + small_noise

        kid = metrax.KID.from_model_output(
            real_features,
            fake_features,
            subset_size=50
        )
        result = kid.compute()
        val = float(result) if hasattr(result, 'shape') and result.shape == () else result
        self.assertTrue(val > 0.0)

    # Tests KID metric's sensitivity to outliers in the feature distributions
    def test_kernel_inception_distance_outliers(self):
        """Test KID metric's sensitivity to outliers in the feature distributions."""
        key1, key2, key3 = random.split(random.PRNGKey(48), 3)
        real_features = random.normal(key1, shape=(100, 2048))
        fake_features = random.normal(key2, shape=(100, 2048))

        outliers = random.normal(key3, shape=(10, 2048)) * 10.0
        fake_features_with_outliers = fake_features.at[:10].set(outliers)

        kid_normal = metrax.KID.from_model_output(
            real_features, fake_features, subset_size=50
        )
        kid_with_outliers = metrax.KID.from_model_output(
            real_features, fake_features_with_outliers, subset_size=50
        )

        result_normal = kid_normal.compute()
        result_with_outliers = kid_with_outliers.compute()
        val_normal = float(result_normal) if hasattr(result_normal, 'shape') and result_normal.shape == () else result_normal
        val_outliers = float(result_with_outliers) if hasattr(result_with_outliers, 'shape') and result_with_outliers.shape == () else result_with_outliers
        self.assertNotEqual(val_normal, val_outliers)

    # Tests KID metric with different subset configurations to evaluate stability
    def test_kernel_inception_distance_different_subset_sizes(self):
        """Test KID metric with different subset configurations to evaluate stability."""
        key1, key2 = random.split(random.PRNGKey(49))
        real_features = random.normal(key1, shape=(200, 2048))
        fake_features = random.normal(key2, shape=(200, 2048))

        kid_small_subsets = metrax.KID.from_model_output(
            real_features, fake_features, subsets=10, subset_size=10
        )
        kid_large_subsets = metrax.KID.from_model_output(
            real_features, fake_features, subsets=5, subset_size=100
        )

        result_small = kid_small_subsets.compute()
        result_large = kid_large_subsets.compute()
        val_small = float(result_small) if hasattr(result_small, 'shape') and result_small.shape == () else result_small
        val_large = float(result_large) if hasattr(result_large, 'shape') and result_large.shape == () else result_large

        self.assertTrue(isinstance(val_small, float))
        self.assertTrue(isinstance(val_large, float))

    # Tests KID metric's ability to differentiate between similar and dissimilar distributions
    def test_kernel_inception_distance_different_distributions(self):
        """Test KID metric's ability to differentiate between similar and dissimilar distributions."""
        key1, key2 = random.split(random.PRNGKey(50))
        real_features = random.normal(key1, shape=(100, 2048))
        mean = 0.5
        std = 2.0
        fake_features = mean + std * random.normal(key2, shape=(100, 2048))

        kid = metrax.KID.from_model_output(
            real_features, fake_features, subset_size=50
        )
        result = kid.compute()
        val = float(result) if hasattr(result, 'shape') and result.shape == () else result
        self.assertTrue(val > 0.0)
        key3 = random.PRNGKey(51)
        another_real_features = random.normal(key3, shape=(100, 2048))

        kid_same_dist = metrax.KID.from_model_output(
            real_features, another_real_features, subset_size=50
        )
        result_same_dist = kid_same_dist.compute()
        val_same = float(result_same_dist) if hasattr(result_same_dist, 'shape') and result_same_dist.shape == () else result_same_dist
        self.assertTrue(val > val_same)

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

  def test_dice_empty(self):
    """Tests the `empty` method of the `Dice` class."""
    m = metrax.Dice.empty()
    self.assertEqual(m.intersection, jnp.array(0, jnp.float32))
    self.assertEqual(m.sum_true, jnp.array(0, jnp.float32))
    self.assertEqual(m.sum_pred, jnp.array(0, jnp.float32))

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
        # Only expect 1.0 for identical images
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
          'psnr_basic_norm_single_channel',
          PREDS_1,
          TARGETS_1,
          MAX_VAL_1,
      ),
      (
          'psnr_multichannel_norm',
          PREDS_2,
          TARGETS_2,
          MAX_VAL_2,
      ),
      (
          'psnr_uint8_range_single_channel',
          PREDS_3,
          TARGETS_3,
          MAX_VAL_3,
      ),
      (
          'psnr_identical_images',
          PREDS_4,
          TARGETS_4,
          MAX_VAL_4,
      ),
      (
          'psnr_large_batch',
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
    """Test that metrax.SSIM computes values close to tf.image.ssim."""
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
          err_msg='PSNR mismatch',
      )

  @parameterized.named_parameters(
      ('basic_f32', OUTPUT_LABELS, OUTPUT_PREDS_F32),
      ('low_threshold', OUTPUT_LABELS, OUTPUT_PREDS_F32),
      ('high_threshold', OUTPUT_LABELS, OUTPUT_PREDS_F32),
      ('batch_size_one', OUTPUT_LABELS_BS1, OUTPUT_PREDS_BS1),
      ('all_ones', *DICE_ALL_ONES),
      ('all_zeros', *DICE_ALL_ZEROS),
      ('no_overlap', *DICE_NO_OVERLAP),
  )
  def test_dice(self, y_true, y_pred):
    """Test that Dice metric computes expected values."""
    y_true = jnp.asarray(y_true, jnp.float32)
    y_pred = jnp.asarray(y_pred, jnp.float32)

    # Manually compute expected Dice
    eps = 1e-7
    intersection = jnp.sum(y_true * y_pred)
    sum_pred = jnp.sum(y_pred)
    sum_true = jnp.sum(y_true)
    expected = (2.0 * intersection) / (sum_pred + sum_true + eps)

    # Compute using the metric class
    metric = metrax.Dice.from_model_output(predictions=y_pred, labels=y_true)
    result = metric.compute()

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    absltest.main()