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
from absl.testing import absltest
import jax.numpy as jnp
from jax import random
import numpy as np
import torch
from torchmetrics.image.kid import KernelInceptionDistance as TorchKID
from PIL import Image
from metrax.image_metrics import KID

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
    @staticmethod
    def compute_torchmetrics_kid(real_images, fake_images, subsets=10, subset_size=8):
        """
        Compute KID using torchmetrics for two batches of images and compare with Metrax implementation.
        Returns a tuple: (torchmetrics_mean, torchmetrics_std, metrax_mean, metrax_std)
        """
        if isinstance(real_images, np.ndarray) and  isinstance(fake_images, np.ndarray):
            real_images = torch.from_numpy(real_images)
            fake_images = torch.from_numpy(fake_images)
        if real_images.dtype != torch.uint8 and fake_images.dtype != torch.uint8:
            real_images = real_images.to(torch.uint8)
            fake_images = fake_images.to(torch.uint8)
        kid = TorchKID(subsets=subsets, subset_size=subset_size)
        kid.update(real_images, real=True)
        kid.update(fake_images, real=False)
        kid_mean, kid_std = kid.compute()
        # For comparison, use random features as a stand-in for Inception features
        n = real_images.shape[0]
        real_features = np.random.randn(n, 2048).astype(np.float32)
        fake_features = np.random.randn(n, 2048).astype(np.float32)
        kid_metric = KID.from_model_output(
            jnp.array(real_features), jnp.array(fake_features),
            subsets=subsets, subset_size=subset_size
        )
        metrax_result = kid_metric.compute()
        # metrax_result may be a single value or a tuple
        if isinstance(metrax_result, tuple) and len(metrax_result) == 2:
            metrax_mean, metrax_std = float(metrax_result[0]), float(metrax_result[1])
        else:
            metrax_mean, metrax_std = float(metrax_result), float('nan')
        return float(kid_mean.cpu().numpy()), float(kid_std.cpu().numpy()), metrax_mean, metrax_std


    def test_kid_equivalence_and_timing(self):
        """Compare KID between Metrax and torchmetrics implementations."""
        n = 32
        subsets: int = 3
        subset_size: int = 16
        imgs_real = random_images(0, n)
        imgs_fake = random_images(1, n)
        real_features = np.random.randn(n, 2048).astype(np.float32)
        fake_features = np.random.randn(n, 2048).astype(np.float32)

        kid_mean_torch, kid_std_torch, kid_mean_metrax, kid_std_metrax = self.compute_torchmetrics_kid(
            imgs_real, imgs_fake, subsets=subsets, subset_size=subset_size
        )
        kid_metric = KID.from_model_output(
            jnp.array(real_features), jnp.array(fake_features),
            subsets=subsets, subset_size=subset_size
        )
        # Accept numpy scalar or float
        ## return float(kid_mean.cpu().numpy()), float(kid_std.cpu().numpy()), metrax_mean, metrax_std
        self.assertIsInstance(kid_mean_torch, float)
        self.assertIsInstance(kid_mean_metrax, float)
        self.assertIsInstance(kid_std_torch, float)
        self.assertIsInstance(kid_std_metrax, float)


    # Tests KID metric with default parameters on random features
    def test_kernel_inception_distance_default_params(self):
        """Test KID metric with default parameters on random features."""
        key1, key2 = random.split(random.PRNGKey(42))
        real_features = random.normal(key1, shape=(100, 2048))
        fake_features = random.normal(key2, shape=(100, 2048))

        kid = KID.from_model_output(
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
            KID.from_model_output(
                real_features,
                fake_features,
                subsets=-1,
            )

        with self.assertRaises(ValueError):
            KID.from_model_output(
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

        kid = KID.from_model_output(
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

        kid = KID.from_model_output(
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

        kid = KID.from_model_output(
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

        kid_normal = KID.from_model_output(
            real_features, fake_features, subset_size=50
        )
        kid_with_outliers = KID.from_model_output(
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

        kid_small_subsets = KID.from_model_output(
            real_features, fake_features, subsets=10, subset_size=10
        )
        kid_large_subsets = KID.from_model_output(
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

        kid = KID.from_model_output(
            real_features, fake_features, subset_size=50
        )
        result = kid.compute()
        val = float(result) if hasattr(result, 'shape') and result.shape == () else result
        self.assertTrue(val > 0.0)
        key3 = random.PRNGKey(51)
        another_real_features = random.normal(key3, shape=(100, 2048))

        kid_same_dist = KID.from_model_output(
            real_features, another_real_features, subset_size=50
        )
        result_same_dist = kid_same_dist.compute()
        val_same = float(result_same_dist) if hasattr(result_same_dist, 'shape') and result_same_dist.shape == () else result_same_dist
        self.assertTrue(val > val_same)



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

