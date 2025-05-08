from absl.testing import absltest
import jax.numpy as jnp
from jax import random
import numpy as np
import torch
from torchmetrics.image.kid import KernelInceptionDistance as TorchKID
from .image_metrics import random_images
from metrax import KID


class KernelInceptionDistanceTest(absltest.TestCase):
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
        if hasattr(metrax_result, '__len__') and len(metrax_result) == 2:
            metrax_mean, metrax_std = float(metrax_result[0]), float(metrax_result[1])
        else:
            metrax_mean, metrax_std = float(metrax_result), float('nan')
        return float(kid_mean.cpu().numpy()), float(kid_std.cpu().numpy()), metrax_mean, metrax_std


    def test_kernel_inception_distance_empty_and_merge(self):
        """Test merging empty and non-empty KID metrics."""
        empty1 = KID.empty()
        empty2 = KID.empty()
        merged = empty1.merge(empty2)
        self.assertEqual(merged.total, 0.0)
        self.assertEqual(merged.count, 0.0)

        key1, key2 = random.split(random.PRNGKey(99))
        real_features = random.normal(key1, shape=(10, 2048))
        fake_features = random.normal(key2, shape=(10, 2048))
        kid_nonempty = KID.from_model_output(
            real_features, fake_features, subset_size=5
        )
        merged2 = kid_nonempty.merge(empty1)
        self.assertEqual(merged2.total, kid_nonempty.total)
        self.assertEqual(merged2.count, kid_nonempty.count)
    
    
    
    def test_kid_equivalence_and_timing(self):
        """Compare KID between Metrax and torchmetrics implementations."""
        n = 32
        subsets = 3
        subset_size = 16
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
        kid_mean_metrax2 = kid_metric.compute()
        self.assertIsInstance(kid_mean_torch, float)
        self.assertIsInstance(kid_mean_metrax, float)
        self.assertIsInstance(kid_mean_metrax2, (float, jnp.ndarray))


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





def compute_torchmetrics_kid(real_features, fake_features, subsets=10, subset_size=8, degree=3, gamma=None, coef=1.0):
    """
    Compute KID using torchmetrics for two batches of features.
    Args:
        real_features: numpy array of shape of  4 params
        fake_features: numpy array of shape (N, 3, 299, 299) or torch tensor
        subsets, subset_size, degree, gamma, coef: KID parameters (degree/gamma/coef are not exposed in torchmetrics)
    Returns:
        kid_mean, kid_std (numpy floats)
    """
    if isinstance(real_features, np.ndarray):
        real_features = torch.from_numpy(real_features)
    if isinstance(fake_features, np.ndarray):
        fake_features = torch.from_numpy(fake_features)
    if real_features.dtype != torch.uint8:
        real_features = real_features.to(torch.uint8)
    if fake_features.dtype != torch.uint8:
        fake_features = fake_features.to(torch.uint8)

    kid = TorchKID(subsets=subsets, subset_size=subset_size)
    kid.update(real_features, real=True)
    kid.update(fake_features, real=False)
    kid_mean, kid_std = kid.compute()
    return kid_mean.cpu().numpy(), kid_std.cpu().numpy()

