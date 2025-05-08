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
        import logging
        logger = logging.getLogger("metrax.KID_test")
        logger.info("[TEST] test_kernel_inception_distance_empty_and_merge: Start")
        empty1 = KID.empty()
        empty2 = KID.empty()
        merged = empty1.merge(empty2)
        logger.info(f"  empty1: total={empty1.total}, count={empty1.count}")
        logger.info(f"  empty2: total={empty2.total}, count={empty2.count}")
        logger.info(f"  merged: total={merged.total}, count={merged.count}")
        self.assertEqual(merged.total, 0.0)
        self.assertEqual(merged.count, 0.0)

        key1, key2 = random.split(random.PRNGKey(99))
        real_features = random.normal(key1, shape=(10, 2048))
        fake_features = random.normal(key2, shape=(10, 2048))
        kid_nonempty = KID.from_model_output(
            real_features, fake_features, subset_size=5
        )
        merged2 = kid_nonempty.merge(empty1)
        logger.info(f"  kid_nonempty: total={kid_nonempty.total}, count={kid_nonempty.count}")
        logger.info(f"  merged2: total={merged2.total}, count={merged2.count}")
        self.assertEqual(merged2.total, kid_nonempty.total)
        self.assertEqual(merged2.count, kid_nonempty.count)
        logger.info("[TEST] test_kernel_inception_distance_empty_and_merge: End\n")
    def test_kid_equivalence_and_timing(self):
        import logging
        logger = logging.getLogger("metrax.KID_test")
        logger.info("[TEST] test_kid_equivalence_and_timing: Start")
        n = 32
        subsets = 3
        subset_size = 16
        # Generate random data
        imgs_real = random_images(0, n)
        imgs_fake = random_images(1, n)
        # For Metrax, use random features (simulate Inception features)
        real_features = np.random.randn(n, 2048).astype(np.float32)
        fake_features = np.random.randn(n, 2048).astype(np.float32)

        # Torchmetrics timing
        import time
        t0 = time.time()
        kid_mean_torch, kid_std_torch = compute_torchmetrics_kid(imgs_real, imgs_fake, subsets=subsets, subset_size=subset_size)
        t1 = time.time()
        logger.info(f"Torchmetrics KID: mean={kid_mean_torch}, std={kid_std_torch}, time={t1-t0:.3f}s")

        # Metrax timing
        t2 = time.time()
        kid_metric = KID.from_model_output(
            jnp.array(real_features), jnp.array(fake_features),
            subsets=subsets, subset_size=subset_size
        )
        kid_mean_metrax = kid_metric.compute()
        t3 = time.time()
        logger.info(f"  Metrax KID: mean={kid_mean_metrax}, time={t3-t2:.3f}s")
        logger.info("[TEST] test_kid_equivalence_and_timing: End\n")

        # Note: The results will not be numerically identical, since torchmetrics uses Inception features from images,
        # while Metrax here uses random features. For a true equivalence test, both must use the same features.
        # This test is for timing and API demonstration.

    # Tests KID metric with default parameters on random features
    def test_kernel_inception_distance_default_params(self):
        import logging
        logger = logging.getLogger("metrax.KID_test")
        logger.info("[TEST] test_kernel_inception_distance_default_params: Start")
        key1, key2 = random.split(random.PRNGKey(42))
        real_features = random.normal(key1, shape=(100, 2048))
        fake_features = random.normal(key2, shape=(100, 2048))

        kid = KID.from_model_output(
            real_features, 
            fake_features, 
            subset_size=50  # Using smaller subset size for testing
        )

        result = kid.compute()
        logger.info(f"  result: {result}")
        self.assertTrue(isinstance(result, (float, int, jnp.ndarray)))
        self.assertGreaterEqual(float(result), 0.0)
        logger.info("[TEST] test_kernel_inception_distance_default_params: End\n")
    # Tests that invalid parameters raise appropriate exceptions
    def test_kernel_inception_distance_invalid_params(self):
        import logging
        logger = logging.getLogger("metrax.KID_test")
        logger.info("[TEST] test_kernel_inception_distance_invalid_params: Start")
        key1, key2 = random.split(random.PRNGKey(44))
        real_features = random.normal(key1, shape=(100, 2048))
        fake_features = random.normal(key2, shape=(100, 2048))

        with self.assertRaises(ValueError):
            KID.from_model_output(
                real_features,
                fake_features,
                subsets=-1,  # Invalid
            )

        with self.assertRaises(ValueError):
            KID.from_model_output(
                real_features,
                fake_features,
                subset_size=0,  # Invalid
            )
        logger.info("[TEST] test_kernel_inception_distance_invalid_params: End\n")

    # Tests KID metric with very small sample sizes
    def test_kernel_inception_distance_small_sample_size(self):
        import logging
        logger = logging.getLogger("metrax.KID_test")
        logger.info("[TEST] test_kernel_inception_distance_small_sample_size: Start")
        key1, key2 = random.split(random.PRNGKey(45))
        real_features = random.normal(key1, shape=(10, 2048))
        fake_features = random.normal(key2, shape=(10, 2048))

        kid = KID.from_model_output(
            real_features,
            fake_features,
            subset_size=5,  
        )
        result = kid.compute()
        logger.info(f"  result: {result}")
        # Should be a scalar (float or 0-dim array)
        self.assertTrue(isinstance(result, (float, int, jnp.ndarray)))
        logger.info("[TEST] test_kernel_inception_distance_small_sample_size: End\n")

    # Tests that identical feature sets produce KID values close to zero
    def test_kernel_inception_distance_identical_sets(self):
        import logging
        logger = logging.getLogger("metrax.KID_test")
        logger.info("[TEST] test_kernel_inception_distance_identical_sets: Start")
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
        logger.info(f"  result: {result}, val: {val}")
        self.assertTrue(val < 1e-3, f"Expected KID close to zero, got {val}")
        logger.info("[TEST] test_kernel_inception_distance_identical_sets: End\n")

    # Tests KID metric when the fake features exhibit mode collapse (low variance)
    def test_kernel_inception_distance_mode_collapse(self):
        import logging
        logger = logging.getLogger("metrax.KID_test")
        logger.info("[TEST] test_kernel_inception_distance_mode_collapse: Start")
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
        logger.info(f"  result: {result}, val: {val}")
        self.assertTrue(val > 0.0)
        logger.info("[TEST] test_kernel_inception_distance_mode_collapse: End\n")

    # Tests KID metric's sensitivity to outliers in the feature distributions
    def test_kernel_inception_distance_outliers(self):
        import logging
        logger = logging.getLogger("metrax.KID_test")
        logger.info("[TEST] test_kernel_inception_distance_outliers: Start")
        key1, key2, key3 = random.split(random.PRNGKey(48), 3)
        real_features = random.normal(key1, shape=(100, 2048))
        fake_features = random.normal(key2, shape=(100, 2048))

        outliers = random.normal(key3, shape=(10, 2048)) * 10.0
        fake_features_with_outliers = fake_features.at[:10].set(outliers)

        kid_normal = KID.from_model_output(
            real_features, fake_features, subset_size=50  # Using smaller subset size for testing
        )
        kid_with_outliers = KID.from_model_output(
            real_features, fake_features_with_outliers, subset_size=50  # Using smaller subset size for testing
        )

        result_normal = kid_normal.compute()
        result_with_outliers = kid_with_outliers.compute()
        val_normal = float(result_normal) if hasattr(result_normal, 'shape') and result_normal.shape == () else result_normal
        val_outliers = float(result_with_outliers) if hasattr(result_with_outliers, 'shape') and result_with_outliers.shape == () else result_with_outliers
        logger.info(f"  val_normal: {val_normal}, val_outliers: {val_outliers}")
        self.assertNotEqual(val_normal, val_outliers)
        logger.info("[TEST] test_kernel_inception_distance_outliers: End\n")

    # Tests KID metric with different subset configurations to evaluate stability
    def test_kernel_inception_distance_different_subset_sizes(self):
        import logging
        logger = logging.getLogger("metrax.KID_test")
        logger.info("[TEST] test_kernel_inception_distance_different_subset_sizes: Start")
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
        logger.info(f"  val_small: {val_small}, val_large: {val_large}")
        
        self.assertTrue(isinstance(val_small, float))
        self.assertTrue(isinstance(val_large, float))
        
        logger.info("[TEST] test_kernel_inception_distance_different_subset_sizes: End\n")

    # Tests KID metric's ability to differentiate between similar and dissimilar distributions
    def test_kernel_inception_distance_different_distributions(self):
        import logging
        logger = logging.getLogger("metrax.KID_test")
        logger.info("[TEST] test_kernel_inception_distance_different_distributions: Start")
        key1, key2 = random.split(random.PRNGKey(50))
        
        real_features = random.normal(key1, shape=(100, 2048))
        
        mean = 0.5
        std = 2.0
        fake_features = mean + std * random.normal(key2, shape=(100, 2048))
        
        kid = KID.from_model_output(
            real_features, fake_features, subset_size=50  # Using smaller subset size for testing
        )
        result = kid.compute()
        val = float(result) if hasattr(result, 'shape') and result.shape == () else result
        logger.info(f"  val (real vs fake): {val}")
        self.assertTrue(val > 0.0)
        key3 = random.PRNGKey(51)
        another_real_features = random.normal(key3, shape=(100, 2048))
        
        kid_same_dist = KID.from_model_output(
            real_features, another_real_features, subset_size=50  # Using smaller subset size for testing
        )
        result_same_dist = kid_same_dist.compute()
        val_same = float(result_same_dist) if hasattr(result_same_dist, 'shape') and result_same_dist.shape == () else result_same_dist
        logger.info(f"  val_same (real vs real): {val_same}")
        self.assertTrue(val > val_same)
        logger.info("[TEST] test_kernel_inception_distance_different_distributions: End\n")





def compute_torchmetrics_kid(real_features, fake_features, subsets=10, subset_size=8, degree=3, gamma=None, coef=1.0):
    """
    Compute KID using torchmetrics for two batches of features.
    Args:
        real_features: numpy array of shape (N, 3, 299, 299) or torch tensor
        fake_features: numpy array of shape (N, 3, 299, 299) or torch tensor
        subsets, subset_size, degree, gamma, coef: KID parameters (degree/gamma/coef are not exposed in torchmetrics)
    Returns:
        kid_mean, kid_std (numpy floats)
    """
    # torchmetrics expects uint8 images in (N, 3, 299, 299)
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

