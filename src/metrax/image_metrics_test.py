from absl.testing import absltest
import jax.numpy as jnp
from jax import random
from . import image_metrics


class KernelImageMetricsTest(absltest.TestCase):

    # Tests empty instantiation and merge of KID metric
    def test_kernel_inception_distance_empty_and_merge(self):
        empty1 = image_metrics.KernelInceptionDistanceMetric.empty()
        empty2 = image_metrics.KernelInceptionDistanceMetric.empty()
        merged = empty1.merge(empty2)
        # Should still be empty and not error
        self.assertEqual(merged.real_features.shape[0], 0)
        self.assertEqual(merged.fake_features.shape[0], 0)

        # Now merge with non-empty
        key1, key2 = random.split(random.PRNGKey(99))
        real_features = random.normal(key1, shape=(10, 2048))
        fake_features = random.normal(key2, shape=(10, 2048))
        kid_nonempty = image_metrics.KernelInceptionDistanceMetric.from_model_output(
            real_features, fake_features, subset_size=5
        )
        merged2 = kid_nonempty.merge(empty1)
        self.assertEqual(merged2.real_features.shape[0], 10)
        self.assertEqual(merged2.fake_features.shape[0], 10)

    # Tests KID metric with default parameters on random features
    def test_kernel_inception_distance_default_params(self):
        key1, key2 = random.split(random.PRNGKey(42))
        real_features = random.normal(key1, shape=(100, 2048))
        fake_features = random.normal(key2, shape=(100, 2048))

        kid = image_metrics.KernelInceptionDistanceMetric.from_model_output(
            real_features, 
            fake_features, 
            subset_size=50  # Using smaller subset size for testing
        )

        result = kid.compute()
        self.assertEqual(result.shape, (2,))  
        self.assertTrue(result[0] >= 0)  

    # Tests that invalid parameters raise appropriate exceptions
    def test_kernel_inception_distance_invalid_params(self):
        key1, key2 = random.split(random.PRNGKey(44))
        real_features = random.normal(key1, shape=(100, 2048))
        fake_features = random.normal(key2, shape=(100, 2048))

        with self.assertRaises(ValueError):
            image_metrics.KernelInceptionDistanceMetric.from_model_output(
                real_features,
                fake_features,
                subsets=-1,  # Invalid
            )

        with self.assertRaises(ValueError):
            image_metrics.KernelInceptionDistanceMetric.from_model_output(
                real_features,
                fake_features,
                subset_size=0,  # Invalid
            )

    # Tests KID metric with very small sample sizes
    def test_kernel_inception_distance_small_sample_size(self):
        key1, key2 = random.split(random.PRNGKey(45))
        real_features = random.normal(key1, shape=(10, 2048))
        fake_features = random.normal(key2, shape=(10, 2048))

        kid = image_metrics.KernelInceptionDistanceMetric.from_model_output(
            real_features,
            fake_features,
            subset_size=5,  
        )
        result = kid.compute()
        self.assertEqual(result.shape, (2,))

    # Tests that identical feature sets produce KID values close to zero
    def test_kernel_inception_distance_identical_sets(self):
        key = random.PRNGKey(46)
        features = random.normal(key, shape=(100, 2048))

        kid = image_metrics.KernelInceptionDistanceMetric.from_model_output(
            features,  
            features,
            subsets=50,
            subset_size=50,
        )
        result = kid.compute()
        self.assertTrue(result[0] < 1e-3, f"Expected KID close to zero, got {result[0]}")

    # Tests KID metric when the fake features exhibit mode collapse (low variance)
    def test_kernel_inception_distance_mode_collapse(self):
        key1, key2 = random.split(random.PRNGKey(47))
        real_features = random.normal(key1, shape=(100, 2048))

        base_feature = random.normal(key2, shape=(1, 2048))
        repeated_base = jnp.repeat(base_feature, 100, axis=0)
        small_noise = random.normal(key2, shape=(100, 2048)) * 0.01
        fake_features = repeated_base + small_noise

        kid = image_metrics.KernelInceptionDistanceMetric.from_model_output(
            real_features,
            fake_features,
            subset_size=50 
        )
        result = kid.compute()
        self.assertTrue(result[0] > 0.0)

    # Tests KID metric's sensitivity to outliers in the feature distributions
    def test_kernel_inception_distance_outliers(self):
        key1, key2, key3 = random.split(random.PRNGKey(48), 3)
        real_features = random.normal(key1, shape=(100, 2048))
        fake_features = random.normal(key2, shape=(100, 2048))

        outliers = random.normal(key3, shape=(10, 2048)) * 10.0
        fake_features_with_outliers = fake_features.at[:10].set(outliers)

        kid_normal = image_metrics.KernelInceptionDistanceMetric.from_model_output(
            real_features, fake_features, subset_size=50  # Using smaller subset size for testing
        )
        kid_with_outliers = image_metrics.KernelInceptionDistanceMetric.from_model_output(
            real_features, fake_features_with_outliers, subset_size=50  # Using smaller subset size for testing
        )

        result_normal = kid_normal.compute()
        result_with_outliers = kid_with_outliers.compute()

        self.assertNotEqual(result_normal[0], result_with_outliers[0])

    # Tests KID metric with different subset configurations to evaluate stability
    def test_kernel_inception_distance_different_subset_sizes(self):
        key1, key2 = random.split(random.PRNGKey(49))
        real_features = random.normal(key1, shape=(200, 2048))
        fake_features = random.normal(key2, shape=(200, 2048))

        kid_small_subsets = image_metrics.KernelInceptionDistanceMetric.from_model_output(
            real_features, fake_features, subsets=10, subset_size=10
        )
        kid_large_subsets = image_metrics.KernelInceptionDistanceMetric.from_model_output(
            real_features, fake_features, subsets=5, subset_size=100
        )

        result_small = kid_small_subsets.compute()
        result_large = kid_large_subsets.compute()

        self.assertEqual(result_small.shape, (2,))
        self.assertEqual(result_large.shape, (2,))

    # Tests KID metric's ability to differentiate between similar and dissimilar distributions
    def test_kernel_inception_distance_different_distributions(self):
        key1, key2 = random.split(random.PRNGKey(50))
        
        real_features = random.normal(key1, shape=(100, 2048))
        
        mean = 0.5
        std = 2.0
        fake_features = mean + std * random.normal(key2, shape=(100, 2048))
        
        kid = image_metrics.KernelInceptionDistanceMetric.from_model_output(
            real_features, fake_features, subset_size=50  # Using smaller subset size for testing
        )
        result = kid.compute()

        self.assertTrue(result[0] > 0.0)
        key3 = random.PRNGKey(51)
        another_real_features = random.normal(key3, shape=(100, 2048))
        
        kid_same_dist = image_metrics.KernelInceptionDistanceMetric.from_model_output(
            real_features, another_real_features, subset_size=50  # Using smaller subset size for testing
        )
        result_same_dist = kid_same_dist.compute()
        
        self.assertTrue(result[0] > result_same_dist[0])