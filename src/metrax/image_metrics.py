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
# ==============================================================================
## credits to the https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/kid.py for the refernece implementation.


import jax.numpy as jnp
from jax import random
import flax
import jax
from clu import metrics as clu_metrics

KID_DEFAULT_SUBSETS = 100
KID_DEFAULT_SUBSET_SIZE = 1000
KID_DEFAULT_DEGREE = 3
KID_DEFAULT_GAMMA = None
KID_DEFAULT_COEF = 1.0


@flax.struct.dataclass
class KernelInceptionDistanceMetric(clu_metrics.Metric):
    r"""Computes Kernel Inception Distance (KID) for asses quality of generated images.
    KID is a metric used to evaluate the quality of generated images by comparing
    the distribution of generated images to the distribution of real images.
    It is based on the Inception Score (IS) and uses a kernelized version of the
    Maximum Mean Discrepancy (MMD) to measure the distance between two
    distributions.

    The KID is computed as follows:

    .. math::
        KID = MMD(f_{real}, f_{fake})^2

    Where :math:`MMD` is the maximum mean discrepancy and :math:`I_{real}, I_{fake}` are extracted features
    from real and fake images, see `kid ref1`_ for more details. In particular, calculating the MMD requires the
    evaluation of a polynomial kernel function :math:`k`.

    .. math::
        k(x,y) = (\gamma * x^T y + coef)^{degree}

    Args:
        subsets: Number of subsets to use for KID calculation.
        subset_size: Number of samples in each subset.
        degree: Degree of the polynomial kernel.
        gamma: Kernel coefficient for the polynomial kernel.
        coef: Independent term in the polynomial kernel.
    """

    subsets: int
    subset_size: int
    degree: int
    gamma: float
    coef: float

    real_features: jax.Array = flax.struct.field(default_factory=lambda: jnp.array([], dtype=jnp.float32))
    fake_features: jax.Array = flax.struct.field(default_factory=lambda: jnp.array([], dtype=jnp.float32))


    @classmethod
    def from_model_output(
        cls,
        real_features: jax.Array,
        fake_features: jax.Array,
        subsets: int = KID_DEFAULT_SUBSETS,
        subset_size: int = KID_DEFAULT_SUBSET_SIZE,
        degree: int = KID_DEFAULT_DEGREE,
        gamma: float = KID_DEFAULT_GAMMA,
        coef: float = KID_DEFAULT_COEF,
    ):
        # checks for the valid inputs
        if subsets <= 0 or subset_size <= 0 or degree <= 0 or (gamma is not None and gamma <= 0) or coef <= 0:
            raise ValueError("All parameters must be positive and non-zero.")
        return cls(
            subsets=subsets,
            subset_size=subset_size,
            degree=degree,
            gamma=gamma,
            coef=coef,
            real_features=real_features,
            fake_features=fake_features,
        )

    @classmethod
    def empty(cls) -> "KernelInceptionDistanceMetric":
        """
        Create an empty instance of KernelInceptionDistanceMetric.
        """
        return cls(
            subsets=KID_DEFAULT_SUBSETS,
            subset_size=KID_DEFAULT_SUBSET_SIZE,
            degree=KID_DEFAULT_DEGREE,
            gamma=KID_DEFAULT_GAMMA,
            coef=KID_DEFAULT_COEF,
            real_features=jnp.empty((0, 2048), dtype=jnp.float32),
            fake_features=jnp.empty((0, 2048), dtype=jnp.float32),
        )

    def compute_mmd(self, f_real: jax.Array, f_fake: jax.Array) -> float:
        """
        Compute the Maximum Mean Discrepancy (MMD) using a polynomial kernel.
        Args:
            f_real: Features from real images.
            f_fake: Features from fake images.
        Returns:
            MMD value in order to compute KID
        """
        k_11 = self.polynomial_kernel(f_real, f_real)
        k_22 = self.polynomial_kernel(f_fake, f_fake)
        k_12 = self.polynomial_kernel(f_real, f_fake)

        m = f_real.shape[0]
        diag_x = jnp.diag(k_11)
        diag_y = jnp.diag(k_22)

        kt_xx_sum = jnp.sum(k_11, axis=-1) - diag_x
        kt_yy_sum = jnp.sum(k_22, axis=-1) - diag_y
        k_xy_sum = jnp.sum(k_12, axis=0)

        value = (jnp.sum(kt_xx_sum) + jnp.sum(kt_yy_sum)) / (m * (m - 1))
        value -= 2 * jnp.sum(k_xy_sum) / (m**2)
        return value

    def polynomial_kernel(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """
        Compute the polynomial kernel between two sets of features.
        Args:
            x: First set of features.
            y: another set of features to be computed with.
        Returns:
            Polynomial kernel value of Array type .
        """
        gamma = self.gamma if self.gamma is not None else 1.0 / x.shape[1]
        return (jnp.dot(x, y.T) * gamma + self.coef) ** self.degree

    def compute(self) -> jax.Array:
        """
        Compute the KID mean and standard deviation from accumulated features.
        """
        if self.real_features.shape[0] < self.subset_size or self.fake_features.shape[0] < self.subset_size:
            raise ValueError("Subset size must be smaller than the number of samples.")

        master_key = random.PRNGKey(42)
        
        kid_scores = []
        for i in range(self.subsets):
            # Split the key for each iteration to ensure different random samples
            key_real, key_fake = random.split(random.fold_in(master_key, i))
            real_indices = random.choice(key_real, self.real_features.shape[0], (self.subset_size,), replace=False)
            fake_indices = random.choice(key_fake, self.fake_features.shape[0], (self.subset_size,), replace=False)

            f_real_subset = self.real_features[real_indices]
            f_fake_subset = self.fake_features[fake_indices]

            kid = self.compute_mmd(f_real_subset, f_fake_subset)
            kid_scores.append(kid)

        kid_mean = jnp.mean(jnp.array(kid_scores))
        kid_std = jnp.std(jnp.array(kid_scores))
        return jnp.array([kid_mean, kid_std])
    

    def merge(self, other: "KernelInceptionDistanceMetric") -> "KernelInceptionDistanceMetric":
        """
        Merge two KernelInceptionDistanceMetric instances.
        Args:
            other: Another instance of KernelInceptionDistanceMetric.
        Returns:
            A new instance of KernelInceptionDistanceMetric with combined features.
        """
        return type(self)(
            subsets=self.subsets,
            subset_size=self.subset_size,
            degree=self.degree,
            gamma=self.gamma,
            coef=self.coef,
            real_features=jnp.concatenate([self.real_features, other.real_features]),
            fake_features=jnp.concatenate([self.fake_features, other.fake_features]),
        )
