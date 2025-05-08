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
from metrax import base
import numpy as np
from PIL import Image

KID_DEFAULT_SUBSETS = 100
KID_DEFAULT_SUBSET_SIZE = 1000
KID_DEFAULT_DEGREE = 3
KID_DEFAULT_GAMMA = None
KID_DEFAULT_COEF = 1.0



def polynomial_kernel(x: jax.Array, y: jax.Array, degree: int, gamma: float, coef: float) -> jax.Array:
    """
    Compute the polynomial kernel between two sets of features.
    Args:
        x: First set of features.
        y: Another set of features to be computed with.
        degree: Degree of the polynomial kernel.
        gamma: Kernel coefficient for the polynomial kernel. If None, uses 1 / x.shape[1].
        coef: Independent term in the polynomial kernel.
    Returns:
        Polynomial kernel value of Array type.
    """
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (jnp.dot(x, y.T) * gamma + coef) ** degree



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

@flax.struct.dataclass
class KernelInceptionDistance(base.Average):
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

    subsets: int = KID_DEFAULT_SUBSETS
    subset_size: int = KID_DEFAULT_SUBSET_SIZE
    degree: int = KID_DEFAULT_DEGREE
    gamma: float = KID_DEFAULT_GAMMA
    coef: float = KID_DEFAULT_COEF


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
        # Compute KID for this batch
        if real_features.shape[0] < subset_size or fake_features.shape[0] < subset_size:
            raise ValueError("Subset size must be smaller than the number of samples.")
        master_key = random.PRNGKey(42)
        kid_scores = []
        for i in range(subsets):
            key_real, key_fake = random.split(random.fold_in(master_key, i))
            real_indices = random.choice(key_real, real_features.shape[0], (subset_size,), replace=False)
            fake_indices = random.choice(key_fake, fake_features.shape[0], (subset_size,), replace=False)
            f_real_subset = real_features[real_indices]
            f_fake_subset = fake_features[fake_indices]
            kid = cls.__compute_mmd_static(f_real_subset, f_fake_subset, degree, gamma, coef)
            kid_scores.append(kid)
        kid_mean = jnp.mean(jnp.array(kid_scores))
        # Accumulate sum and count for averaging
        return cls(
            total=kid_mean,
            count=1.0,
            subsets=subsets,
            subset_size=subset_size,
            degree=degree,
            gamma=gamma,
            coef=coef,
        )

    @classmethod
    def empty(cls) -> "KernelInceptionDistance":
        """
        Create an empty instance of KernelInceptionDistance.
        """
        return cls(
            total=0.0,
            count=0.0,
            subsets=KID_DEFAULT_SUBSETS,
            subset_size=KID_DEFAULT_SUBSET_SIZE,
            degree=KID_DEFAULT_DEGREE,
            gamma=KID_DEFAULT_GAMMA,
            coef=KID_DEFAULT_COEF,
        )


    @staticmethod
    def __compute_mmd_static(f_real: jax.Array, f_fake: jax.Array, degree: int, gamma: float, coef: float) -> float:
        k_11 = polynomial_kernel(f_real, f_real, degree, gamma, coef)
        k_22 = polynomial_kernel(f_fake, f_fake, degree, gamma, coef)
        k_12 = polynomial_kernel(f_real, f_fake, degree, gamma, coef)

        m = f_real.shape[0]
        diag_x = jnp.diag(k_11)
        diag_y = jnp.diag(k_22)

        kt_xx_sum = jnp.sum(k_11, axis=-1) - diag_x
        kt_yy_sum = jnp.sum(k_22, axis=-1) - diag_y
        k_xy_sum = jnp.sum(k_12, axis=0)

        value = (jnp.sum(kt_xx_sum) + jnp.sum(kt_yy_sum)) / (m * (m - 1))
        value -= 2 * jnp.sum(k_xy_sum) / (m**2)
        return value

    
    def compute(self) -> jax.Array:
        """
        Compute the average KID value from accumulated batches.
        Always returns a scalar (0-dim array or float).
        """
        result = base.divide_no_nan(self.total, self.count)
        # If result is a 0-dim array, convert to float for easier downstream use
        if hasattr(result, 'shape') and result.shape == ():
            return float(result)
        return result
    

    def merge(self, other: "KernelInceptionDistance") -> "KernelInceptionDistance":
        """
        Merge two KernelInceptionDistance instances by summing totals and counts.
        """
        return type(self)(
            total=self.total + other.total,
            count=self.count + other.count,
            subsets=self.subsets,
            subset_size=self.subset_size,
            degree=self.degree,
            gamma=self.gamma,
            coef=self.coef,
        )
