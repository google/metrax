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

"""A collection of different metrics for image models."""

import jax.numpy as jnp
from jax import random, lax
import flax
import jax
from clu import metrics as clu_metrics
from metrax import base

KID_DEFAULT_SUBSETS = 100
KID_DEFAULT_SUBSET_SIZE = 1000
KID_DEFAULT_DEGREE = 3
KID_DEFAULT_GAMMA = None
KID_DEFAULT_COEF = 1.0


def _gaussian_kernel1d(sigma, radius):
  r"""Generates a 1D normalized Gaussian kernel.

  This function creates a 1D Gaussian kernel, which can be used for smoothing
  operations. The kernel is centered at zero and its values are determined by
  the Gaussian function:

  .. math::
    \phi(x) = e^{-\frac{x^2}{2\sigma^2}}

  The resulting kernel :math:`\phi(x)` is then normalized by dividing each
  element by the sum of all elements, so that the sum of the kernel's elements
  is 1. This function assumes an order of 0 for the Gaussian derivative (i.e.,
  a standard smoothing kernel).

  Args:
    sigma (float): The standard deviation (:math:`\sigma`) of the Gaussian
      distribution. This controls the "width" or "spread" of the kernel.
    radius (int): The radius of the kernel. The kernel will include points from
      :math:`-radius` to :math:`+radius`. The total size of the kernel will be
      :math:`2 \times radius + 1`.

  Returns:
    jnp.ndarray: A 1D JAX array representing the normalized Gaussian kernel.
  """
  sigma2 = sigma * sigma
  x = jnp.arange(-radius, radius + 1)
  phi_x = jnp.exp(-0.5 / sigma2 * x**2)
  phi_x = phi_x / phi_x.sum()
  return phi_x


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


@flax.struct.dataclass
class KID(base.Average):
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
        """
        Create a KID instance from model output.
        also it computes average output and then store it in the instance.        
        """
        if subsets <= 0 or subset_size <= 0 or degree <= 0 or (gamma is not None and gamma <= 0) or coef <= 0:
            raise ValueError("All parameters must be positive and non-zero.")
        # Compute KID for this batch and then store the aggregated response.
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
            kid = cls._compute_mmd_static(f_real_subset, f_fake_subset, degree, gamma, coef)
            kid_scores.append(kid)
        kid_mean = jnp.mean(jnp.array(kid_scores))

        return cls(
            total=kid_mean,
            count=1.0,
            subsets=subsets,
            subset_size=subset_size,
            degree=degree,
            gamma=gamma,
            coef=coef,
        )


    @staticmethod
    def _compute_mmd_static(f_real: jax.Array, f_fake: jax.Array, degree: int, gamma: float, coef: float) -> float:
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

@flax.struct.dataclass
class SSIM(base.Average):
  r"""SSIM (Structural Similarity Index Measure) Metric.

  This class calculates the structural similarity between predicted and target
  images and averages it over a dataset. SSIM is a perception-based model that
  considers changes in structural information, luminance, and contrast.

  The general SSIM formula considers three components: luminance (l),
      contrast (c), and structure (s):

      .. math::
        SSIM(x, y) = [l(x, y)]^\alpha \cdot [c(x, y)]^\beta \cdot [s(x,
        y)]^\gamma

      Where:
        - Luminance comparison:
          :math:`l(x, y) = \frac{2\mu_x\mu_y + c_1}{\mu_x^2 + \mu_y^2 + c_1}`
        - Contrast comparison:
          :math:`c(x, y) = \frac{2\sigma_x\sigma_y + c_2}{\sigma_x^2 +
          \sigma_y^2 + c_2}`
        - Structure comparison:
          :math:`s(x, y) = \frac{\sigma_{xy} + c_3}{\sigma_x\sigma_y + c_3}`

      This implementation uses a common simplified form where :math:`\alpha =
      \beta = \gamma = 1` and :math:`c_3 = c_2 / 2`.

      This leads to the combined formula:

      .. math::
        SSIM(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 +
        \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}

      In these formulas:
        - :math:`\mu_x` and :math:`\mu_y` are the local means of :math:`x` and
        :math:`y`.
        - :math:`\sigma_x^2` and :math:`\sigma_y^2` are the local variances of
        :math:`x` and :math:`y`.
        - :math:`\sigma_{xy}` is the local covariance of :math:`x` and
        :math:`y`.
        - :math:`c_1 = (K_1 L)^2` and :math:`c_2 = (K_2 L)^2` are stabilization
        constants,
          where :math:`L` is the dynamic range of pixel values, and :math:`K_1,
          K_2` are small constants (e.g., 0.01 and 0.03).
  """

  @staticmethod
  def _calculate_ssim(
      img1: jnp.ndarray,
      img2: jnp.ndarray,
      max_val: float,
      filter_size: int = 11,
      filter_sigma: float = 1.5,
      k1: float = 0.01,
      k2: float = 0.03,
  ) -> jnp.ndarray:
    """Computes SSIM (Structural Similarity Index Measure) values for a batch of images.

    This function calculates the SSIM between two batches of images (`img1` and
    `img2`). If the images have multiple channels, SSIM is calculated for each
    channel independently, and then the mean SSIM across channels is returned.

    Args:
      img1: The first batch of images, expected shape ``(batch, height, width,
        channels)``.
      img2: The second batch of images, expected shape ``(batch, height, width,
        channels)``.
      max_val: The dynamic range of the pixel values (e.g., 1.0 for images
        normalized to [0,1] or 255 for uint8 images).
      filter_size: The size of the Gaussian filter window used for calculating
        local statistics. Must be an odd integer.
      filter_sigma: The standard deviation of the Gaussian filter.
      k1: A small constant used in the SSIM formula to stabilize the luminance
        comparison.
      k2: A small constant used in the SSIM formula to stabilize the
        contrast/structure comparison.

    Returns:
      A 1D JAX array of shape ``(batch,)`` containing the SSIM value for each
      image pair in the batch.
    """
    if img1.shape != img2.shape:
      raise ValueError(
          f'Input images must have the same shape, but got {img1.shape} and'
          f' {img2.shape}'
      )
    if img1.ndim != 4:  # (batch, H, W, C)
      raise ValueError(
          'Input images must be 4D tensors (batch, height, width, channels),'
          f' but got {img1.ndim}D'
      )
    if img1.shape[-3] < filter_size or img1.shape[-2] < filter_size:
      raise ValueError(
          f'Image dimensions ({img1.shape[-3]}x{img1.shape[-2]}) must be at'
          f' least filter_size x filter_size ({filter_size}x{filter_size}).'
      )

    num_channels = img1.shape[-1]
    img1 = img1.astype(jnp.float32)
    img2 = img2.astype(jnp.float32)

    gaussian_kernal_1d = _gaussian_kernel1d(
        filter_sigma, (filter_size - 1) // 2
    )
    gaussian_kernel_2d = jnp.outer(gaussian_kernal_1d, gaussian_kernal_1d)
    # Kernel for convolution: (H_k, W_k, C_in=1, C_out=1)
    kernel_conv = gaussian_kernel_2d[:, :, jnp.newaxis, jnp.newaxis]

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    def _calculate_ssim_for_channel(x_ch, y_ch, conv_kernel, c1, c2):
      r"""Calculates the Structural Similarity Index (SSIM) for a single channel.

      This function computes the SSIM between two single-channel image arrays
      (:math:`x_{ch}` and :math:`y_{ch}`) using a precomputed Gaussian kernel
      for local statistics. The SSIM metric quantifies image quality
      degradation based on perceived changes in structural information, also
      incorporating important perceptual phenomena like luminance and contrast
      masking.

      The general SSIM formula considers three components: luminance (l),
      contrast (c), and structure (s):

      .. math::
        SSIM(x, y) = [l(x, y)]^\alpha \cdot [c(x, y)]^\beta \cdot [s(x,
        y)]^\gamma

      Where:
        - Luminance comparison:
          :math:`l(x, y) = \frac{2\mu_x\mu_y + c_1}{\mu_x^2 + \mu_y^2 + c_1}`
        - Contrast comparison:
          :math:`c(x, y) = \frac{2\sigma_x\sigma_y + c_2}{\sigma_x^2 +
          \sigma_y^2 + c_2}`
        - Structure comparison:
          :math:`s(x, y) = \frac{\sigma_{xy} + c_3}{\sigma_x\sigma_y + c_3}`

      This implementation uses a common simplified form where :math:`\alpha =
      \beta = \gamma = 1` and :math:`c_3 = c_2 / 2`.

      This leads to the combined formula:

      .. math::
        SSIM(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 +
        \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}

      In these formulas:
        - :math:`\mu_x` and :math:`\mu_y` are the local means of :math:`x` and
        :math:`y`.
        - :math:`\sigma_x^2` and :math:`\sigma_y^2` are the local variances of
        :math:`x` and :math:`y`.
        - :math:`\sigma_{xy}` is the local covariance of :math:`x` and
        :math:`y`.
        - :math:`c_1 = (K_1 L)^2` and :math:`c_2 = (K_2 L)^2` are stabilization
        constants,
          where :math:`L` is the dynamic range of pixel values, and :math:`K_1,
          K_2` are small constants (e.g., 0.01 and 0.03).

      Args:
        x_ch (jnp.ndarray): The first input image channel. Expected shape is
          ``(batch, Height, Width, 1)``.
        y_ch (jnp.ndarray): The second input image channel. Expected shape is
          ``(batch, Height, Width, 1)``.
        conv_kernel (jnp.ndarray): The 2D Gaussian kernel, reshaped to 4D, used
          for calculating local windowed statistics (mean, variance,
          covariance). Expected shape is ``(Kernel_H, Kernel_W, 1, 1)``.
        c1 (float): Stabilization constant for the luminance and mean component,
          :math:`(K_1 L)^2`.
        c2 (float): Stabilization constant for the variance and covariance
          component, :math:`(K_2 L)^2`.

      Returns:
        jnp.ndarray: A scalar JAX array (or an array of scalars if batch size >
        1)
        representing the mean SSIM value(s) for the input channel(s).
      """
      # x_ch, y_ch are (batch, H, W, 1)
      dn = lax.conv_dimension_numbers(
          x_ch.shape, conv_kernel.shape, ('NHWC', 'HWIO', 'NHWC')
      )

      mu_x = lax.conv_general_dilated(
          x_ch,
          conv_kernel,
          window_strides=(1, 1),
          padding='VALID',
          dimension_numbers=dn,
      )
      mu_y = lax.conv_general_dilated(
          y_ch,
          conv_kernel,
          window_strides=(1, 1),
          padding='VALID',
          dimension_numbers=dn,
      )

      mu_x_sq = mu_x**2
      mu_y_sq = mu_y**2
      mu_x_mu_y = mu_x * mu_y

      sigma_x_sq = (
          lax.conv_general_dilated(
              x_ch**2,
              conv_kernel,
              window_strides=(1, 1),
              padding='VALID',
              dimension_numbers=dn,
          )
          - mu_x_sq
      )
      sigma_y_sq = (
          lax.conv_general_dilated(
              y_ch**2,
              conv_kernel,
              window_strides=(1, 1),
              padding='VALID',
              dimension_numbers=dn,
          )
          - mu_y_sq
      )
      sigma_xy = (
          lax.conv_general_dilated(
              x_ch * y_ch,
              conv_kernel,
              window_strides=(1, 1),
              padding='VALID',
              dimension_numbers=dn,
          )
          - mu_x_mu_y
      )

      numerator1 = 2 * mu_x_mu_y + c1
      numerator2 = 2 * sigma_xy + c2
      denominator1 = mu_x_sq + mu_y_sq + c1
      denominator2 = sigma_x_sq + sigma_y_sq + c2

      ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
      return jnp.mean(
          ssim_map, axis=(1, 2, 3)
      )  # Mean over H, W, C (which is 1 here for the map)

    ssim_per_channel_list = []
    for i in range(num_channels):
      img1_c = lax.dynamic_slice_in_dim(
          img1, i * 1, 1, axis=3
      )  # (batch, H, W, 1)
      img2_c = lax.dynamic_slice_in_dim(
          img2, i * 1, 1, axis=3
      )  # (batch, H, W, 1)

      ssim_for_channel = _calculate_ssim_for_channel(
          img1_c, img2_c, kernel_conv, c1, c2
      )
      ssim_per_channel_list.append(ssim_for_channel)

    ssim_scores_stacked = jnp.stack(
        ssim_per_channel_list, axis=-1
    )  # (batch, num_channels)
    return jnp.mean(ssim_scores_stacked, axis=-1)  # (batch,)

  @classmethod
  def from_model_output(  # type: ignore[override]
      cls,
      predictions: jax.Array,  # Represents predicted images (y_pred)
      targets: jax.Array,  # Represents ground truth images (y_true)
      max_val: float,  # Dynamic range of pixel values
      filter_size: int = 11,
      filter_sigma: float = 1.5,
      k1: float = 0.01,
      k2: float = 0.03,
  ) -> 'SSIM':
    """Computes SSIM for a batch of images and creates an SSIM metric instance.

    This method takes batches of predicted and target images, calculates their
    SSIM values, and then initializes an SSIM metric object suitable for
    aggregation across multiple batches.

    Args:
        predictions: A JAX array of predicted images, with shape ``(batch,
          height, width, channels)``.
        targets: A JAX array of ground truth images, with shape ``(batch,
          height, width, channels)``.
        max_val: The maximum possible pixel value (dynamic range) of the images
          (e.g., 1.0 for float images in [0,1], 255 for uint8 images).
        filter_size: The size of the Gaussian filter window used in SSIM
          calculation (default is 11).
        filter_sigma: The standard deviation of the Gaussian filter (default is
          1.5).
        k1: SSIM stability constant for the luminance term (default is 0.01).
        k2: SSIM stability constant for the contrast/structure term (default is
          0.03).

    Returns:
        An SSIM instance containing the SSIM values for the current batch,
        ready for averaging.
    """
    # shape (batch_size,)
    batch_ssim_values = cls._calculate_ssim(
        predictions,
        targets,
        max_val=max_val,
        filter_size=filter_size,
        filter_sigma=filter_sigma,
        k1=k1,
        k2=k2,
    )
    return super().from_model_output(values=batch_ssim_values)