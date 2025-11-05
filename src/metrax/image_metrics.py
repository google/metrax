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

from clu import metrics as clu_metrics
import flax
import jax
from jax import lax
import jax.numpy as jnp
from metrax import base


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
  def from_model_output(
      cls,
      predictions: jax.Array,
      targets: jax.Array,
      max_val: float,
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


@flax.struct.dataclass
class IoU(base.Average):
  r"""Measures Intersection over Union (IoU) for semantic segmentation.

  The general formula for IoU for a single class is:
  $IoU_{class} = \frac{TP}{TP + FP + FN}$
  where TP, FP, FN are True Positives, False Positives, and False Negatives.

  **Per-Batch Processing:**
  For each input batch, a mean IoU is calculated. This involves:
  1. Aggregating TP, FP, and FN pixel counts for each specified target class
     (from the required `target_class_ids` list) across all samples within the
     batch.
  2. Computing IoU for each of these classes using the batch-aggregated counts:
     $IoU_{class} = \frac{TP}{TP + FP + FN + \epsilon}$.
  3. Averaging these per-class IoU scores to get a single value for the batch.
     - If `target_class_ids` is empty, an array of zeros of shape `(B,)`
       (where `B` is batch size) is produced by `_calculate_iou`.
     - Otherwise, a scalar `jnp.ndarray` (shape `()`) representing the mean
       IoU is produced.

  **Accumulation & Final Metric:**
  This class inherits from `base.Average`. It accumulates the results from
  per-batch processing and `compute()` returns the final mean IoU as a scalar
  `jnp.ndarray` (shape `()`).
  """

  @staticmethod
  def _calculate_iou(
      targets: jnp.ndarray,
      predictions: jnp.ndarray,
      target_class_ids: jnp.ndarray,
      epsilon: float = 1e-7,
  ) -> jnp.ndarray:
    r"""Computes mean IoU for a processed batch by class-wise aggregation using jax.vmap.

    Per-batch processing: For each target class in the provided
    `target_class_ids` list, True Positives (TP), False Positives (FP), and
    False Negatives (FN) are summed across all items in the input batch.
    The IoU for that class is $TP / (TP + FP + FN + \epsilon)$.
    These per-class IoU scores are then averaged. If `target_class_ids` is
    empty, a scalar 0.0 is returned.

    Args:
      targets: Ground truth segmentation masks. Shape is `(B, H, W)`, integer
        class labels. (B: batch size, H: height, W: width)
      predictions: Predicted segmentation masks. Shape is `(B, H, W)`, integer
        class labels.
      target_class_ids: An array of integer class IDs for which to compute IoU.
      epsilon: Small float added to the denominator for numerical stability.
        Default is `1e-7`.

    Returns:
      scalar `jnp.ndarray` (shape `()`) mean IoU for the batch. Returns 0.0
      if `target_class_ids` is empty.
    """
    if target_class_ids.shape[0] == 0:
      return jnp.array(0.0, dtype=jnp.float32)

    def _calculate_iou_for_single_class(
        class_id: jnp.ndarray,
    ) -> jnp.ndarray:
      target_is_class = (targets == class_id)
      pred_is_class = (predictions == class_id)
      intersection = jnp.sum(jnp.logical_and(target_is_class, pred_is_class))
      union = jnp.sum(jnp.logical_or(target_is_class, pred_is_class))
      return intersection / (union + epsilon)

    iou_scores_per_class = jax.vmap(_calculate_iou_for_single_class)(
        target_class_ids
    )

    return jnp.mean(iou_scores_per_class)

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      targets: jax.Array,
      num_classes: int,
      target_class_ids: jax.Array,
      from_logits: bool = False,
      epsilon: float = 1e-7,
  ) -> 'IoU':
    """Creates an `IoU` instance from a batch of model outputs.

    Per-batch processing:
    1. Preprocesses `predictions` and `targets` into integer label masks of
       shape `(B, H, W)`. (B: batch size, H: height, W: width).
    2. Calls `_calculate_iou` using the provided `target_class_ids` to compute
       the batch's mean IoU.

    Args:
      predictions: `jax.Array`. Model predictions. - If `from_logits` is `True`:
        shape `(B, H, W, C)` (C: `num_classes`). - If `from_logits` is `False`:
        shape `(B, H, W)` or `(B, H, W, 1)`.
      targets: `jax.Array`. Ground truth segmentation masks. Shape `(B, H, W)`
        or `(B, H, W, 1)`, integer class labels.
      num_classes: Total number of distinct classes (`C`). Integer.
      target_class_ids: An array of integer class IDs for which to compute IoU.
      from_logits: `bool`. If `True`, `predictions` are logits and argmax is
        applied. Default is `False`.
      epsilon: `float`. Small value for stable IoU calculation. Default is
        `1e-7`.

    Returns:
      An `IoU` metric instance updated with the IoU score from this batch.
    """
    # Preprocessing predictions and targets to be (batch, H, W) integer labels
    if from_logits:
      if predictions.ndim != 4 or predictions.shape[-1] != num_classes:
        raise ValueError(
            'Logit predictions must be 4D (batch, H, W, num_classes) with last'
            f' dim matching num_classes. Got shape {predictions.shape} and'
            f' num_classes {num_classes}'
        )
      processed_predictions = jnp.argmax(predictions, axis=-1).astype(jnp.int32)
    else:
      if predictions.ndim == 4 and predictions.shape[-1] == 1:
        processed_predictions = jnp.squeeze(predictions, axis=-1).astype(
            jnp.int32
        )
      elif predictions.ndim == 3:
        processed_predictions = predictions.astype(jnp.int32)
      else:
        raise ValueError(
            'Predictions (if not from_logits) must be 3D (batch, H, W) or '
            f'4D (batch, H, W, 1). Got shape {predictions.shape}'
        )
    if targets.ndim == 4 and targets.shape[-1] == 1:
      processed_targets = jnp.squeeze(targets, axis=-1).astype(jnp.int32)
    elif targets.ndim == 3:
      processed_targets = targets.astype(jnp.int32)
    else:
      raise ValueError(
          'Targets must be 3D (batch, H, W) or 4D (batch, H, W, 1). '
          f'Got shape {targets.shape}'
      )

    iou_score = cls._calculate_iou(
        targets=processed_targets,
        predictions=processed_predictions,
        target_class_ids=target_class_ids,
        epsilon=epsilon,
    )
    return super().from_model_output(values=iou_score)


@flax.struct.dataclass
class PSNR(base.Average):
  r"""PSNR (Peak Signal-to-Noise Ratio)  Metric.

  This class calculates the Peak Signal-to-Noise Ratio (PSNR) between two images
  to measure the quality of a reconstructed image compared to a reference.

  .. math::

      \text{PSNR}(I, J) = 10 \cdot \log_{10} \left(
      \frac{\max(I)^2}{\text{MSE}(I, J)} \right)

  Where:
    - :math:`\max(I)` is the maximum possible pixel value of the input image.
    - :math:`\text{MSE}(I, J)` is the mean squared error between images
    :math:`I` and :math:`J`.
  """

  @staticmethod
  def _calculate_psnr(
      img1: jnp.ndarray,
      img2: jnp.ndarray,
      max_val: float,
      eps: float = 0,
  ) -> jnp.ndarray:
    """Computes PSNR (Peak Signal-to-Noise Ratio) values.

    Args:
            img1: Predicted images, shape ``(batch, H, W, C)``.
            img2: Ground‑truth images, same shape as ``img1``.
            max_val: Dynamic range of the images (e.g. ``1.0`` or ``255``).
            eps: Small constant to avoid ``log(0)`` when images are identical.

        Returns:
          A 1D JAX array of shape ``(batch,)`` containing PSNR in dB.
    """
    if img1.shape != img2.shape:
      raise ValueError(
          f'Input images must have the same shape, got {img1.shape} and'
          f' {img2.shape}.'
      )
    if img1.ndim != 4:  # (batch, H, W, C)
      raise ValueError(
          'Inputs must be 4‑D (batch, height, width, channels), got'
          f' {img1.ndim}‑D.'
      )

    img1 = img1.astype(jnp.float32)
    img2 = img2.astype(jnp.float32)

    # Mean‑squared error per image.
    mse = jnp.mean(jnp.square(img1 - img2), axis=(1, 2, 3))
    mse = jnp.maximum(mse, eps)

    psnr = 20.0 * jnp.log10(max_val) - 10.0 * jnp.log10(mse)
    return psnr

  @classmethod
  def from_model_output(
      cls,
      predictions: jnp.ndarray,
      targets: jnp.ndarray,
      max_val: float,
  ) -> 'PSNR':
    """Computes PSNR for a batch of images and creates an PSNR metric instance.

    Args:
        predictions: A JAX array of predicted images, with shape ``(batch, H, W,
          C)``.
        targets: A JAX array of ground truth images, with shape ``(batch, H, W,
          C)``.
        max_val: The maximum possible pixel value (dynamic range) of the images
          (e.g., 1.0 for float images in [0,1], 255 for uint8 images).

    Returns:
        A ``PSNR`` instance containing per‑image PSNR values.
    """
    batch_psnr = cls._calculate_psnr(predictions, targets, max_val=max_val)
    return super().from_model_output(values=batch_psnr)


@flax.struct.dataclass
class Dice(clu_metrics.Metric):
  r"""Computes the Dice coefficient between `y_true` and `y_pred`.

  Dice is a similarity measure used to measure overlap between two samples.
  A Dice score of 1 indicates perfect overlap; 0 indicates no overlap.

  The formula is:

  .. math::

      \text{Dice} = \frac{2 \cdot \sum (y_{\text{true}} \cdot y_{\text{pred}})}
                          {\sum y_{\text{true}} + \sum y_{\text{pred}} +
                          \epsilon}

  Attributes:
      intersection: Sum of element-wise product between `y_true` and `y_pred`.
      sum_true: Sum of y_true across all examples.
      sum_pred: Sum of y_pred across all examples.
  """

  intersection: jax.Array
  sum_pred: jax.Array
  sum_true: jax.Array

  @classmethod
  def empty(cls) -> 'Dice':
    return cls(
        intersection=jnp.array(0.0, jnp.float32),
        sum_pred=jnp.array(0.0, jnp.float32),
        sum_true=jnp.array(0.0, jnp.float32),
    )

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
  ) -> 'Dice':
    """Updates the metric.

    Args:
        predictions: A floating point vector whose values are in the range [0,
          1]. The shape should be (batch_size,).
        labels: True value. The value is expected to be 0 or 1. The shape should
          be (batch_size,).

    Returns:
        Updated Dice metric.
    """
    predictions = jnp.asarray(predictions, jnp.float32)
    labels = jnp.asarray(labels, jnp.float32)

    intersection = jnp.sum(predictions * labels)
    sum_pred = jnp.sum(predictions)
    sum_true = jnp.sum(labels)

    return cls(
        intersection=intersection,
        sum_pred=sum_pred,
        sum_true=sum_true,
    )

  def merge(self, other: 'Dice') -> 'Dice':
    return type(self)(
        intersection=self.intersection + other.intersection,
        sum_pred=self.sum_pred + other.sum_pred,
        sum_true=self.sum_true + other.sum_true,
    )

  def compute(self) -> jax.Array:
    """Returns the final Dice coefficient."""
    epsilon = 1e-7
    return (2.0 * self.intersection) / (self.sum_pred + self.sum_true + epsilon)
