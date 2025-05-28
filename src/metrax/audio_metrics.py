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

"""A collection of different metrics for audio models."""

import flax
import jax
import jax.numpy as jnp
from metrax import base


@flax.struct.dataclass
class SNR(base.Average):
  r"""SNR (Signal-to-Noise Ratio) Metric for audio.

  This class calculates the Signal-to-Noise Ratio (SNR) in decibels (dB)
  between a predicted audio signal and a ground truth audio signal,
  and averages it over a dataset.

  The SNR is defined as:

  .. math::

      SNR_{dB} = 10 \cdot \log_{10} \left( \frac{P_{signal}}{P_{noise}}
      \right)

  Where:
    - :math:`P_{signal}` is the power of the ground truth signal (`target`).
      By default (`zero_mean=False`), this is the mean of the squared `target`
      values.
      If `zero_mean=True`, it's the variance of the `target` values.
    - :math:`P_{noise}` is the power of the noise component, which is defined as
      the difference between the `target` and `preds` (`target - preds`).
      By default (`zero_mean=False`), this is the mean of the squared noise
      values.
      If `zero_mean=True`, it's the variance of the noise values.
  """

  @staticmethod
  def _calculate_snr(
      preds: jax.Array,
      target: jax.Array,
      zero_mean: bool = False,
  ) -> jax.Array:
    """Computes SNR (Signal-to-Noise Ratio) values for a batch of audio signals.

    Args:
        preds: The estimated or predicted audio signal. JAX Array.
        target: The ground truth audio signal. JAX Array.
        zero_mean: If True, subtracts the mean from the signal and noise before
          calculating their respective powers. Defaults to False.

    Returns:
        A 1D JAX array representing the SNR in decibels (dB) for each example
        in the batch.
    """
    if preds.shape != target.shape:
      raise ValueError(
          f'Input signals must have the same shape, but got {preds.shape} and'
          f' {target.shape}'
      )

    target_processed, preds_processed = jax.lax.cond(
        zero_mean,
        lambda t, p: (
            t - jnp.mean(t, axis=-1, keepdims=True),
            p - jnp.mean(p, axis=-1, keepdims=True),
        ),
        lambda t, p: (t, p),
        target,
        preds,
    )
    noise = target_processed - preds_processed
    eps = jnp.finfo(preds.dtype).eps
    signal_power = jnp.sum(target_processed**2, axis=-1) + eps
    noise_power = jnp.sum(noise**2, axis=-1) + eps

    snr = 10 * jnp.log10(base.divide_no_nan(signal_power, noise_power))
    return snr

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      targets: jax.Array,
      zero_mean: bool = False,
  ) -> 'SNR':
    """Computes SNR for a batch of audio signals and creates an SNR metric instance.

    Args:
        predictions: A JAX array of predicted audio signals.
        targets: A JAX array of ground truth audio signals.
        zero_mean: If True, subtracts the mean from the signal and noise before
          calculating their respective powers.

    Returns:
        An SNR instance containing the SNR value for the current batch,
        ready for averaging.
    """
    batch_snr_value = cls._calculate_snr(
        predictions,
        targets,
        zero_mean=zero_mean,
    )
    return super().from_model_output(values=batch_snr_value)
