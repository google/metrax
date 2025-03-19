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

"""A collection of different metrics for ranking models."""

import flax
import jax
import jax.numpy as jnp
from metrax import base


@flax.struct.dataclass
class AveragePrecisionAtK(base.Average):
  r"""Computes AP@k (average precision at k) metrics in JAX.

  Average precision at k (AP@k) is a metric used to evaluate the performance of
  ranking models. It measures the sum of precision at k where the item at
  the kth rank is relevant, divided by the total number of relevant items.

  Given the top :math:`K` recommendations, AP@K is calculated as:

  .. math::
      AP@K = \frac{1}{r} \sum_{k=1}^{K} Precision@k * rel(k) \\
      rel(k) =
        \begin{cases}
          1 & \text{if the item at rank } k \text{ is relevant} \\
          0 & \text{otherwise}
        \end{cases}
  """

  @classmethod
  def average_precision_at_ks(
      cls, predictions: jax.Array, labels: jax.Array, ks: jax.Array
  ):
    """Computes AP@k (average precision at k) metrics for each of k in ks.

    Args:
      predictions: A floating point 2D vector representing the prediction
        generated from the model. The shape should be (batch_size, vocab_size).
      labels: A multi-hot encoding of the true label. The shape should be
        (batch_size, vocab_size).
      ks: A 1D vector of integers representing the k's to compute the MAP@k
        metrics. The shape should be (|ks|).

    Returns:
      Rank-2 tensor of shape [batch, |ks|] containing AP@k metrics.
    """
    top_k_indices = jnp.argsort(-predictions, axis=1)[:, : jnp.max(ks)]
    labels = jnp.array(labels >= 1, dtype=jnp.float32)
    total_relevant = labels.sum(axis=1)

    def compute_ap_at_k_single(relevant_labels, total_relevant, ks):
      cumulative_precision = jnp.where(
          relevant_labels,
          base.divide_no_nan(
              jnp.cumsum(relevant_labels),
              jnp.arange(1, len(relevant_labels) + 1),
          ),
          0,
      )
      return jnp.array([
          base.divide_no_nan(jnp.sum(cumulative_precision[:k]), total_relevant)
          for k in ks
      ])

    vmap_compute_ap_at_k = jax.vmap(
        compute_ap_at_k_single, in_axes=(0, 0, None), out_axes=0
    )

    ap_at_ks = vmap_compute_ap_at_k(
        jnp.take_along_axis(labels, top_k_indices, axis=1), total_relevant, ks
    )
    return ap_at_ks

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      ks: jax.Array,
  ) -> 'AveragePrecisionAtK':
    """Updates the metric.

    Args:
      predictions: A floating point 2D vector representing the prediction
        generated from the model. The shape should be (batch_size, vocab_size).
      labels: A multi-hot encoding of the true label. The shape should be
        (batch_size, vocab_size).
      ks: A 1D vector of integers representing the k's to compute the MAP@k
        metrics. The shape should be (|ks|).

    Returns:
      The AveragePrecisionAtK metric. The shape should be (|ks|).

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `predictions`
      and `labels` are incompatible.
    """
    ap_at_ks = cls.average_precision_at_ks(predictions, labels, ks)
    count = jnp.ones((labels.shape[0], 1), dtype=jnp.float32)
    return cls(
        total=ap_at_ks.sum(axis=0),
        count=count.sum(),
    )
