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

import abc
import flax
import jax
import jax.numpy as jnp
from metrax import base


@flax.struct.dataclass
class DCGAtK(base.Average):
  r"""Computes Discounted Cumulative Gain at k metric.

  DCG tells how good a list of search results or recommendations is, based on
  the relevance of the items and their positions in the list.

  This implementation calculates :math:`DCG@k` based on the following formula:

  .. math::
      DCG@K(y, s) = \sum_{i=1}^{K} \text{gain}(y_i) \times
      \text{rank_discount}(\text{rank}(s_i))

  where

    - :math:`y_i` is the relevance label from the labels,
    - :math:`s_i` is its score from the prediction,
    - :math:`\text{rank}(s_i)` is the 1-based rank of item :math:`i`.
    - :math:`\text{gain}(y_i) = 2^{y_i} - 1`.
    - :math:`\text{rank_discount}(\text{rank}(s_i)) = \frac{1}{\log_2(\text{rank}(s_i) + 1)}`.

  We get the final formula:

  .. math::
      DCG@K(y, s) = \sum_{i=1}^{K} \frac{2^{y_i} - 1}{\log_2(\text{rank}(s_i) +
      1)}
  """

  @classmethod
  def _calculate_dcg_at_ks(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      ks: jax.Array,
  ) -> jax.Array:
    """Computes DCG@k for each example and for each k, using 'exp2' gain.

    This function is JIT-compiled. The gain calculation is fixed to 'exp2'.
    It uses jax.vmap to compute DCG for multiple k values efficiently.

    Args:
      predictions: A floating point 2D array (batch_size, vocab_size)
        representing prediction scores. Higher scores mean higher rank.
      labels: A 2D array (batch_size, vocab_size) of graded relevance scores.
      ks: A 1D array of integers representing the k values for which DCG is
        computed (e.g., jnp.array([1, 5, 10])). Shape: (|ks|,).

    Returns:
      A 2D array (batch_size, |ks|) containing DCG@k values.
    """
    gains = jnp.power(2.0, labels.astype(jnp.float32)) - 1.0
    score_ranks = jnp.argsort(jnp.argsort(-predictions, axis=1), axis=1) + 1
    score_rank_discounts = 1.0 / jnp.log2(score_ranks.astype(jnp.float32) + 1.0)
    item_contributions = gains * score_rank_discounts

    def _compute_dcg_at_k(k, current_item_contributions, current_score_ranks):
      """Computes DCG for a single k value across all examples in a batch.

      Args:
        k: A scalar array representing the single 'k' (top-k) value for which
          DCG is to be computed.
        current_item_contributions: A 2D array containing the pre-calculated
          contribution (gain * discount) for each item in each example of the
          batch. The shape should be (batch_size, vocab_size).
        current_score_ranks: A 2D array containing the 1-based rank for each
          item in each example of the batch. The shape should be (batch_size,
          vocab_size).

      Returns:
        A 1D array containing the DCG@k for each example in the batch.
        The shape should be (batch_size, ).
      """
      mask_for_k = current_score_ranks <= k
      dcg_at_k = jnp.sum(current_item_contributions * mask_for_k, axis=1)
      return dcg_at_k

    dcg_at_ks = jax.vmap(
        _compute_dcg_at_k,
        in_axes=(0, None, None),
        out_axes=1,  # Place the mapped axis(from ks) as the second axis
    )(ks, item_contributions, score_ranks)

    return dcg_at_ks

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      ks: jax.Array,
  ) -> 'DCGAtK':
    """Creates a DCGAtK metric instance from model output."""
    dcg_at_ks = cls._calculate_dcg_at_ks(predictions, labels, ks)
    num_examples = jnp.array(labels.shape[0], dtype=jnp.float32)
    return cls(
        total=jnp.sum(dcg_at_ks, axis=0),
        count=num_examples,
    )


@flax.struct.dataclass
class NDCGAtK(DCGAtK):
  r"""Computes Normalized Discounted Cumulative Gain at k (NDCG@k) metrics.

  NDCG@k normalizes DCG@k by the Ideal DCG@k (IDCG@k), which is the DCG@k
  score of a perfectly ranked list (items sorted by their true relevance).

  This implementation calculates :math:`NDCG@k` based on the following formula:

  .. math::
      NDCG@k = \frac{DCG@k}{IDCG@k}

  where

    - If :math:`IDCG@k` is 0, then :math:`NDCG@k` is defined as 0.
    - The :math:`DCG@k` calculation uses :math:`exp2` gain
      (:math:`2^{\text{relevance}} - 1`) and standard logarithmic discount
      (:math:`\frac{1}{\log_2(\text{rank} + 1)}`).
  """

  @classmethod
  def _calculate_dcg_at_ks(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      ks: jax.Array,
  ) -> jax.Array:
    """Computes NDCG@k for each example and for each k.

    Args:
      predictions: A floating point 2D array (batch_size, vocab_size)
        representing prediction scores.
      labels: A 2D array (batch_size, vocab_size) of graded relevance scores.
      ks: A 1D array of integers representing the k values. The shape should be
        (|ks|, ).

    Returns:
      A 2D array (batch_size, |ks|) containing NDCG@k values.
    """
    actual_dcg_at_ks = super()._calculate_dcg_at_ks(predictions, labels, ks)
    ideal_dcg_at_ks = super()._calculate_dcg_at_ks(
        predictions=labels,  # Use labels to determine ideal DCG.
        labels=labels,
        ks=ks,
    )
    ndcg_at_ks = base.divide_no_nan(actual_dcg_at_ks, ideal_dcg_at_ks)
    # Ensure NDCG is not greater than 1.
    ndcg_at_ks = jnp.minimum(ndcg_at_ks, 1.0)
    return ndcg_at_ks


@flax.struct.dataclass
class AveragePrecisionAtK(base.Average):
  r"""Computes AP@k (average precision at k) metrics.

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
      Rank-2 tensor of shape (batch, |ks|) containing AP@k metrics.
    """
    indices_by_rank = jnp.argsort(-predictions, axis=1)
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
          base.divide_no_nan(
              jnp.sum(
                  jnp.where(
                      jnp.arange(cumulative_precision.shape[0]) < k,
                      cumulative_precision,
                      0.0,
                  )
              ),
              total_relevant,
          )
          for k in ks
      ])

    vmap_compute_ap_at_k = jax.vmap(
        compute_ap_at_k_single, in_axes=(0, 0, None), out_axes=0
    )

    ap_at_ks = vmap_compute_ap_at_k(
        jnp.take_along_axis(labels, indices_by_rank, axis=1), total_relevant, ks
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
    num_examples = jnp.array(labels.shape[0], dtype=jnp.float32)
    return cls(
        total=ap_at_ks.sum(axis=0),
        count=num_examples,
    )


@flax.struct.dataclass
class MRR(base.Average):
  r"""Computes Mean Reciprocal Rank (MRR), supporting MRR@k for multiple k values.

  MRR is the average of the reciprocal ranks of the first relevant item
  for a set of queries.

  The mean reciprocal rank for a group of queries :math:`q` in :math:`Q` is defined as follows:

  .. math::
      MRR = \frac{1}{|Q|} \sum_{q \in Q} RR_q

  Where :math:`RR_q` is the reciprocal rank for query :math:`q`, defined as:

  .. math::
      RR_q =
        \begin{cases}
          \frac{1}{\text{rank}} & \text{if a revelant item is found} \\
          0 & \text{if no relevant item is found.}
        \end{cases}

  This implementation assumes binary relevance labels (1 for relevant, 0 for not
  relevant).
  """

  @classmethod
  def _calculate_rr_at_ks(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      ks: jax.Array,
  ) -> jax.Array:
    """Calculates the Reciprocal Rank@k for each query in a batch, for each k.

    Args:
      predictions: A 2D array of prediction scores. Higher scores indicate
        higher rank. The shape should be (batch_size, vocab_size).
      labels: A 2D array of binary relevance labels (0 or 1). The shape should
        be (batch_size, vocab_size).
      ks: A 1D array of integers representing the k cutoffs. The shape should be
        (|ks|,).

    Returns:
      A 2D array containing the Reciprocal Rank@k for each query and each k.
      Shape: (batch_size, |ks|).
    """
    indices_by_rank = jnp.argsort(-predictions, axis=1)
    labels_by_rank = jnp.take_along_axis(labels, indices_by_rank, axis=1)
    first_relevant_indices = jnp.argmax(labels_by_rank, axis=1)
    first_relevant_ranks = first_relevant_indices + 1
    # Check if a relevant item exists for each query.
    reciprocal_rank = jnp.where(
        jnp.any(labels_by_rank > 0, axis=1),
        1.0 / first_relevant_ranks.astype(jnp.float32),
        0.0,
    )

    def _compute_rr_at_k(k, reciprocal_rank, first_relevant_rank):
      rr_at_k = jnp.where(first_relevant_rank <= k, reciprocal_rank, 0.0)
      return rr_at_k

    rr_at_ks = jax.vmap(_compute_rr_at_k, in_axes=(0, None, None), out_axes=1)(
        ks, reciprocal_rank, first_relevant_ranks
    )

    return rr_at_ks

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      ks: jax.Array,
  ) -> 'MRR':
    """Creates an MRR metric instance from model output, calculating MRR@k for each k.

    Args:
      predictions: A 2D array of prediction scores. Higher scores indicate
        higher rank. The shape should be (batch_size, vocab_size).
      labels: A 2D array of binary relevance labels (0 or 1). The shape should
        be (batch_size, vocab_size).
      ks: A 1D array of integers representing the k cutoffs. The shape should be
        (|ks|, ).

    Returns:
      An MRR metric object. The 'total' field will be an array of shape
      (|ks|, ).
    """
    rr_at_ks = cls._calculate_rr_at_ks(predictions, labels, ks)
    num_queries = jnp.array(labels.shape[0], dtype=jnp.float32)
    return cls(
        total=rr_at_ks.sum(axis=0),
        count=num_queries,
    )


@flax.struct.dataclass
class TopKRankingMetric(base.Average, abc.ABC):
  """Abstract base class for Top-K ranking metrics like Precision@k and Recall@k.

  This class provides common functionality for calculating metrics that evaluate
  the quality of the top k items in a ranked list. Subclasses must implement
  the `_calculate_metric_at_ks` method to define the specific metric
  computation (e.g., precision, recall).

  The `from_model_output` method is a factory method that computes the metric
  values for a batch of predictions and labels, and aggregates them.
  """

  @staticmethod
  def _get_relevant_at_k(
      predictions: jax.Array, labels: jax.Array, ks: jax.Array
  ) -> jax.Array:
    """Computes the number of relevant items at each k.

    This static method processes predictions and labels to determine the
    number of relevant items at specified k-values.

    Args:
      predictions: A floating point 2D array representing the prediction scores
        from the model. Higher scores indicate higher relevance. The shape
        should be (batch_size, vocab_size).
      labels: A multi-hot encoding (0 or 1) of the true labels. The shape should
        be (batch_size, vocab_size).
      ks: A 1D array of integers representing the k's to compute the P@k
        metrics. The shape should be (|ks|).

    Returns:
      relevant_at_k: A 2D array of shape (batch_size, |ks|). Each element [i, j]
      is the number of relevant items among the top ks[j] recommendations for
      the i-th example in the batch.
    """
    labels = jnp.array(labels >= 1, dtype=jnp.float32)
    indices_by_rank = jnp.argsort(-predictions, axis=1)
    labels_by_rank = jnp.take_along_axis(labels, indices_by_rank, axis=1)
    relevant_by_rank = jnp.cumsum(labels_by_rank, axis=1)

    vocab_size = predictions.shape[1]
    k_indices = jnp.minimum(ks - 1, vocab_size - 1)
    relevant_at_k = relevant_by_rank[:, k_indices]

    return relevant_at_k

  @classmethod
  @abc.abstractmethod
  def _calculate_metric_at_ks(
      cls, predictions: jax.Array, labels: jax.Array, ks: jax.Array
  ) -> jax.Array:
    """Computes the specific metric (e.g., P@k, R@k) values per example for each k.

    This method must be implemented by concrete subclasses (e.g., PrecisionAtK,
    RecallAtK) to define the actual calculation of the metric based on
    predictions, labels, and k-values.

    Args:
      predictions: A floating point 2D array representing the prediction scores
        from the model.
      labels: A multi-hot encoding of the true labels.
      ks: A 1D array of integers representing the k's.

    Returns:
      A rank-2 array of shape (batch_size, |ks|) containing the metric
      values for each example in the batch and each specified k.
    """
    raise NotImplementedError('Subclasses must implement this method.')

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      ks: jax.Array,
  ) -> 'TopKRankingMetric':
    """Creates a metric instance from model output.

    This class method computes the specific ranking metric (defined by the
    subclass's implementation of `_calculate_metric_at_ks`) for each example
    in the batch. It then aggregates these values (sum of metric values and
    count of examples) into a metric object. This object can later be used
    to compute the mean metric value (e.g., Mean Precision@k) by calling
    its `.compute()` method (inherited from `base.Average`).

    Args:
      predictions: A floating point 2D array representing the prediction scores
        from the model. The shape should be (batch_size, vocab_size).
      labels: A multi-hot encoding (0 or 1, or counts) of the true labels. The
        shape should be (batch_size, vocab_size).
      ks: A 1D array of integers representing the k's to compute the metrics.
        The shape should be (|ks|).

    Returns:
      An instance of the calling class (e.g., PrecisionAtK, RecallAtK)
      with `total` and `count` fields populated. The `total` field will
      have shape (|ks|), representing the sum of metric values for each k
      across the batch, and `count` will be a scalar representing the
      number of examples in the batch.
    """
    metric_at_ks = cls._calculate_metric_at_ks(predictions, labels, ks)
    num_examples = jnp.array(labels.shape[0], dtype=jnp.float32)
    return cls(
        total=metric_at_ks.sum(axis=0),
        count=num_examples,
    )


@flax.struct.dataclass
class PrecisionAtK(TopKRankingMetric):
  r"""Computes P@k (precision at k) metrics.

  Precision at k (P@k) is a metric that measures the proportion of
  relevant items found in the top k recommendations. It answers the question:
  "Out of the K items recommended, how many are actually relevant?"

  Given the top :math:`K` recommendations, P@K is calculated as:

  .. math::
      Precision@K = \frac{\text{Number of relevant items in top K}}{K}
  """

  @classmethod
  def _calculate_metric_at_ks(
      cls, predictions: jax.Array, labels: jax.Array, ks: jax.Array
  ) -> jax.Array:
    """Computes P@k (precision at k) metrics for each of k in ks for each example.

    This method implements the core logic for calculating Precision@k.
    It utilizes the `_get_relevant_at_k` helper from the base
    class to get the number of relevant items at each k, and then divides
    by k (clamped by vocabulary size) to get the precision.

    Args:
      predictions: A floating point 2D array representing the prediction scores
        from the model. The shape should be (batch_size, vocab_size).
      labels: A multi-hot encoding (0 or 1, or counts) of the true labels. The
        shape should be (batch_size, vocab_size).
      ks: A 1D array of integers representing the k's to compute the P@k
        metrics. The shape should be (|ks|).

    Returns:
      A rank-2 array of shape (batch_size, |ks|) containing P@k metrics
      for each example and each k.
    """
    relevant_at_k = cls._get_relevant_at_k(predictions, labels, ks)
    vocab_size = labels.shape[1]
    denominator_p_at_k = jnp.minimum(ks.astype(jnp.float32), vocab_size)
    return base.divide_no_nan(relevant_at_k, denominator_p_at_k[jnp.newaxis, :])


@flax.struct.dataclass
class RecallAtK(TopKRankingMetric):
  r"""Computes R@k (recall at k) metrics.

  Recall at k (R@k) is a metric that measures the proportion of
  relevant items that are found in the top k recommendations, out of the
  total number of relevant items for a given user/query. It answers the
  question:
  "Out of all the items that are truly relevant, how many did we find in the top
  K?"

  Given the top :math:`K` recommendations, R@K is calculated as:

  .. math::
      Recall@K = \frac{\text{Number of relevant items in top K}}{\text{Total
      number of relevant items}}
  """

  @classmethod
  def _calculate_metric_at_ks(
      cls, predictions: jax.Array, labels: jax.Array, ks: jax.Array
  ) -> jax.Array:
    """Computes R@k (recall at k) metrics for each of k in ks for each example.

    This method implements the core logic for calculating Recall@k.
    It utilizes the `_get_relevant_at_k` helper from the base
    class to get the number of relevant items at each k and the binarized
    labels.
    The number of relevant items at k is then divided by the total number of
    relevant items for that example to get the recall.

    Args:
      predictions: A floating point 2D array representing the prediction scores
        from the model. The shape should be (batch_size, vocab_size).
      labels: A multi-hot encoding (0 or 1, or counts) of the true labels. The
        shape should be (batch_size, vocab_size).
      ks: A 1D array of integers representing the k's to compute the R@k
        metrics. The shape should be (|ks|).

    Returns:
      A rank-2 array of shape (batch_size, |ks|) containing R@k metrics
      for each example and each k.
    """
    relevant_at_k = cls._get_relevant_at_k(predictions, labels, ks)
    total_relevant = jnp.sum(jnp.array(labels >= 1, dtype=jnp.float32), axis=1)
    return base.divide_no_nan(relevant_at_k, total_relevant[:, jnp.newaxis])
