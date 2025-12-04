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

"""A collection of different metrics for NLP models."""

import abc
import collections
import math
from clu import metrics as clu_metrics
import flax
import jax
import jax.numpy as jnp
from metrax import base


def _get_single_n_grams(segment: list[str], order: int):
  """Generates a counter of n-grams from a list of tokens for a specific n.

  Args:
    segment: list. Text segment from which n-grams will be extracted.
    order: The order of n-grams.

  Returns:
    A collections.Counter mapping n-gram tuples to their counts.
  """
  return collections.Counter(zip(*[segment[i:] for i in range(order)]))


def _get_ngrams(segment: list[str], max_order: int):
  """Extracts all n-grams up to a given maximum order from an input segment.

  Args:
      segment: list. Text segment from which n-grams will be extracted.
      max_order: int. Maximum length in tokens of the n-grams returned by this
        method.

  Returns:
    A collections.Counter mapping n-gram tuples to their counts for all orders.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    ngram_counts.update(_get_single_n_grams(segment, order))
  return ngram_counts


def _lcs_length(str1: list[str], str2: list[str]) -> int:
  """Computes the length of the Longest Common Subsequence (LCS)."""
  lengths = [[0 for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
  for i, x in enumerate(str1):
    for j, y in enumerate(str2):
      if x == y:
        lengths[i + 1][j + 1] = lengths[i][j] + 1
      else:
        lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
  return lengths[len(str1)][len(str2)]


@flax.struct.dataclass
class BLEU(clu_metrics.Metric):
  r"""Computes the BLEU score for sequence generation.

  BLEU measures the similarity between a machine-generated candidate translation
  and one or more human reference translations, focusing on matching n-grams.

  It's calculated as:

  .. math::
      BLEU = \text{BP} \cdot \exp\left( \sum_{n=1}^{N} w_n \log p_n \right)

  where:
    - :math:`p_n` is the modified n-gram precision for n-grams of order n.
    - :math:`N` is the maximum n-gram order considered (typically 4).
    - :math:`w_n` are weights for each order (typically uniform, 1/N).
    - :math:`\text{BP}` is the Brevity Penalty.

  This implementation uses uniform weights and calculates statistics
  incrementally.

  Attributes:
    max_order: Maximum n-gram order to consider.
    matches_by_order: Accumulated sum of clipped n-gram matches for each order.
    possible_matches_by_order: Accumulated sum of total n-grams in predictions
      for each order.
    translation_length: Accumulated total length of predictions.
    reference_length: Accumulated total 'effective' reference length (closest
      length match for each prediction).
  """

  max_order: int
  matches_by_order: jax.Array
  possible_matches_by_order: jax.Array
  translation_length: jax.Array
  reference_length: jax.Array

  @classmethod
  def empty(cls) -> 'BLEU':
    return cls(
        max_order=4,
        matches_by_order=jnp.array(0, jnp.float32),
        possible_matches_by_order=jnp.array(0, jnp.float32),
        translation_length=jnp.array(0, jnp.float32),
        reference_length=jnp.array(0, jnp.float32),
    )

  @classmethod
  def from_model_output(
      cls,
      predictions: list[str],
      references: list[list[str]],
      max_order: int = 4,
  ) -> 'BLEU':
    """Computes BLEU statistics for a batch of predictions and references.

    Args:
      predictions: A list of predicted strings. The shape should be (batch_size,
        ).
      references: A list of lists of reference strings. The shape should be
        (batch_size, num_references).
      max_order: The maximum order of n-grams to consider.

    Returns:
      A BLEU metric instance containing the statistics for this batch.

    Raises:
      ValueError: If the shapes of `predictions` and `references` are
      incompatible.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    pred_length = 0
    ref_length = 0

    for pred, ref_list in zip(predictions, references):
      pred = pred.split()
      ref_list = [r.split() for r in ref_list]
      pred_length += len(pred)
      ref_length += min(len(r) for r in ref_list)
      prediction_ngram_counts = _get_ngrams(pred, max_order)
      reference_ngram_counts = collections.Counter()
      for ref in ref_list:
        reference_ngram_counts |= _get_ngrams(ref, max_order)
      overlap = prediction_ngram_counts & reference_ngram_counts
      for ngram in overlap:
        matches_by_order[len(ngram) - 1] += overlap[ngram]
      for order in range(1, max_order + 1):
        possible_matches = len(pred) - order + 1
        if possible_matches > 0:
          possible_matches_by_order[order - 1] += possible_matches

    return cls(
        max_order=max_order,
        matches_by_order=jnp.array(matches_by_order, dtype=jnp.float32),
        possible_matches_by_order=jnp.array(
            possible_matches_by_order, dtype=jnp.float32
        ),
        translation_length=jnp.array(pred_length, dtype=jnp.float32),
        reference_length=jnp.array(ref_length, dtype=jnp.float32),
    )

  def merge(self, other: 'BLEU') -> 'BLEU':
    if self.max_order != other.max_order:
      raise ValueError(
          'BLEU metrics with different max_order cannot be merged.'
      )
    return type(self)(
        max_order=self.max_order,
        matches_by_order=(self.matches_by_order + other.matches_by_order),
        possible_matches_by_order=(
            self.possible_matches_by_order + other.possible_matches_by_order
        ),
        translation_length=(self.translation_length + other.translation_length),
        reference_length=(self.reference_length + other.reference_length),
    )

  def compute(self) -> jax.Array:
    precisions = [0] * self.max_order
    for i in range(0, self.max_order):
      precisions[i] = base.divide_no_nan(
          self.matches_by_order[i], self.possible_matches_by_order[i]
      )
    geo_mean = (
        math.exp(sum((1.0 / self.max_order) * math.log(p) for p in precisions))
        if precisions and min(precisions) > 0
        else 0
    )
    ratio = base.divide_no_nan(self.translation_length, self.reference_length)
    bp = 1.0 if ratio > 1.0 else math.exp(1 - 1.0 / ratio)
    bleu = geo_mean * bp
    return jnp.array(bleu)


@flax.struct.dataclass
class Perplexity(clu_metrics.Metric):
  r"""Computes perplexity for sequence generation.

  Perplexity is a measurement of how well a probability distribution predicts a
  sample. It is defined as the exponentiation of the cross-entropy. A low
  perplexity indicates the probability distribution is good at predicting the
  sample.

  For language models, it can be interpreted as the weighted average branching
  factor of the model - how many equally likely words can be selected at each
  step.

  Given a sequence of :math:`N` tokens, perplexity is calculated as:

  .. math::
      Perplexity = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i|x_{<i})\right)

  When sample weights :math:`w_i` are provided:

  .. math::
      Perplexity = \exp\left(-\frac{\sum_{i=1}^{N} w_i\log P(x_i|x_{<i})}{\sum_{i=1}^{N} w_i}\right)

  where:
      - :math:`P(x_i|x_{<i})` is the predicted probability of token :math:`x_i`
        given previous tokens
      - :math:`w_i` are sample weights
      - :math:`N` is the sequence length

  Lower perplexity indicates better prediction - the model is less "perplexed" by the data.
  """

  aggregate_crossentropy: jax.Array
  num_samples: jax.Array

  @classmethod
  def empty(cls) -> 'Perplexity':
    return cls(
      aggregate_crossentropy=jnp.array(0, jnp.float32),
      num_samples=jnp.array(0, jnp.float32))

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      sample_weights: jax.Array | None = None,
      from_logits: bool = False,
  ) -> 'Perplexity':
    """Updates the metric.

    Args:
      predictions: A floating point tensor representing the prediction
      generated from the model. The shape should be (batch_size, seq_len,
      vocab_size).
      labels: True value. The shape should be (batch_size, seq_len).
      sample_weights: An optional tensor representing the
        weight of each token. The shape should be (batch_size, seq_len).
      from_logits: Whether the predictions are logits. If True, the predictions
        are converted to probabilities using a softmax. If False, all values
        outside of [0, 1] are clipped to 0 or 1.

    Returns:
      Updated Perplexity metric.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `predictions`
      and `labels` are incompatible.
    """
    if from_logits:
      log_prob = jax.nn.log_softmax(predictions, axis=-1)
    else:
      predictions = base.divide_no_nan(
          predictions, jnp.sum(predictions, axis=-1, keepdims=True)
      )
      epsilon = 1e-7
      predictions = jnp.clip(predictions, epsilon, 1.0 - epsilon)
      log_prob = jnp.log(predictions)

    labels_one_hot = jax.nn.one_hot(labels, predictions.shape[-1], axis=-1)
    crossentropy = -jnp.sum(labels_one_hot * log_prob, axis=-1)

    # Sum across sequence length dimension first.
    if sample_weights is not None:
      crossentropy = crossentropy * sample_weights
      # Normalize by the sum of weights for each sequence.
      crossentropy = base.divide_no_nan(
          jnp.sum(crossentropy), jnp.sum(sample_weights)
      )
    else:
      crossentropy = jnp.mean(crossentropy)

    batch_size = jnp.array(labels.shape[0])
    return cls(
        aggregate_crossentropy=(batch_size * crossentropy),
        num_samples=batch_size,
    )

  def merge(self, other: 'Perplexity') -> 'Perplexity':
    return type(self)(
        aggregate_crossentropy=(
            self.aggregate_crossentropy + other.aggregate_crossentropy
        ),
        num_samples=self.num_samples + other.num_samples,
    )

  def compute(self) -> jax.Array:
    return jnp.exp(
        base.divide_no_nan(self.aggregate_crossentropy, self.num_samples)
    )


@flax.struct.dataclass
class RougeBase(clu_metrics.Metric, abc.ABC):
  """Abstract base class for ROUGE metrics using macro-averaging."""

  total_precision: jax.Array
  total_recall: jax.Array
  total_f1: jax.Array
  num_examples: jax.Array

  @classmethod
  @abc.abstractmethod
  def empty(cls, **kwargs) -> 'RougeBase':
    """Creates an empty Rouge metric. Implemented by subclasses."""
    raise NotImplementedError('Subclasses must implement the empty method.')

  @staticmethod
  @abc.abstractmethod
  def _calculate_instance_scores(
      pred_tokens: list[str], ref_tokens: list[str], **kwargs
  ) -> tuple[float, float, float]:
    """Calculates precision, recall, and F1 for a single prediction-reference pair.

    kwargs may contain 'order' for RougeN.

    Returns:
        A tuple (precision, recall, f1_score).
    """
    raise NotImplementedError(
        'Subclasses must implement _calculate_instance_scores.'
    )

  @classmethod
  def _get_common_initial_values(cls) -> dict[str, jax.Array]:
    """Returns a dictionary of common metric attributes initialized to zero."""
    return {
        'total_precision': jnp.array(0.0, jnp.float32),
        'total_recall': jnp.array(0.0, jnp.float32),
        'total_f1': jnp.array(0.0, jnp.float32),
        'num_examples': jnp.array(0.0, jnp.float32),
    }

  @classmethod
  def _get_specific_constructor_args_for_class(cls) -> dict[str, int]:
    """Returns subclass-specific keyword arguments for the constructor."""
    return {}

  @classmethod
  def from_model_output(
      cls, predictions: list[str], references: list[str], **kwargs
  ) -> 'RougeBase':
    """Computes sums of per-instance ROUGE scores for a batch.

    Subclasses implement _calculate_instance_scores and helper methods for
    specific constructor arguments and validation.

    Args:
      predictions: A list of predicted strings. The shape should be (batch_size,
        ).
      references: A list of reference strings. The shape should be (batch_size,
        ).
      **kwargs: Additional keyword arguments passed to
        _calculate_instance_scores and to the class constructor.

    Returns:
      An instance of the ROUGE metric with accumulated scores.
    """
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    num_examples = 0.0

    for pred_str, ref_str in zip(predictions, references):
      pred_tokens = pred_str.split()
      ref_tokens = ref_str.split()

      precision, recall, f1 = cls._calculate_instance_scores(
          pred_tokens, ref_tokens, **kwargs
      )

      total_precision += precision
      total_recall += recall
      total_f1 += f1
      num_examples += 1

    constructor_args = {
        'total_precision': jnp.array(total_precision, dtype=jnp.float32),
        'total_recall': jnp.array(total_recall, dtype=jnp.float32),
        'total_f1': jnp.array(total_f1, dtype=jnp.float32),
        'num_examples': jnp.array(num_examples, dtype=jnp.float32),
    }
    constructor_args.update(
        cls._get_specific_constructor_args_for_class(**kwargs)
    )
    return cls(**constructor_args)

  def merge(self, other: 'RougeBase') -> 'RougeBase':
    """Merges this Rouge metric with another."""
    if not isinstance(other, type(self)):
      raise TypeError(
          'Cannot merge instances of different types:'
          f' {type(self).__name__} and {type(other).__name__}'
      )

    self._validate_merge_specifics(other)

    merged_data = {
        'total_precision': self.total_precision + other.total_precision,
        'total_recall': self.total_recall + other.total_recall,
        'total_f1': self.total_f1 + other.total_f1,
        'num_examples': self.num_examples + other.num_examples,
    }
    merged_data.update(self._get_specific_fields_for_merge_constructor())
    return type(self)(**merged_data)  # type: ignore[call-arg]

  def _validate_merge_specifics(self, other: 'RougeBase'):
    """Hook for subclass-specific validation during merge (e.g., check order).

    Args:
      other: The other instance of RougeBase to merge with.

    Raises:
      ValueError if validation fails.
    """
    pass

  def _get_specific_fields_for_merge_constructor(self) -> dict[str, int]:
    """Returns subclass-specific fields for constructing a merged instance.

    Returns:
      A dictionary of subclass-specific fields to merge. These fields (e.g.,
      `order` for RougeN) are taken from `self`.
    """
    return {}

  def compute(self) -> jax.Array:
    """Computes macro-averaged recall, precision, and F1-score."""
    macro_avg_precision = base.divide_no_nan(
        self.total_precision, self.num_examples
    )
    macro_avg_recall = base.divide_no_nan(self.total_recall, self.num_examples)
    macro_avg_f1score = base.divide_no_nan(self.total_f1, self.num_examples)

    return jnp.stack([macro_avg_precision, macro_avg_recall, macro_avg_f1score])


@flax.struct.dataclass
class RougeL(RougeBase):
  r"""Computes macro-averaged ROUGE-L recall, precision, and F1-score.

  ROUGE-L measures the longest common subsequence (LCS) between a prediction
  and a reference. This metric calculates ROUGE-L precision, recall, and
  F1-score for each individual prediction compared against its single
  corresponding reference. These per-instance scores are then averaged.

  How ROUGE-L scores are calculated for each individual prediction-reference
  pair:

  For a single prediction P and reference R:

  .. math::
      LCS(P, R) = \text{length of the Longest Common Subsequence}
  .. math::
      \text{Recall}_{\text{LCS}} = \frac{LCS(P, R)}{|R|}
  .. math::
      \text{Precision}_{\text{LCS}} = \frac{LCS(P, R)}{|P|}
  .. math::
      \text{F1}_{\text{LCS}} = 2 \times \frac{\text{Precision} \times
      \text{Recall}}{\text{Precision} + \text{Recall}}

  Final Macro-Averaged Metrics:

  .. math::
      \text{MacroAvgPrecision} =
      \frac{\text{total_precision}}{\text{num_examples}}
  .. math::
      \text{MacroAvgRecall} = \frac{\text{total_recall}}{\text{num_examples}}
  .. math::
      \text{MacroAvgF1} = \frac{\text{total_f1}}{\text{num_examples}}

  Attributes:
    total_precision: Accumulated sum of LCS precision scores from each instance.
    total_recall: Accumulated sum of LCS recall scores from each instance.
    total_f1: Accumulated sum of LCS F1 scores from each instance.
    num_examples: The number of instances (prediction-reference pairs)
      processed.
  """

  @classmethod
  def empty(cls, **kwargs) -> 'RougeL':
    common_values = super()._get_common_initial_values()
    return cls(**common_values)

  @staticmethod
  def _calculate_instance_scores(
      pred_tokens: list[str],
      ref_tokens: list[str],
      **kwargs,
  ) -> tuple[float, float, float]:
    lcs = jnp.array(_lcs_length(pred_tokens, ref_tokens))
    pred_len = jnp.array(len(pred_tokens))
    ref_len = jnp.array(len(ref_tokens))

    precision = base.divide_no_nan(lcs, pred_len).item()
    recall = base.divide_no_nan(lcs, ref_len).item()
    f1 = base.divide_no_nan(2 * precision * recall, precision + recall).item()
    return float(precision), float(recall), float(f1)


@flax.struct.dataclass
class RougeN(RougeBase):
  r"""Computes macro-averaged ROUGE-N recall, precision, and F1-score.

  This metric first calculates ROUGE-N precision, recall, and F1-score for each
  individual prediction compared against its single corresponding reference.
  ROUGE-N scores are based on the number of overlapping n-grams (sequences of n
  words) between the prediction and the reference text. These per-instance
  precision, recall, and F1-scores are then averaged across all instances in
  the dataset/batch.

  How ROUGE-N scores are calculated for each individual prediction-reference
  pair:

  .. math::
      \text{Precision} = \frac{N_o}{N_p}
  .. math::
      \text{Recall} = \frac{N_o}{N_r}
  .. math::
      \text{F1} = 2 \times \frac{\text{Precision} \times
      \text{Recall}}{\text{Precision} + \text{Recall}}
  where:
      - :math:`N_o` be the number of n-grams that overlap between the prediction
      and the reference.
      - :math:`N_p` be the total number of n-grams in the prediction.
      - :math:`N_r` be the total number of n-grams in the reference.

  Final Macro-Averaged Metrics:

  .. math::
      \text{MacroAvgPrecision} =
      \frac{\text{total_precision}}{\text{num_examples}}
  .. math::
      \text{MacroAvgRecall} = \frac{\text{total_recall}}{\text{num_examples}}
  .. math::
      \text{MacroAvgF1} = \frac{\text{total_f1}}{\text{num_examples}}

  Attributes:
    order: The specific 'N' in ROUGE-N (e.g., 1 for ROUGE-1, 2 for ROUGE-2).
    total_precision: Accumulated sum of precision scores from each instance.
    total_recall: Accumulated sum of recall scores from each instance.
    total_f1: Accumulated sum of f1 scores from each instance.
    num_examples: The number of instances (prediction-reference pairs)
      processed.
  """

  order: int

  @classmethod
  def empty(cls, order: int = 2) -> 'RougeN':
    common_values = super()._get_common_initial_values()
    return cls(order=order, **common_values)

  @classmethod
  def _get_specific_constructor_args_for_class(cls, **kwargs) -> dict[str, int]:
    return {'order': kwargs.get('order', 2)}

  @staticmethod
  def _calculate_instance_scores(
      pred_tokens: list[str], ref_tokens: list[str], **kwargs
  ) -> tuple[float, float, float]:
    order = kwargs.get('order', 2)

    pred_ngrams_counts = _get_single_n_grams(pred_tokens, order)
    ref_ngrams_counts = _get_single_n_grams(ref_tokens, order)
    overlap_counts = pred_ngrams_counts & ref_ngrams_counts

    prediction_ngrams = jnp.array(sum(pred_ngrams_counts.values()))
    reference_ngrams = jnp.array(sum(ref_ngrams_counts.values()))
    overlapping_ngrams = jnp.array(sum(overlap_counts.values()))

    precision = base.divide_no_nan(overlapping_ngrams, prediction_ngrams)
    recall = base.divide_no_nan(overlapping_ngrams, reference_ngrams)
    f1 = base.divide_no_nan(2 * precision * recall, precision + recall)
    return float(precision), float(recall), float(f1)

  def _validate_merge_specifics(self, other: 'RougeBase'):
    if not isinstance(other, RougeN) or self.order != other.order:
      raise ValueError(
          'RougeN metrics with different orders cannot be merged. '
          f'Got {self.order} and {getattr(other, "order", "N/A")}.'
      )

  def _get_specific_fields_for_merge_constructor(self) -> dict[str, int]:
    return {'order': self.order}


@flax.struct.dataclass
class WER(base.Average):
  r"""Computes Word Error Rate (WER) for speech recognition or text generation tasks.

  Word Error Rate measures the edit distance between reference texts and
  predictions,
  normalized by the length of the reference texts. It is calculated as:

  .. math::
      WER = \frac{S + D + I}{N}

  where:
      - S is the number of substitutions
      - D is the number of deletions
      - I is the number of insertions
      - N is the number of words in the reference

  A lower WER indicates better performance, with 0 being perfect.

  This implementation accepts both pre-tokenized inputs (lists of tokens) and
  untokenized
  strings. When strings are provided, they are tokenized by splitting on
  whitespace.
  """

  @classmethod
  def from_model_output(
      cls,
      predictions: list[str],
      references: list[str],
  ) -> 'WER':
    """Updates the metric.

    Args:
        prediction: Either a string or a list of tokens in the predicted
          sequence.
        reference: Either a string or a list of tokens in the reference
          sequence.

    Returns:
        New WER metric instance.

    Raises:
        ValueError: If inputs are not properly formatted or are empty.
    """
    if not predictions or not references:
      raise ValueError('predictions and references must not be empty')

    if isinstance(predictions, str):
      predictions = predictions.split()
    if isinstance(references, str):
      references = references.split()

    edit_distance = cls._levenshtein_distance(predictions, references)
    reference_length = len(references)

    return cls(
        total=jnp.array(edit_distance, dtype=jnp.float32),
        count=jnp.array(reference_length, dtype=jnp.float32),
    )

  @staticmethod
  def _levenshtein_distance(prediction: list, reference: list) -> int:
    """Computes the Levenshtein (edit) distance between two token sequences.

    Args:
        prediction: List of tokens in the predicted sequence.
        reference: List of tokens in the reference sequence.

    Returns:
        The minimum number of edits needed to transform prediction into
        reference.
    """
    m, n = len(prediction), len(reference)

    # Handle edge cases
    if m == 0:
      return n
    if n == 0:
      return m

    # Create distance matrix
    distance_matrix = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Initialize first row and column
    for i in range(m + 1):
      distance_matrix[i][0] = i
    for j in range(n + 1):
      distance_matrix[0][j] = j

    # Fill the matrix
    for i in range(1, m + 1):
      for j in range(1, n + 1):
        cost = 0 if prediction[i - 1] == reference[j - 1] else 1

        distance_matrix[i][j] = min(
            distance_matrix[i - 1][j] + 1,  # deletion
            distance_matrix[i][j - 1] + 1,  # insertion
            distance_matrix[i - 1][j - 1] + cost,  # substitution
        )

    return distance_matrix[m][n]
