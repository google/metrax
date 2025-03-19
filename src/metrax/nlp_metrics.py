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

from clu import metrics as clu_metrics
import flax
import jax
import jax.numpy as jnp
from metrax import base


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
  ) -> 'Perplexity':
    """Updates the metric.

    Args:
      predictions: A floating point tensor representing the prediction
      generated from the model. The shape should be (batch_size, seq_len,
      vocab_size).
      labels: True value. The shape should be (batch_size, seq_len).
      sample_weights: An optional tensor representing the
        weight of each token. The shape should be (batch_size, seq_len).

    Returns:
      Updated Perplexity metric.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `predictions`
      and `labels` are incompatible.
    """
    predictions = base.divide_no_nan(
        predictions, jnp.sum(predictions, axis=-1, keepdims=True)
    )
    labels_one_hot = jax.nn.one_hot(labels, predictions.shape[-1], axis=-1)
    log_prob = jnp.log(predictions)
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
        if prediction[i - 1] == reference[j - 1]:
          cost = 0
        else:
          cost = 1

        distance_matrix[i][j] = min(
            distance_matrix[i - 1][j] + 1,  # deletion
            distance_matrix[i][j - 1] + 1,  # insertion
            distance_matrix[i - 1][j - 1] + cost,  # substitution
        )

    return distance_matrix[m][n]
