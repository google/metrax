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

"""A collection of different CLU metrics for the training library."""

from clu import metrics as clu_metrics
import flax
import jax
import jax.numpy as jnp


def _default_threshold(num_thresholds: int) -> jax.Array:
  """Returns evenly distributed `num_thresholds` between 0.0 and 1.0.

  Args:
    num_thresholds: The number of thresholds to return.

  Returns:
    Evently distributed `num_thresholds` between 0.0 and 1.0.
  """
  if num_thresholds < 2:
    raise ValueError(
        f'num_thresholds must be at least 2, but got {num_thresholds}.'
    )
  epsilon = 1e-5
  thresholds = jnp.arange(num_thresholds, dtype=jnp.float32) / (
      num_thresholds - 1
  )
  thresholds = thresholds.at[0].set(-epsilon)
  thresholds = thresholds.at[-1].set(1.0 + epsilon)
  return thresholds


def _divide_no_nan(x: jax.Array, y: jax.Array) -> jax.Array:
  """Computes a safe divide which returns 0 if the y is zero."""
  return jnp.where(y != 0, jnp.divide(x, y), 0.0)


@flax.struct.dataclass
class MSE(clu_metrics.Average):
  """Computes the mean squared error for regression problems given `predictions` and `labels`."""

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      sample_weights: jax.Array | None = None,
  ) -> 'MSE':
    """Updates the metric.

    Args:
      predictions: A floating point 1D vector representing the prediction
        generated from the model. The shape should be (batch_size,).
      labels: True value. The shape should be (batch_size,).
      sample_weights: An optional floating point 1D vector representing the
        weight of each sample. The shape should be (batch_size,).

    Returns:
      Updated MSE metric. The shape should be a single scalar.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `predictions`
      and `labels` are incompatible.
    """
    squared_error = jnp.square(predictions - labels)
    count = jnp.ones_like(labels, dtype=jnp.int32)
    if sample_weights is not None:
      squared_error = squared_error * sample_weights
      count = count * sample_weights
    return cls(
        total=squared_error.sum(),
        count=count.sum(),
    )


@flax.struct.dataclass
class RMSE(MSE):
  """Computes the root mean squared error for regression problems given `predictions` and `labels`."""

  def compute(self) -> jax.Array:
    return jnp.sqrt(super().compute())


@flax.struct.dataclass
class RSQUARED(clu_metrics.Metric):
  """Computes the r-squared score of a scalar or a batch of tensors.

  R-squared is a measure of how well the regression model fits the data. It
  measures the proportion of the variance in the dependent variable that is
  explained by the independent variable(s). It is defined as 1 - SSE / SST,
  where SSE is the sum of squared errors and SST is the total sum of squares.
  """

  total: jax.Array
  count: jax.Array
  sum_of_squared_error: jax.Array
  sum_of_squared_label: jax.Array
  

  @classmethod
  def empty(cls) -> 'RSQUARED':
    return cls(
      total=jnp.array(0, jnp.float32),
      count=jnp.array(0, jnp.float32),
      sum_of_squared_error=jnp.array(0, jnp.float32),
      sum_of_squared_label=jnp.array(0, jnp.float32))

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      sample_weights: jax.Array | None = None,
  ) -> 'RSQUARED':
    """Updates the metric.

    Args:
      predictions: A floating point 1D vector representing the prediction
        generated from the model. The shape should be (batch_size,).
      labels: True value. The shape should be (batch_size,).
      sample_weights: An optional floating point 1D vector representing the
        weight of each sample. The shape should be (batch_size,).

    Returns:
      Updated RSQUARED metric. The shape should be a single scalar.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `predictions`
      and `labels` are incompatible.
    """
    count = jnp.ones_like(labels, dtype=jnp.int32)
    squared_error = jnp.power(labels - predictions, 2)
    squared_label = jnp.power(labels, 2)
    if sample_weights is not None:
      labels = labels * sample_weights
      count = count * sample_weights
      squared_error = squared_error * sample_weights
      squared_label = squared_label * sample_weights
    return cls(
        total=labels.sum(),
        count=count.sum(),
        sum_of_squared_error=squared_error.sum(),
        sum_of_squared_label=squared_label.sum(),
    )

  def merge(self, other: 'RSQUARED') -> 'RSQUARED':
    return type(self)(
        total=self.total + other.total,
        count=self.count + other.count,
        sum_of_squared_error=self.sum_of_squared_error
        + other.sum_of_squared_error,
        sum_of_squared_label=self.sum_of_squared_label
        + other.sum_of_squared_label,
    )

  def compute(self) -> jax.Array:
    """Computes the r-squared score.

    Since we don't know the mean of the labels before we aggregate all of the
    data, we will manipulate the formula to be:
    sst = \sum_i (x_i - mean)^2
        = \sum_i (x_i^2 - 2 x_i mean + mean^2)
        = \sum_i x_i^2 - 2 mean \sum_i x_i + N * mean^2
        = \sum_i x_i^2 - 2 mean * N * mean + N * mean^2
        = \sum_i x_i^2 - N * mean^2

    Returns:
      The r-squared score.
    """
    mean = self.total / self.count
    sst = self.sum_of_squared_label - self.count * jnp.power(mean, 2)
    return 1 - _divide_no_nan(self.sum_of_squared_error, sst)


@flax.struct.dataclass
class Precision(clu_metrics.Metric):
  """Computes precision for binary classification given `predictions` and `labels`.

  Attributes:
    true_positives: The count of true positive instances from the given data,
      label, and threshold.
    false_positives: The count of false positive instances from the given data,
      label, and threshold.
  """

  true_positives: jax.Array
  false_positives: jax.Array

  @classmethod
  def empty(cls) -> 'Precision':
    return cls(
      true_positives=jnp.array(0, jnp.float32),
      false_positives=jnp.array(0, jnp.float32))

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      threshold: float = 0.5,
  ) -> 'Precision':
    """Updates the metric.

    Args:
      predictions: A floating point 1D vector whose values are in the range [0,
        1]. The shape should be (batch_size,).
      labels: True value. The value is expected to be 0 or 1. The shape should
        be (batch_size,).
      threshold: The threshold to use for the binary classification.

    Returns:
      Updated Precision metric. The shape should be a single scalar.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `predictions`
      and `labels` are incompatible.
    """
    predictions = jnp.where(predictions >= threshold, 1, 0)
    true_positives = jnp.sum((predictions == 1) & (labels == 1))
    false_positives = jnp.sum((predictions == 1) & (labels == 0))

    return cls(true_positives=true_positives, false_positives=false_positives)

  def merge(self, other: 'Precision') -> 'Precision':
    return type(self)(
        true_positives=self.true_positives + other.true_positives,
        false_positives=self.false_positives + other.false_positives,
    )

  def compute(self) -> jax.Array:
    return _divide_no_nan(
        self.true_positives, (self.true_positives + self.false_positives)
    )


@flax.struct.dataclass
class Recall(clu_metrics.Metric):
  """Computes recall for binary classification given `predictions` and `labels`.

  Attributes:
    true_positives: The count of true positive instances from the given data,
      label, and threshold.
    false_negatives: The count of false negative instances from the given data,
      label, and threshold.
  """

  true_positives: jax.Array
  false_negatives: jax.Array

  @classmethod
  def empty(cls) -> 'Recall':
    return cls(
      true_positives=jnp.array(0, jnp.float32),
      false_negatives=jnp.array(0, jnp.float32))

  @classmethod
  def from_model_output(
      cls, predictions: jax.Array, labels: jax.Array, threshold: float = 0.5
  ) -> 'Recall':
    """Updates the metric.

    Args:
      predictions: A floating point 1D vector whose values are in the range [0,
        1]. The shape should be (batch_size,).
      labels: True value. The value is expected to be 0 or 1. The shape should
        be (batch_size,).
      threshold: The threshold to use for the binary classification.

    Returns:
      Updated Recall metric. The shape should be a single scalar.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `predictions`
      and `labels` are incompatible.
    """
    predictions = jnp.where(predictions >= threshold, 1, 0)
    true_positives = jnp.sum((predictions == 1) & (labels == 1))
    false_negatives = jnp.sum((predictions == 0) & (labels == 1))

    return cls(true_positives=true_positives, false_negatives=false_negatives)

  def merge(self, other: 'Recall') -> 'Recall':
    return type(self)(
        true_positives=self.true_positives + other.true_positives,
        false_negatives=self.false_negatives + other.false_negatives,
    )

  def compute(self) -> jax.Array:
    return _divide_no_nan(
        self.true_positives, (self.true_positives + self.false_negatives)
    )


@flax.struct.dataclass
class AUCPR(clu_metrics.Metric):
  """Computes area under the precision-recall curve for binary classification given `predictions` and `labels`.

  AUC-PR Curve metric have a number of known issues so use it with caution.
  - PR curves are highly class balance sensitive.
  - PR is a non-monotonic function and thus its "area" is not directly
    proportional to performance.
  - PR-AUC has no standard implementation and different libraries will give
    different results. Some libraries will interpolate between points, others
    will assume a step function (or trapezoidal as sklearn does). Some libraries
    will compute the convex hull of the PR curve, others will not. Because PR is
    non monotonic, its value is sensitive to the number of samples along the
    curve (more so than ROC-AUC).

  Attributes:
    true_positives: The count of true positive instances from the given data and
      label at each threshold.
    false_positives: The count of false positive instances from the given data
      and label at each threshold.
    false_negatives: The count of false negative instances from the given data
      and label at each threshold.
  """

  # shape: (threshold, 1)
  true_positives: jax.Array
  false_positives: jax.Array
  false_negatives: jax.Array
  num_thresholds: int
  
  @classmethod
  def empty(cls) -> 'AUCPR':
    return cls(
      true_positives=jnp.array(0, jnp.float32),
      false_positives=jnp.array(0, jnp.float32),
      false_negatives=jnp.array(0, jnp.float32),
      num_thresholds=0)

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      sample_weights: jax.Array | None = None,
      num_thresholds: int = 200,
  ) -> 'AUCPR':
    """Updates the metric.

    Args:
      predictions: A floating point 1D vector whose values are in the range [0,
        1]. The shape should be (batch_size,).
      labels: True value. The value is expected to be 0 or 1. The shape should
        be (batch_size,).
      sample_weights: An optional floating point 1D vector representing the
        weight of each sample. The shape should be (batch_size,).
      num_thresholds: The number of thresholds to use. Default is 200.

    Returns:
      The area under the precision-recall curve. The shape should be a single
      scalar.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `predictions`
      and `labels` are incompatible.
    """
    pred_is_pos = jnp.greater(
        predictions,
        _default_threshold(num_thresholds=num_thresholds)[..., None],
    )
    pred_is_neg = jnp.logical_not(pred_is_pos)
    label_is_pos = jnp.equal(labels, 1)
    label_is_neg = jnp.equal(labels, 0)

    true_positives = pred_is_pos * label_is_pos
    false_positives = pred_is_pos * label_is_neg
    false_negatives = pred_is_neg * label_is_pos

    if sample_weights is not None:
      true_positives = true_positives * sample_weights
      false_positives = false_positives * sample_weights
      false_negatives = false_negatives * sample_weights

    return cls(
        true_positives=true_positives.sum(axis=-1),
        false_positives=false_positives.sum(axis=-1),
        false_negatives=false_negatives.sum(axis=-1),
        num_thresholds=num_thresholds,
    )

  def merge(self, other: 'AUCPR') -> 'AUCPR':
    return type(self)(
        true_positives=self.true_positives + other.true_positives,
        false_positives=self.false_positives + other.false_positives,
        false_negatives=self.false_negatives + other.false_negatives,
        num_thresholds=self.num_thresholds,
    )

  def interpolate_pr_auc(self) -> jax.Array:
    """Interpolation formula inspired by section 4 of Davis & Goadrich 2006.

    https://minds.wisconsin.edu/handle/1793/60482

    Note here we derive & use a closed formula not present in the paper
    as follows:

      Precision = TP / (TP + FP) = TP / P

    Modeling all of TP (true positive), FP (false positive) and their sum
    P = TP + FP (predicted positive) as varying linearly within each
    interval [A, B] between successive thresholds, we get

      Precision slope = dTP / dP
                      = (TP_B - TP_A) / (P_B - P_A)
                      = (TP - TP_A) / (P - P_A)
      Precision = (TP_A + slope * (P - P_A)) / P

    The area within the interval is (slope / total_pos_weight) times

      int_A^B{Precision.dP} = int_A^B{(TP_A + slope * (P - P_A)) * dP / P}
      int_A^B{Precision.dP} = int_A^B{slope * dP + intercept * dP / P}

    where intercept = TP_A - slope * P_A = TP_B - slope * P_B, resulting in

      int_A^B{Precision.dP} = TP_B - TP_A + intercept * log(P_B / P_A)

    Bringing back the factor (slope / total_pos_weight) we'd put aside, we
    get

      slope * [dTP + intercept *  log(P_B / P_A)] / total_pos_weight

    where dTP == TP_B - TP_A.

    Note that when P_A == 0 the above calculation simplifies into

      int_A^B{Precision.dTP} = int_A^B{slope * dTP} = slope * (TP_B - TP_A)

    which is really equivalent to imputing constant precision throughout the
    first bucket having >0 true positives.

    Returns:
      pr_auc: A float scalar jax.Array that is an approximation of the area
      under the P-R curve.
    """
    dtp = (
        self.true_positives[: self.num_thresholds - 1] - self.true_positives[1:]
    )
    p = self.true_positives + self.false_positives
    dp = p[: self.num_thresholds - 1] - p[1:]
    prec_slope = _divide_no_nan(dtp, jnp.maximum(dp, 0))
    intercept = self.true_positives[1:] - prec_slope * p[1:]

    # recall_relative_ratio
    safe_p_ratio = jnp.where(
        jnp.multiply(p[: self.num_thresholds - 1] > 0, p[1:] > 0),
        _divide_no_nan(
            p[: self.num_thresholds - 1],
            jnp.maximum(p[1:], 0),
        ),
        jnp.ones_like(p[1:]),
    )
    # pr_auc_increment
    pr_auc_increment = _divide_no_nan(
        prec_slope * (dtp + intercept * jnp.log(safe_p_ratio)),
        jnp.maximum(self.true_positives[1:] + self.false_negatives[1:], 0),
    )
    return jnp.sum(pr_auc_increment)

  def compute(self) -> jax.Array:
    # Use interpolation to compute the area under the PR curve to match Keras.
    return self.interpolate_pr_auc()


@flax.struct.dataclass
class AUCROC(clu_metrics.Metric):
  """Computes area under the receiver operation characteristic curve for binary classification given `predictions` and `labels`.

  Attributes:
    true_positives: The count of true positive instances from the given data and
      label at each threshold.
    false_positives: The count of false positive instances from the given data
      and label at each threshold.
    total_count: The count of every data point.
  """

  # shape: (threshold, 1)
  true_positives: jax.Array
  true_negatives: jax.Array
  false_positives: jax.Array
  false_negatives: jax.Array
  num_thresholds: int

  @classmethod
  def empty(cls) -> 'AUCROC':
    return cls(
      true_positives=jnp.array(0, jnp.float32),
      true_negatives=jnp.array(0, jnp.float32),
      false_positives=jnp.array(0, jnp.float32),
      false_negatives=jnp.array(0, jnp.float32),
      num_thresholds=0)

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      sample_weights: jax.Array | None = None,
      num_thresholds: int = 200,
  ) -> 'AUCROC':
    """Updates the metric.

    Args:
      predictions: A floating point 1D vector whose values are in the range [0,
        1]. The shape should be (batch_size,).
      labels: True value. The value is expected to be 0 or 1. The shape should
        be (batch_size,).
      sample_weights: An optional floating point 1D vector representing the
        weight of each sample. The shape should be (batch_size,).
      num_thresholds: The number of thresholds to use. Default is 200.

    Returns:
      The area under the receiver operation characteristic curve. The shape
      should be a single scalar.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `predictions`
      and `labels` are incompatible.
    """
    pred_is_pos = jnp.greater(
        predictions,
        _default_threshold(num_thresholds=num_thresholds)[..., None],
    )
    pred_is_neg = jnp.logical_not(pred_is_pos)
    label_is_pos = jnp.equal(labels, 1)
    label_is_neg = jnp.equal(labels, 0)

    true_positives = pred_is_pos * label_is_pos
    true_negatives = pred_is_neg * label_is_neg
    false_positives = pred_is_pos * label_is_neg
    false_negatives = pred_is_neg * label_is_pos

    if sample_weights is not None:
      true_positives = true_positives * sample_weights
      true_negatives = true_negatives * sample_weights
      false_positives = false_positives * sample_weights
      false_negatives = false_negatives * sample_weights

    return cls(
        true_positives=true_positives.sum(axis=-1),
        true_negatives=true_negatives.sum(axis=-1),
        false_positives=false_positives.sum(axis=-1),
        false_negatives=false_negatives.sum(axis=-1),
        num_thresholds=num_thresholds,
    )

  def merge(self, other: 'AUCROC') -> 'AUCROC':
    return type(self)(
        true_positives=self.true_positives + other.true_positives,
        true_negatives=self.true_negatives + other.true_negatives,
        false_positives=self.false_positives + other.false_positives,
        false_negatives=self.false_negatives + other.false_negatives,
        num_thresholds=self.num_thresholds,
    )

  def compute(self) -> jax.Array:
    tp_rate = _divide_no_nan(
        self.true_positives, self.true_positives + self.false_negatives
    )
    fp_rate = _divide_no_nan(
        self.false_positives, self.false_positives + self.true_negatives
    )
    # Threshold goes from 0 to 1, so trapezoid is negative.
    return jnp.trapezoid(tp_rate, fp_rate) * -1


@flax.struct.dataclass
class Perplexity(clu_metrics.Metric):
  """Computes perplexity for sequence generation.

  Perplexity is a measurement of how well a probability distribution predicts a
  sample. It is defined as the exponentiation of the cross-entropy. A low
  perplexity indicates the probability distribution is good at predicting the
  sample.

  For language models, it can be interpreted as the weighted average branching
  factor of the model - how many equally likely words can be selected at each
  step.
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
    predictions = predictions / jnp.sum(predictions, axis=-1, keepdims=True)
    labels_one_hot = jax.nn.one_hot(labels, predictions.shape[-1], axis=-1)
    log_prob = jnp.log(predictions)
    crossentropy = -jnp.sum(labels_one_hot * log_prob, axis=-1)

    # Sum across sequence length dimension first.
    if sample_weights is not None:
      crossentropy = crossentropy * sample_weights
      # Normalize by the sum of weights for each sequence.
      crossentropy = jnp.sum(crossentropy) / jnp.sum(sample_weights)
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
    return jnp.exp(self.aggregate_crossentropy / self.num_samples)
