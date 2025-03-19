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

"""A collection of different metrics for regression models."""

from clu import metrics as clu_metrics
import flax
import jax
import jax.numpy as jnp
from metrax import base


@flax.struct.dataclass
class MSE(base.Average):
  r"""Computes the mean squared error for regression problems given `predictions` and `labels`.

  The mean squared error without sample weights is defined as:

  .. math::
      MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2

  When sample weights :math:`w_i` are provided, the weighted mean squared error
  is:

  .. math::
      MSE = \frac{\sum_{i=1}^{N} w_i(y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} w_i}

  where:
      - :math:`y_i` are true values
      - :math:`\hat{y}_i` are predictions
      - :math:`w_i` are sample weights
      - :math:`N` is the number of samples
  """

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
  r"""Computes the root mean squared error for regression problems given `predictions` and `labels`.

  The root mean squared error without sample weights is defined as:

  .. math::
      RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}

  When sample weights :math:`w_i` are provided, the weighted root mean squared
  error is:

  .. math::
      RMSE = \sqrt{\frac{\sum_{i=1}^{N} w_i(y_i - \hat{y}_i)^2}{\sum_{i=1}^{N}
      w_i}}

  where:
      - :math:`y_i` are true values
      - :math:`\hat{y}_i` are predictions
      - :math:`w_i` are sample weights
      - :math:`N` is the number of samples
  """

  def compute(self) -> jax.Array:
    return jnp.sqrt(super().compute())


@flax.struct.dataclass
class RSQUARED(clu_metrics.Metric):
  r"""Computes the r-squared score of a scalar or a batch of tensors.

  R-squared is a measure of how well the regression model fits the data. It
  measures the proportion of the variance in the dependent variable that is
  explained by the independent variable(s). It is defined as 1 - SSE / SST,
  where SSE is the sum of squared errors and SST is the total sum of squares.

  .. math::
      R^2 = 1 - \frac{SSE}{SST}

  where:
      .. math::
          SSE = \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
      .. math::
          SST = \sum_{i=1}^{N} (y_i - \bar{y})^2

  When sample weights :math:`w_i` are provided:

  .. math::
      R^2 = 1 - \frac{\sum_{i=1}^{N} w_i(y_i - \hat{y}_i)^2}{\sum_{i=1}^{N}
      w_i(y_i - \bar{y})^2}

  where:
      - :math:`y_i` are true values
      - :math:`\hat{y}_i` are predictions
      - :math:`\bar{y}` is the mean of true values
      - :math:`w_i` are sample weights
      - :math:`N` is the number of samples

  The score ranges from -∞ to 1, where 1 indicates perfect prediction and 0
  indicates
  that the model performs no better than a horizontal line.
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
        sum_of_squared_label=jnp.array(0, jnp.float32),
    )

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
    mean = base.divide_no_nan(self.total, self.count)
    sst = self.sum_of_squared_label - self.count * jnp.power(mean, 2)
    return 1 - base.divide_no_nan(self.sum_of_squared_error, sst)
