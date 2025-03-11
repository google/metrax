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

"""Tests for metrax nlp metrics."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import keras_hub
import metrax
import numpy as np


class NlpMetricsTest(parameterized.TestCase):

  def test_perplexity_empty(self):
    """Tests the `empty` method of the `Perplexity` class."""
    m = metrax.Perplexity.empty()
    self.assertEqual(m.aggregate_crossentropy, jnp.array(0, jnp.float32))
    self.assertEqual(m.num_samples, jnp.array(0, jnp.float32))

  @parameterized.named_parameters(
      (
          'basic',
          np.random.randint(10, size=[2, 5, 10]),
          np.random.uniform(size=(2, 5, 10, 20)),
          None,
      ),
      (
          'weighted',
          np.random.randint(10, size=[2, 5, 10]),
          np.random.uniform(size=(2, 5, 10, 20)),
          np.random.randint(2, size=(2, 5, 10)).astype(np.float32),
      ),
  )
  def test_perplexity(self, y_true, y_pred, sample_weights):
    """Test that `Perplexity` Metric computes correct values."""
    keras_metric = keras_hub.metrics.Perplexity()
    metrax_metric = None
    for index, (labels, logits) in enumerate(zip(y_true, y_pred)):
      weights = sample_weights[index] if sample_weights is not None else None
      keras_metric.update_state(labels, logits, sample_weight=weights)
      update = metrax.Perplexity.from_model_output(
          predictions=logits,
          labels=labels,
          sample_weights=weights,
      )
      metrax_metric = update if metrax_metric is None else metrax_metric.merge(
          update
      )

    expected = keras_metric.result()
    np.testing.assert_allclose(
        metrax_metric.compute(),
        expected,
        rtol=1e-05,
        atol=1e-05,
    )


if __name__ == '__main__':
  absltest.main()
