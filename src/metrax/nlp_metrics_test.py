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

  def test_wer_empty(self):
    """Tests the `empty` method of the `WER` class."""
    m = metrax.WER.empty()
    self.assertEqual(m.total, jnp.array(0, jnp.float32))
    self.assertEqual(m.count, jnp.array(0, jnp.float32))

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

  def test_wer(self):
    """Tests that WER metric computes correct values with tokenized and untokenized inputs."""
    string_preds = [
      "the cat sat on the mat",
      "a quick brown fox jumps over the lazy dog",
      "hello world"
    ]
    string_refs = [
      "the cat sat on the hat",
      "the quick brown fox jumps over the lazy dog",
      "hello beautiful world"
    ]
    tokenized_preds = [sentence.split() for sentence in string_preds]
    tokenized_refs = [sentence.split() for sentence in string_refs]

    metrax_token_metric = None
    keras_metric = keras_hub.metrics.EditDistance(normalize=True)
    for pred, ref in zip(tokenized_preds, tokenized_refs):
      metrax_update = metrax.WER.from_model_output(pred,ref)
      keras_metric.update_state(ref, pred)
      metrax_token_metric = metrax_update if metrax_token_metric is None else metrax_token_metric.merge(metrax_update)

    np.testing.assert_allclose(
        metrax_token_metric.compute(),
        keras_metric.result(),
        rtol=1e-05,
        atol=1e-05,
        err_msg="String-based WER should match keras_hub EditDistance"
    )

    metrax_string_metric = None
    for pred, ref in zip(string_preds, string_refs):
      update = metrax.WER.from_model_output(predictions=pred, references=ref)
      metrax_string_metric = update if metrax_string_metric is None else metrax_string_metric.merge(update)

    np.testing.assert_allclose(
    metrax_string_metric.compute(),
    metrax_token_metric.compute(),
    rtol=1e-05,
    atol=1e-05,
    err_msg="String input and tokenized input should produce the same WER"
    )


if __name__ == '__main__':
  absltest.main()
