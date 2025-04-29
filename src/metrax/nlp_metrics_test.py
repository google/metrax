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

import os
os.environ['KERAS_BACKEND'] = 'jax'

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import keras_hub
import keras_nlp
import metrax
import numpy as np

np.random.seed(42)


class NlpMetricsTest(parameterized.TestCase):

  def test_bleu_empty(self):
    """Tests the `empty` method of the `BLEU` class."""
    m = metrax.BLEU.empty()
    self.assertEqual(m.max_order, 4)
    self.assertEqual(m.matches_by_order, jnp.array(0, jnp.float32))
    self.assertEqual(m.possible_matches_by_order, jnp.array(0, jnp.float32))
    self.assertEqual(m.translation_length, jnp.array(0, jnp.float32))
    self.assertEqual(m.reference_length, jnp.array(0, jnp.float32))

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

  def test_bleu(self):
    """Tests that BLEU metric computes correct values."""
    references = [
        ["He eats a sweet apple", "He is eating a tasty apple, isn't he"],
        [
            "Silicon Valley is one of my favourite shows",
            "Silicon Valley is the best show ever",
        ],
    ]
    predictions = [
        "He He He eats sweet apple which is a fruit",
        "I love Silicon Valley it is one of my favourite shows",
    ]
    keras_metric = keras_nlp.metrics.Bleu()
    keras_metric.update_state(references, predictions)
    metrax_metric = metrax.BLEU.from_model_output(predictions, references)

    np.testing.assert_allclose(
        metrax_metric.compute(),
        keras_metric.result(),
        rtol=1e-05,
        atol=1e-05,
    )

  def test_bleu_merge(self):
    """Tests that BLEU metric computes correct values using merge."""
    references = [
        ["He eats a sweet apple", "He is eating a tasty apple, isn't he"],
        [
            "Silicon Valley is one of my favourite shows",
            "Silicon Valley is the best show ever",
        ],
    ]
    predictions = [
        "He He He eats sweet apple which is a fruit",
        "I love Silicon Valley it is one of my favourite shows",
    ]
    keras_metric = keras_nlp.metrics.Bleu()
    keras_metric.update_state(references, predictions)
    metrax_metric = None
    for ref_list, pred in zip(references, predictions):
      update = metrax.BLEU.from_model_output([pred], [ref_list])
      metrax_metric = (
          update if metrax_metric is None else metrax_metric.merge(update)
      )

    np.testing.assert_allclose(
        metrax_metric.compute(),
        keras_metric.result(),
        rtol=1e-05,
        atol=1e-05,
    )

  def test_bleu_merge_fails_on_different_max_order(self):
    """Tests that error is raised when BLEU metrics with different max_order are merged."""
    references = [
        ["He eats a sweet apple", "He is eating a tasty apple, isn't he"],
    ]
    predictions = [
        "He He He eats sweet apple which is a fruit",
    ]
    order_3_metric = metrax.BLEU.from_model_output(
        predictions, references, max_order=3
    )
    order_4_metric = metrax.BLEU.from_model_output(
        predictions, references, max_order=4
    )

    np.testing.assert_raises(
        ValueError, lambda: order_3_metric.merge(order_4_metric)
    )

  def test_rougen(self):
    """Tests that ROUGE-N metric computes correct values."""
    references = [
        "He eats a sweet apple",
        "Silicon Valley is one of my favourite shows",
    ]
    predictions = [
        "He He He eats sweet apple which is a fruit",
        "I love Silicon Valley it is one of my favourite shows",
    ]
    keras_metric = keras_nlp.metrics.RougeN()
    keras_metric.update_state(references, predictions)
    keras_metric_array = jnp.stack(list(keras_metric.result().values()))
    metrax_metric = metrax.RougeN.from_model_output(predictions, references)

    np.testing.assert_allclose(
        metrax_metric.compute(),
        keras_metric_array,
        rtol=1e-05,
        atol=1e-05,
    )

  def test_rougen_merge(self):
    """Tests that ROUGE-N metric computes correct values using merge."""
    references = [
        "He eats a sweet apple",
        "Silicon Valley is one of my favourite shows",
    ]
    predictions = [
        "He He He eats sweet apple which is a fruit",
        "I love Silicon Valley it is one of my favourite shows",
    ]
    keras_metric = keras_nlp.metrics.RougeN()
    keras_metric.update_state(references, predictions)
    keras_metric_array = jnp.stack(list(keras_metric.result().values()))

    metrax_metric = None
    for ref, pred in zip(references, predictions):
      update = metrax.RougeN.from_model_output([pred], [ref])
      metrax_metric = (
          update if metrax_metric is None else metrax_metric.merge(update)
      )

    np.testing.assert_allclose(
        metrax_metric.compute(),
        keras_metric_array,
        rtol=1e-05,
        atol=1e-05,
    )

  def test_rougen_merge_fails_on_different_max_order(self):
    """Tests that error is raised when ROUGE-N metrics with different max_order are merged."""
    references = [
        "He eats a sweet apple",
    ]
    predictions = [
        "He He He eats sweet apple which is a fruit",
    ]
    order_3_metric = metrax.RougeN.from_model_output(
        predictions, references, order=3
    )
    order_4_metric = metrax.RougeN.from_model_output(
        predictions, references, order=4
    )

    np.testing.assert_raises(
        ValueError, lambda: order_3_metric.merge(order_4_metric)
    )

  @parameterized.named_parameters(
      (
          'basic',
          np.random.randint(10, size=[2, 5, 10]),
          np.random.uniform(size=(2, 5, 10, 20)),
          None,
          False,
      ),
      (
          'weighted',
          np.random.randint(10, size=[2, 5, 10]),
          np.random.uniform(size=(2, 5, 10, 20)),
          np.random.randint(2, size=(2, 5, 10)).astype(np.float32),
          False,
      ),
      (
          'negative_values',
          np.random.randint(10, size=[2, 5, 10]),
          np.random.uniform(size=(2, 5, 10, 20), low=-2, high=2),
          None,
          False,
      ),
      (
          'from_logits',
          np.random.randint(10, size=[2, 5, 10]),
          np.random.uniform(size=(2, 5, 10, 20), low=-2, high=2),
          None,
          True,
      ),
  )
  def test_perplexity(self, y_true, y_pred, sample_weights, from_logits):
    """Test that `Perplexity` Metric computes correct values."""
    keras_metric = keras_hub.metrics.Perplexity(from_logits=from_logits)
    metrax_metric = None
    for index, (labels, logits) in enumerate(zip(y_true, y_pred)):
      weights = sample_weights[index] if sample_weights is not None else None
      keras_metric.update_state(labels, logits, sample_weight=weights)
      update = metrax.Perplexity.from_model_output(
          predictions=logits,
          labels=labels,
          sample_weights=weights,
          from_logits=from_logits,
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
