Metrax: A JAX-native High Performance Eval Metrics Library
==========================================================

.. toctree::
   :hidden:
   :caption: Currently Supported Metrics

   Metrax metrics <metrax>

**Metrax** provides common evaluation metric implementations for JAX.


Getting Started
---------------

All metrics inherit from the `clu.Metric
<https://github.com/google/CommonLoopUtils/blob/main/clu/metrics.py>`_ base
class, which provides a functional API consisting of three main methods:

* ``from_model_output`` to create the metric dataclass from inputs
* ``merge`` to update the results
* ``compute`` to get the final result

.. code-block:: python

    # Run model:
    y_true, y_pred = model(inputs)

    # Create metric class:
    metric = metrax.Precision.from_model_output(
        predictions=y_pred,
        labels=y_true,
    )

    # Update metric with new inputs:
    metric = metric.merge(
        metrax.Precision.from_model_output(
            predictions=y_pred,
            labels=y_true,
        )
    )

    # Get result:
    result = metric.compute()

Integrate into your training loop
---------------------------------

Metrax metrics that take arrays are jittable (they can be used within a
``jax.jit`` function). If your custom metric uses standard JAX operations and no
dynamic shapes, it should be jittable. You can test with the following:

.. code-block:: python

   logits = jnp.ones((2, 3))
   labels = jnp.ones((2, 3))
   jax.jit(metrax.MSE.from_model_output)(logits, labels)

Jittable metrics can be added directly to your train or eval step.
Non-jittable metrics (such as the string metrics ``BLEU`` and ``WER``) need to
go outside the jitted function.

.. code-block:: python

   @jax.jit
   def eval_step(logits, labels):
     ...
     outputs['mse'] = metrax.MSE.from_model_output(logits, labels)
     outputs['rmse'] = metrax.RMSE.from_model_output(logits, labels)
     return outputs

   def run_eval():
     for logits, labels, pred_texts, ref_texts in eval_dataset:
       # Jittable metrics
       outputs = eval_step(logits, labels)
       # Non-jittable metrics
       outputs['bleu'] = metrax.BLEU.from_model_output(pred_texts, ref_texts)
