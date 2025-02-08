metrax Documentation
=====================

**metrax** provides common evaluation metric implementations for JAX.

Getting Started
---------------

Metrics are based on `clu.Metric`.

.. code-block::

    # Run model:
    y_true, y_pred = model(inputs)

    # Create metric class:
    metric = metrics.Precision.from_model_output(
        predictions=y_pred,
        labels=y_true,
    )

    # Update metric with new inputs:
    metric = metric.merge(
        metrics.Precision.from_model_output(
            predictions=y_pred,
            labels=y_true,
        )
    )

    # Get result:
    result = metric.compute()


Metrax API
==========

.. toctree::
   :maxdepth: 2

   metrax API <metrax>
