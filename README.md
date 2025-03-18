# metrax

`metrax` is a library with standard eval metrics implementations in JAX.

## Vision

While the [JAX ecosystem](https://docs.jax.dev/en/latest/) is powerful, it currently lacks a core evaluation metrics library. This absence presents challenges for users transitioning from other frameworks like TensorFlow or PyTorch, which offer built-in metrics reporting capabilities.

To address these challenges, we introduced metrax, a standalone JAX model evaluation library that:
*   Provides essential predefined metrics: Includes metrics commonly used to evaluate various machine learning models (classification, regression, recommendation, and language modeling), with the flexibility to add more in the future.
*   Leverages existing library as a foundation: Builds upon the robust [CLU](https://github.com/google/CommonLoopUtils) library, ensuring compatibility and consistency in distributed and scaled training environments.
*   Is developed on GitHub first and is used by several Google core products.

Please refer to the [readthedocs page](http://metrax.readthedocs.io/) of the library for more information.

## Installation

Install the package by installing the PyPi release.

```
pip install google-metrax
```

## Development

Run the tests:

```sh
pytest src/metrax
```

Develop the docs locally:

```
pip install -r ./docs/requirements.txt
sphinx-build ./docs /tmp/metrax_docs
python -m http.server --directory /tmp/metrax_docs
```
