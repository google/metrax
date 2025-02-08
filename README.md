# metrax

`metrax` is a library with standard eval metrics implementations in JAX.

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
sphinx-build ./docs /tmp/metrax_docs
python -m http.server --directory /tmp/metrax_docs
```
