# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Testing

Please make sure that your PR passes all tests by running `pytest ./src/` on your
local machine. Also, you can run only tests that are affected by your code
changes, but you will need to select them manually.

Metrax uses [ruff](https://github.com/astral-sh/ruff) for linting. Before
sending a PR please run `ruff check` to catch any issues.

## Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google.com/conduct/).

## Release Checklist

1. Bump version in `pyproject.toml` according to [semver](https://semver.org/)
1. Commit change
1. [Generate and upload to PyPI](https://packaging.python.org/en/latest/tutorials/packaging-projects/#generating-distribution-archives)

```
# pip install --upgrade build twine
python -m build
python -m twine upload --repository pypi dist/*
python -m twine upload --repository testpypi dist/*
```
