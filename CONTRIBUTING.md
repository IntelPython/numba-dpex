# Contributing

## Python code style

### black

We use [black](https://black.readthedocs.io/en/stable/) code formatter.

- Revision: `20.8b1`.
- See configuration in `pyproject.toml`.

Install:
```bash
python -m pip install black
```

Run before each commit:
```bash
black .
```

### License

We use [addlicense](https://github.com/google/addlicense) license checker.

Install:
```bash
conda install go
export PATH=${PATH}:`go env GOPATH`/bin
go get -u github.com/google/addlicense
```

Run before each commit:
```bash
export PATH=${PATH}:`go env GOPATH`/bin
addlicense -l apache -c "Intel Corporation" numba_dppy/**/*.py numba_dppy/*.py setup.py
```

## Security

### Bandit

We use [Bandit](https://github.com/PyCQA/bandit) to find common security issues in Python code.

Install: `pip install bandit`

- Revision: `1.7.0`

Run before each commit: `bandit -r numba_dppy -lll`

## Documentation

### Generating documentation

Install Sphinx and plugins:
```bash
pip install sphinx autodoc recommonmark sphinx-rtd-theme
```

Generate HTML:
```bash
cd docs
make html
```

Generated documentation will be in `docs/_build/html`.

### Uploading to GitHub Pages

Documentation for GitHub Pages is placed in following branch
[`gh-pages`](https://github.com/IntelPython/numba-dppy/tree/gh-pages).

Folders:
- `dev` folder contains current documentation for default branch.
- `0.12.0` folder and other similar folders contain documentation for releases.
- `latest` folder is a link to the latest release folder.

Copy generated documentation into corresponding folder and create pull request
to `gh-pages` branch.

## Code Coverage

Implement python file coverage using `coverage` and `pytest-cov` packages.

### Using coverage

Install Coverage:
```bash
conda install coverage
```

Run Coverage:
```bash
coverage run -m pytest
```

Show report:
```bash
coverage report
```

- For each module executed, the report shows the count of executable statements, the number of those statements missed, and the resulting coverage, expressed as a percentage.

The `-m` flag also shows the line numbers of missing statements:
```bash
coverage report -m
```

Produce annotated HTML listings with coverage results:
```bash
coverage html
```

- The htmlcov folder will appear in the root folder of the project. It contains reports on python file coverage in html format.

Erase previously collected coverage data:
```bash
coverage erase
```

### Using pytest-cov

This plugin provides a clean minimal set of command line options that are added to pytest.

You must have `coverage` package installed to use pytest-cov.

Install pytest-cov:
```bash
conda install pytest-cov
```

Run pytest-cov:
```bash
pytest --cov=numba_dppy
```

The complete list of command line options is:
- `--cov=PATH`

Measure coverage for filesystem path. (multi-allowed)
- `--cov-report=type`

Type of report to generate: term(the terminal report without line numbers (default)), term-missing(the terminal report with line numbers), annotate, html, xml (multi-allowed).
