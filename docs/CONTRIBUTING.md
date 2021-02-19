# Contributing

## Python code style

### black

We use [black](https://black.readthedocs.io/en/stable/) code formatter.

- Revision: `20.8b1` or branch `stable`.
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
