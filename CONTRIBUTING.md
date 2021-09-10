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
pip install sphinx autodoc recommonmark sphinx-rtd-theme sphinxcontrib-apidoc
```

Generate HTML:
```bash
cd docs && make html
```

Run HTTP server:
```bash
cd docs/_build/html && python -m http.server 8000
```

Don't forget to change the version in `docs/conf.py` before generating.
```python
release = "<VERSION>"
```

Generated documentation will be in `docs/_build/html`.

#### Documentation common issues

1. Use `:language: shell-session` for GDB shell sessions:
```
.. literalinclude:: <...>
  :language: shell-session
```
2. Use `:language: bash` for commands which could be inserted in shell or script:

```
.. code-block:: bash
    export IGC_ShaderDumpEnable=1
```
3. Use `:lineno-match:` if line numbers matter and example file contains license header:
```
.. literalinclude:: <...>
    :linenos:
    :lineno-match:
```

### Uploading to GitHub Pages

Documentation for GitHub Pages is placed in following branch
[`gh-pages`](https://github.com/IntelPython/numba-dppy/tree/gh-pages).

Folders:
- `dev` folder contains current documentation for default branch.
- `0.12.0` folder and other similar folders contain documentation for releases.
- `latest` folder is a link to the latest release folder.

Copy generated documentation into corresponding folder and create pull request
to `gh-pages` branch.
