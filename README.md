This repo is to reproduce the CPython issue https://github.com/python/cpython/issues/105134.

Repro steps:
1. Clone this repo and `cd` into it
2. Run `python -m venv .venv`
3. Activate the `.venv` virtual environment
4. Run `pip install -e ./wrapt`
5. Run `python repro.py`
