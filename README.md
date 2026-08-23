# GNOVI Studio

[![CI](https://github.com/wavicles/Gnovi-Plot/actions/workflows/ci.yml/badge.svg)](https://github.com/wavicles/Gnovi-Plot/actions/workflows/ci.yml)
[![CodeQL](https://github.com/wavicles/Gnovi-Plot/actions/workflows/codeql.yml/badge.svg)](https://github.com/wavicles/Gnovi-Plot/actions/workflows/codeql.yml)
[![Python](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fwavicles%2FGnovi-Plot%2Fmain%2Fpyproject.toml&query=%24.project.requires-python&label=python)](pyproject.toml)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](LICENSE)

Scientific Plotting & Analysis Studio.

A cross-platform Python desktop application for scientific plotting, experimental
data visualization, equation graphing, and publication-quality figure creation.

## Status

GNOVI Studio is under active development. Core scientific plotting,
multi-panel Workbenches, project persistence, graph reuse, curve fitting,
fit diagnostics, residual analysis, and panel-scoped analysis history are
implemented and covered by automated tests on Linux (Ubuntu) and Windows.

## Platform Support

Linux is the primary development platform. Ubuntu is the reference Linux
environment validated by continuous integration. Windows is also CI-tested on
every push and pull request.

GNOVI Studio is designed to remain distribution-independent and is intended
to work on other Linux distributions, although Ubuntu is currently the Linux
distribution validated automatically.

## Development setup (Linux)

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project (with test dependencies):

```bash
pip install -e ".[test]"
```

## Running the application

```bash
python -m gnovi_plot
```

## Running tests

```bash
pytest
```
