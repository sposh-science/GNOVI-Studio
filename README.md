# GNOVI PLOT

Scientific Plotting & Analysis Studio.

A cross-platform Python desktop application for scientific plotting, experimental
data visualization, equation graphing, and publication-quality figure creation.

## Status

Early development. Milestone 1: a minimal, runnable PySide6 application with an
embedded Matplotlib canvas — the foundation the rest of the application will be
built on.

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
