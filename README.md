# GNOVI Studio

[![CI](https://github.com/wavicles/GNOVI-Studio/actions/workflows/ci.yml/badge.svg)](https://github.com/wavicles/GNOVI-Studio/actions/workflows/ci.yml)
[![CodeQL](https://github.com/wavicles/GNOVI-Studio/actions/workflows/codeql.yml/badge.svg)](https://github.com/wavicles/GNOVI-Studio/actions/workflows/codeql.yml)
[![Python](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fwavicles%2FGNOVI-Studio%2Fmain%2Fpyproject.toml&query=%24.project.requires-python&label=python)](pyproject.toml)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22066802.svg)](https://doi.org/10.5281/zenodo.22066802)
[![Software Heritage](https://archive.softwareheritage.org/badge/directory/788767f5421078a3a97b3c9cc94a4c310c90c075/)](https://archive.softwareheritage.org/swh:1:dir:788767f5421078a3a97b3c9cc94a4c310c90c075;origin=https://doi.org/10.5281/zenodo.22066802;visit=swh:1:snp:e3fdd6a23a82d9ec656225cd5c3612e15d68e528;anchor=swh:1:rel:0938fb937d72a19cbe55c00b33e3cb9b4ec486c6;path=wavicles-GNOVI-Studio-ad7c865)

**Scientific Plotting & Analysis Studio** — a cross-platform Python desktop
application for scientific plotting, experimental data analysis, and
publication-quality figure creation.

GNOVI Studio helps researchers, students, and scientific Python users import
experimental data, build multi-panel figures, and run reproducible
curve-fitting analysis in a single open-source desktop application.

## Overview

GNOVI Studio is built on NumPy, SciPy, pandas, and Matplotlib, and is
designed to keep the full analysis workflow — from imported data, through
curve fitting, to a finished figure — transparent and reproducible. It
targets researchers and students who want a dedicated plotting and analysis
tool rather than assembling one from scripts and notebooks each time.

## Features

**Data import**
- CSV, TXT, TSV, and DAT import
- Preview-driven import with automatic header/data-row detection
- Raw and working-data workflow, so imported data is never modified in place
- Calculated/derived columns using mathematical expressions

**Plotting & figures**
- Multi-series plotting
- Multi-panel figures
- Workbenches for organizing related plots and datasets
- Graph Library for saving and reusing graph definitions
- Panel/layout and figure customization

**Analysis**
- Curve fitting
- Fit diagnostics and residual analysis
- Panel-scoped analysis history
- Add/Remove Fit Curve on a figure

**Project & output**
- Project save/open
- Undo/redo
- Publication-quality figure export (PNG, TIFF, SVG, PDF)

## Scientific Analysis

GNOVI Studio's curve fitting is built around a small, well-tested set of
models:

- Linear
- Polynomial
- Exponential
- Gaussian

For each fit, GNOVI reports R², adjusted R², and parameter uncertainty
estimates, and provides residual diagnostics to help assess fit quality.
Analysis results are kept in a persistent, panel-scoped history, and fitted
curves can be added to or removed from a figure directly.

## Platform Support

Linux is the primary development platform, with Ubuntu as the reference
Linux environment.

Continuous integration currently validates:

- Ubuntu
- Fedora
- Windows

macOS is not currently CI-validated.

## Installation & Running from Source

GNOVI Studio is currently run from source. It is not yet published on PyPI,
and no packaged installer or executable is provided.

**Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
python -m gnovi_plot
```

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[test]"
python -m gnovi_plot
```

**Running tests**

```bash
pytest
```

## Development Status

GNOVI Studio is under active development and is currently in Beta. The
application identifies itself internally as `v0.9.0 Beta`.

A corresponding `v0.9.0` release has not yet been published on GitHub; that
release is being prepared separately.

## Testing & Quality

GNOVI Studio is developed with an automated pytest test suite covering data
import, plotting, analysis, project persistence, and the GUI. Continuous
integration runs on Ubuntu, Fedora, and Windows on every push and pull
request, alongside CodeQL static analysis.

## Project Philosophy

GNOVI Studio is built as open, reproducible scientific software:

- Reproducible scientific workflows, from raw data to figure
- Transparent analysis, with visible fit statistics and residual diagnostics
- Publication-quality output as a first-class goal
- A modular, domain-organized codebase
- Cross-platform development, with Linux as the primary platform

See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for the full development guide,
including architecture, conventions, and contribution rules.

## Citation

If you use GNOVI Studio in your work, please cite it using the information
in [`CITATION.cff`](CITATION.cff). GitHub's "Cite this repository" feature,
available in the repository sidebar, can generate a citation from this file
in several formats.

## License

GNOVI Studio is licensed under the [GPL-3.0-or-later](LICENSE).

## Contributing & Issues

Bug reports, feature requests, and contributions are welcome. Please open an
issue at [github.com/wavicles/GNOVI-Studio/issues](https://github.com/wavicles/GNOVI-Studio/issues).

For development setup, architecture, and contribution conventions, see
[PROJECT_GUIDE.md](PROJECT_GUIDE.md).
