# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

GNOVI Studio (repository: Gnovi-Plot, package name `gnovi-plot`, currently version 0.9.0 "Beta") is a cross-platform, open-source Python desktop application for scientific plotting and analysis. It combines experimental data visualization, mathematical equation graphing, and publication-quality figure creation in a single tool.

The project is past the pre-code phase: it is a working PySide6 application (`python -m gnovi_plot`) with an implemented `gui`/`data`/`plotting`/`equations`/`analysis`/`export`/`core` package structure, a 58-file pytest suite, and cross-platform CI + CodeQL running on every push/PR. This file still records the agreed vision and constraints so implementation stays consistent as the codebase grows; see "Current implementation status" below for what's actually built versus still planned.

## Technology stack

- **Python** (>=3.9, CI runs 3.12) — implementation language
- **PySide6** — GUI framework
- **Pandas** — standard internal representation for all tabular experimental data
- **NumPy** — numerical calculations (used directly in `analysis/cycles.py` and `equations/evaluator.py`; not yet a declared top-level dependency, it currently arrives transitively via pandas)
- **Matplotlib** — authoritative scientific plotting and publication-quality rendering backend. `mplot3d`/3D plotting is part of the long-term vision (see architectural principles) but is not implemented yet — only 2D line/scatter/histogram plotting exists today.
- **SciPy** — curve fitting and numerical analysis. Planned per the architectural principles below, but not yet a dependency and not yet used anywhere in the codebase (no `analysis/fitting.py` exists yet).
- **SymPy** — equation parsing and symbolic mathematics (equation input must go through SymPy, never raw `eval()`)
- **pytest** / **pytest-cov** — testing, with coverage collected in CI

## Architectural principles

These constraints were established deliberately and should guide all design and implementation decisions, not just the initial scaffold:

- **Modular by domain, not monolithic.** GUI, data management, plotting, equation handling, and scientific analysis are separate modules/packages. There is no monolithic `main.py` — it should only wire things together.
- **Pandas DataFrame is the canonical internal representation** for tabular experimental data. Other layers (plotting, analysis) consume DataFrames rather than raw arrays/dicts where the data is tabular in nature.
- **Matplotlib stays authoritative for publication-quality rendering.** Any interactive/alternate rendering path is additive, not a replacement.
- **Multiple experimental datasets can coexist and overlap on the same axes.** The data/plotting layer must support this from the start.
- **Equation curves and experimental data are meant to eventually coexist on the same graph.** Design the plotting layer so this integration doesn't require rework later, even if not implemented immediately.
- **Equation plotting must eventually support both `y = f(x)` and `z = f(x, y)`.** Keep the equation-handling module general enough for both, even when only one is implemented first.
- **No unrestricted `eval()` on user-entered equations, ever.** Equation parsing/evaluation must go through a safe SymPy-based approach.
- **3D plotting is a core feature, not an add-on.** Start with Matplotlib `mplot3d`, but structure the plotting layer so an alternate interactive 3D backend (e.g. PyVista) can be added later without rewriting the application — i.e. keep a backend-agnostic plotting interface/abstraction rather than coupling the app directly to `mplot3d` calls everywhere.
- **Specialized scientific analysis (e.g. cyclic voltammetry) lives in its own module(s)**, separate from the general plotting engine. The general plotting/analysis core must not be hard-coded with domain-specific logic.
- **Reproducibility is a major design goal.** Favor designs where a plotting/analysis session's inputs and parameters are enough to reproduce its output deterministically.
- **GUI-created graphs should eventually be exportable as equivalent Python code.** Keep the plotting layer's API something a generated script could call directly (i.e. avoid GUI-only state that can't be expressed as code).
- **Cross-platform: Linux, Windows, macOS.** Linux is the primary development platform — verify assumptions don't silently depend on Linux-only behavior.

## Development philosophy

- **Build incrementally.** Prefer small, working increments over large upfront builds.
- **Don't implement roadmap items early just because they're planned.** Only build what's needed for the current step. As of now this still applies to: `z = f(x,y)` equation support, an interactive 3D/`mplot3d`/PyVista backend, `EquationSeries` on the same graph as data, `code_export.py`, SciPy-based curve fitting, and the `modules/` package (e.g. cyclic voltammetry) — none of these are built yet and shouldn't be started without being explicitly taken on.
- **Discuss before major architectural changes or new major dependencies.** Don't introduce a new core dependency or restructure module boundaries without raising it first.

## Current implementation status

The package layout below reflects the repository as it exists now (not a plan to scaffold). Top-level packages under `gnovi_plot/`:

- `gui/` — PySide6 presentation layer. `main_window.py` (~1,900 lines) is the main application window: multi-Workbench UI (tabs, each with its own figure/panels/datasets), side drawers for dataset/plot-series/figure-layout/properties panels, and the recent cross-platform layout-robustness work (auto-collapsing drawers instead of shrinking below a usable minimum, normalized sidebar/control minimum widths, Workbench-vs-drawer width budgeting) that CI now exercises on both Ubuntu and Windows. Also `styles.py` (theming) and `undo_manager.py`. No `controllers/` submodule — GUI/core coordination goes through normal Qt signals/slots, as originally intended; still no need for a controller layer.
- `data/` — `dataset.py` (`Dataset`: a pandas DataFrame plus metadata — name, units, source, color — with calculated-column support via `equations/evaluator.py`), `dataset_manager.py` (`DatasetManager`, holds multiple coexisting datasets, no Qt dependency), `numeric.py`, `transforms.py` (row-range/calculated-column transformations), and `importers/text_importer.py` (CSV/TSV/TXT/DAT import — the only place file formats are known about).
- `plotting/` — backend-agnostic 2D plotting engine: `figure.py` (`GnoviFigure` and `Panel`, declarative multi-panel plot description — themes, axis labels/limits, grid, legend), `series.py` (`PlotSeries` with `PlotType.LINE/SCATTER/HISTOGRAM` — there is no `EquationSeries` yet, so equation curves and experimental data don't yet coexist on a graph), `graph.py`/`graph_library.py` (saved reusable graph definitions), `stacking.py`, `units.py`, and `backends/` (`base.py`'s `PlotBackend` Protocol plus `matplotlib_backend.py`, the only implementation — 2D only, no `mplot3d` usage yet). There is no `axes.py`; axis state currently lives on `GnoviFigure`/`Panel`.
- `equations/` — `parser.py` (safe SymPy parsing via `parse_expr`, never raw `eval`) and `evaluator.py` (`evaluate_formula`: evaluates a formula against a DataFrame's columns to produce a new Series). Today this powers dataset calculated columns only; grid evaluation for standalone `y=f(x)`/`z=f(x,y)` equation plotting is not implemented.
- `analysis/` — generic, domain-independent analysis: `cycles.py` (`detect_cycles`, turning-point-based repeating-sweep-cycle detection — explicitly documented as domain-independent, shaped for but not limited to cyclic-voltammetry data) and `segments.py` (`contiguous_row_range`, generic row-range selection). No `fitting.py` / SciPy usage yet. No domain-specific submodules live here.
- `modules/` — not created yet. Specialized domain analyses (e.g. cyclic voltammetry) will live here, kept out of `analysis/`/`plotting/`, when that work is taken on.
- `export/` — `figure_export.py` (`export_figure`, `export_live_figure` — rendered-figure image/file export, with WYSIWYG parity tests against the live canvas). No `code_export.py` yet (`GnoviFigure` → standalone Python script export is planned, not built).
- `core/` — `app_info.py` (single source of truth for `APP_NAME = "GNOVI Studio"`, tagline, `__version__`, About text — kept in sync with `pyproject.toml`'s version via `tests/test_app_info.py`), `project.py` (`Project`, owns one or more `Workbench`es), `workbench.py` (`Workbench`: a figure + its datasets + identity), and `project_io.py` (reads/writes the versioned `.gnovi` ZIP project container). No separate `session.py`/`config.py` yet — `Project`/`Workbench`/`project_io.py` currently serve the reproducibility role session.py was meant for.
- `app.py` — thin composition root that wires the above together and launches the app via `python -m gnovi_plot`.

## Testing and CI

- `tests/` has 58 test files covering datasets, transforms, figures/panels, series, graph library, workbenches/projects, project I/O, equation evaluation, cycle detection, export (including WYSIWYG/typography parity against the live canvas), and GUI behavior (responsiveness, aspect-ratio, legend fit, drawer/panel layout, theming, undo).
- `.github/workflows/ci.yml`: on push/PR to `main`, runs the suite on a `[ubuntu-latest, windows-latest]` matrix (Python 3.12, Qt offscreen platform). Ubuntu run collects coverage (`pytest-cov`, uploaded to Codecov, non-blocking) and installs `libegl1`/`libgl1`/`libxkbcommon0`. Windows run currently also executes `scripts/_windows_qt_diagnostics.py` as a temporary, non-blocking diagnostic step left over from the PR #2 cross-platform layout investigation.
- `.github/workflows/codeql.yml`: CodeQL static analysis for Python on push/PR to `main` plus a weekly schedule.
