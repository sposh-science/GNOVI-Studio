# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Gnovi-Plot (GNOVI PLOT) is a cross-platform, open-source Python desktop application for scientific plotting and analysis. It combines experimental data visualization, mathematical equation graphing, 2D/3D visualization, and publication-quality figure creation in a single tool.

The project is in the pre-code architecture phase: no application code exists yet. This file records the agreed vision and constraints so implementation stays consistent as the codebase grows.

## Technology stack

- **Python** — implementation language
- **PySide6** — GUI framework
- **Pandas** — standard internal representation for all tabular experimental data
- **NumPy** — numerical calculations
- **Matplotlib** — authoritative scientific plotting and publication-quality rendering backend (including `mplot3d` for 3D)
- **SciPy** — curve fitting and numerical analysis
- **SymPy** — equation parsing and symbolic mathematics (equation input must go through SymPy, never raw `eval()`)
- **pytest** — testing

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
- **Don't implement roadmap items early just because they're planned.** Only build what's needed for the current step (e.g. `z = f(x,y)` support, PyVista backend, code export, and CV analysis are all planned but should only be built when explicitly taken on).
- **Discuss before major architectural changes or new major dependencies.** Don't introduce a new core dependency or restructure module boundaries without raising it first.

## Approved package architecture

No code exists yet; this is the agreed folder layout to scaffold when implementation starts. Top-level packages:

- `gui/` — PySide6 presentation layer only. No `controllers/` submodule for now — use normal Qt signal/slot coordination between widgets and the layers below. Introduce a controller layer later only if GUI/core coordination complexity actually justifies it; don't pre-build it.
- `data/` — `Dataset` (a pandas DataFrame plus metadata: name, units, source, color) and `DatasetManager` (holds multiple coexisting datasets). Pandas DataFrame stays the canonical tabular representation. `importers/` is the only place file formats are known about.
- `plotting/` — backend-agnostic plotting engine only: `figure.py` (`GnoviFigure`, declarative plot description), `series.py` (`DataSeries` / `EquationSeries`), `axes.py` (axis configuration/state), and `backends/` (`base.py` defines a minimal `PlotBackend` interface; `matplotlib_backend.py` is the only implementation for now, covering 2D and `mplot3d`). Keep `PlotBackend` minimal — only what `matplotlib_backend.py` actually needs today. Don't design interface surface for a hypothetical PyVista backend before it exists; the swap should be possible later without requiring the abstraction to be elaborate now.
- `equations/` — `parser.py` (safe SymPy parsing of user expressions — the only place equation text is parsed, never raw `eval`) and `evaluator.py` (numeric evaluation over grids for `y=f(x)` and `z=f(x,y)`).
- `analysis/` — generic, domain-independent scientific analysis only, e.g. `fitting.py` (SciPy-based curve fitting shared across analyses). No domain-specific submodules live here.
- `modules/` — top-level package for specialized domain analyses (e.g. `cyclic_voltammetry/`), kept out of `analysis/` and `plotting/` entirely so the general engine stays domain-agnostic.
- `export/` — top-level package, separate from `plotting/`: `figure_export.py` (export a rendered figure as an image/file) and `code_export.py` (`GnoviFigure` → equivalent standalone Python/matplotlib script).
- `core/` — cross-cutting concerns: `session.py` (session state — datasets, figures, equations, params — for reproducibility) and `config.py`.
- `app.py` — thin composition root that wires the above together and launches the app. Not a monolithic `main.py`.
