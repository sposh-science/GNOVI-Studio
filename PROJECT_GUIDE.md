# GNOVI Studio Project Guide

This file documents GNOVI Studio's agreed architecture, conventions, and
constraints for anyone working on this repository -- human contributors and
any development tool alike. Treat it as the authoritative source for the
project's design decisions, and follow it for all work in this repository.

## Project overview

GNOVI Studio (repository: GNOVI-Studio, package name `gnovi-plot`, currently version 0.9.0 "Beta", tagged as `v0.9.0` with a citable DOI in `CITATION.cff`) is a cross-platform, open-source Python desktop application for scientific plotting and analysis. It combines experimental data visualization, mathematical equation graphing, and publication-quality figure creation in a single tool.

The project is past the pre-code phase: it is a working PySide6 application (`python -m gnovi_plot`) with an implemented `gui`/`data`/`plotting`/`equations`/`analysis`/`export`/`core` package structure, an 82-file pytest suite (more than 1,600 automated tests -- see "Testing and CI" for how to get an exact current count), and cross-platform CI + CodeQL running on every push/PR. 2D plotting, multi-panel figures, and a 3D foundation (scatter, line, grouped curve families, and publication-oriented styling) are all implemented and merged -- see "Current implementation status" below for the precise, current feature baseline, and "Roadmap" for what's next. This file still records the agreed vision and constraints so implementation stays consistent as the codebase grows.

## Technology stack

- **Python** (>=3.9, CI runs 3.12) — implementation language
- **PySide6** — GUI framework
- **Pandas** — standard internal representation for all tabular experimental data
- **NumPy** — numerical calculations (used directly in `analysis/cycles.py`, `analysis/fitting.py`, `data/numeric.py`, and `equations/evaluator.py`; not yet a declared top-level dependency, it currently arrives transitively via pandas and SciPy)
- **Matplotlib** — authoritative scientific plotting and publication-quality rendering backend, for both 2D (line/scatter/histogram) and 3D (scatter/line/line+markers via `mpl_toolkits.mplot3d`). See "2D / 3D model architecture" and "Rendering architecture" below for exactly what's implemented. 3D surface/wireframe/mesh rendering is not implemented yet.
- **SciPy** — curve fitting and numerical analysis, via `scipy.optimize.curve_fit` in `analysis/fitting.py` (`fit_curve`/`FitResult`) and, for XRD, `scipy.signal.find_peaks`/`savgol_filter` in `modules/xrd/`. A declared `pyproject.toml` dependency.
- **SymPy** — equation parsing and symbolic mathematics (equation input must go through SymPy, never raw `eval()`)
- **pybaselines** — arPLS baseline correction in `modules/xrd/preprocessing.py` only. Optional (`xrd` extra), not core — see "Analysis" below.
- **pytest** / **pytest-cov** — testing, with coverage collected in CI

## Architectural principles

These constraints were established deliberately and should guide all design and implementation decisions, not just the initial scaffold:

- **Modular by domain, not monolithic.** GUI, data management, plotting, equation handling, and scientific analysis are separate modules/packages. There is no monolithic `main.py` — it should only wire things together.
- **Pandas DataFrame is the canonical internal representation** for tabular experimental data. Other layers (plotting, analysis) consume DataFrames rather than raw arrays/dicts where the data is tabular in nature. This still holds for every dataset in the app today: there is no grid/matrix (2D array/mesh) data representation yet -- see "Data model: current limitation and future direction" below.
- **Matplotlib stays authoritative for publication-quality rendering.** Any interactive/alternate rendering path is additive, not a replacement.
- **Multiple experimental datasets can coexist and overlap on the same axes.** The data/plotting layer supports this for both 2D `Panel`/`PlotSeries` and 3D `Panel3D`/`Series3D`.
- **Equation curves and experimental data are meant to eventually coexist on the same graph.** Design the plotting layer so this integration doesn't require rework later. Not implemented yet -- there is still no `EquationSeries`; see "Analysis" and the development-philosophy note below.
- **Equation plotting must eventually support both `y = f(x)` and `z = f(x, y)`.** Keep the equation-handling module general enough for both, even when only one is implemented first. Neither is implemented yet -- `equations/evaluator.py` only evaluates a formula against an existing DataFrame's columns (dataset calculated columns), not standalone equation plotting.
- **No unrestricted `eval()` on user-entered equations, ever.** Equation parsing/evaluation must go through a safe SymPy-based approach.
- **3D plotting is a core feature, not an add-on.** Matplotlib `mplot3d` now backs a real, merged 3D scatter/line/curve-family/publication-styling feature set (see "2D / 3D model architecture"). The plotting layer keeps a backend-agnostic `PlotBackend` interface (`plotting/backends/base.py`) rather than coupling the app directly to Matplotlib calls everywhere in `gui`/`core`, but see "Rendering architecture" below for exactly how far that abstraction currently goes -- do not describe `PlotBackend` as a general swappable-renderer system beyond what its actual Protocol provides today. 3D surface/wireframe/mesh is future work, likely dependent on a structured grid data model (see "Data model" and "Roadmap").
- **Specialized scientific analysis (e.g. cyclic voltammetry, XRD) lives in its own module(s)**, separate from the general plotting engine. The general plotting/analysis core must not be hard-coded with domain-specific logic. No such domain-specific module exists yet (`gnovi_plot/modules/` has not been created).
- **Reproducibility is a major design goal.** Favor designs where a plotting/analysis session's inputs and parameters are enough to reproduce its output deterministically.
- **GUI-created graphs should eventually be exportable as equivalent Python code.** Keep the plotting layer's API something a generated script could call directly (i.e. avoid GUI-only state that can't be expressed as code). Not implemented yet (no `code_export.py`).
- **Cross-platform: Linux, Windows, macOS.** Linux is the primary development platform — verify assumptions don't silently depend on Linux-only behavior.

## Development philosophy

- **Build incrementally.** Prefer small, working increments over large upfront builds.
- **Don't implement roadmap items early just because they're planned.** Only build what's needed for the current step. The Matplotlib `mplot3d` backend and a 3D scatter/line/curve-family/publication-styling feature set have now been built and merged, so they no longer belong on this "not yet started" list. As of now the list still applies to: `z = f(x,y)` (and standalone `y = f(x)`) equation plotting, `EquationSeries` on the same graph as data, `code_export.py`, the `modules/` package (e.g. cyclic voltammetry, XRD), 3D surface/wireframe/mesh rendering, and a structured `GridDataset`/matrix-data model — none of these are built yet and shouldn't be started without being explicitly taken on.
- **Discuss before major architectural changes or new major dependencies.** Contributors should not introduce a new core dependency or restructure module boundaries without raising it first.

## Scientific Python Library Policy

GNOVI Studio should use modern, actively maintained, stable Python scientific libraries wherever appropriate rather than reimplementing established numerical or scientific algorithms.

Prefer the current stable, mutually compatible releases of:

- **NumPy** for numerical arrays and fundamental numerical operations
- **SciPy** for optimization, curve fitting, signal processing, interpolation, statistics, and other scientific algorithms
- **pandas** for tabular scientific data handling
- **Matplotlib** for publication-quality scientific plotting and vector/raster export
- **PySide6** / modern Qt APIs for the desktop GUI

For future scientific capabilities such as XRD, Raman, UV–Vis, CV, spectroscopy, peak analysis, smoothing, baseline correction, and SEM/TEM image analysis, first evaluate established actively maintained Python libraries before implementing algorithms from scratch.

Do not add a dependency merely because it exists. A new dependency should provide a meaningful scientific, numerical, performance, reliability, or maintenance advantage.

Use **latest stable and compatible**, not blindly latest. Before introducing or upgrading a major dependency, verify:

- Python-version compatibility
- Linux compatibility
- Windows compatibility
- macOS compatibility
- availability of binary wheels where relevant
- compatibility with the existing GNOVI dependency stack
- licensing compatibility with GNOVI Studio
- suitability for future application packaging

Avoid obsolete, abandoned, or unnecessarily niche libraries when a well-maintained scientific-Python alternative exists.

Keep numerical/scientific logic separated from the Qt GUI so scientific algorithms can be independently tested and replaced or upgraded without redesigning the user interface — this is already how `analysis/`/`equations/` are structured relative to `gui/` (see "Current implementation status" below), and new analysis tools should keep following that split.

Do not change any existing working dependency solely to satisfy this policy. Apply it to new development and evaluate dependency upgrades separately.

## Current implementation status

The package layout below reflects the repository as it exists now (not a plan to scaffold). Top-level packages under `gnovi_plot/`:

- `gui/` — PySide6 presentation layer. `main_window.py` is the main application window: multi-Workbench UI (tabs, each with its own figure/panels/datasets), a grouped sidebar drawer (see "Sidebar / workspace" below), and cross-platform layout-robustness work (auto-collapsing drawers instead of shrinking below a usable minimum, normalized sidebar/control minimum widths, Workbench-vs-drawer width budgeting) that CI exercises on both Ubuntu and Windows. `widgets/analysis_panel.py` is the Curve Fitting / Analysis History drawer page (Run Fit, Add/Remove Fit Curve, a per-panel selectable result history); `widgets/analysis_result_view.py` and `widgets/residual_window.py` show fit results and residual diagnostics. `widgets/plot3d_panel.py` is the "3D" sidebar page (add/configure `Series3D`). `widgets/figure_properties_panel.py` (the "Axes" page) adapts to whichever panel type is active, rendering 3D-specific controls (camera, grid, panes, aspect, per-axis tick spacing) when the active panel is a `Panel3D`, instead of a separate duplicated 3D destination. Also `styles.py` (theming) and `undo_manager.py`. No `controllers/` submodule — GUI/core coordination goes through normal Qt signals/slots, as originally intended; still no need for a controller layer.
- `data/` — `dataset.py` (`Dataset`: a pandas DataFrame plus metadata — name, units, source, color — with calculated-column support via `equations/evaluator.py`), `dataset_manager.py` (`DatasetManager`, holds multiple coexisting datasets, no Qt dependency), `numeric.py` (numeric-validity helpers for both 2D (`numeric_xy`, `numeric_column`) and 3D (`numeric_xyz`) series, plus `group_row_positions`, which backs `Series3D`'s "Group by" curve-family grouping — see below), `transforms.py` (row-range/calculated-column transformations), and `importers/text_importer.py` (CSV/TSV/TXT/DAT import — the only place file formats are known about). All of this is still exclusively tabular/column-based — see "Data model" below.
- `plotting/` — backend-agnostic 2D+3D plotting engine: `figure.py` (`GnoviFigure`, `Panel`, and `Panel3D`; `series.py`'s `PlotSeries` with `PlotType.LINE/SCATTER/HISTOGRAM`; `series3d.py`'s `Series3D`), `graph.py`/`graph_library.py` (saved reusable graph definitions, for both `Panel` and `Panel3D`), `stacking.py`, `units.py`, and `backends/` (`base.py`'s `PlotBackend` Protocol plus `matplotlib_backend.py`, the only implementation). See "2D / 3D model architecture" and "Rendering architecture" below for detail. There is no `EquationSeries` yet, so equation curves and experimental data don't yet coexist on a graph. There is no `axes.py`; axis state lives on `GnoviFigure`/`Panel`/`Panel3D`.
- `equations/` — `parser.py` (safe SymPy parsing via `parse_expr`, never raw `eval`) and `evaluator.py` (`evaluate_formula`: evaluates a formula against a DataFrame's columns to produce a new Series). Today this powers dataset calculated columns only; grid evaluation for standalone `y=f(x)`/`z=f(x,y)` equation plotting is not implemented.
- `analysis/` — generic, domain-independent analysis only; see "Analysis" below.
- `modules/` — specialized domain analyses, kept out of `analysis/`/`plotting/`. `modules/xrd/` is the first: a native numerical foundation (radiation/wavelength, Bragg's-law d-spacing, background/smoothing preprocessing, peak detection) — see "Analysis" below for exactly what exists and what doesn't yet. No other `modules/` submodule exists yet (e.g. cyclic voltammetry).
- `export/` — see "Export" below.
- `core/` — `app_info.py` (single source of truth for `APP_NAME = "GNOVI Studio"`, tagline, `__version__`, About text — kept in sync with `pyproject.toml`'s version via `tests/test_app_info.py`), `project.py` (`Project`, owns one or more `Workbench`es, including duplicate/remove/extract-panel-to-workbench), `workbench.py` (`Workbench`: a figure + its datasets + identity + its own `analysis_results` — a `PanelResultHistory`), and `project_io.py` (reads/writes the versioned `.gnovi` ZIP project container — see "Project format / persistence" below). No separate `session.py`/`config.py` yet — `Project`/`Workbench`/`project_io.py` currently serve the reproducibility role session.py was meant for.
- `app.py` — thin composition root that wires the above together and launches the app via `python -m gnovi_plot`.

### 2D / 3D model architecture

`GnoviFigure.panels` is a `list[Panel | Panel3D]` — a figure can freely mix 2D and 3D panels, and both panel kinds render into the same figure (see "Rendering architecture").

- **`Panel`** (`plotting/figure.py`) — a 2D subplot: axis labels/limits/scale, grid, legend, tick spacing, scientific-notation toggles, `panel_aspect_preset`, and a list of `PlotSeries`.
- **`Panel3D`** (`plotting/figure.py`) — the 3D sibling of `Panel`, rendered as a Matplotlib `mpl_toolkits.mplot3d.Axes3D`. It is deliberately **not a subclass of `Panel`**: `GnoviFigure.panels` uses plain structural duck typing (both expose `.id`/`.panel_label`/`.title`/`.series`/`.add_series`/`.remove_series`/`.get_series`/`.invalidate_series_for_dataset`/`.to_dict`), not a formal Protocol/ABC — the two dataclasses' actual field sets barely overlap (a 3D panel has no use for `xscale`/`tick_direction`; a 2D panel has no use for a third axis or a camera). `Panel3D` holds a list of `Series3D`.
- **`PlotSeries`** (`plotting/series.py`) — a 2D series (`PlotType.LINE`/`SCATTER`/`HISTOGRAM`).
- **`Series3D`** (`plotting/series3d.py`) — the 3D sibling of `PlotSeries`, deliberately not a variant of it. Supported kinds are `Plot3DType.SCATTER`, `LINE`, and `LINE_MARKER` — there is **no surface/mesh/wireframe/trisurf kind**; that is explicitly out of scope for the current milestone (see "Roadmap"). A `Series3D` optionally restricts itself to an explicit, ordered subset of its `Dataset`'s row positions via `row_indices`, which is how a "Group by" grouped curve family (e.g. one `Series3D` per distinct temperature in a sweep dataset) is represented: every series in the family shares the same live `Dataset`, only the row selection differs, and row order is always the dataset's own original source order (grouping never sorts by X or any other column, since a sweep can be genuinely non-monotonic by design).

### Rendering architecture

Matplotlib remains the sole, authoritative rendering backend. `plotting/backends/matplotlib_backend.py` provides:

- `render_figure(axes_list, figure, ...)` — draws every panel (2D or 3D) into its corresponding pre-built `Axes`.
- `build_projection_aware_axes(mpl_figure, rows, cols, panels)` — constructs each subplot's `Axes` with the correct Matplotlib projection (`"3d"` for a `Panel3D` entry, the default 2D projection otherwise), so a figure with mixed 2D/3D panels lays out and renders correctly in one pass. This is exercised by both the live GUI canvas and export.
- `render_panel_3d`/`_draw_series_3d` — the 3D counterpart of the existing 2D per-series drawing, using `Axes3D.scatter` (SCATTER) and `Axes3D.plot` (LINE/LINE_MARKER).
- `_apply_3d_grid_style` — applies `Panel3D`'s per-panel grid styling through a documented Matplotlib private-API workaround (mplot3d has no public per-axis grid-style API); pane styling (`pane_visible`/`pane_color`/`pane_alpha`), by contrast, uses Matplotlib's fully public `Axis.pane` API.

`plotting/backends/base.py` defines `PlotBackend` as a minimal `Protocol` with a single method, `render_figure(axes_list, figure)`. This keeps the app's GUI/export code decoupled from calling Matplotlib directly everywhere, but it is **not** a general, freely swappable multi-backend system today — `matplotlib_backend.py` is the only implementation, and the Protocol is intentionally scoped to only what that one implementation needs, not designed against a hypothetical future backend (e.g. PyVista) before one exists.

### Sidebar / workspace

The left tool drawer is grouped into four visually separated sections (`ToolDrawer.add_section` headings), in this order:

- **DATA** — `Data` (dataset import/list)
- **PLOT** — `2D` and `3D` (add/configure `PlotSeries` / `Series3D`)
- **FORMAT** — `Series`, `Axes`, `Figure`, `Layout`
- **ANALYZE** — `Analysis` (curve fitting / analysis history)

`Series` and `Axes` are single destinations, not duplicated per panel type: both adapt their controls to whichever panel (`Panel` or `Panel3D`) is currently active, rather than the sidebar exposing separate 2D-Axes/3D-Axes or 2D-Series/3D-Series pages.

### 3D publication controls

`Panel3D` supports, at an architectural level (see `plotting/figure.py` for the full field list and each field's own docstring):

- Axis labels and limits (`x_label`/`y_label`/`z_label`, `xlim`/`ylim`/`zlim`)
- Scientific/data aspect (`aspect_mode`: `"auto"` or `"equal"`, mapped to `Axes3D.set_aspect(...)` — deliberately not `set_box_aspect()`, which is a separate, unimplemented physical-shape concern)
- Per-axis major/minor tick spacing (`major_tick_spacing_x/y/z`, `minor_tick_spacing_x/y/z`)
- Grid styling (`grid_linestyle`/`grid_linewidth`/`grid_alpha`/`grid_color`) and pane styling (`pane_visible`/`pane_color`/`pane_alpha`)
- Legend (`legend_visible`/`legend_loc`/`legend_ncol`/`legend_frameon`)
- Camera (`elevation`/`azimuth`, the two `Axes3D.view_init(elev=, azim=)` parameters — deliberately not a full orientation; `roll` is not implemented)

Interactive mouse rotation of a 3D panel in the GUI is **transient, session-only view state** — it never writes back into `Panel3D.elevation`/`.azimuth` on its own, exactly like interactive 2D pan/zoom never writes into `Panel.xlim`/`.ylim`. The "Set Current View" control in the Axes page explicitly captures the live rendered view's current elevation/azimuth into the persistent `Panel3D` fields; a separate reset action restores the Matplotlib defaults. Only the explicitly-set (or default) values are ever exported or saved.

### Export

`export/figure_export.py` provides `export_figure` (from a `GnoviFigure` model, headless) and `export_live_figure` (WYSIWYG-tested against the live GUI canvas), plus panel-scoped export (`build_panel_export_figure`/`export_panel`, typed `panel: Panel | Panel3D`) reachable from the panel context menu ("Export Panel…") and the Panels menu ("Export Active Panel…"). Both full-figure and single-panel export work for 3D panels — `build_projection_aware_axes` (see "Rendering architecture") is used on the export path too, so 3D projection and camera state render the same way in an exported file as in the live canvas.

Supported formats (`SUPPORTED_FORMATS` in `export/figure_export.py`): raster `png`, `tiff`; vector `svg`, `pdf`. DPI is configurable for the raster formats. No claim beyond this should be made about vector-specific behavior (e.g. per-element vector editability) without checking what the export tests actually establish.

### Focus / Extract

Both are implemented generically over `Panel | Panel3D` — neither the "Focus Panel" nor "Extract Panel to New Workbench" code path contains a 2D-only type check:

- **Focus** (`MainWindow._focus_panel`) tracks the focused panel purely by `.id` (a field both `Panel` and `Panel3D` expose); it never clones or type-checks the panel object, so a focused `Panel3D` is the exact same live object before, during, and after "Restore Multi-Panel View" — any modification made while focused persists, by construction.
- **Extract** (`Project.extract_panel_to_workbench`) clones the source panel via `plotting.graph.clone_panel_with_shared_datasets`, which is explicitly documented to work unchanged for either `Panel` or `Panel3D` (a plain `copy.deepcopy` plus a fresh `.id`). For a `Panel3D`, this deep-copies its `Series3D` list (including grouped-curve `row_indices`) and all publication-styling fields (camera, grid, panes, legend, aspect, tick spacing), assigns the extracted panel a fresh `id`, and preserves 2D Extract's existing `Dataset`-sharing semantics (the extracted panel's series still reference the same live `Dataset` objects, not copies).

Note: this behavior is proven correct for `Panel3D` today by code inspection and by Graph Library's dedicated `Panel3D` tests (which exercise the same shared `clone_panel_with_shared_datasets` function — see "Graph Library" below), plus a full existing Focus/Extract test suite. There is currently no *dedicated* `Panel3D`-specific Focus/Extract GUI test scenario; that is a test-coverage gap worth closing, not a functional limitation.

### Graph Library

`plotting/graph.py`'s `Graph.panel` is typed `Panel | Panel3D`, loaded polymorphically (kind-aware) via `panel_from_dict`, so a saved `Panel3D` reloads as a `Panel3D`. `graph_library.py`'s `save_panel_as_graph`/`load_graph_into_panel` route through the same `clone_panel_with_shared_datasets` function Extract uses. This full round trip — save a `Panel3D` graph, reload the library, load it into a figure as a fresh `Panel3D` — is covered by dedicated passing tests (`tests/test_panel3d_model.py`), including grouped-curve `row_indices`, publication-styling fields (camera/grid/pane/legend/aspect/tick spacing), and `Dataset`-sharing preservation.

### Project format / persistence

`core/project_io.py` defines `PROJECT_FORMAT_VERSION = 3`. The version history is tracked in-file: v1→v2 flattened `figures`/`active_figure_index` into named workbenches; v2→v3 allowed a `GnoviFigure.panels` list to contain `Panel3D` entries alongside `Panel`. Newer `Series3D`/`Panel3D` fields added since the initial 3D milestone are plain optional keys with safe defaults in each `from_dict`, so an older save loads without a further format-version bump and without misparsing. Loading refuses (with a clear error) a project whose `project_format_version` is newer than the running app's `PROJECT_FORMAT_VERSION`. Backward-compatibility claims beyond what's described here should be limited to what `tests/test_project_io.py`/`tests/test_project_io_3d.py` actually verify.

### Analysis

`analysis/` contains only generic, domain-independent tooling: `cycles.py` (`detect_cycles`, turning-point-based repeating-sweep-cycle detection — documented as domain-independent, shaped for but not limited to cyclic-voltammetry data), `segments.py` (`contiguous_row_range`, generic row-range selection), `fitting.py` (SciPy-based curve fitting — `fit_curve`/`FitResult`, plus fit-quality/residual computation for the residual diagnostics window), `results.py` (`AnalysisResult`, the generic base and polymorphic `kind`-based persistence registry any analysis tool's result type registers into — see below for its engine-neutral provenance fields), and `panel_results.py` (`PanelResultHistory`: per-panel, multi-result analysis history with an explicit current-selection marker, owned by `core.workbench.Workbench`).

`AnalysisResult` carries `engine`/`engine_version`/`operation`/`parameters` alongside its existing dataset/series/panel provenance — added ahead of any tool that needs a non-native value so every result (present and future) records *how* it was produced the same way. Every result in this codebase today sets `engine="gnovi"` (`ENGINE_GNOVI`); no external scientific engine (GSAS-II, pyFAI, BGMN, ...) is integrated, so nothing sets it to anything else yet.

**XRD (`modules/xrd/`)** is the first domain-specific analysis, and the first consumer of that provenance mechanism. What exists: an explicit `Radiation` model (`radiation.py`, presets for Cu/Co/Mo K-alpha1 and their weighted-K-alpha averages, kept distinct — never silently assumed for a dataset), first-order Bragg's-law d-spacing (`bragg.py`), background correction (`preprocessing.py`: a low-order polynomial primitive fit to caller-specified baseline points, and arPLS via the optional `pybaselines` dependency, installed with the `xrd` extra (`pip install gnovi-plot[xrd]`) rather than core — its absence raises a clear `PybaselinesNotAvailableError` at call time, never breaks GNOVI startup), optional Savitzky-Golay smoothing (`preprocessing.py`, opt-in only, never automatic), and peak detection (`peaks.py`, a small wrapper around `scipy.signal.find_peaks` returning `XRDPeakSeed` candidates — automatic or manual, enabled/disabled, never a final measured position) feeding into `results.py`'s `XRDAnalysisResult` (`AnalysisResult` kind `"xrd_peaks"`). None of this preprocessing ever mutates a `Dataset` in place; every function returns new arrays/results, matching the "Add Fit Curve to Plot" convention `analysis.fitting.FitResult` already established for turning an analysis result into a derived `Dataset`, which XRD does not yet do itself (no XRD-specific derived-`Dataset` creation exists yet).

**Explicitly not implemented, anywhere in the app**: peak-profile fitting (Gaussian/Lorentzian/pseudo-Voigt), FWHM/integrated-area/uncertainty from a fit, Scherrer crystallite-size calculation, any XRD GUI/sidebar page (`modules/xrd/` is usable only from Python/tests today), phase identification, Rietveld refinement, quantitative phase analysis, Raman analysis, or any external scientific-engine integration (GSAS-II, pyFAI, Profex, BGMN) — all future work; see "Roadmap".

### Data model: current limitation and future direction

`Dataset` (and every current series type) is strictly tabular/column-based: a pandas DataFrame plus metadata, addressed by column name and row position. There is **no** grid/matrix (structured 2D array/mesh) data abstraction anywhere in the codebase today — no `GridDataset` or equivalent exists.

This matters architecturally because several likely future features are naturally grid-shaped rather than tabular-row-shaped: 2D heatmaps/image-style scientific plots, 3D surfaces, and 3D wireframes. A structured grid data model is expected to be a shared prerequisite for those features rather than something each reinvents independently — see "Roadmap" below. This document does not design that abstraction; it only records that current `Dataset` cannot represent gridded data and that this is a known, deliberate gap.

## Roadmap

This section records current near-term thinking, not a committed sequence or a v1.0 requirement list — items here are not all required to ship before any particular release, and priority may change as work proceeds.

Candidate near-term areas of work (not a strict order):

- **XRD Phase 1** — the first domain-specific scientific workflow. Deliberately modest in scope, aimed at what's broadly useful to materials-science researchers rather than reproducing a commercial package (e.g. HighScore). The native numerical foundation now exists (`modules/xrd/` — radiation/wavelength, Bragg's-law d-spacing, background/smoothing preprocessing, peak detection; see "Analysis" above for exactly what's implemented and what isn't). Still to come, in this same Phase 1: peak-profile fitting (Gaussian/Lorentzian/pseudo-Voigt), FWHM/area/uncertainty, Scherrer crystallite-size calculation, an Analysis-page XRD section (GUI), graph peak labels, and results export. Explicitly **not** Phase 1: reference-database phase identification, Rietveld refinement, and automated quantitative phase analysis — those remain later, more involved work.
- **Structured `GridDataset` foundation** — a minimal grid/matrix data abstraction (see "Data model" above). Expected to unblock both of the next two items rather than each building its own ad hoc grid representation.
- **2D Heatmap / image-style scientific plotting** — depends on `GridDataset`.
- **3D Surface / Wireframe** — extends `Panel3D`/the mplot3d backend with `plot_surface`/`plot_wireframe`; depends on `GridDataset` for real (non-toy) datasets.
- **Documentation / examples / reference datasets** — usage docs and reproducible example projects, useful independent of which feature above lands next.
- **Release / publication preparation** — keeping the release tag, `CITATION.cff`, and any manuscript work (e.g. a SoftwareX submission) in step with the actual shipped feature set.

## Testing and CI

- `tests/` has 88 test files (1,789 tests passing as of this writing via a full `pytest` run; treat any exact count in this file as a snapshot, not a maintained invariant — prefer running pytest's own collection for a current number) covering datasets, transforms, 2D and 3D figures/panels/series, graph library (2D and 3D), workbenches/projects, project I/O (2D and 3D and XRD), equation evaluation, cycle detection, curve fitting/fit diagnostics/residual analysis, panel-scoped analysis history (including save/reopen persistence of history and the current-selection marker), XRD radiation/Bragg d-spacing/preprocessing/peak-detection (synthetic-pattern and independently-derived-analytical validation), export (2D and 3D, including WYSIWYG/typography parity against the live canvas), and GUI behavior (responsiveness, aspect-ratio, legend fit, drawer/panel layout, sidebar navigation, theming, undo, focus/extract).
- `.github/workflows/ci.yml`: on push/PR to `main`, runs the suite on a `[ubuntu-latest, windows-latest]` matrix (Python 3.12, Qt offscreen platform). Ubuntu run collects coverage (`pytest-cov`, uploaded to Codecov, non-blocking) and installs `libegl1`/`libgl1`/`libxkbcommon0`. Windows run currently also executes `scripts/_windows_qt_diagnostics.py` as a temporary, non-blocking diagnostic step left over from the PR #2 cross-platform layout investigation.
- `.github/workflows/codeql.yml`: CodeQL static analysis for Python on push/PR to `main` plus a weekly schedule.
