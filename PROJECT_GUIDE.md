# GNOVI Studio Project Guide

This guide documents the architecture, development conventions, scientific
principles, and current direction of GNOVI Studio. It is intended to keep
development consistent as the project grows. Treat it as the authoritative
source for the project's design decisions, and follow it for all work in this
repository.

## Project overview

GNOVI Studio (package name `gnovi-plot`, version 0.9.0 "Beta", tagged `v0.9.0`
with a citable DOI in `CITATION.cff`) is a cross-platform, open-source Python
desktop application for scientific plotting and analysis. It combines
experimental data visualization, mathematical equation graphing, and
publication-quality figure creation in a single tool.

It is a working PySide6 application, launched with `python -m gnovi_plot`, with
an implemented `gui`/`data`/`plotting`/`equations`/`analysis`/`modules`/`export`/
`core` package structure and cross-platform CI plus CodeQL on every push and
pull request. 2D plotting, multi-panel figures, and a 3D foundation (scatter,
line, grouped curve families, and publication-oriented styling) are implemented.
"Current implementation status" describes the feature baseline; "Roadmap"
describes what is planned next.

## Technology stack

- **Python** (>=3.9; CI runs 3.12) — implementation language
- **PySide6** — GUI framework
- **pandas** — the canonical internal representation for all tabular
  experimental data
- **NumPy** — numerical calculations, used directly in `analysis/cycles.py`,
  `analysis/fitting.py`, `data/numeric.py`, `equations/evaluator.py`,
  `modules/xrd/`, and `modules/electrochemistry/`. A direct `pyproject.toml`
  dependency.
- **Matplotlib** — the authoritative scientific plotting and publication-quality
  rendering backend, for both 2D (line/scatter/histogram) and 3D
  (scatter/line/line+markers via `mpl_toolkits.mplot3d`). 3D
  surface/wireframe/mesh rendering is not implemented. See "2D / 3D model
  architecture" and "Rendering architecture".
- **SciPy** — curve fitting and numerical analysis: `scipy.optimize.curve_fit`
  in `analysis/fitting.py`, and `scipy.signal.find_peaks` / `savgol_filter` in
  `modules/xrd/`. A direct `pyproject.toml` dependency.
- **SymPy** — equation parsing and symbolic mathematics. Equation input always
  goes through SymPy, never raw `eval()`.
- **pybaselines** — arPLS baseline correction in `modules/xrd/preprocessing.py`
  only. Optional (`xrd` extra), not a core dependency. See "Analysis".
- **pytest** / **pytest-cov** — testing, with coverage collected in CI

## Architectural principles

These constraints guide all design and implementation decisions.

- **Modular by domain, not monolithic.** GUI, data management, plotting,
  equation handling, and scientific analysis are separate packages. There is no
  monolithic `main.py`; the composition root only wires things together.
- **The pandas DataFrame is the canonical internal representation** for tabular
  experimental data. Plotting and analysis layers consume DataFrames rather than
  raw arrays or dicts where the data is tabular. There is no grid/matrix data
  representation yet — see "Data model".
- **Matplotlib stays authoritative for publication-quality rendering.** Any
  interactive or alternate rendering path is additive, not a replacement.
- **Multiple experimental datasets can coexist and overlap on the same axes.**
  The data and plotting layers support this for both 2D (`Panel`/`PlotSeries`)
  and 3D (`Panel3D`/`Series3D`).
- **Equation curves and experimental data are meant to coexist on the same
  graph.** The plotting layer should not need rework to integrate this. Not yet
  implemented — there is no `EquationSeries`.
- **Equation plotting should support both `y = f(x)` and `z = f(x, y)`.** Keep
  the equation-handling module general enough for both. Neither standalone form
  is implemented; `equations/evaluator.py` currently evaluates a formula against
  an existing DataFrame's columns (dataset calculated columns) only.
- **No unrestricted `eval()` on user-entered equations, ever.** Parsing and
  evaluation must go through the safe SymPy-based path.
- **3D plotting is a core feature, not an add-on.** Matplotlib `mplot3d` backs a
  real 3D scatter/line/curve-family/publication-styling feature set. The
  plotting layer keeps a backend interface (`plotting/backends/base.py`) rather
  than calling Matplotlib directly throughout `gui`/`core`. See "Rendering
  architecture" for how far that abstraction currently goes.
- **Specialized scientific analysis lives in its own module(s)**, separate from
  the general plotting engine. The generic `analysis/` and `plotting/` layers
  must stay domain-agnostic. Domain code lives under `gnovi_plot/modules/`
  (`xrd/`, `electrochemistry/`).
- **Reproducibility is a major design goal.** Favor designs where a session's
  inputs and parameters are enough to reproduce its output deterministically.
- **GUI-created graphs should be exportable as equivalent Python code.** Keep the
  plotting API something a generated script could call directly; avoid GUI-only
  state that cannot be expressed as code. Not yet implemented.
- **Cross-platform: Linux, Windows, macOS.** Linux is the primary development
  platform; verify assumptions don't silently depend on Linux-only behavior.

## Development philosophy

- **Build incrementally.** Prefer small, working increments over large upfront
  builds.
- **Don't implement roadmap items early just because they're planned.** Build
  what the current step needs. Not yet started: standalone `y = f(x)` and
  `z = f(x, y)` equation plotting, `EquationSeries` on the same graph as data,
  code export (`code_export.py`), 3D surface/wireframe/mesh rendering, and a
  structured `GridDataset`/matrix data model.
- **Discuss before major architectural changes or new core dependencies.** Don't
  introduce a new core dependency or restructure module boundaries without
  raising it first.

## Scientific Python library policy

Use modern, actively maintained, stable scientific-Python libraries rather than
reimplementing established numerical or scientific algorithms.

Prefer the current stable, mutually compatible releases of:

- **NumPy** — numerical arrays and fundamental operations
- **SciPy** — optimization, curve fitting, signal processing, interpolation,
  statistics
- **pandas** — tabular scientific data handling
- **Matplotlib** — publication-quality plotting and vector/raster export
- **PySide6** / modern Qt APIs — the desktop GUI

For future scientific capabilities (XRD, Raman, UV–Vis, CV, spectroscopy, peak
analysis, smoothing, baseline correction, SEM/TEM image analysis), evaluate
established libraries before implementing algorithms from scratch.

Do not add a dependency merely because it exists. A new dependency should
provide a meaningful scientific, numerical, performance, reliability, or
maintenance advantage.

Use the latest stable *and compatible* release, not blindly the latest. Before
introducing or upgrading a major dependency, verify:

- Python-version compatibility
- Linux, Windows, and macOS compatibility
- availability of binary wheels where relevant
- compatibility with the existing GNOVI dependency stack
- license compatibility
- suitability for future application packaging

Avoid obsolete, abandoned, or unnecessarily niche libraries when a
well-maintained alternative exists.

Keep numerical and scientific logic separated from the Qt GUI so algorithms can
be tested and upgraded independently of the interface — this is how `analysis/`
and `equations/` are structured relative to `gui/`, and new analysis tools
should follow the same split.

Do not change an existing working dependency solely to satisfy this policy.
Apply it to new development, and evaluate upgrades separately.

## Current implementation status

Top-level packages under `gnovi_plot/`:

- **`gui/`** — PySide6 presentation layer. `main_window.py` is the main window:
  a multi-Workbench UI (tabs, each with its own figure, panels, and datasets), a
  grouped sidebar drawer (see "Sidebar / workspace"), and cross-platform
  layout-robustness handling (auto-collapsing drawers rather than shrinking
  below a usable minimum, normalized sidebar/control minimum widths,
  Workbench-vs-drawer width budgeting) that CI exercises on Ubuntu and Windows.
  `widgets/analysis_panel.py` is the Curve Fitting / Analysis History drawer
  page; `widgets/analysis_result_view.py` and `widgets/residual_window.py` show
  fit results and residual diagnostics. `widgets/plot3d_panel.py` is the "3D"
  sidebar page. `widgets/figure_properties_panel.py` (the "Axes" page) adapts to
  the active panel type, showing 3D-specific controls (camera, grid, panes,
  aspect, per-axis tick spacing) when the active panel is a `Panel3D`. Also
  `styles.py` (theming) and `undo_manager.py`. GUI/core coordination goes
  through Qt signals and slots; there is no `controllers/` submodule.
- **`data/`** — `dataset.py` (`Dataset`: a pandas DataFrame plus metadata —
  name, units, source, color — with calculated-column support via
  `equations/evaluator.py`); `dataset_manager.py` (`DatasetManager`, holds
  multiple coexisting datasets, no Qt dependency); `numeric.py` (numeric-validity
  helpers for 2D series (`numeric_xy`, `numeric_column`) and 3D series
  (`numeric_xyz`), plus `group_row_positions`, which backs `Series3D`'s
  "Group by" curve-family grouping); `transforms.py` (row-range and
  calculated-column transformations); and `importers/text_importer.py`
  (CSV/TSV/TXT/DAT/XY/XYE import — the only place file formats are known). All of
  this is tabular and column-based — see "Data model".
- **`plotting/`** — the backend-agnostic 2D+3D plotting engine: `figure.py`
  (`GnoviFigure`, `Panel`, `Panel3D`); `series.py` (`PlotSeries`, with
  `PlotType.LINE`/`SCATTER`/`HISTOGRAM`); `series3d.py` (`Series3D`);
  `graph.py`/`graph_library.py` (saved reusable graph definitions for `Panel`
  and `Panel3D`); `stacking.py`; `units.py`; and `backends/` (`base.py`'s
  `PlotBackend` Protocol plus `matplotlib_backend.py`, the only implementation).
  There is no `EquationSeries` and no `axes.py`; axis state lives on
  `GnoviFigure`/`Panel`/`Panel3D`.
- **`equations/`** — `parser.py` (safe SymPy parsing via `parse_expr`, never raw
  `eval`) and `evaluator.py` (`evaluate_formula`: evaluates a formula against a
  DataFrame's columns to produce a new Series). This powers dataset calculated
  columns only; grid evaluation for standalone equation plotting is not
  implemented.
- **`analysis/`** — generic, domain-independent analysis only. See "Analysis".
- **`modules/`** — specialized domain analyses, kept out of `analysis/` and
  `plotting/`. `modules/xrd/` is a native numerical foundation
  (radiation/wavelength, Bragg's-law d-spacing, background/smoothing
  preprocessing, peak detection). `modules/electrochemistry/` is the cyclic
  voltammetry foundation (unit and sign-convention helpers, sweep/cycle
  segmentation, candidate peak detection, a local-linear baseline primitive,
  peak measurement, ΔEp/E½/couple-ratio metrics, charge integration, and the
  `CVCycleAnalysisResult` persistence type). See "Analysis".
- **`export/`** — see "Export".
- **`core/`** — `app_info.py` (single source of truth for `APP_NAME`, tagline,
  `__version__`, About text — kept in sync with `pyproject.toml` via
  `tests/test_app_info.py`); `project.py` (`Project`, owns one or more
  `Workbench`es, including duplicate, remove, and extract-panel-to-workbench);
  `workbench.py` (`Workbench`: a figure, its datasets, its identity, and its own
  `analysis_results` `PanelResultHistory`); and `project_io.py` (reads and
  writes the versioned `.gnovi` ZIP project container — see "Project format /
  persistence"). There is no separate `session.py` or `config.py` layer:
  `Project`, `Workbench`, and `project_io.py` together hold the reproducible
  session state — inputs, parameters, and results — that such a layer would
  otherwise own.
- **`app.py`** — thin composition root that wires the packages together and
  launches the app via `python -m gnovi_plot`.

### 2D / 3D model architecture

`GnoviFigure.panels` is a `list[Panel | Panel3D]`: a figure can freely mix 2D
and 3D panels, and both render into the same figure (see "Rendering
architecture").

- **`Panel`** (`plotting/figure.py`) — a 2D subplot: axis labels/limits/scale,
  grid, legend, tick spacing, scientific-notation toggles, `panel_aspect_preset`,
  and a list of `PlotSeries`.
- **`Panel3D`** (`plotting/figure.py`) — the 3D sibling of `Panel`, rendered as a
  Matplotlib `mpl_toolkits.mplot3d.Axes3D`. It is not a subclass of `Panel`:
  their field sets barely overlap (a 3D panel has no `xscale`/`tick_direction`; a
  2D panel has no third axis or camera). `GnoviFigure.panels` relies on
  structural duck typing — both expose
  `.id`/`.panel_label`/`.title`/`.series`/`.add_series`/`.remove_series`/
  `.get_series`/`.invalidate_series_for_dataset`/`.to_dict` — rather than a
  shared Protocol or ABC. `Panel3D` holds a list of `Series3D`.
- **`PlotSeries`** (`plotting/series.py`) — a 2D series
  (`PlotType.LINE`/`SCATTER`/`HISTOGRAM`).
- **`Series3D`** (`plotting/series3d.py`) — the 3D sibling of `PlotSeries`, not a
  variant of it. Supported kinds are `Plot3DType.SCATTER`, `LINE`, and
  `LINE_MARKER`. There is no surface/mesh/wireframe/trisurf kind — that is out of
  scope for now (see "Roadmap"). A `Series3D` can restrict itself to an explicit,
  ordered subset of its `Dataset`'s row positions via `row_indices`. That is how
  a "Group by" grouped curve family is represented (for example, one `Series3D`
  per distinct temperature in a sweep dataset): every series in the family shares
  the same live `Dataset` and differs only in row selection. Row order is always
  the dataset's original source order — grouping never sorts by X or any other
  column, because a sweep can be genuinely non-monotonic.

### Rendering architecture

Matplotlib is the sole authoritative rendering backend.
`plotting/backends/matplotlib_backend.py` provides:

- `render_figure(axes_list, figure, ...)` — draws every panel (2D or 3D) into
  its pre-built `Axes`.
- `build_projection_aware_axes(mpl_figure, rows, cols, panels)` — constructs each
  subplot's `Axes` with the correct projection (`"3d"` for a `Panel3D`, the
  default 2D projection otherwise), so a figure with mixed 2D/3D panels lays out
  and renders in one pass. Used by both the live GUI canvas and export.
- `render_panel_3d` / `_draw_series_3d` — the 3D counterpart of the 2D per-series
  drawing, using `Axes3D.scatter` (SCATTER) and `Axes3D.plot`
  (LINE/LINE_MARKER).
- `_apply_3d_grid_style` — applies `Panel3D`'s per-panel grid styling through a
  documented Matplotlib private-API workaround (mplot3d has no public per-axis
  grid-style API). Pane styling (`pane_visible`/`pane_color`/`pane_alpha`) uses
  Matplotlib's public `Axis.pane` API.

`plotting/backends/base.py` defines `PlotBackend` as a minimal `Protocol` with a
single method, `render_figure(axes_list, figure)`. This keeps GUI and export
code from calling Matplotlib directly everywhere. It is not a general swappable
multi-backend system: `matplotlib_backend.py` is the only implementation, and
the Protocol is scoped to what that implementation needs.

### Sidebar / workspace

The left tool drawer has four visually separated sections
(`ToolDrawer.add_section` headings), in order:

- **DATA** — `Data` (dataset import and list)
- **PLOT** — `2D` and `3D` (add and configure `PlotSeries` / `Series3D`)
- **FORMAT** — `Series`, `Axes`, `Figure`, `Layout`
- **ANALYZE** — `Analysis` (curve fitting and analysis history)

`Series` and `Axes` are single destinations, not duplicated per panel type: each
adapts its controls to whichever panel (`Panel` or `Panel3D`) is active, rather
than exposing separate 2D and 3D pages.

### 3D publication controls

`Panel3D` supports the following at an architectural level (see
`plotting/figure.py` for the full field list and per-field docstrings):

- Axis labels and limits (`x_label`/`y_label`/`z_label`, `xlim`/`ylim`/`zlim`)
- Aspect (`aspect_mode`: `"auto"` or `"equal"`, mapped to `Axes3D.set_aspect`;
  not `set_box_aspect`, which is a separate unimplemented concern)
- Per-axis major and minor tick spacing
  (`major_tick_spacing_x/y/z`, `minor_tick_spacing_x/y/z`)
- Grid styling (`grid_linestyle`/`grid_linewidth`/`grid_alpha`/`grid_color`) and
  pane styling (`pane_visible`/`pane_color`/`pane_alpha`)
- Legend (`legend_visible`/`legend_loc`/`legend_ncol`/`legend_frameon`)
- Camera (`elevation`/`azimuth`, the `Axes3D.view_init` parameters; `roll` is not
  implemented)

Interactive mouse rotation of a 3D panel is transient live-view state: it never
writes back into `Panel3D.elevation`/`.azimuth`, never marks the project dirty,
and never adds an undo checkpoint — the same rule interactive 2D pan/zoom
follows for `Panel.xlim`/`.ylim`. Rotation is Matplotlib `Axes3D`'s own
click-drag navigation; GNOVI observes those events without consuming them.

The renderer re-applies stored camera state only when it has actually changed,
so an incidental redraw — a grid or label edit, a panel switch, undo/redo, an
overlay refresh — never snaps an interactively rotated view back. After a drag,
the Axes page's elevation/azimuth readout updates to match the screen while the
stored fields stay put.

Two explicit controls act on the live view: **Set Current View** commits it into
the persistent `Panel3D` fields as one undoable change, and **Reset View**
restores Matplotlib's default orientation. Both also clear the renderer's
change-tracking, so the chosen orientation is re-applied even when it equals the
one last committed — this is what makes "Reset View" take effect immediately
after a mouse rotation.

On export, headless figure export renders from the stored `Panel3D` fields on a
fresh Axes; live-figure export saves the on-screen Axes as-is.

### 2D view navigation

The Matplotlib navigation toolbar (`_CursorSafeNavigationToolbar` —
Home/Back/Forward/Pan/Zoom/Save) drives all interactive 2D view changes. GNOVI
adds one button beside "Zoom": **Zoom Out** (`MainWindow._on_zoom_out`). It
widens the active 2D panel's current X and Y view about their centers by a fixed
step (`plotting/navigation.py::ZOOM_OUT_FACTOR`, 1.25× per click). It is
incremental — never a jump to full data extent, which is Home's job.

`plotting/navigation.py` is pure view-limit math: each axis keeps its own order
(an inverted axis stays inverted) and its own scale (a `"log"` axis widens
multiplicatively in log space). Like all toolbar navigation, Zoom Out is
transient view state — it never writes `Panel.xlim`/`.ylim` or any other model
field, never marks the project dirty, and never adds a figure-content undo
checkpoint. It participates in the normal Home/Back/Forward history via
`NavigationToolbar2.push_current()`. Zoom Out is disabled for a `Panel3D`; 3D
navigation is `Axes3D`'s own mouse handling.

### Export

`export/figure_export.py` provides `export_figure` (from a `GnoviFigure` model,
headless) and `export_live_figure` (WYSIWYG-tested against the live GUI canvas),
plus panel-scoped export (`build_panel_export_figure`/`export_panel`, typed
`panel: Panel | Panel3D`) reachable from the panel context menu and the Panels
menu. Full-figure and single-panel export both work for 3D panels;
`build_projection_aware_axes` is used on the export path, so 3D projection and
camera state render the same in a file as on the live canvas.

Supported formats (`SUPPORTED_FORMATS`): raster `png` and `tiff`; vector `svg`
and `pdf`. DPI is configurable for the raster formats. Vector output is standard
Matplotlib SVG/PDF, with no separate per-element editability guarantee.

### Focus / Extract

Both work generically over `Panel | Panel3D`; neither path contains a 2D-only
type check.

- **Focus** (`MainWindow._focus_panel`) tracks the focused panel by `.id` only.
  It never clones or type-checks the panel object, so a focused `Panel3D` is the
  same live object before, during, and after "Restore Multi-Panel View" — any
  change made while focused persists.
- **Extract** (`Project.extract_panel_to_workbench`) clones the source panel via
  `plotting.graph.clone_panel_with_shared_datasets` (a `copy.deepcopy` plus a
  fresh `.id`), which works unchanged for `Panel` or `Panel3D`. For a `Panel3D`
  it deep-copies the `Series3D` list (including grouped-curve `row_indices`) and
  all publication-styling fields (camera, grid, panes, legend, aspect, tick
  spacing), and preserves 2D Extract's `Dataset`-sharing semantics: the
  extracted panel's series still reference the same live `Dataset` objects.

`Panel3D` Focus/Extract behavior is currently covered by code inspection, by
Graph Library's `Panel3D` tests (which exercise the same
`clone_panel_with_shared_datasets` function), and by the existing Focus/Extract
suite. A dedicated `Panel3D` Focus/Extract GUI test scenario is a known
coverage gap, not a functional limitation.

### Graph Library

`plotting/graph.py`'s `Graph.panel` is typed `Panel | Panel3D` and loaded
kind-aware via `panel_from_dict`, so a saved `Panel3D` reloads as a `Panel3D`.
`graph_library.py`'s `save_panel_as_graph` / `load_graph_into_panel` route
through the same `clone_panel_with_shared_datasets` function Extract uses. The
full round trip — save a `Panel3D` graph, reload the library, load it into a
figure — is covered by `tests/test_panel3d_model.py`, including grouped-curve
`row_indices`, publication-styling fields, and `Dataset`-sharing preservation.

### Project format / persistence

`core/project_io.py` defines `PROJECT_FORMAT_VERSION = 3`. The version history:

- **v1 → v2** — flattened `figures`/`active_figure_index` into named workbenches.
- **v2 → v3** — allowed a `GnoviFigure.panels` list to contain `Panel3D` entries
  alongside `Panel`.

Newer `Series3D`/`Panel3D` fields are plain optional keys with safe defaults in
each `from_dict`, so an older save loads without a further version bump and
without misparsing. Loading refuses, with a clear error, any project whose
`project_format_version` is newer than the running app's
`PROJECT_FORMAT_VERSION`. The persistence test suite
(`tests/test_project_io.py`, `tests/test_project_io_3d.py`) exercises this
round trip for both 2D and 3D project state.

## Analysis

### Generic analysis (`analysis/`)

`analysis/` contains only generic, domain-independent tooling:

- `cycles.py` — `detect_cycles`, turning-point-based detection of repeating
  sweep cycles. Domain-independent, shaped for but not limited to cyclic
  voltammetry. Its noise-tolerant, plateau-carrying direction primitive,
  `carried_step_directions`, is factored out and reused by
  `modules/electrochemistry/common.py`.
- `segments.py` — `contiguous_row_range`, generic row-range selection.
- `fitting.py` — SciPy-based curve fitting (`fit_curve`/`FitResult`), plus
  fit-quality and residual computation for the residual diagnostics window.
- `results.py` — `AnalysisResult`, the generic base and polymorphic,
  `kind`-based persistence registry that every analysis tool's result type
  registers into; also `ResidualData`, the domain-neutral `observed − fitted`
  container any result type with residual support returns (a curve fit, an XRD
  peak fit), rendered by the shared residual diagnostics window.
- `panel_results.py` — `PanelResultHistory`: per-panel, multi-result analysis
  history with an explicit current-selection marker, owned by
  `core.workbench.Workbench`.

**Provenance.** `AnalysisResult` carries `engine`, `engine_version`,
`operation`, and `parameters` alongside its dataset/series/panel provenance, so
every result records *how* it was produced. Every result in this codebase sets
`engine="gnovi"` (`ENGINE_GNOVI`); no external scientific engine (GSAS-II,
pyFAI, BGMN, ...) is integrated.

### XRD (`modules/xrd/`)

The native numerical foundation:

- `radiation.py` — an explicit `Radiation` model, with presets for Cu/Co/Mo
  K-alpha1 and their weighted K-alpha averages, kept distinct and never silently
  assumed for a dataset.
- `bragg.py` — first-order Bragg's-law d-spacing.
- `preprocessing.py` — background correction (a low-order polynomial primitive
  fit to caller-specified baseline points, and arPLS via the optional
  `pybaselines` dependency, installed with `pip install "gnovi-plot[xrd]"`),
  plus optional Savitzky–Golay smoothing (opt-in, never automatic). Without
  `pybaselines`, arPLS raises a clear `PybaselinesNotAvailableError` at call
  time and never breaks startup.
- `peaks.py` — a small wrapper around `scipy.signal.find_peaks` returning
  `XRDPeakSeed` candidates (automatic or manual, enabled or disabled). A
  candidate is never a final measured position.
- `fitting.py` — single-peak profile fitting: area-normalized Gaussian,
  Lorentzian, and pseudo-Voigt shapes plus an optional local baseline (none /
  constant / linear), fitted with `scipy.optimize.curve_fit` inside an explicit
  2θ window. See "Peak profile fitting" below.

None of this preprocessing mutates a `Dataset` in place; every function returns
new arrays or results. Detection feeds `results.py`'s `XRDAnalysisResult`
(`AnalysisResult` kind `"xrd_peaks"`); a fit feeds `fitting.py`'s
`XRDPeakFitResult` (kind `"xrd_peak_fit"`).

**Peak profile fitting.** `modules/xrd/fitting.py` is the XRD-3A numerical
foundation — pure NumPy/SciPy, no Qt. `fit_xrd_peak(...)` fits ONE symmetric
profile plus a local baseline and returns an `XRDPeakFitResult`.

- **Canonical parameterization.** The profiles are area-normalized, so the
  fitted amplitude is the integrated peak area `A` directly (there is no
  independent height parameter). The common parameters are `A`, center `x0`,
  and FWHM `Γ`. Pseudo-Voigt is `(1 − η)·Gaussian + η·Lorentzian` with the two
  components sharing one center and one FWHM: **`η` is the Lorentzian fraction —
  `η = 0` is a pure Gaussian, `η = 1` a pure Lorentzian.** The exact convention
  is recorded verbatim in `parameters["profile_convention"]`. Peak height is
  derived from `A`, `Γ`, and the model.
- **Analytical area.** `A` is the full analytical, infinite-domain integrated
  intensity of the peak component above the fitted local baseline. Because the
  profiles are area-normalized, a finite fit window does not directly contain
  all of `A` — the model constrains the wings through the profile shape.
  Negligible for a Gaussian; not for a Lorentzian (a ±4·FWHM window around a
  pure Lorentzian encloses ~92% of the reported `A`). A reported `A` is
  therefore sensitive to the profile model, the fit window, and the baseline
  model; quantitative area comparisons should use a consistent fitting
  procedure and profile convention where possible, and the model choice should
  be reported alongside the value.
- **Local baseline** (`none` / `constant` / `linear`, default `linear`) is a fit
  term, conceptually separate from `preprocessing.py`'s whole-pattern
  background. The reported `A` never includes a baseline contribution. A
  `none`-baseline fit warns when the data clearly do not return near zero at
  the window edges (for a positive or a negative offset alike).
- **FWHM** is the fitted `Γ`, reported in degrees 2θ (`fwhm_units =
  "degrees_2theta"`) — never `find_peaks`' detection width, and never silently
  converted to radians.
- **Fit window** is always explicit. `propose_fit_window(...)` derives an
  initial `center ± 4·FWHM` window from a seed's detection width and the local x
  spacing, then clips it to the data range and to the midpoints toward
  neighbouring detected peaks. A fit needs `max(2·P, 10)` finite points for `P`
  free parameters — a numerical minimum, not proof the fit is scientifically
  sound.
- **Standard errors** are covariance-derived (`sqrt(diag(pcov))`), reported as
  fit standard errors — not measurement uncertainties or confidence intervals.
  A parameter's standard error is `None` when the whole covariance is singular
  or non-finite (then all are `None`), or when that parameter sits at a fit
  bound. Extreme parameter correlation (a dimensionless
  correlation-matrix check, not `cond(pcov)`) and low degrees of freedom add a
  caution but do **not** null otherwise-finite standard errors. The area
  standard error is the covariance value for `A` directly; the derived-height
  standard error is propagated through the `(A, Γ[, η])` sub-covariance with
  cross-terms. When `η` converges to 0 or 1 the fit stays valid — only `η`'s
  standard error is withheld, with a neutral "converged to the Gaussian /
  Lorentzian endpoint" note.
- **d-spacing** comes from the fitted centre via `bragg.d_spacing` when a
  `Radiation` context is supplied, with the centre standard error propagated
  analytically through Bragg's law; without radiation, d-spacing is simply
  absent rather than computed from an assumed wavelength.
- **Diagnostics:** RSS, RMSE, R² (a descriptive fit statistic only — not a
  profile-model selection criterion), point count, parameter count, degrees of
  freedom, and convergence state. No reduced χ² — there is no justified
  per-point measurement variance. SciPy-internal solver strings are not stored
  in the reproducibility parameter dict.
- **Overlap** is flagged only from explicitly supplied neighbouring detected-peak
  positions (a conservative, hedged "may not represent an isolated reflection").
  There is no residual-based automatic overlap detector.
- The fitted curve is regenerated from the model, parameters, baseline, and
  window (`evaluate_total` / `sample_fit_curve`) — dense arrays are never
  stored on the result.

`XRDPeakFitResult` fits ONE peak component. Overlapping-peak deconvolution, if
built, gets its own result kind rather than a components list here.

**XRD Peak Analysis (GUI).** `gui/widgets/xrd_analysis_section.py`'s
`XRDAnalysisSection` is a `CollapsibleSection` on the Analysis page, selected via
`AnalysisPanel`'s "Analysis Tool" combo — one Analysis destination, several
workflows sharing one Analysis History. It operates on a 2D `PlotSeries` in the
active panel (disabled, with an explanation, when the active panel is a
`Panel3D`). It supports:

- source-series and radiation selection
- background/smoothing preview
- an explicit detection-input chain (Raw / Background-corrected / Smoothed raw /
  Smoothed background-corrected — only options actually available are offered)
- `find_peaks`-based detection with a peak table (seed 2θ, observed intensity,
  prominence, d-spacing, origin, enabled). There are no fitted-center/FWHM/area
  columns — the profile-fitting workspace that would surface those is not built
  yet (see "Peak profile fitting" above for the numerical layer, and "Roadmap").
- manual peak add/remove/enable-disable
- CSV peak-table export

Background and smoothing previews are transient — never registered as a
`Dataset`/`PlotSeries` until explicitly accepted via "Add Corrected/Smoothed
Curve to Plot", which follows the same derived-`Dataset` pattern as
`FitResult`'s "Add Fit Curve to Plot". Peak markers and labels on the graph are
a live-only overlay, reconstructed each redraw from the current
`XRDAnalysisResult`; they are not part of the saved figure and do not appear in
export. "Find Peaks" always creates a new Analysis History entry; manual peak
edits mutate the current entry in place. `.xy`/`.xye` pattern files are handled
by the existing text importer alongside `.csv`/`.txt`/`.tsv`/`.dat` — no
dedicated diffraction parser was added.

**Implemented numerically, no GUI yet:** single-peak profile fitting
(`modules/xrd/fitting.py`, above) — the researcher-facing peak-fitting workspace
that would drive it, show the fitted parameters in the Results tab, overlay the
fitted curve, and offer "add fitted curve to plot" is not built.

**Not implemented, anywhere in the app:** multi-peak / overlapping-peak
deconvolution, Scherrer crystallite-size calculation, instrumental broadening
correction, Williamson–Hall analysis, Kα1/Kα2 doublet modelling, asymmetric
peak profiles, Poisson-weighted fitting / reduced χ², phase identification,
Rietveld refinement, quantitative phase analysis, Raman analysis, and external
scientific-engine integration (GSAS-II, pyFAI, Profex, BGMN). See "Roadmap".

### Electrochemistry / Cyclic Voltammetry (`modules/electrochemistry/`)

The package is named for the family — LSV, chronoamperometry, GCD, and EIS are
intended later members — rather than `cv/`. There is deliberately no
`ElectrochemicalExperiment`/`ElectrochemicalResult` base class and no plugin
system: one technique does not justify an abstraction layer.

The foundation is pure NumPy/SciPy (pandas only where the `Dataset` layer
already uses it), with no Qt and no Matplotlib.

**`common.py` — reusable primitives:**

- Unit helpers (V/mV; A/mA/µA/nA; V/s ↔ mV/s; C/mC). Canonical internal units
  are V, A, V/s, C. This is not a units framework.
- `CurrentSignConvention` (`ANODIC_POSITIVE` default / `CATHODIC_POSITIVE`) — an
  interpretation layer only. The imported `Dataset`/array current is never
  modified or flipped, and the chosen convention is stored in every result.
- `ElectrodeContext` — an all-optional dataclass
  (`area_cm2`/`n`/`concentration_mol_cm3`/`temperature_k`/electrode
  identities/electrolyte). None is required for basic peak analysis, none is
  silently defaulted, and a supplied non-positive numeric value raises.
- `segment_sweeps` — deterministic rising/falling segmentation, reusing
  `analysis.cycles.carried_step_directions`. Tolerates noise, plateaus, and an
  arbitrary initial direction. A monotonic LSV-like trace is one segment, not an
  error. Incomplete first/last sweeps are allowed and never assumed anodic or
  cathodic.
- `integrate_current` — `Q = ∫ I dt` via `scipy.integrate.trapezoid`:
  time-domain when a time array is given, else `Q = (1/|v|) ∫ I d|E−E₀|` for a
  strictly monotonic constant-rate sweep. Rejects a zero or negative scan rate
  and a non-monotonic potential (never integrates across a reversal). Sign is
  preserved; inputs are never mutated.

**`cv.py` — CV-specific:**

- `pair_cycles` — 2-by-2 sweep pairing on top of `segment_sweeps`. A truncated
  leading sweep is emitted alone as an incomplete cycle so it cannot misalign
  later cycles. Incomplete cycles are flagged `complete=False`, never dropped or
  mispaired.
- `CVPeakSeed` — a candidate model mirroring `XRDPeakSeed`'s pattern (stable
  `id`, automatic/manual `origin`, `enabled` soft-exclude), but not a subclass
  and with no shared superclass. `process` (anodic/cathodic/unassigned) is
  independent of `sweep` (rising/falling).
- `detect_cv_peaks` — `scipy.signal.find_peaks` run per sweep, never on a
  concatenated cycle. Sign-convention-aware (oxidative candidates in
  `+I·oxidative_sign`, reductive in `−I·oxidative_sign`). `process` is assigned
  from the current direction, not the sweep direction. Small parameter surface:
  `prominence` primary, `distance`/`width` optional.
- `local_linear_baseline` — a straight line through the recorded current at only
  the caller-specified anchor ranges, evaluable across any region. Returns a new
  representation, never mutates the source, and is never an automatic or opaque
  background (no LOWESS/spline/arPLS).
- `measure_peak` — **detection is not measurement.** Locates the extremum on the
  *unsmoothed* signal, on the baseline-corrected curve when a baseline is given.
  `Epa`/`Epc` is the recorded potential there, with no sub-sample fitting.
  Returns `i_peak_raw_a` always, and `i_peak_corrected_a` only when a baseline
  was supplied. Without a baseline the value is explicitly a raw extremum, never
  presented as a baseline-corrected `Ipa`/`Ipc`.
- `couple_metrics` — `ΔEp = |Epa − Epc|` and `E½ = (Epa + Epc)/2` in canonical
  volts. `E½` is documented as the **midpoint potential** — an estimate of the
  formal potential only under reversibility, never called `E°'` here.
  Peak-current ratios are the explicitly labelled `|Ipa|/|Ipc|` and its
  reciprocal `|Ipc|/|Ipa|`, never an anonymous forward/reverse ratio, with a
  `ratio_basis` recording baseline-corrected vs raw-extremum.

**`results.py`:** `CVCycleAnalysisResult` (`AnalysisResult` kind `"cv_peaks"`,
`@register_result_kind`) carries the sign convention, the cycle
index/confidence/completeness, the sweep segmentation, the measured
`CVPeakResult`s (each with `CVBaselineInfo`), the couple metrics, and the
`peak_id`s of the two candidates forming the couple. It has a bounded
`details()` and an unbounded `detail_table()` (one row per peak, rendered in the
wide bottom Results tab). `assign_couple` / `couple_from_peak_results` pick each
process's couple member deterministically from the *enabled* candidates
(`unassigned` and disabled peaks are never members): the largest-`prominence`
candidate when any carries one (ties broken by earliest position), otherwise the
candidate added last. Raw current magnitude is never used for ranking — it is
dominated by the charging background. Registered in `core/project_io.py` with no
`PROJECT_FORMAT_VERSION` bump (additive within the already-polymorphic
`analysis_results` structure; the couple-member id fields are defaulted, so
older `from_dict` calls are unaffected).

**Cyclic Voltammetry Analysis (GUI).** `gui/widgets/cv_analysis_section.py`'s
`CVAnalysisSection` is a `CollapsibleSection` on the Analysis page, selected via
`AnalysisPanel`'s "Analysis Tool" combo (`["Curve Fitting", "XRD Peak
Analysis", "Cyclic Voltammetry"]`) — one Analysis destination, three workflows
sharing one Analysis History. It operates on a 2D `PlotSeries` in the active
panel (disabled, with an explanation, when the active panel is a `Panel3D`).

Sidebar sections:

- **Source** — series, read-only potential/current columns, current sign
  convention, optional scan rate, and a collapsed all-optional "Physical
  context" (`ElectrodeContext`).
- **Cycle Selection** — source precedence: metadata column, auto-detect, or
  manual row ranges. A status line reports "N sweeps → M cycle(s), K complete",
  an ambiguity warning, or a graceful single-sweep message for monotonic data.
  The picker defaults to the last complete cycle and shows a confidence chip.
- **Sweep Selection** — both / rising / falling, with a rows-and-E-span readout.
  Never labelled "forward = anodic".
- **Peak Detection** — a free-text prominence field with a data-scaled default
  (`cv.default_prominence`: 3.5·MAD with a 2%-of-range floor, per sweep,
  provisional pending human validation); a minimum separation in mV converted
  via `cv.mv_to_sample_distance`; an optional width. **Find Peaks** produces a
  fresh `CVCycleAnalysisResult` and a new History entry. Manual candidate Add
  (click graph, wrong-panel-guarded like XRD), Remove, Enable/Disable, and Set
  Process all act on the Results-tab table's row selection.

The per-candidate table and the anodic/cathodic couple summary (ΔEp, E½
midpoint, `|Ipa|/|Ipc|` and its reciprocal with an explicit "raw extremum" or
"baseline-corrected" basis, and which peak numbers form the couple) render in
the bottom Results tab (`AnalysisResultView`, from `detail_table()`/
`details()`), never the narrow sidebar.

Graph aids — a selected-cycle line split by sweep tint, a switching-potential
line, open-circle candidate markers, and filled process-shaped enabled markers —
are a live-only overlay via `PlotCanvas.set_cv_overlay` (payload from
`CVAnalysisSection.overlay_payload`). The overlay is reconstructed each render
from the current result and gated on the active panel, the selected source
series, and the selected cycle, so a result computed on series A never floats
its markers over series B or a different cycle. It is stripped from every
figure-to-disk path — the Export dialog and Matplotlib's own toolbar Save —
through the shared `PlotCanvas.clear_gui_only_overlays` boundary (reference
cursor, XRD overlay, CV overlay); headless `export_figure` builds a fresh figure
and never sees them.

Editing behavior:

- A sign-convention change, manual candidate edit, enable/disable, process
  reassignment, or Remove edits the current result in place: the project is
  marked dirty and the result redisplayed, with no new History entry and no
  figure-undo checkpoint.
- Changing the cycle or sweep re-arms detection but adds no History entry until
  the next Find Peaks.
- Changing the source series also detaches the working result and clears the
  overlay; the persisted History entry is untouched and stays selectable.
- Manual Add Peak snaps to the nearest point on the selected cycle's trace in
  normalised (potential, current) space, so proximity to the curve chooses the
  sweep; a click not on that cycle's curve is rejected with a status message.
- Selecting a CV History entry restores its source series, sign convention,
  cycle, sweep, and detection settings without rerunning detection.

Terminology is deliberate: "candidate" until enabled, then "enabled candidate"
and "couple member" — never "accepted peak"; "E½ (midpoint potential)", never
an unqualified "formal potential" or "E°'"; raw current is never labelled
"Ipa"/"Ipc". `AnalysisResultView`'s button is "Copy Summary" (it uses the
generic base `report_text()`). `gui/widgets/dataset_panel.py` has a "CV
(Potential vs Current)" plot preset (axis labels "Potential (V)" / "Current
(A)", forces a line plot, keeps "plot by cycles" available) with a small
potential/current column-name matcher (`_match_cv_columns`; CHI/BioLogic/Autolab
and plain names; silent fallback to X = column 0 / Y = column 1) applied only
when that preset is chosen.

**Not implemented for Cyclic Voltammetry yet** (see "Roadmap" for phasing):
interactive baseline anchoring (the `local_linear_baseline` primitive exists but
has no UI; current baselines are "None (raw extremum)" only), the
raw↔corrected current workflow beyond what `detail_table()` renders,
derived-curve actions ("Add baseline-corrected trace" / "Add selected cycle"),
CSV peak-table export, a smoothing workflow, the Nicholson switching-potential
peak-current ratio (deferred, not approximated — it needs interactive
switching-potential context), polynomial/arPLS/spline baselines,
multi-scan-rate aggregation (`scan_rate.py` / `CVScanRateSeriesResult`),
Randles–Ševčík (regression or diffusion coefficient), reversibility
classification, LSV/GCD/EIS, vendor import parsers
(CHI/BioLogic/Gamry/PalmSens/VersaStudio) and ixdat integration,
current-density normalization, and reference-electrode potential conversion.

### Data model: current limitation and future direction

`Dataset` and every current series type are strictly tabular and column-based: a
pandas DataFrame plus metadata, addressed by column name and row position. There
is no grid/matrix (structured 2D array/mesh) data abstraction anywhere in the
codebase — no `GridDataset` or equivalent.

This matters architecturally because several likely future features are
naturally grid-shaped: 2D heatmaps and image-style plots, 3D surfaces, and 3D
wireframes. A structured grid data model is expected to be a shared prerequisite
for those, rather than something each feature reinvents. This document does not
design that abstraction; it records that current `Dataset` cannot represent
gridded data and that this is a known, deliberate gap.

## Roadmap

This section records current near-term thinking, not a committed sequence or a
v1.0 requirement list. Priorities may change as work proceeds.

- **XRD Phase 1** — the first domain-specific workflow, deliberately modest in
  scope: broadly useful analysis for materials-science researchers, not a
  reimplementation of a full commercial diffraction-analysis package. The
  numerical foundation (`modules/xrd/`) and the Analysis-page workspace
  (`XRDAnalysisSection`) exist: radiation selection, background/smoothing
  preview, peak detection with a peak table, manual peak editing, live overlays,
  derived corrected/smoothed curves, and CSV export. Single-peak profile fitting
  (Gaussian/Lorentzian/pseudo-Voigt, area/FWHM/height/η, local baseline, fit
  standard errors, propagated d-spacing error) is implemented as a numerical
  layer (`modules/xrd/fitting.py`). Still to come in Phase 1: the
  researcher-facing peak-fitting workspace over that layer, then Scherrer
  crystallite-size calculation (which also needs instrumental broadening
  correction). Not Phase 1: multi-peak deconvolution, reference-database phase
  identification, Rietveld refinement, and automated quantitative phase
  analysis.
- **Electrochemistry / Cyclic Voltammetry** — the second domain-specific family.
  CV-1 (the `modules/electrochemistry/` numerical foundation) and CV-2A (the
  "Cyclic Voltammetry" Analysis workspace: source/sign-convention/cycle/sweep
  selection, candidate detection with manual curation, the auto anodic/cathodic
  couple summary on the raw-extremum basis, transient overlays, and
  History/persistence wiring) are done. Still to come: **CV-2B** (interactive
  per-peak baseline anchoring via graph-click anchors on the existing
  `local_linear_baseline` primitive, the baseline-corrected current basis in the
  couple summary, derived-curve actions, and CSV export), then **CV-3**
  (multi-scan-rate grouping, Ip-vs-√v regression, Randles–Ševčík, charge
  integration UI, reversibility diagnostics). LSV, GCD, and EIS are later
  members of the same family, each in its own module file.
- **Structured `GridDataset` foundation** — a minimal grid/matrix data
  abstraction (see "Data model"). Expected to unblock the next two items.
- **2D heatmap / image-style plotting** — depends on `GridDataset`.
- **3D surface / wireframe** — extends `Panel3D` and the mplot3d backend with
  `plot_surface`/`plot_wireframe`; depends on `GridDataset` for real datasets.
- **Documentation, examples, reference datasets** — usage docs and reproducible
  example projects, useful independent of which feature lands next.
- **Release / publication preparation** — keeping the release tag,
  `CITATION.cff`, and any manuscript work in step with the shipped feature set.

## Testing and CI

`tests/` is the authoritative record of coverage, and `pytest` and CI report the
current test count. The suite covers:

- datasets, transforms, and numeric-validity helpers
- 2D and 3D figures, panels, and series
- Graph Library (2D and 3D) and project I/O (2D, 3D, and both XRD result kinds),
  including save/reopen persistence of analysis history and the current-selection
  marker
- workbenches and projects
- equation evaluation and cycle detection
- curve fitting, fit diagnostics, and residual analysis
- XRD radiation, Bragg d-spacing, preprocessing, and peak detection, validated
  against synthetic patterns and independently derived analytical values
- XRD single-peak profile fitting (`modules/xrd/fitting.py`) — profile
  normalization and FWHM checked against closed-form values and `scipy.integrate`
  over the real line; parameter recovery (area, centre, FWHM, η, baseline) from
  deterministic synthetic Gaussian / Lorentzian / pseudo-Voigt data with and
  without seeded noise; fit-window proposal, clipping, and ascending/descending
  2θ equivalence; irregular x spacing; the failure and edge cases
  (reversed/empty window, too few points, flat or negative-only signal,
  non-finite data, parameter at a bound, low degrees of freedom, non-convergence,
  singular covariance); the pseudo-Voigt Gaussian/Lorentzian endpoints as valid
  fits; d-spacing error propagation; deterministic curve regeneration; result
  serialization round trip; a realistic multi-peak pattern fitted one isolated
  peak at a time through a local window; a characterization that a wrong profile
  model can bias the fitted area while R² stays high; and an approximate
  standard-error calibration check
- the `modules/electrochemistry/` CV numerical foundation — unit conversion,
  sign convention, `ElectrodeContext`, sweep/cycle segmentation, candidate
  detection, local-linear baseline, peak measurement, couple metrics, charge
  integration, couple assignment, and `CVCycleAnalysisResult`
  serialization/history/persistence — using synthetic deterministic fixtures
  with independently derived expected values plus a non-golden real-ferricyanide
  sanity check
- the CV analysis GUI workspace (`test_cv_analysis_workspace_gui.py`) — tool
  selector, `Panel3D` guard, sign-convention reinterpretation, cycle-source
  precedence and the last-complete default, sweep restriction, one History entry
  per Find Peaks, the manual-add wrong-panel guard, Results-driven
  enable/disable/set-process/remove, restoring a History entry without rerunning,
  overlay gating, and project dirty / no-source-mutation / save-reopen behavior —
  plus the "CV" plot preset and column matcher
- export (2D and 3D), including WYSIWYG and typography parity against the live
  canvas
- GUI behavior — responsiveness, aspect ratio, legend fit, drawer/panel layout,
  sidebar navigation, theming, undo, and focus/extract

Continuous integration:

- `.github/workflows/ci.yml` — on push/PR to `main`, runs the full suite on a
  `[ubuntu-latest, windows-latest]` matrix (Python 3.12, Qt offscreen platform).
  The Ubuntu run collects coverage (`pytest-cov`, uploaded to Codecov,
  non-blocking) and installs `libegl1`/`libgl1`/`libxkbcommon0`. The Windows run
  also executes `scripts/_windows_qt_diagnostics.py` as a temporary, non-blocking
  Qt/geometry diagnostic step.
- `.github/workflows/codeql.yml` — CodeQL static analysis for Python on push/PR
  to `main` plus a weekly schedule.
