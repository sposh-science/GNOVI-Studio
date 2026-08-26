from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

# The `engine` value every result GNOVI computes itself uses -- see
# `AnalysisResult.engine`'s own docstring. Every result in this codebase
# sets `engine=ENGINE_GNOVI` today; a future external-engine integration
# (none exists yet) would use its own short identifier instead.
ENGINE_GNOVI = "gnovi"


@dataclass
class AnalysisResult:
    """Base for the output of any analysis tool run against a Dataset's
    data -- curve fitting today (see `analysis.fitting.FitResult`), and
    later tools (statistics, peak analysis, FFT, smoothing, ...) that don't
    exist yet.

    Provenance is stable-id-first: `source_dataset_id` / `source_series_id`
    are `Dataset.id` / `PlotSeries.id`, never a filename or display name,
    so the link survives a later rename. `source_dataset_name` /
    `source_series_label` are a *descriptive snapshot* of what the
    researcher actually fitted, captured at analysis time -- purely
    informational, resolved fresh by whatever produced the result (never
    reconstructed from the id), and always safe to be `None` (older
    results, or a dataset/series with no obvious name yet). Display code
    (see `gui.widgets.analysis_result_view.AnalysisResultView`) prefers
    this stored snapshot, falls back to live id resolution, and only shows
    the raw id itself as a last resort.

    `source_series_id`/`source_series_label` are optional: a result can be
    computed directly against a dataset's column pair with no live plotted
    series involved.

    `row_range` records which rows of the source were actually used
    (positional, end-exclusive, `DataFrame.iloc`-style), mirroring
    `PlotSeries.row_range`; `None` means the whole dataset.

    `source_panel_id` is `Panel.id` (see `plotting.figure.Panel`) -- the
    panel this result was produced against, stable across renumbering/
    layout changes/undo-redo. Used by `analysis.panel_results.
    PanelResultHistory` (owned by `core.workbench.Workbench`) to restore
    the right result when the active panel/Workbench changes -- never a
    panel index or "Panel N" display text, both of which are positional
    and can silently point at the wrong panel after a layout edit. `None`
    for a result not associated with any panel (defensive/future case;
    such a result is simply never added to a `PanelResultHistory`).

    `result_id` is this result's *own* stable identity -- unlike every
    other id field here (which all point at something else), this is
    what lets other state point *at* this result and still find it after
    a save/reload round trip, when Python object identity is gone. Today
    that's the derived fit Dataset's `metadata["result_id"]` (see
    `gui.widgets.analysis_panel._on_add_fit_curve_clicked`), which is how
    "Remove Fit Curve from Plot" finds exactly the `PlotSeries` a given
    `FitResult` produced -- deliberately never a name/label match, since
    labels are user-editable and non-unique. Named `result_id` rather
    than the bare `id` every other domain object (`Dataset`/`PlotSeries`/
    `Panel`/`Workbench`/`Graph`) uses for its own identity, specifically
    because this one is primarily consumed as a cross-reference embedded
    inside a *different* object's metadata, where a bare `"id"` would be
    ambiguous against that object's own.

    Both `source_panel_id` and `result_id` are required (not defaulted)
    like every other provenance field here -- `AnalysisResult` has no
    defaulted fields, since `FitResult` (and any later subclass) adds its
    own required fields after these, and a dataclass can't mix the two
    (see PEP 557's default-argument-ordering rule); every field here is
    always passed explicitly by its caller regardless. `fit_curve()`
    generates a fresh `result_id` for every new fit; `FitResult.from_dict`
    restores the one that was persisted, so identity survives reload.

    `engine`/`engine_version`/`operation`/`parameters` are engine-neutral
    provenance, added ahead of any tool that actually needs them (see
    `modules.xrd.results.XRDAnalysisResult`, the first consumer) so every
    `AnalysisResult` -- present and future -- records the same four things
    about *how* it was produced, not just *from what data*. `engine` is a
    short, stable identifier: `"gnovi"` (`ENGINE_GNOVI`, below) for every
    result this app computes itself (all of `FitResult` today), reserved
    for a future non-native value (`"gsas2"`, `"pyfai"`, `"bgmn"`, ...) once
    GNOVI actually calls out to an external scientific engine -- which it
    does not yet; nothing in this codebase sets `engine` to anything but
    `ENGINE_GNOVI` today. `engine_version` is that engine's own version
    string (GNOVI's own `__version__` for `ENGINE_GNOVI`), `None` only for
    a result persisted before this field existed. `operation` is a short,
    engine-scoped label for which computation actually ran (e.g.
    `"curve_fit"`, `"xrd_peak_detection"`) -- coarser than `kind` (which
    identifies the *result type* for polymorphic reconstruction) and not a
    substitute for it. `parameters` is a small, JSON-safe dict of the
    inputs that operation actually ran with; values should be plain
    `str`/`int`/`float`/`bool`/`None`/nested `dict`/`list` (never a NumPy
    scalar -- callers should `float(...)`/`int(...)` first) so two results
    computed from identical parameters always compare equal after a
    save/reload round trip. Deliberately no timestamp: nothing here
    depends on wall-clock time for its scientific identity, and adding one
    would only make otherwise-deterministic results/tests non-reproducible
    for no present benefit.

    These four are required (no default), for the same reason every other
    field on this class is: a dataclass can't place a defaulted field
    ahead of a subclass's own required fields (`FitResult.model` and
    friends), so making them optional-with-a-default here would force
    *every* subclass field after them to also default -- backward
    compatibility for an *old saved project* missing these keys belongs in
    each subclass's own `from_dict` (`data.get("engine", ENGINE_GNOVI)`
    etc.), not in the dataclass field declaration; see `FitResult.
    from_dict` for exactly that pattern. No `PROJECT_FORMAT_VERSION` bump
    was needed to add these: an older app's `from_dict` simply never
    reads the new keys (extra dict keys are harmless), and this app's own
    `from_dict` methods default them for a file saved before they existed
    -- the same one-directional-compatibility precedent `Panel3D`'s own
    fields-added-since-v3 already established.

    Pure data plus a small display contract (`summary`/`details`/
    `provenance_details`/`report_text`) any results view can render
    without knowing the concrete subclass. No Qt, no plotting, no project
    I/O.
    """

    source_dataset_id: str
    source_dataset_name: str | None
    source_series_id: str | None
    source_series_label: str | None
    x_column: str
    y_column: str
    row_range: tuple[int, int] | None
    source_panel_id: str | None
    result_id: str
    engine: str
    engine_version: str | None
    operation: str
    parameters: dict

    kind: ClassVar[str] = "analysis"

    def summary(self) -> str:
        """One-line description for a compact header/list entry."""
        raise NotImplementedError

    def details(self) -> list[tuple[str, str]]:
        """Label/value rows for the main results display -- the
        model/measurement itself (e.g. fitted parameters, goodness of
        fit). Provenance (ids, columns, row range) lives in
        `provenance_details()` instead, so a results view can show this
        list prominently and that one collapsed/secondary."""
        raise NotImplementedError

    def detail_table(self) -> tuple[list[str], list[list[str]]] | None:
        """An optional wide, row-per-record tabular view of this result --
        `(column headers, row values)` -- for a results view able to render
        a scrolling table below the bounded `details()` summary. `None`
        (the default) means this result type has no such table and
        `details()` alone is shown.

        Unlike `details()`, this list MAY be long (one row per XRD peak
        candidate, potentially thousands -- see `modules.xrd.results.
        XRDAnalysisResult`). The rendering view is responsible for bounding
        its own height and scrolling internally so row count never drives
        parent layout (see `gui.widgets.analysis_result_view.
        AnalysisResultView` / `gui.widgets.bottom_panel.BottomPanel`)."""
        return None

    def detail_table_title(self) -> str:
        """Heading shown above `detail_table()` when it has content -- only
        consulted when `detail_table()` returns non-`None`."""
        return "Details"

    def provenance_details(self) -> list[tuple[str, str]]:
        """Label/value rows for raw structural provenance -- always ids,
        never names (see the class docstring / `AnalysisResultView` for
        name resolution). One correct default implementation every
        subclass shares, since the fields it reads are already on this
        base class. Includes engine/operation last -- for `ENGINE_GNOVI`
        this is simply confirming a native result, but once a non-native
        `engine` value exists this is exactly what makes it "obvious [...]
        whether a result came from GNOVI native [or] an external engine"
        in a saved project, directly from this one shared method."""
        rows = [("Source dataset ID", self.source_dataset_id)]
        if self.source_series_id is not None:
            rows.append(("Source series ID", self.source_series_id))
        rows.append(("Columns", f"{self.x_column} → {self.y_column}"))
        if self.row_range is not None:
            rows.append(("Row range", f"{self.row_range[0]}–{self.row_range[1]}"))
        engine_text = f"{self.engine} {self.engine_version}" if self.engine_version else self.engine
        rows.append(("Engine", engine_text))
        rows.append(("Operation", self.operation))
        return rows

    def supports_residuals(self) -> bool:
        """Whether `compute_residuals` is implemented for this result type.
        `False` by default -- "observed minus predicted, per point" is
        meaningful for curve fitting (see `FitResult`) but not necessarily
        for every future analysis tool (e.g. a peak table)."""
        return False

    def compute_residuals(self, x, y):
        """Per-point residuals for `(x, y)`. Only called after
        `supports_residuals()` returns `True`; the base implementation is
        never actually invoked. Return type is subclass-specific (see
        `analysis.fitting.FitResult.compute_residuals` ->
        `analysis.fitting.ResidualData`) -- left unannotated here to avoid
        a dependency back onto a subclass's own module."""
        raise NotImplementedError

    def residual_window_subtitle(self) -> str:
        """Short '<model> fit'-style label for a residual diagnostic
        window's title (see `gui.widgets.residual_window.ResidualWindow`).
        Only meaningful when `supports_residuals()` is `True`; the default
        (`self.kind`) is never actually shown today since no subclass
        without residual support opens a residual window."""
        return self.kind

    def report_text(self, *, dataset_name: str | None = None, series_label: str | None = None) -> str:
        """Concise, human-readable text for clipboard/export -- names
        only, never raw ids (see `provenance_details()` for those).
        Generic: works for any subclass via `summary()`/`details()` alone,
        so this never needs a subclass override."""
        lines = [self.summary()]
        name_lines = []
        if dataset_name:
            name_lines.append(f"Dataset: {dataset_name}")
        if series_label:
            name_lines.append(f"Series: {series_label}")
        if name_lines:
            lines.append("")
            lines.extend(name_lines)
        lines.append("")
        lines.extend(f"{label}: {value}" for label, value in self.details())
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-safe representation -- base provenance fields plus `kind`.
        Subclasses extend this (see `FitResult.to_dict`) so the combined
        dict can be embedded directly in a derived `Dataset.metadata` with
        no `project.json` schema change: `metadata` is already a free-form
        dict that's serialized whole.
        """
        return {
            "kind": self.kind,
            "source_dataset_id": self.source_dataset_id,
            "source_dataset_name": self.source_dataset_name,
            "source_series_id": self.source_series_id,
            "source_series_label": self.source_series_label,
            "x_column": self.x_column,
            "y_column": self.y_column,
            "row_range": list(self.row_range) if self.row_range is not None else None,
            "source_panel_id": self.source_panel_id,
            "result_id": self.result_id,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "operation": self.operation,
            "parameters": dict(self.parameters),
        }


# --- Polymorphic (kind -> subclass) reconstruction --------------------------
#
# `AnalysisResult` stays fully generic (never imports `FitResult` or any
# other concrete subclass); a concrete subclass registers itself here
# instead (see `analysis.fitting.FitResult`'s `@register_result_kind`).
# `result_from_dict` is the one place that dispatches on `kind` -- callers
# (see `analysis.panel_results.PanelResultHistory.from_dict`) never
# hard-code which concrete types exist. Adding a later analysis tool
# (peak analysis, statistics, FFT, smoothing, ...) means giving it its own
# `AnalysisResult` subclass with `@register_result_kind` and its own
# `from_dict`, nothing here changes.
_RESULT_KIND_REGISTRY: dict[str, type["AnalysisResult"]] = {}


def register_result_kind(result_cls: type["AnalysisResult"]) -> type["AnalysisResult"]:
    """Class decorator: register a concrete `AnalysisResult` subclass's
    `kind` for polymorphic `result_from_dict` dispatch."""
    _RESULT_KIND_REGISTRY[result_cls.kind] = result_cls
    return result_cls


def result_from_dict(data: dict) -> "AnalysisResult":
    """Reconstruct the correct concrete `AnalysisResult` subclass from
    `data["kind"]` (see `to_dict()`). Raises `ValueError` for an
    unrecognized kind -- callers persisting a whole history (see
    `PanelResultHistory.from_dict`) are expected to catch this per-entry
    and skip/log rather than fail the whole load, the same tolerance
    `core.project_io.load_project` already gives a stale `PlotSeries`
    reference; a *result* subclass this app doesn't know about yet
    (e.g. saved by a newer version) is not this app's fault to crash
    over."""
    kind = data.get("kind")
    result_cls = _RESULT_KIND_REGISTRY.get(kind)
    if result_cls is None:
        raise ValueError(f"Unknown analysis result kind: {kind!r}")
    return result_cls.from_dict(data)
