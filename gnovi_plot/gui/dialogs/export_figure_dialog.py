from __future__ import annotations

import io
from pathlib import Path

from matplotlib.transforms import Bbox
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from gnovi_plot.export.figure_export import ExportError, export_live_figure
from gnovi_plot.gui.widgets.plot_canvas import PlotCanvas
from gnovi_plot.plotting.figure import GnoviFigure

_RASTER_FORMATS = ["PNG", "TIFF"]
_VECTOR_FORMATS = ["SVG", "PDF"]
_DPI_PRESETS = ["150", "300", "600", "1200", "Custom"]
_DEFAULT_DPI = 300

_SCOPE_COMPLETE = "Complete Figure"
_SCOPE_ACTIVE = "Active Panel"

_BACKGROUND_AS_SHOWN = "As shown"
_BACKGROUND_OPAQUE = "Opaque"
_BACKGROUND_TRANSPARENT = "Transparent"

_BBOX_NORMAL = "Normal"
_BBOX_TIGHT = "Tight"

# The preview always renders at a fixed, modest DPI for speed -- it exists
# to show layout/background/crop, never as a stand-in for judging actual
# export sharpness (which is governed by the dialog's own DPI setting and
# shown numerically via `pixel_size_label`, not visually, since a screen
# can't really show a difference between e.g. 300 and 1200 DPI at preview
# size anyway).
_PREVIEW_DPI = 100
_PREVIEW_MAX_WIDTH = 380
_PREVIEW_MAX_HEIGHT = 220
_CHECKER_SIZE = 8
_CHECKER_LIGHT = QColor("#e4e6ea")
_CHECKER_DARK = QColor("#c9cdd4")


class ExportFigureDialog(QDialog):
    """Collects export settings and saves the LIVE on-screen Matplotlib
    Figure (`plot_canvas.figure`) directly, via `export.figure_export.
    export_live_figure` -- no rendering/export logic lives here.

    WYSIWYG by construction: this is an "enhanced save dialog" around the
    exact same Figure object the Matplotlib navigation toolbar's own
    "Save" button would save (see `MainWindow`'s toolbar setup) -- the
    same axes, legends, typography, scientific notation, panel geometry,
    margins, spacing, Figure/Panel Aspect Ratio, series styles, grid,
    labels and titles, exactly as currently rendered, not a second,
    independently reconstructed figure. GNOVI's own options (format/DPI/
    background/bounding box/scope) only ever change `savefig()` arguments,
    never re-render or reposition anything. "Active Panel" scope crops to
    the active Axes' own tight bounding box (`Axes.get_tightbbox`) rather
    than screenshotting GUI pixels or rebuilding a second figure -- still
    the one live Figure, just a smaller `bbox_inches`.
    """

    def __init__(self, figure: GnoviFigure, plot_canvas: PlotCanvas, parent=None):
        super().__init__(parent)
        self._figure = figure
        self._plot_canvas = plot_canvas
        self.setWindowTitle("Export Figure")
        self.setModal(True)

        self.scope_combo = QComboBox()
        self.scope_combo.addItems([_SCOPE_COMPLETE, _SCOPE_ACTIVE])

        self.format_combo = QComboBox()
        self.format_combo.addItems(_RASTER_FORMATS + _VECTOR_FORMATS)

        self.dpi_preset_combo = QComboBox()
        self.dpi_preset_combo.addItems(_DPI_PRESETS)
        self.dpi_preset_combo.setCurrentText(str(_DEFAULT_DPI))

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(1, 4800)
        self.dpi_spin.setValue(_DEFAULT_DPI)
        self.dpi_spin.setEnabled(False)

        self.background_combo = QComboBox()
        self.background_combo.addItems([_BACKGROUND_AS_SHOWN, _BACKGROUND_OPAQUE, _BACKGROUND_TRANSPARENT])
        self.background_combo.setToolTip(
            "As shown: the figure's current Plot Theme background, exactly as displayed.\n"
            "Opaque: a plain, solid background regardless of Plot Theme.\n"
            "Transparent: no background (alpha channel)."
        )

        # Off by default -- matches the on-screen preview and Matplotlib's
        # own toolbar "Save" (neither ever crops the configured margins);
        # see `export.figure_export.export_figure`'s docstring for why a
        # silently-on tight bbox made every text element look uniformly
        # oversized compared to those two. Still available for anyone who
        # explicitly wants whitespace trimmed for publication.
        self.bbox_combo = QComboBox()
        self.bbox_combo.addItems([_BBOX_NORMAL, _BBOX_TIGHT])

        self.padding_spin = QDoubleSpinBox()
        self.padding_spin.setRange(0.0, 5.0)
        self.padding_spin.setSingleStep(0.05)
        self.padding_spin.setValue(0.1)
        self.padding_spin.setSuffix(" in")

        self.pixel_size_label = QLabel()

        self.path_edit = QLineEdit()
        self.browse_button = QPushButton("Browse…")

        form = QFormLayout()
        form.addRow("Scope", self.scope_combo)
        form.addRow("Format", self.format_combo)
        form.addRow("DPI preset", self.dpi_preset_combo)
        form.addRow("Custom DPI", self.dpi_spin)
        form.addRow("Background", self.background_combo)
        form.addRow("Bounding box", self.bbox_combo)
        form.addRow("Padding", self.padding_spin)
        form.addRow("Figure size", self.pixel_size_label)

        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit)
        path_row.addWidget(self.browse_button)
        form.addRow("Save to", path_row)

        # Export Preview: renders through the exact same `export_live_figure`
        # call the real export uses (see `_render_preview_pixmap`), with the
        # dialog's OWN current scope/background/bounding-box/padding
        # settings applied -- so what's shown here is what actually gets
        # written to disk, not just the live on-screen Plot Theme preview.
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFrameShape(QFrame.StyledPanel)
        self.preview_label.setMinimumSize(_PREVIEW_MAX_WIDTH, _PREVIEW_MAX_HEIGHT)
        self.preview_label.setScaledContents(False)

        content_row = QHBoxLayout()
        content_row.addLayout(form, 1)
        content_row.addWidget(self.preview_label, 1)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(content_row)
        layout.addWidget(self.button_box)

        self.scope_combo.currentTextChanged.connect(self._on_scope_or_bbox_changed)
        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        self.dpi_preset_combo.currentTextChanged.connect(self._on_dpi_preset_changed)
        self.dpi_spin.valueChanged.connect(self._update_pixel_size_label)
        self.background_combo.currentTextChanged.connect(self._refresh_preview)
        self.bbox_combo.currentTextChanged.connect(self._on_scope_or_bbox_changed)
        self.padding_spin.valueChanged.connect(self._refresh_preview)
        self.browse_button.clicked.connect(self._on_browse)

        self._on_format_changed(self.format_combo.currentText())
        self._sync_padding_enabled()
        self._refresh_preview()
        # Small, fixed-purpose dialog -- comfortably within a 1366x768 screen.
        self.resize(760, 440)

    # --- Scope / format / DPI --------------------------------------------

    def _on_scope_or_bbox_changed(self, *_args) -> None:
        self._sync_padding_enabled()
        self._refresh_preview()

    def _padding_enabled(self) -> bool:
        if self.scope_combo.currentText() == _SCOPE_ACTIVE:
            return True  # always meaningful around a per-panel crop
        return self.bbox_combo.currentText() == _BBOX_TIGHT

    def _sync_padding_enabled(self) -> None:
        self.padding_spin.setEnabled(self._padding_enabled())

    def _on_format_changed(self, fmt: str) -> None:
        is_raster = fmt in _RASTER_FORMATS
        self.dpi_preset_combo.setEnabled(is_raster)
        self.dpi_spin.setEnabled(is_raster and self.dpi_preset_combo.currentText() == "Custom")
        self._sync_path_extension(fmt.lower())
        # The preview always rasterizes to PNG regardless of export format
        # (see `_render_preview_pixmap`), so switching between e.g. PNG and
        # SVG doesn't change it -- only the pixel-size label does. Only
        # refresh if the widget already exists (this can fire from
        # `__init__` before `pixel_size_label` is built).
        if hasattr(self, "pixel_size_label"):
            self._update_pixel_size_label()

    def _sync_path_extension(self, ext: str) -> None:
        text = self.path_edit.text()
        if not text:
            return
        self.path_edit.setText(str(Path(text).with_suffix(f".{ext}")))

    def _on_dpi_preset_changed(self, text: str) -> None:
        if text == "Custom":
            self.dpi_spin.setEnabled(True)
            return
        self.dpi_spin.setEnabled(False)
        self.dpi_spin.setValue(int(text))

    def _on_browse(self) -> None:
        fmt = self.format_combo.currentText().lower()
        path, _filter = QFileDialog.getSaveFileName(self, "Export Figure", f"figure.{fmt}", f"*.{fmt}")
        if path:
            self.path_edit.setText(path)

    # --- Resolving savefig() arguments from the dialog's own state -------

    def _background_kwargs(self) -> dict:
        choice = self.background_combo.currentText()
        if choice == _BACKGROUND_TRANSPARENT:
            return {"transparent": True, "facecolor": None}
        if choice == _BACKGROUND_OPAQUE:
            return {"transparent": False, "facecolor": "white"}
        return {"transparent": False, "facecolor": None}  # As shown

    def _active_panel_bbox_inches(self, *, padded: bool) -> Bbox | None:
        """The active Axes' own current rendered extent, in Figure-inches
        -- `Axes.get_tightbbox()` includes that Axes' own title/labels/
        ticks/legend/panel-label, never a neighboring panel's. `padded`
        adds `padding_spin`'s value uniformly on every side."""
        renderer = self._plot_canvas.get_renderer()
        if renderer is None:
            return None
        ax = self._plot_canvas.active_axes(self._figure)
        bbox_px = ax.get_tightbbox(renderer)
        bbox_in = bbox_px.transformed(self._plot_canvas.figure.dpi_scale_trans.inverted())
        if not padded:
            return bbox_in
        pad = self.padding_spin.value()
        return Bbox.from_extents(bbox_in.x0 - pad, bbox_in.y0 - pad, bbox_in.x1 + pad, bbox_in.y1 + pad)

    def _savefig_kwargs(self, *, dpi: int) -> dict:
        kwargs: dict = dict(dpi=dpi, **self._background_kwargs())
        if self.scope_combo.currentText() == _SCOPE_ACTIVE:
            bbox = self._active_panel_bbox_inches(padded=(self.bbox_combo.currentText() == _BBOX_NORMAL))
            if bbox is not None:
                kwargs["bbox_inches"] = bbox
        elif self.bbox_combo.currentText() == _BBOX_TIGHT:
            kwargs["bbox_inches"] = "tight"
            kwargs["pad_inches"] = self.padding_spin.value()
        return kwargs

    def _current_physical_size_in(self) -> tuple[float, float] | None:
        if self.scope_combo.currentText() == _SCOPE_ACTIVE:
            bbox = self._active_panel_bbox_inches(padded=(self.bbox_combo.currentText() == _BBOX_NORMAL))
            if bbox is None:
                return None
            return (bbox.width, bbox.height)
        return tuple(self._plot_canvas.figure.get_size_inches())

    def _update_pixel_size_label(self) -> None:
        size_in = self._current_physical_size_in()
        if size_in is None:
            self.pixel_size_label.setText("—")
            return
        width_in, height_in = size_in
        text = f"{width_in:.2f} × {height_in:.2f} in"
        if self.format_combo.currentText() in _RASTER_FORMATS:
            dpi = self.dpi_spin.value()
            text += f"  @ {dpi} DPI  →  {round(width_in * dpi)} × {round(height_in * dpi)} px"
        self.pixel_size_label.setText(text)

    def _hide_gui_only_overlays(self) -> None:
        """Strip GUI-only interaction aids that are real Matplotlib artists
        on the live figure -- unlike the active-panel badge (a separate Qt
        widget that was never added to the Figure to begin with, see
        `PlotCanvas._ActivePanelBadge`), the reference cursor IS drawn as
        actual `Line2D` artists on the active Axes (see `PlotCanvas.
        update_reference_cursor`), so saving the live figure as-is would
        otherwise export whatever crosshair/reference line happens to be
        showing. Called immediately before every `export_live_figure` call
        (both the real export and the live preview) -- the cursor simply
        reappears on the user's next mouse move over the canvas, same as
        any other redraw."""
        self._plot_canvas.clear_reference_cursor()

    def _on_accept(self) -> None:
        if not self.path_edit.text():
            QMessageBox.warning(self, "Export Figure", "Choose a file to save to.")
            return
        self._hide_gui_only_overlays()
        try:
            export_live_figure(
                self._plot_canvas.figure,
                self.path_edit.text(),
                fmt=self.format_combo.currentText().lower(),
                **self._savefig_kwargs(dpi=self.dpi_spin.value()),
            )
        except (ExportError, OSError) as exc:
            QMessageBox.critical(self, "Export Figure", str(exc))
            return
        self.accept()

    # --- Export Preview ------------------------------------------------------

    def _refresh_preview(self) -> None:
        """Render a small PNG through the exact same `export_live_figure`
        call the real export uses, with this dialog's current scope/
        background/bounding-box/padding settings applied, and show it.
        Best-effort: a broken preview must never block choosing settings or
        exporting (the real export still runs its own typed error handling
        in `_on_accept`), so failures here are shown as text in the preview
        area rather than raised -- rendering an arbitrary live figure can
        fail in ways beyond this dialog's control.
        """
        try:
            pixmap = self._render_preview_pixmap()
        except Exception as exc:  # noqa: BLE001 -- best-effort preview, see docstring
            self.preview_label.setText(f"Preview unavailable:\n{exc}")
            self._update_pixel_size_label()
            return
        self.preview_label.setPixmap(pixmap)
        self._update_pixel_size_label()

    def _render_preview_pixmap(self) -> QPixmap:
        self._hide_gui_only_overlays()
        buffer = io.BytesIO()
        export_live_figure(
            self._plot_canvas.figure,
            buffer,
            fmt="png",
            **self._savefig_kwargs(dpi=_PREVIEW_DPI),
        )

        rendered = QPixmap()
        rendered.loadFromData(buffer.getvalue(), "PNG")
        rendered = rendered.scaled(
            _PREVIEW_MAX_WIDTH, _PREVIEW_MAX_HEIGHT, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        if self.background_combo.currentText() != _BACKGROUND_TRANSPARENT:
            return rendered

        # Composite over a checkerboard so a transparent export is visibly
        # distinguishable from a plain white one -- a transparent PNG
        # otherwise just looks identical to a white-background PNG here.
        canvas = QPixmap(rendered.size())
        canvas.fill(Qt.transparent)
        painter = QPainter(canvas)
        painter.drawTiledPixmap(canvas.rect(), _checkerboard_tile())
        painter.drawPixmap(0, 0, rendered)
        painter.end()
        return canvas


def _checkerboard_tile() -> QPixmap:
    tile = QPixmap(_CHECKER_SIZE * 2, _CHECKER_SIZE * 2)
    painter = QPainter(tile)
    painter.fillRect(0, 0, _CHECKER_SIZE * 2, _CHECKER_SIZE * 2, _CHECKER_LIGHT)
    painter.fillRect(0, 0, _CHECKER_SIZE, _CHECKER_SIZE, _CHECKER_DARK)
    painter.fillRect(_CHECKER_SIZE, _CHECKER_SIZE, _CHECKER_SIZE, _CHECKER_SIZE, _CHECKER_DARK)
    painter.end()
    return tile
