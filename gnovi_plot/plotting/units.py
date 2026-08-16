from __future__ import annotations

_MM_PER_INCH = 25.4
_CM_PER_INCH = 2.54

_UNIT_TO_INCH = {"mm": 1.0 / _MM_PER_INCH, "cm": 1.0 / _CM_PER_INCH, "in": 1.0}


def to_inches(value: float, unit: str) -> float:
    if unit not in _UNIT_TO_INCH:
        raise ValueError(f"Unknown length unit: {unit!r}")
    return value * _UNIT_TO_INCH[unit]


def from_inches(value_in: float, unit: str) -> float:
    if unit not in _UNIT_TO_INCH:
        raise ValueError(f"Unknown length unit: {unit!r}")
    return value_in / _UNIT_TO_INCH[unit]


# width/height ratio (None = no fixed ratio -- width and height stay
# independently editable). Figure ratio governs page/export geometry only;
# it is never used as a Matplotlib Axes aspect ratio.
ASPECT_RATIO_PRESETS: dict[str, float | None] = {
    "Auto / Fit workspace": None,
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "3:2": 3 / 2,
    "2:3": 2 / 3,
    "16:9": 16 / 9,
    "Custom": None,
}

# Generic, practical defaults -- deliberately NOT tied to any specific
# journal's submission requirements. Adjust these constants if a specific
# target size is later configured.
PUBLICATION_PRESETS_MM: dict[str, tuple[float, float]] = {
    "Square figure": (100.0, 100.0),
    "Presentation 16:9": (254.0, 142.9),
    "Journal single column": (85.0, 65.0),
    "Journal double column": (178.0, 110.0),
}

# Panel Aspect Ratio: the physical width/height shape of each individual
# Axes box, independent of the outer Figure Aspect Ratio above (which
# governs the complete multi-panel page/canvas) and independent of data-unit
# scaling (never `ax.set_aspect("equal")` -- see `panel_box_aspect` below).
# Same width/height ratio values as `ASPECT_RATIO_PRESETS` (deliberately, so
# "1:1"/"4:3"/etc. mean the same physical shape whichever control picked
# them) minus the figure-only "Auto / Fit workspace" and "Custom" entries,
# which have no panel-level equivalent -- "Auto" is Panel Aspect Ratio's own
# only no-op value.
PANEL_ASPECT_RATIO_PRESETS: dict[str, float | None] = {
    "Auto": None,
    **{
        name: ratio
        for name, ratio in ASPECT_RATIO_PRESETS.items()
        if name not in ("Auto / Fit workspace", "Custom")
    },
}


def panel_box_aspect(preset: str) -> float | None:
    """`Axes.set_box_aspect()` value (height/width) for a Panel Aspect Ratio
    preset name, or None for "Auto"/unknown (no box-aspect constraint --
    Matplotlib's own default layout behavior). `PANEL_ASPECT_RATIO_PRESETS`
    stores width/height (matching `ASPECT_RATIO_PRESETS`'s convention), so
    this is the reciprocal -- e.g. "4:3" (width:height 4:3) -> box_aspect
    3/4 (a box 3 units tall per 4 wide).
    """
    ratio = PANEL_ASPECT_RATIO_PRESETS.get(preset)
    return None if ratio is None else 1.0 / ratio
