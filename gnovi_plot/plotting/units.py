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
