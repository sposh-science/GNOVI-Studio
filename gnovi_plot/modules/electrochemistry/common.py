"""Reusable electrochemistry primitives shared across techniques.

Pure NumPy/SciPy numerical code -- no Qt, no Matplotlib, no ``Dataset``.
Every function here treats its inputs as immutable: arrays are copied on
entry (``np.array(..., dtype=float)``) and never written to in place.

Canonical internal units (everything downstream assumes these):

======== ==========
potential  volt (V)
current    ampere (A)
scan rate  volt / second (V/s)
charge     coulomb (C)
======== ==========

Unit handling is deliberately a small set of scale factors plus explicit
helpers -- NOT a general units framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.integrate import trapezoid

from gnovi_plot.analysis.cycles import DEFAULT_TOLERANCE_FRACTION, carried_step_directions


class ElectrochemistryError(Exception):
    """Base class for every error raised by ``gnovi_plot.modules.electrochemistry``."""


class UnknownUnitError(ElectrochemistryError, ValueError):
    """Raised for a unit string not in the relevant conversion table."""


class SweepSegmentationError(ElectrochemistryError):
    """Raised when a potential signal cannot be segmented into sweeps
    (e.g. it is flat / has no monotonic progression at all)."""


class ChargeIntegrationError(ElectrochemistryError, ValueError):
    """Raised for invalid charge-integration input: mismatched shapes,
    non-finite data, a non-monotonic axis where one is required, an
    invalid/zero scan rate, or an ambiguous domain selection."""


class InvalidElectrodeContextError(ElectrochemistryError, ValueError):
    """Raised for a non-physical value in :class:`ElectrodeContext`
    (a non-positive area / n / concentration / temperature that was
    actually supplied -- ``None`` is always allowed)."""


# --------------------------------------------------------------------------
# Unit conversion helpers
# --------------------------------------------------------------------------

#: Scale factor -> canonical volts.
POTENTIAL_UNITS: dict[str, float] = {"V": 1.0, "mV": 1e-3}
#: Scale factor -> canonical amperes. ``µA`` and ``uA`` are accepted spellings.
CURRENT_UNITS: dict[str, float] = {"A": 1.0, "mA": 1e-3, "µA": 1e-6, "uA": 1e-6, "nA": 1e-9}
#: Scale factor -> canonical volts / second.
SCAN_RATE_UNITS: dict[str, float] = {"V/s": 1.0, "mV/s": 1e-3}
#: Scale factor -> canonical coulombs.
CHARGE_UNITS: dict[str, float] = {"C": 1.0, "mC": 1e-3}

_UNIT_TABLES: dict[str, dict[str, float]] = {
    "potential": POTENTIAL_UNITS,
    "current": CURRENT_UNITS,
    "scan_rate": SCAN_RATE_UNITS,
    "charge": CHARGE_UNITS,
}


def convert_units(value: float, from_unit: str, to_unit: str, quantity: str) -> float:
    """Convert ``value`` from ``from_unit`` to ``to_unit`` for ``quantity``
    (one of ``"potential"``, ``"current"``, ``"scan_rate"``, ``"charge"``).

    Uses simple scale factors relative to the canonical unit. Raises
    :class:`UnknownUnitError` for an unknown quantity or unit string --
    never silently guesses.
    """
    table = _UNIT_TABLES.get(quantity)
    if table is None:
        raise UnknownUnitError(
            f"Unknown quantity {quantity!r}; expected one of {sorted(_UNIT_TABLES)}"
        )
    try:
        from_factor = table[from_unit]
    except KeyError:
        raise UnknownUnitError(
            f"Unknown {quantity} unit {from_unit!r}; expected one of {sorted(table)}"
        ) from None
    try:
        to_factor = table[to_unit]
    except KeyError:
        raise UnknownUnitError(
            f"Unknown {quantity} unit {to_unit!r}; expected one of {sorted(table)}"
        ) from None
    return value * from_factor / to_factor


def potential_to_volts(value: float, unit: str) -> float:
    """``value`` (in ``unit``, e.g. ``"mV"``) as canonical volts."""
    return convert_units(value, unit, "V", "potential")


def current_to_amperes(value: float, unit: str) -> float:
    """``value`` (in ``unit``, e.g. ``"µA"``/``"uA"``) as canonical amperes."""
    return convert_units(value, unit, "A", "current")


def scan_rate_to_v_per_s(value: float, unit: str) -> float:
    """``value`` (in ``unit``, e.g. ``"mV/s"``) as canonical V/s."""
    return convert_units(value, unit, "V/s", "scan_rate")


# --------------------------------------------------------------------------
# Current sign convention
# --------------------------------------------------------------------------


class CurrentSignConvention(str, Enum):
    """How the sign of a recorded current maps to oxidation / reduction.

    This is an INTERPRETATION LAYER ONLY. The imported ``Dataset``/array
    current values are never modified or flipped -- the convention only
    tells the analysis code which direction of current is oxidative.
    """

    #: IUPAC / Bard & Faulkner: oxidation (anodic) current is positive.
    ANODIC_POSITIVE = "anodic_positive"
    #: Polarographic convention: reduction (cathodic) current is positive.
    CATHODIC_POSITIVE = "cathodic_positive"


DEFAULT_SIGN_CONVENTION = CurrentSignConvention.ANODIC_POSITIVE


def oxidative_sign(convention: CurrentSignConvention) -> int:
    """``+1`` if oxidative (anodic) current is positive under ``convention``,
    ``-1`` if it is negative.

    Multiply a recorded current by this to get an "oxidation points up"
    signal, or by its negation for "reduction points up".
    """
    return 1 if CurrentSignConvention(convention) == CurrentSignConvention.ANODIC_POSITIVE else -1


# --------------------------------------------------------------------------
# ElectrodeContext -- optional physical / descriptive metadata
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ElectrodeContext:
    """Optional physical and descriptive metadata about a CV measurement.

    Every field is optional. None of it is required for basic CV peak
    analysis (cycle/sweep segmentation, peak detection/measurement,
    Epa/Epc/Ipa/Ipc, ΔEp, E½). The numeric fields gate specific later
    calculations (current density, Randles-Sevcik diffusion coefficient --
    NOT implemented in CV-1) and are NEVER silently defaulted: a missing
    area/n/concentration/temperature disables the calculation that needs
    it, it does not become 1 cm² / n=1 / 1 mM / 298 K.

    A supplied numeric value must be physically positive; a supplied
    non-positive value raises :class:`InvalidElectrodeContextError`.
    """

    area_cm2: float | None = None
    n: float | None = None
    concentration_mol_cm3: float | None = None
    temperature_k: float | None = None
    reference_electrode: str | None = None
    working_electrode: str | None = None
    counter_electrode: str | None = None
    electrolyte: str | None = None

    def __post_init__(self) -> None:
        for name in ("area_cm2", "n", "concentration_mol_cm3", "temperature_k"):
            value = getattr(self, name)
            if value is None:
                continue
            if not np.isfinite(value) or value <= 0:
                raise InvalidElectrodeContextError(
                    f"ElectrodeContext.{name} must be a positive finite number when supplied "
                    f"(got {value!r})"
                )

    def is_empty(self) -> bool:
        """``True`` if no field is set."""
        return all(getattr(self, f.name) is None for f in self.__dataclass_fields__.values())

    def to_dict(self) -> dict:
        return {
            "area_cm2": self.area_cm2,
            "n": self.n,
            "concentration_mol_cm3": self.concentration_mol_cm3,
            "temperature_k": self.temperature_k,
            "reference_electrode": self.reference_electrode,
            "working_electrode": self.working_electrode,
            "counter_electrode": self.counter_electrode,
            "electrolyte": self.electrolyte,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ElectrodeContext":
        return cls(
            area_cm2=data.get("area_cm2"),
            n=data.get("n"),
            concentration_mol_cm3=data.get("concentration_mol_cm3"),
            temperature_k=data.get("temperature_k"),
            reference_electrode=data.get("reference_electrode"),
            working_electrode=data.get("working_electrode"),
            counter_electrode=data.get("counter_electrode"),
            electrolyte=data.get("electrolyte"),
        )


# --------------------------------------------------------------------------
# Sweep segmentation
# --------------------------------------------------------------------------

SWEEP_RISING = "rising"
SWEEP_FALLING = "falling"


@dataclass(frozen=True)
class SweepSegment:
    """One monotonic-direction segment of a potential sweep.

    ``start`` / ``end`` are POSITIONAL row indices into the source arrays,
    end-exclusive (``arr[start:end]``, ``DataFrame.iloc`` style). Adjacent
    segments SHARE the turning-point row (segment ``k``'s ``end`` is
    ``k+1``'s ``start + 1``), matching ``analysis.cycles.detect_cycles``'s
    own overlapping-at-the-vertex convention.

    ``direction`` is pure geometry: :data:`SWEEP_RISING` (potential
    increasing) or :data:`SWEEP_FALLING` (potential decreasing). It carries
    NO electrochemical meaning -- rising is not "anodic", falling is not
    "cathodic" (see :mod:`gnovi_plot.modules.electrochemistry.cv`).

    ``e_start`` / ``e_end`` are the potential values at the segment
    endpoints, so ``potential_span`` is available without re-passing the
    array (used by cycle pairing to flag truncated first/last sweeps).
    """

    start: int
    end: int
    direction: str
    e_start: float
    e_end: float

    @property
    def potential_span(self) -> float:
        """``|e_end - e_start|`` -- how far the potential travelled."""
        return abs(self.e_end - self.e_start)

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "direction": self.direction,
            "e_start": self.e_start,
            "e_end": self.e_end,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SweepSegment":
        return cls(
            start=int(data["start"]),
            end=int(data["end"]),
            direction=data["direction"],
            e_start=float(data["e_start"]),
            e_end=float(data["e_end"]),
        )


def segment_sweeps(
    potential,
    *,
    noise_tolerance: float | None = None,
    tolerance_fraction: float = DEFAULT_TOLERANCE_FRACTION,
) -> list[SweepSegment]:
    """Deterministically segment ``potential`` into rising / falling sweeps.

    Reuses ``analysis.cycles.carried_step_directions`` -- the same
    noise-tolerant, plateau-carrying turning-point primitive
    ``detect_cycles`` uses -- so this never reimplements "which way is the
    potential going". Non-numeric / NaN samples are ignored when finding
    turning points but the returned ranges are positional into the ORIGINAL
    input (a range may therefore contain a few NaN rows a caller's numeric
    extraction will drop), exactly as ``detect_cycles`` does.

    Handles:

    * arbitrary initial sweep direction;
    * small numerical noise (sub-tolerance jitter is absorbed);
    * short plateaus / duplicate potentials at a vertex (direction carried);
    * a monotonic (LSV-like) trace -> a single segment, no error;
    * incomplete first/last sweeps -> just shorter segments (completeness is
      :func:`gnovi_plot.modules.electrochemistry.cv.pair_cycles`'s call).

    Raises :class:`SweepSegmentationError` only when there is no directional
    progression at all (flat data), or fewer than 2 numeric samples.

    Never mutates ``potential``.
    """
    values_full = np.array(potential, dtype=float)
    if values_full.ndim != 1:
        raise SweepSegmentationError("potential must be 1-dimensional")

    valid_positions = np.flatnonzero(np.isfinite(values_full))
    if valid_positions.size < 2:
        raise SweepSegmentationError(
            f"Not enough numeric potential samples to segment (found {valid_positions.size}, need 2)."
        )

    values = values_full[valid_positions]
    sign = carried_step_directions(values, noise_tolerance, tolerance_fraction)

    if not np.any(sign != 0):
        raise SweepSegmentationError(
            "The potential signal has no rising or falling progression (it appears flat)."
        )

    # Interior turning points: where the outgoing direction differs from the
    # incoming one. Boundaries in local (`values`) index space.
    turning_points = [i for i in range(1, len(sign)) if sign[i] != sign[i - 1] and sign[i - 1] != 0]
    boundaries = [0, *turning_points, len(values) - 1]

    segments: list[SweepSegment] = []
    for a, b in zip(boundaries, boundaries[1:]):
        if b <= a:
            continue
        # Direction of this span = its first genuine step direction.
        span_signs = sign[a:b]
        nonzero = span_signs[span_signs != 0]
        if nonzero.size == 0:
            continue
        direction = SWEEP_RISING if nonzero[0] > 0 else SWEEP_FALLING
        start = int(valid_positions[a])
        end = int(valid_positions[b]) + 1
        segments.append(
            SweepSegment(
                start=start,
                end=end,
                direction=direction,
                e_start=float(values[a]),
                e_end=float(values[b]),
            )
        )

    if not segments:  # pragma: no cover - defensive; sign has a non-zero entry above
        raise SweepSegmentationError("No sweep segments could be formed from the potential signal.")
    return segments


# --------------------------------------------------------------------------
# Charge integration primitive:  Q = ∫ I dt
# --------------------------------------------------------------------------

CHARGE_DOMAIN_TIME = "time"
CHARGE_DOMAIN_POTENTIAL = "potential"


def integrate_current(
    current,
    *,
    time=None,
    potential=None,
    scan_rate_v_per_s: float | None = None,
) -> float:
    """Charge ``Q = ∫ I dt`` in coulombs (SciPy trapezoidal integration).

    Exactly one integration domain must be supplied:

    * ``time`` (seconds) -- the preferred path: integrate ``current``
      directly against time. ``time`` must be strictly monotonic.
    * ``potential`` (volts) + ``scan_rate_v_per_s`` (> 0) -- for a
      monotonic, constant-rate sweep only. Uses ``dt = |dE| / v``, i.e.
      ``Q = (1/v) ∫ I d|E - E0|``. ``potential`` must be strictly
      monotonic (the function refuses to integrate across a reversal).

    The sign of the result follows the sign of ``current`` (a net
    oxidative current in an anodic-positive dataset gives ``Q > 0``).

    Raises :class:`ChargeIntegrationError` for: both/neither domain given,
    shape mismatch, non-finite data, fewer than 2 points, a non-monotonic
    axis where monotonicity is required, or a missing / non-positive /
    non-finite scan rate.

    Never mutates the input arrays.
    """
    i = np.array(current, dtype=float)
    if i.ndim != 1:
        raise ChargeIntegrationError("current must be 1-dimensional")
    if i.size < 2:
        raise ChargeIntegrationError(f"Need at least 2 current samples to integrate (got {i.size}).")
    if not np.all(np.isfinite(i)):
        raise ChargeIntegrationError("current must be entirely finite.")

    if (time is None) == (potential is None):
        raise ChargeIntegrationError(
            "Supply exactly one of `time` (time-domain) or `potential` (+ `scan_rate_v_per_s`)."
        )

    if time is not None:
        t = np.array(time, dtype=float)
        if t.shape != i.shape:
            raise ChargeIntegrationError(
                f"time and current must have the same shape (got {t.shape} and {i.shape})."
            )
        if not np.all(np.isfinite(t)):
            raise ChargeIntegrationError("time must be entirely finite.")
        dt = np.diff(t)
        if not (np.all(dt > 0) or np.all(dt < 0)):
            raise ChargeIntegrationError(
                "time must be strictly monotonic to integrate charge over it."
            )
        return float(trapezoid(i, t))

    # potential / (E, v) path
    if scan_rate_v_per_s is None:
        raise ChargeIntegrationError(
            "`scan_rate_v_per_s` is required when integrating charge over potential."
        )
    if not np.isfinite(scan_rate_v_per_s) or scan_rate_v_per_s <= 0:
        raise ChargeIntegrationError(
            f"scan rate must be a positive finite number of V/s (got {scan_rate_v_per_s!r})."
        )
    e = np.array(potential, dtype=float)
    if e.shape != i.shape:
        raise ChargeIntegrationError(
            f"potential and current must have the same shape (got {e.shape} and {i.shape})."
        )
    if not np.all(np.isfinite(e)):
        raise ChargeIntegrationError("potential must be entirely finite.")
    de = np.diff(e)
    if not (np.all(de > 0) or np.all(de < 0)):
        raise ChargeIntegrationError(
            "potential must be strictly monotonic for E/v charge integration -- "
            "do not integrate across a sweep reversal; split into single sweeps first."
        )
    # s = distance travelled in potential, monotonically increasing with time
    # regardless of sweep direction:  ds = |dE| = v * dt  ->  Q = (1/v) ∫ I ds
    s = np.abs(e - e[0])
    return float(trapezoid(i, s) / scan_rate_v_per_s)
