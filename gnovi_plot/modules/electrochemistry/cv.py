"""Cyclic-voltammetry numerical core: cycle pairing, candidate peak
detection, a transparent local-linear baseline primitive, quantitative
peak measurement, and couple metrics (ΔEp, E½, anodic/cathodic peak-current
ratio).

Pure NumPy/SciPy code -- no Qt, no Matplotlib, no ``Dataset``, no CV GUI.
Every function copies its array inputs and never mutates them.

Design boundaries held in CV-1:

* **Detection is not measurement.** ``detect_cv_peaks`` returns
  :class:`CVPeakSeed` CANDIDATES. Quantitative ``Epa``/``Epc``/``Ipa``/
  ``Ipc`` come from ``measure_peak`` on the raw (optionally
  baseline-corrected) signal. CV-1 does no smoothing at all.
* **Geometry is not interpretation.** A sweep's ``direction`` (rising /
  falling) is independent of a peak's ``process`` (anodic / cathodic).
  Rising is never assumed anodic.
* **A raw extremum is not a peak current.** Without a baseline,
  ``measure_peak`` returns the raw extremum labelled as such
  (``i_peak_corrected_a is None``); it is never presented as a
  baseline-corrected Ipa/Ipc.

Deferred past CV-1 (see PROJECT_GUIDE.md): the Nicholson switching-potential
ratio (needs interactive switching-potential context to do correctly),
polynomial/arPLS/spline baselines, multi-scan-rate regression, and
Randles-Sevcik.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from gnovi_plot.modules.electrochemistry.common import (
    SWEEP_FALLING,
    SWEEP_RISING,
    CurrentSignConvention,
    ElectrochemistryError,
    SweepSegment,
    oxidative_sign,
)

# Peak "process" -- the ELECTROCHEMICAL interpretation, independent of sweep
# direction.
PROCESS_ANODIC = "anodic"
PROCESS_CATHODIC = "cathodic"
PROCESS_UNASSIGNED = "unassigned"

# Peak "origin" -- how the candidate came to exist.
ORIGIN_AUTOMATIC = "automatic"
ORIGIN_MANUAL = "manual"

# Which current values a couple ratio was computed from.
RATIO_BASIS_CORRECTED = "baseline_corrected"
RATIO_BASIS_RAW = "raw_extremum"


class CVAnalysisError(ElectrochemistryError):
    """Base class for CV-specific analysis errors."""


class InvalidCVInputError(CVAnalysisError, ValueError):
    """Raised for invalid CV numerical input: mismatched array shapes,
    non-finite data, an out-of-bounds search/anchor range, or too few
    baseline anchor points."""


# --------------------------------------------------------------------------
# Cycle model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cycle:
    """One CV cycle: an ordered group of one or more consecutive sweeps.

    ``index`` is the 1-based cycle number in ACQUISITION ORDER -- so if a
    run begins with a truncated partial sweep, that stub is cycle 1 and the
    first full cycle is cycle 2.
    ``start`` / ``end`` are positional row indices (end-exclusive) spanning
    every sweep in the cycle. ``sweeps`` are the :class:`SweepSegment`\\ s,
    in order.

    ``complete`` is ``True`` only when the cycle contains BOTH a rising and
    a falling sweep AND none of its sweeps is a truncated fragment (a
    potential span below :data:`PARTIAL_SWEEP_FRACTION` of the widest sweep
    in the whole trace). A lone leading / trailing sweep, or a cycle whose
    first/last sweep is a stub, is ``complete=False`` -- it is kept and
    reported, never silently dropped or merged into a neighbour.
    """

    index: int
    start: int
    end: int
    sweeps: tuple[SweepSegment, ...]
    complete: bool

    @property
    def rising_sweep(self) -> SweepSegment | None:
        return next((s for s in self.sweeps if s.direction == SWEEP_RISING), None)

    @property
    def falling_sweep(self) -> SweepSegment | None:
        return next((s for s in self.sweeps if s.direction == SWEEP_FALLING), None)


#: A sweep whose potential span is below this fraction of the widest sweep
#: in the trace is treated as a truncated fragment, making its cycle
#: incomplete.
PARTIAL_SWEEP_FRACTION = 0.6


def pair_cycles(sweeps: list[SweepSegment]) -> list[Cycle]:
    """Group ``sweeps`` (as returned by
    :func:`gnovi_plot.modules.electrochemistry.common.segment_sweeps`, in
    acquisition order) into CV cycles.

    Rule: if the very first sweep is a truncated fragment (a run that
    started mid-sweep), it is emitted alone as an incomplete cycle so it
    cannot steal the following full sweep and misalign every later cycle.
    The remaining sweeps are then consumed two at a time in order; each
    consecutive opposite-direction pair is one cycle, and a single leftover
    sweep at the end is its own (incomplete) cycle. ``n`` identical
    complete cycles (``2n`` full sweeps, no leading stub) therefore yield
    exactly ``n`` complete cycles.

    Completeness (see :class:`Cycle`): a cycle is complete only if it has
    both directions and neither of its sweeps is a truncated fragment
    (span < :data:`PARTIAL_SWEEP_FRACTION` of the widest sweep overall) --
    so an experiment that begins or ends mid-sweep produces a clearly
    ``complete=False`` first/last cycle rather than a silent mispairing.

    Never assumes the first sweep is anodic (or rising).
    """
    if not sweeps:
        return []

    reference_span = max((s.potential_span for s in sweeps), default=0.0)
    threshold = PARTIAL_SWEEP_FRACTION * reference_span

    def _is_stub(seg: SweepSegment) -> bool:
        return reference_span > 0 and seg.potential_span < threshold

    def _make(index: int, group: tuple[SweepSegment, ...]) -> Cycle:
        directions = {s.direction for s in group}
        has_both = SWEEP_RISING in directions and SWEEP_FALLING in directions
        complete = has_both and not any(_is_stub(s) for s in group)
        return Cycle(
            index=index,
            start=group[0].start,
            end=group[-1].end,
            sweeps=group,
            complete=complete,
        )

    cycles: list[Cycle] = []
    remaining = list(sweeps)
    if len(remaining) > 1 and _is_stub(remaining[0]):
        cycles.append(_make(1, (remaining.pop(0),)))

    for i in range(0, len(remaining), 2):
        cycles.append(_make(len(cycles) + 1, tuple(remaining[i : i + 2])))
    return cycles


# --------------------------------------------------------------------------
# Peak candidate model
# --------------------------------------------------------------------------


@dataclass
class CVPeakSeed:
    """One CV peak CANDIDATE -- either ``scipy.signal.find_peaks`` found it
    (``origin=ORIGIN_AUTOMATIC``), or a caller added it directly
    (``origin=ORIGIN_MANUAL``, ``index=None``, no ``prominence``).

    A seed is a starting point for measurement ("analyse a peak near
    here"), never a final scientific assignment. ``enabled`` lets a
    candidate stay in the list (a detection pass is never silently lost)
    while being excluded from analysis -- the same soft-exclude convention
    as ``modules.xrd.peaks.XRDPeakSeed``. Deliberately NOT a subclass of
    ``XRDPeakSeed`` and with no shared superclass: CV peaks are
    bidirectional, baseline-relative and sweep-tagged; XRD peaks are not.

    ``potential_v`` / ``current_a`` are canonical volts / amperes at the
    seed location, ``current_a`` signed exactly as imported. ``sweep`` is
    the geometry (:data:`SWEEP_RISING` / :data:`SWEEP_FALLING`);
    ``process`` is the interpretation (:data:`PROCESS_ANODIC` /
    :data:`PROCESS_CATHODIC` / :data:`PROCESS_UNASSIGNED`) -- the two are
    independent.
    """

    potential_v: float
    current_a: float
    sweep: str
    process: str
    origin: str = ORIGIN_AUTOMATIC
    index: int | None = None
    prominence: float | None = None
    enabled: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @classmethod
    def manual(
        cls,
        potential_v: float,
        current_a: float,
        *,
        sweep: str = SWEEP_RISING,
        process: str = PROCESS_UNASSIGNED,
    ) -> "CVPeakSeed":
        """A user-added seed, not tied to any detection-array position."""
        return cls(
            potential_v=float(potential_v),
            current_a=float(current_a),
            sweep=sweep,
            process=process,
            origin=ORIGIN_MANUAL,
        )

    def to_dict(self) -> dict:
        return {
            "potential_v": self.potential_v,
            "current_a": self.current_a,
            "sweep": self.sweep,
            "process": self.process,
            "origin": self.origin,
            "index": self.index,
            "prominence": self.prominence,
            "enabled": self.enabled,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CVPeakSeed":
        return cls(
            potential_v=float(data["potential_v"]),
            current_a=float(data["current_a"]),
            sweep=data["sweep"],
            process=data["process"],
            origin=data.get("origin", ORIGIN_AUTOMATIC),
            index=data.get("index"),
            prominence=data.get("prominence"),
            enabled=data.get("enabled", True),
            id=data.get("id") or uuid.uuid4().hex,
        )


def _finite_pair(potential, current) -> tuple[np.ndarray, np.ndarray]:
    e = np.array(potential, dtype=float)
    i = np.array(current, dtype=float)
    if e.shape != i.shape:
        raise InvalidCVInputError(
            f"potential and current must have the same shape (got {e.shape} and {i.shape})."
        )
    if e.ndim != 1:
        raise InvalidCVInputError("potential and current must be 1-dimensional.")
    return e, i


def detect_cv_peaks(
    potential,
    current,
    sweep: SweepSegment,
    *,
    convention: CurrentSignConvention = CurrentSignConvention.ANODIC_POSITIVE,
    prominence: float | None = None,
    distance: int | None = None,
    width: float | None = None,
) -> list[CVPeakSeed]:
    """Detect anodic and cathodic peak candidates within a single ``sweep``.

    Runs ``scipy.signal.find_peaks`` INDEPENDENTLY on the slice
    ``potential[sweep.start:sweep.end]`` -- never on a concatenated full
    cycle, whose reversal vertex would create a spurious extremum.

    Sign-convention aware: with an oxidation-points-up signal
    ``o = current * oxidative_sign(convention)``, anodic candidates are
    ``find_peaks(o)`` and cathodic candidates are ``find_peaks(-o)``. Each
    candidate's ``process`` is assigned from the CURRENT direction it was
    found in, not from ``sweep.direction`` -- so a cathodic wave that
    happens to fall on a rising sweep is still tagged ``cathodic``.

    ``prominence`` is the primary parameter. ``distance`` (minimum sample
    separation) and ``width`` are optional passthroughs. The parameter
    surface is deliberately small.

    Returns candidates ordered by position. Never mutates the inputs.
    """
    e, i = _finite_pair(potential, current)
    if not (0 <= sweep.start < sweep.end <= e.size):
        raise InvalidCVInputError(
            f"sweep range ({sweep.start}, {sweep.end}) is out of bounds for {e.size} samples."
        )

    e_seg = e[sweep.start : sweep.end]
    i_seg = i[sweep.start : sweep.end]
    ox_sign = oxidative_sign(convention)
    oxidative = i_seg * ox_sign

    if not np.all(np.isfinite(oxidative)):
        raise InvalidCVInputError(
            "current within the sweep must be entirely finite -- clean non-finite values first."
        )

    seeds: list[CVPeakSeed] = []
    for process, signal in ((PROCESS_ANODIC, oxidative), (PROCESS_CATHODIC, -oxidative)):
        try:
            indices, properties = find_peaks(
                signal, prominence=prominence, distance=distance, width=width
            )
        except Exception as exc:  # SciPy raises plain ValueError for bad params
            raise InvalidCVInputError(f"Peak detection failed: {exc}") from exc
        prominences = properties.get("prominences")
        for position, local_idx in enumerate(indices):
            abs_idx = int(sweep.start + local_idx)
            seeds.append(
                CVPeakSeed(
                    potential_v=float(e_seg[local_idx]),
                    current_a=float(i_seg[local_idx]),
                    sweep=sweep.direction,
                    process=process,
                    origin=ORIGIN_AUTOMATIC,
                    index=abs_idx,
                    prominence=(
                        float(prominences[position]) if prominences is not None else None
                    ),
                )
            )
    seeds.sort(key=lambda s: (s.index if s.index is not None else 0))
    return seeds


# --------------------------------------------------------------------------
# Baseline primitive
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CVBaseline:
    """A transparent local-linear peak-current baseline.

    Fitted through the recorded current at ONLY the caller-specified anchor
    row ranges, in the potential domain, then evaluable across any region.
    ``coefficients`` are ``numpy.polyfit`` order (highest power first), so
    :meth:`evaluate` is a plain ``numpy.polyval``.

    Deliberately NOT an automatic / opaque background removal: the anchors
    are an explicit decision by the caller (a human, or a later
    peak-avoiding heuristic -- never this class) about which points are
    background rather than peak.
    """

    method: str  # always "linear" in CV-1
    anchor_ranges: tuple[tuple[int, int], ...]
    coefficients: tuple[float, ...]

    def evaluate(self, potential) -> np.ndarray:
        """The baseline current at each value of ``potential`` (a new array)."""
        return np.polyval(np.asarray(self.coefficients, dtype=float), np.array(potential, dtype=float))


def local_linear_baseline(
    potential,
    current,
    anchor_ranges: list[tuple[int, int]],
) -> CVBaseline:
    """Fit a straight line through ``current`` at the ``anchor_ranges`` only.

    ``anchor_ranges`` is a list of ``(start, end)`` positional row ranges
    (end-exclusive) that the caller has decided represent background either
    side of a peak -- the classic foot-to-foot CV baseline construction.
    At least 2 distinct anchor points are required (a line).

    Returns a :class:`CVBaseline`; never mutates ``potential`` / ``current``
    and never touches any ``Dataset``.

    Raises :class:`InvalidCVInputError` for an empty / out-of-bounds
    selection, or fewer than 2 distinct points.
    """
    e, i = _finite_pair(potential, current)

    idx: set[int] = set()
    for rng in anchor_ranges:
        start, end = int(rng[0]), int(rng[1])
        if start < 0 or end > e.size or start >= end:
            raise InvalidCVInputError(
                f"anchor range ({start}, {end}) is out of bounds for {e.size} samples."
            )
        idx.update(range(start, end))
    if not idx:
        raise InvalidCVInputError("anchor_ranges selected no points.")

    anchor_idx = np.array(sorted(idx), dtype=int)
    anchor_e = e[anchor_idx]
    anchor_i = i[anchor_idx]
    if not (np.all(np.isfinite(anchor_e)) and np.all(np.isfinite(anchor_i))):
        raise InvalidCVInputError("anchor points contain non-finite data.")
    if np.unique(anchor_e).size < 2:
        raise InvalidCVInputError(
            "Need at least 2 anchor points at distinct potentials to fit a linear baseline."
        )

    coeffs = np.polyfit(anchor_e, anchor_i, 1)
    return CVBaseline(
        method="linear",
        anchor_ranges=tuple((int(a), int(b)) for a, b in anchor_ranges),
        coefficients=tuple(float(c) for c in coeffs),
    )


# --------------------------------------------------------------------------
# Peak measurement  (detection != measurement)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CVPeakMeasurement:
    """A quantitative peak measurement produced by :func:`measure_peak`.

    ``potential_v`` is ``Ep`` (``Epa`` for an anodic process, ``Epc`` for
    cathodic) -- the recorded potential at the current extremum, no
    sub-sample interpolation.

    ``i_peak_raw_a`` is the current at that extremum exactly as imported
    (signed). ``i_peak_corrected_a`` is ``i_peak_raw_a - baseline(Ep)``
    when a baseline was supplied, else ``None`` -- in which case
    ``i_peak_raw_a`` is a RAW EXTREMUM, not a baseline-corrected peak
    current, and must be labelled as such downstream.
    """

    potential_v: float
    i_peak_raw_a: float
    i_peak_corrected_a: float | None
    baseline_current_a: float | None
    process: str
    sweep: str
    index: int


def measure_peak(
    potential,
    current,
    *,
    search: tuple[int, int],
    process: str,
    convention: CurrentSignConvention = CurrentSignConvention.ANODIC_POSITIVE,
    baseline: CVBaseline | None = None,
) -> CVPeakMeasurement:
    """Locate and measure the ``process`` extremum within ``search``.

    ``search`` is a ``(start, end)`` positional row range (end-exclusive) --
    normally a single sweep, or a tighter window around a candidate.
    ``process`` is :data:`PROCESS_ANODIC` (oxidative extremum) or
    :data:`PROCESS_CATHODIC` (reductive extremum).

    The extremum is always taken from the UNSMOOTHED signal (CV-1 never
    smooths). When a ``baseline`` is given the extremum is located on the
    BASELINE-CORRECTED current -- ``Ep`` is where the faradaic contribution
    is largest, which on a sloping charging background is not where the raw
    current is largest. Without a baseline the raw extremum is returned and
    ``i_peak_corrected_a`` is ``None``: that value is a raw extremum, not a
    baseline-corrected Ipa/Ipc.

    Never mutates the inputs.
    """
    if process not in (PROCESS_ANODIC, PROCESS_CATHODIC):
        raise InvalidCVInputError(
            f"process must be {PROCESS_ANODIC!r} or {PROCESS_CATHODIC!r} to measure a peak "
            f"(got {process!r})."
        )
    e, i = _finite_pair(potential, current)
    start, end = int(search[0]), int(search[1])
    if start < 0 or end > e.size or start >= end:
        raise InvalidCVInputError(
            f"search range ({start}, {end}) is out of bounds for {e.size} samples."
        )

    e_win = e[start:end]
    i_win = i[start:end]
    if not (np.all(np.isfinite(e_win)) and np.all(np.isfinite(i_win))):
        raise InvalidCVInputError("search window contains non-finite data.")

    ox_sign = oxidative_sign(convention)
    process_sign = ox_sign if process == PROCESS_ANODIC else -ox_sign

    if baseline is not None:
        baseline_win = baseline.evaluate(e_win)
        target = (i_win - baseline_win) * process_sign
    else:
        baseline_win = None
        target = i_win * process_sign

    local = int(np.argmax(target))
    abs_idx = start + local

    ep = float(e_win[local])
    i_raw = float(i_win[local])
    if baseline is not None:
        bl = float(baseline_win[local])
        i_corr: float | None = i_raw - bl
    else:
        bl = None
        i_corr = None

    return CVPeakMeasurement(
        potential_v=ep,
        i_peak_raw_a=i_raw,
        i_peak_corrected_a=i_corr,
        baseline_current_a=bl,
        process=process,
        sweep="",  # sweep tag is a caller/result concern, not derivable here
        index=abs_idx,
    )


# --------------------------------------------------------------------------
# Couple metrics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CVCoupleMetrics:
    """Metrics for one anodic/cathodic peak couple.

    ``delta_ep_v`` and ``e_half_v`` are in canonical volts (mV display is a
    formatting concern for a later GUI/report layer).

    ``e_half_v`` is the MIDPOINT potential ``(Epa + Epc) / 2`` -- it is an
    estimate of the formal potential ``E°'`` only for a reversible couple,
    and this layer never calls it ``E°'``.

    Peak-current ratios preserve electrochemical identity: they are
    explicitly ``|Ipa| / |Ipc|`` and its reciprocal ``|Ipc| / |Ipa|`` --
    never an anonymous "forward / reverse". ``ratio_basis`` records whether
    they were computed from baseline-corrected currents
    (:data:`RATIO_BASIS_CORRECTED`, used only when BOTH peaks have one) or
    raw extrema (:data:`RATIO_BASIS_RAW`).
    """

    epa_v: float
    epc_v: float
    ipa_raw_a: float
    ipc_raw_a: float
    ipa_corrected_a: float | None
    ipc_corrected_a: float | None
    delta_ep_v: float
    e_half_v: float
    ratio_ipa_over_ipc: float | None
    ratio_ipc_over_ipa: float | None
    ratio_basis: str


def couple_metrics(anodic: CVPeakMeasurement, cathodic: CVPeakMeasurement) -> CVCoupleMetrics:
    """ΔEp, E½ (midpoint) and the anodic/cathodic peak-current ratios for a
    couple.

    Uses baseline-corrected currents for the ratios only when BOTH
    measurements carry one; otherwise falls back to raw extrema and says so
    via ``ratio_basis``. A ratio is ``None`` when its denominator is zero.
    """
    if anodic.process != PROCESS_ANODIC or cathodic.process != PROCESS_CATHODIC:
        raise InvalidCVInputError(
            "couple_metrics expects an anodic measurement and a cathodic measurement."
        )

    epa, epc = anodic.potential_v, cathodic.potential_v
    both_corrected = (
        anodic.i_peak_corrected_a is not None and cathodic.i_peak_corrected_a is not None
    )
    if both_corrected:
        ipa_mag = abs(anodic.i_peak_corrected_a)
        ipc_mag = abs(cathodic.i_peak_corrected_a)
        basis = RATIO_BASIS_CORRECTED
    else:
        ipa_mag = abs(anodic.i_peak_raw_a)
        ipc_mag = abs(cathodic.i_peak_raw_a)
        basis = RATIO_BASIS_RAW

    return CVCoupleMetrics(
        epa_v=epa,
        epc_v=epc,
        ipa_raw_a=anodic.i_peak_raw_a,
        ipc_raw_a=cathodic.i_peak_raw_a,
        ipa_corrected_a=anodic.i_peak_corrected_a,
        ipc_corrected_a=cathodic.i_peak_corrected_a,
        delta_ep_v=abs(epa - epc),
        e_half_v=(epa + epc) / 2.0,
        ratio_ipa_over_ipc=(ipa_mag / ipc_mag) if ipc_mag != 0 else None,
        ratio_ipc_over_ipa=(ipc_mag / ipa_mag) if ipa_mag != 0 else None,
        ratio_basis=basis,
    )
