"""gnovi_plot.modules.electrochemistry.cv: cycle pairing, the peak
candidate model, sign-convention-aware detection, the local-linear
baseline primitive, quantitative peak measurement, and couple metrics.

Scientific ground truth is the explicit synthetic model in
``tests/data/generate_synthetic_cv.py`` -- the tests re-derive expected
peak potentials / ΔEp / E½ / peak currents from its documented constants,
never from a hard-coded textbook number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnovi_plot.modules.electrochemistry.common import (
    CurrentSignConvention,
    SWEEP_FALLING,
    SWEEP_RISING,
    segment_sweeps,
)
from gnovi_plot.modules.electrochemistry.cv import (
    ORIGIN_AUTOMATIC,
    ORIGIN_MANUAL,
    PARTIAL_SWEEP_FRACTION,
    PROCESS_ANODIC,
    PROCESS_CATHODIC,
    PROCESS_UNASSIGNED,
    RATIO_BASIS_CORRECTED,
    RATIO_BASIS_RAW,
    CVPeakSeed,
    InvalidCVInputError,
    couple_metrics,
    detect_cv_peaks,
    local_linear_baseline,
    measure_peak,
    pair_cycles,
)
from tests.data import generate_synthetic_cv as model

_DATA = Path(__file__).resolve().parent / "data"
_STEP = model.STEP


def _triangle(cycles, positive_first=True, lo=model.E_LOW, hi=model.E_HIGH):
    n = round((hi - lo) / _STEP)
    rising = np.round(np.linspace(lo, hi, n + 1), 10)
    falling = rising[::-1]
    first, second = (rising, falling) if positive_first else (falling, rising)
    legs = [first]
    for k in range(1, 2 * cycles):
        legs.append((second if k % 2 else first)[1:])
    return np.concatenate(legs)


# --- cycle pairing -----------------------------------------------------------


def test_pair_cycles_one_complete_rising_then_falling():
    cycles = pair_cycles(segment_sweeps(_triangle(1, positive_first=True)))
    assert len(cycles) == 1
    assert cycles[0].complete
    assert cycles[0].index == 1
    assert cycles[0].rising_sweep is not None and cycles[0].falling_sweep is not None


def test_pair_cycles_one_complete_falling_then_rising():
    cycles = pair_cycles(segment_sweeps(_triangle(1, positive_first=False)))
    assert len(cycles) == 1
    assert cycles[0].complete


def test_pair_cycles_n_identical_cycles_yields_n_complete_cycles():
    for n in (2, 3, 5):
        cycles = pair_cycles(segment_sweeps(_triangle(n)))
        assert len(cycles) == n
        assert all(c.complete for c in cycles)
        assert [c.index for c in cycles] == list(range(1, n + 1))


def test_pair_cycles_truncated_trailing_sweep_is_flagged_incomplete():
    e = np.concatenate([_triangle(1), _triangle(1)[1 : 1 + 400]])  # + a partial rising leg
    cycles = pair_cycles(segment_sweeps(e))
    assert len(cycles) == 2
    assert cycles[0].complete
    assert not cycles[1].complete  # lone trailing sweep


def test_pair_cycles_truncated_leading_sweep_is_flagged_incomplete():
    full = _triangle(2)
    e = full[1200:]  # start partway down the second falling leg
    cycles = pair_cycles(segment_sweeps(e))
    assert not cycles[0].complete  # first sweep is a stub, span < fraction of a full sweep
    assert any(c.complete for c in cycles[1:])


def test_pair_cycles_stub_threshold_uses_widest_sweep():
    segs = segment_sweeps(_triangle(2))
    widest = max(s.potential_span for s in segs)
    # every full sweep spans the whole window, so none is a stub
    assert all(s.potential_span >= PARTIAL_SWEEP_FRACTION * widest for s in segs)


def test_pair_cycles_empty_input():
    assert pair_cycles([]) == []


# --- CVPeakSeed model ------------------------------------------------------


def test_cv_peak_seed_manual_factory():
    seed = CVPeakSeed.manual(0.25, 1e-5, sweep=SWEEP_RISING, process=PROCESS_ANODIC)
    assert seed.origin == ORIGIN_MANUAL
    assert seed.index is None
    assert seed.prominence is None
    assert seed.enabled is True
    assert seed.id  # stable id assigned


def test_cv_peak_seed_round_trips_through_dict():
    seed = CVPeakSeed(potential_v=0.25, current_a=1e-5, sweep=SWEEP_FALLING,
                      process=PROCESS_CATHODIC, origin=ORIGIN_AUTOMATIC, index=42, prominence=3e-6,
                      enabled=False)
    restored = CVPeakSeed.from_dict(seed.to_dict())
    assert restored == seed


def test_cv_peak_seed_enabled_flag_is_soft_exclude():
    seed = CVPeakSeed.manual(0.1, 1e-6)
    seed.enabled = False
    assert seed.potential_v == 0.1  # data kept, just excluded


# --- detection -----------------------------------------------------------


def _reversible_arrays():
    df = pd.read_csv(_DATA / "synthetic_cv_reversible.csv")
    return df["Potential/V"].to_numpy(), df["Current/A"].to_numpy()


def test_detect_runs_per_sweep_and_finds_the_anodic_wave_on_the_rising_sweep():
    e, i = _reversible_arrays()
    rising = next(s for s in segment_sweeps(e) if s.direction == SWEEP_RISING)
    seeds = detect_cv_peaks(e, i, rising, prominence=1e-6)
    anodic = [s for s in seeds if s.process == PROCESS_ANODIC]
    assert anodic
    best = max(anodic, key=lambda s: s.current_a)
    assert best.potential_v == pytest.approx(model.EPA_TRUE, abs=2 * _STEP)
    assert best.sweep == SWEEP_RISING
    assert best.origin == ORIGIN_AUTOMATIC


def test_detect_finds_the_cathodic_wave_on_the_falling_sweep():
    e, i = _reversible_arrays()
    falling = next(s for s in segment_sweeps(e) if s.direction == SWEEP_FALLING)
    seeds = detect_cv_peaks(e, i, falling, prominence=1e-6)
    cathodic = [s for s in seeds if s.process == PROCESS_CATHODIC]
    assert cathodic
    best = min(cathodic, key=lambda s: s.current_a)
    assert best.potential_v == pytest.approx(model.EPC_TRUE, abs=2 * _STEP)
    assert best.sweep == SWEEP_FALLING


def test_detect_process_is_independent_of_sweep_direction():
    """A reductive bump placed on a RISING sweep must be tagged cathodic,
    not anodic -- rising != anodic."""
    e = _triangle(1, positive_first=True)
    rising = np.gradient(e) > 0
    i = 1e-6 - 3e-5 * np.exp(-((e - 0.2) / 0.02) ** 2) * rising  # reductive spike while E rises
    seg = next(s for s in segment_sweeps(e) if s.direction == SWEEP_RISING)
    seeds = detect_cv_peaks(e, i, seg, prominence=1e-6)
    strong = [s for s in seeds if abs(s.current_a) > 1e-5]
    assert strong
    assert all(s.process == PROCESS_CATHODIC for s in strong)
    assert all(s.sweep == SWEEP_RISING for s in strong)


def test_detect_cathodic_positive_convention_flips_assignment():
    e, i = _reversible_arrays()
    rising = next(s for s in segment_sweeps(e) if s.direction == SWEEP_RISING)
    # Under CATHODIC_POSITIVE, the positive-going wave on the rising sweep is
    # a REDUCTION current -> tagged cathodic.
    seeds = detect_cv_peaks(e, i, rising, convention=CurrentSignConvention.CATHODIC_POSITIVE,
                            prominence=1e-6)
    strong = max(seeds, key=lambda s: abs(s.current_a))
    assert strong.process == PROCESS_CATHODIC


def test_detect_prominence_filters_candidates():
    e, i = _reversible_arrays()
    rising = next(s for s in segment_sweeps(e) if s.direction == SWEEP_RISING)
    loose = detect_cv_peaks(e, i, rising, prominence=1e-9)
    tight = detect_cv_peaks(e, i, rising, prominence=5e-6)
    assert len(tight) <= len(loose)


def test_detect_rejects_out_of_bounds_sweep_and_non_finite():
    e, i = _reversible_arrays()
    seg = segment_sweeps(e)[0]
    bad = type(seg)(start=0, end=len(e) + 10, direction=seg.direction,
                    e_start=seg.e_start, e_end=seg.e_end)
    with pytest.raises(InvalidCVInputError):
        detect_cv_peaks(e, i, bad, prominence=1e-6)
    i2 = i.copy()
    i2[seg.start + 3] = np.inf
    with pytest.raises(InvalidCVInputError):
        detect_cv_peaks(e, i2, seg, prominence=1e-6)


def test_detect_does_not_mutate_inputs():
    e, i = _reversible_arrays()
    e0, i0 = e.copy(), i.copy()
    detect_cv_peaks(e, i, segment_sweeps(e)[0], prominence=1e-6)
    np.testing.assert_array_equal(e, e0)
    np.testing.assert_array_equal(i, i0)


# --- baseline primitive --------------------------------------------------


def test_local_linear_baseline_recovers_a_known_line():
    e = np.linspace(-0.2, 0.6, 801)
    true_slope, true_intercept = 3.0e-5, 1.0e-6
    i = true_slope * e + true_intercept
    bl = local_linear_baseline(e, i, [(0, 50), (750, 801)])
    assert bl.method == "linear"
    got = bl.evaluate(e)
    np.testing.assert_allclose(got, i, rtol=1e-9, atol=1e-12)


def test_local_linear_baseline_requires_two_distinct_points():
    e = np.linspace(-0.2, 0.6, 100)
    i = np.ones_like(e)
    with pytest.raises(InvalidCVInputError):
        local_linear_baseline(e, i, [(10, 11)])  # single point
    with pytest.raises(InvalidCVInputError):
        local_linear_baseline(e, i, [])
    with pytest.raises(InvalidCVInputError):
        local_linear_baseline(e, i, [(0, 5), (200, 260)])  # out of bounds


def test_local_linear_baseline_does_not_mutate_inputs():
    e = np.linspace(-0.2, 0.6, 200)
    i = 2e-5 * e
    e0, i0 = e.copy(), i.copy()
    local_linear_baseline(e, i, [(0, 20), (180, 200)])
    np.testing.assert_array_equal(e, e0)
    np.testing.assert_array_equal(i, i0)


# --- measurement:  raw extremum vs baseline-corrected peak current -------


def test_measure_peak_raw_extremum_when_no_baseline():
    e, i = _reversible_arrays()
    rising = next(s for s in segment_sweeps(e) if s.direction == SWEEP_RISING)
    m = measure_peak(e, i, search=(rising.start, rising.end), process=PROCESS_ANODIC)
    assert m.i_peak_corrected_a is None  # explicitly a raw extremum
    assert m.baseline_current_a is None
    assert m.potential_v == pytest.approx(model.EPA_TRUE, abs=2 * _STEP)
    # raw extremum includes the flat background offset
    assert m.i_peak_raw_a == pytest.approx(model.PEAK_AMPLITUDE_A + model.FLAT_BACKGROUND_A, rel=1e-3)


def test_measure_peak_baseline_corrected_recovers_true_faradaic_current_on_sloped_background():
    """Fixture B: a real sloping charging background. The RAW extremum is
    badly biased; a local-linear baseline recovers the known faradaic peak
    current within tolerance."""
    df = pd.read_csv(_DATA / "synthetic_cv_sloped_background.csv")
    e, i = df["Potential/V"].to_numpy(), df["Current/A"].to_numpy()
    rising = next(s for s in segment_sweeps(e) if s.direction == SWEEP_RISING)

    m_raw = measure_peak(e, i, search=(rising.start, rising.end), process=PROCESS_ANODIC)
    # the raw extremum sits near Epa but is inflated by the charging
    # background under the peak -- it is NOT the true faradaic peak current
    bg_at_epa = model.SLOPE_A_PER_V * (model.EPA_TRUE - model.E_LOW)
    assert m_raw.potential_v == pytest.approx(model.EPA_TRUE, abs=0.02)
    assert m_raw.i_peak_raw_a == pytest.approx(model.PEAK_AMPLITUDE_A + bg_at_epa, rel=0.1)
    assert m_raw.i_peak_raw_a > 1.3 * model.PEAK_AMPLITUDE_A  # meaningfully biased

    # anchors: a pre-peak window and a post-peak window on the rising sweep,
    # both well away from Epa (~5 sigma)
    peak_pos = int(np.argmin(np.abs(e[rising.start:rising.end] - model.EPA_TRUE))) + rising.start
    pre = (rising.start + 20, peak_pos - 150)
    post = (peak_pos + 150, rising.end - 20)
    bl = local_linear_baseline(e, i, [pre, post])
    m_corr = measure_peak(e, i, search=(rising.start, rising.end), process=PROCESS_ANODIC, baseline=bl)

    assert m_corr.i_peak_corrected_a is not None
    assert m_corr.potential_v == pytest.approx(model.EPA_TRUE, abs=0.01)
    assert m_corr.i_peak_corrected_a == pytest.approx(model.PEAK_AMPLITUDE_A, rel=0.05)
    # the correction genuinely mattered
    assert abs(m_corr.i_peak_raw_a - m_corr.i_peak_corrected_a) > 0.3 * model.PEAK_AMPLITUDE_A


def test_measure_peak_cathodic_process():
    e, i = _reversible_arrays()
    falling = next(s for s in segment_sweeps(e) if s.direction == SWEEP_FALLING)
    m = measure_peak(e, i, search=(falling.start, falling.end), process=PROCESS_CATHODIC)
    assert m.potential_v == pytest.approx(model.EPC_TRUE, abs=2 * _STEP)
    assert m.i_peak_raw_a < 0  # reductive current is negative in the anodic-positive fixture


def test_measure_peak_rejects_bad_process_and_range():
    e, i = _reversible_arrays()
    with pytest.raises(InvalidCVInputError):
        measure_peak(e, i, search=(0, 100), process=PROCESS_UNASSIGNED)
    with pytest.raises(InvalidCVInputError):
        measure_peak(e, i, search=(100, 50), process=PROCESS_ANODIC)


def test_measure_peak_does_not_mutate_inputs():
    e, i = _reversible_arrays()
    e0, i0 = e.copy(), i.copy()
    measure_peak(e, i, search=(0, 800), process=PROCESS_ANODIC)
    np.testing.assert_array_equal(e, e0)
    np.testing.assert_array_equal(i, i0)


# --- couple metrics ----------------------------------------------------


def _reversible_couple(baseline=False):
    e, i = _reversible_arrays()
    segs = segment_sweeps(e)
    rising = next(s for s in segs if s.direction == SWEEP_RISING)
    falling = next(s for s in segs if s.direction == SWEEP_FALLING)
    bl_a = bl_c = None
    if baseline:
        # flat-background fixture -> a 2-point flat baseline recovers amplitude
        bl_a = local_linear_baseline(e, i, [(rising.start + 10, rising.start + 60),
                                            (rising.end - 60, rising.end - 10)])
        bl_c = local_linear_baseline(e, i, [(falling.start + 10, falling.start + 60),
                                            (falling.end - 60, falling.end - 10)])
    a = measure_peak(e, i, search=(rising.start, rising.end), process=PROCESS_ANODIC, baseline=bl_a)
    c = measure_peak(e, i, search=(falling.start, falling.end), process=PROCESS_CATHODIC, baseline=bl_c)
    return a, c


def test_couple_metrics_delta_ep_and_e_half_match_the_model():
    a, c = _reversible_couple()
    m = couple_metrics(a, c)
    # independently derived from the fixture constants
    assert m.delta_ep_v == pytest.approx(model.DELTA_EP_TRUE, abs=2 * _STEP)
    assert m.e_half_v == pytest.approx(model.E_HALF_TRUE, abs=_STEP)
    assert m.delta_ep_v == abs(m.epa_v - m.epc_v)
    assert m.e_half_v == (m.epa_v + m.epc_v) / 2


def test_couple_metrics_ratio_is_labelled_anodic_over_cathodic_not_forward_reverse():
    a, c = _reversible_couple(baseline=True)
    m = couple_metrics(a, c)
    assert m.ratio_basis == RATIO_BASIS_CORRECTED
    # a genuine reversible couple: |Ipa|/|Ipc| ~ 1, and the reciprocal is its inverse
    assert m.ratio_ipa_over_ipc == pytest.approx(1.0, rel=0.05)
    assert m.ratio_ipc_over_ipa == pytest.approx(1.0 / m.ratio_ipa_over_ipc, rel=1e-9)


def test_couple_metrics_falls_back_to_raw_basis_when_a_baseline_is_missing():
    a, c = _reversible_couple(baseline=False)
    m = couple_metrics(a, c)
    assert m.ratio_basis == RATIO_BASIS_RAW
    assert m.ipa_corrected_a is None and m.ipc_corrected_a is None


def test_couple_metrics_rejects_swapped_processes():
    a, c = _reversible_couple()
    with pytest.raises(InvalidCVInputError):
        couple_metrics(c, a)  # cathodic passed as anodic


# --- fixture validation: Randles-Sevcik scaling Ip ∝ sqrt(v) -------------
# This validates the SYNTHETIC MODEL's scaling assumption only; it does NOT
# exercise any CV-3 production code (no scan_rate.py, no regression object,
# no diffusion coefficient).


def test_real_ferricyanide_export_sanity_non_golden():
    """NON-GOLDEN sanity check on the bundled real CHI660E ferricyanide
    export (0.1 V/s). The algorithm is NOT tuned to reproduce textbook
    numbers for this one dataset -- this only asserts import -> arrays ->
    segmentation -> candidate detection all run without crashing and
    without mutating the source. Synthetic/analytical tests are the
    scientific ground truth."""
    from gnovi_plot.data.importers.text_importer import import_table

    df = import_table(_DATA / "real_cv_export.csv").dataframe
    e = df["Potential/V"].to_numpy()
    i = df["Current/A"].to_numpy()
    e0, i0 = e.copy(), i.copy()

    segs = segment_sweeps(e)
    assert len(segs) == 6  # 6 CHI segments -> 6 sweeps
    cycles = pair_cycles(segs)
    assert len(cycles) == 3 and all(c.complete for c in cycles)

    # a plausible, prominence-filtered candidate detection just runs
    mad = 1.4826 * np.median(np.abs(np.diff(i) - np.median(np.diff(i))))
    for seg in segs:
        seeds = detect_cv_peaks(e, i, seg, prominence=5 * mad)
        assert isinstance(seeds, list)

    np.testing.assert_array_equal(e, e0)
    np.testing.assert_array_equal(i, i0)


def test_synthetic_peak_current_scales_with_sqrt_scan_rate():
    scan_rates = np.array([0.01, 0.02, 0.05, 0.1, 0.2])  # V/s
    e = _triangle(1, positive_first=True)
    rising_mask = np.gradient(e) > 0
    seg = next(s for s in segment_sweeps(e) if s.direction == SWEEP_RISING)
    ip = []
    for v in scan_rates:
        # model: peak amplitude proportional to sqrt(v) (Randles-Sevcik form)
        amp = model.PEAK_AMPLITUDE_A * np.sqrt(v / 0.1)
        i = model.FLAT_BACKGROUND_A + amp * np.exp(
            -((e - model.EPA_TRUE) / model.PEAK_SIGMA) ** 2 / 2
        ) * rising_mask
        m = measure_peak(e, i, search=(seg.start, seg.end), process=PROCESS_ANODIC)
        ip.append(m.i_peak_raw_a - model.FLAT_BACKGROUND_A)
    ip = np.array(ip)
    # slope of Ip vs sqrt(v) recovers the model's proportionality constant,
    # and the fit is essentially perfect (noise-free)
    slope, intercept = np.polyfit(np.sqrt(scan_rates), ip, 1)
    predicted = slope * np.sqrt(scan_rates) + intercept
    ss_res = np.sum((ip - predicted) ** 2)
    ss_tot = np.sum((ip - ip.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot
    assert r_squared > 0.9999
    assert slope == pytest.approx(model.PEAK_AMPLITUDE_A / np.sqrt(0.1), rel=1e-3)
