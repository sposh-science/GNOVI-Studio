"""modules.raman.peaks: scipy.signal.find_peaks wrapper, synthetic-
spectrum validation (cases A-H, mirroring test_xrd_peaks.py's structure),
and the manual/automatic seed model."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from gnovi_plot.modules.raman.peaks import (
    ORIGIN_AUTOMATIC,
    ORIGIN_MANUAL,
    InvalidPeakDetectionError,
    RamanPeakSeed,
    detect_raman_peaks,
)


def _gaussian(x: np.ndarray, center: float, amplitude: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2))


# A typical Raman-shift range in cm^-1 -- values are illustrative synthetic
# data only, not a claim about any real material's spectrum.
RAMAN_SHIFT = np.linspace(100.0, 3200.0, 2000)


# --- A: isolated Gaussian peaks with known centers --------------------------


def test_a_isolated_peaks_are_detected_near_their_known_centers():
    centers = [520.0, 1350.0, 1580.0, 2700.0]  # D/G/2D-like spacing, illustrative only
    intensity = np.full_like(RAMAN_SHIFT, 10.0)
    for c in centers:
        intensity = intensity + _gaussian(RAMAN_SHIFT, c, 500.0, 4.0)

    peaks = detect_raman_peaks(RAMAN_SHIFT, intensity, prominence=100.0)

    assert len(peaks) == len(centers)
    detected_shifts = sorted(p.raman_shift for p in peaks)
    for expected, found in zip(centers, detected_shifts):
        # Detection tolerance: within the array's own sample spacing --
        # this is SEED detection, not a fitted center (see this file's
        # own module-under-test docstring), so a coarse index-level
        # tolerance is the correct bar, not a fit-grade one.
        spacing = RAMAN_SHIFT[1] - RAMAN_SHIFT[0]
        assert abs(found - expected) <= spacing


def test_a_peaks_are_returned_as_seeds_not_raw_indices():
    intensity = 10.0 + _gaussian(RAMAN_SHIFT, 1580.0, 500.0, 4.0)
    peaks = detect_raman_peaks(RAMAN_SHIFT, intensity, prominence=100.0)
    assert len(peaks) == 1
    assert isinstance(peaks[0], RamanPeakSeed)
    assert peaks[0].origin == ORIGIN_AUTOMATIC
    assert peaks[0].index is not None


# --- B: peaks of different intensity ----------------------------------------


def test_b_peaks_of_different_intensity_are_all_detected_with_low_enough_prominence():
    intensity = (
        10.0 + _gaussian(RAMAN_SHIFT, 1350.0, 100.0, 4.0) + _gaussian(RAMAN_SHIFT, 1580.0, 900.0, 4.0)
    )
    peaks = detect_raman_peaks(RAMAN_SHIFT, intensity, prominence=50.0)
    assert len(peaks) == 2
    intensities = sorted(p.intensity for p in peaks)
    assert intensities[0] < intensities[1]


# --- C: sloped/curved background (e.g. residual fluorescence) --------------


def test_c_a_peak_is_still_detected_on_top_of_a_sloped_background():
    background = 5.0 + 0.05 * RAMAN_SHIFT
    intensity = background + _gaussian(RAMAN_SHIFT, 1580.0, 300.0, 4.0)
    peaks = detect_raman_peaks(RAMAN_SHIFT, intensity, prominence=100.0)
    assert len(peaks) == 1
    assert peaks[0].raman_shift == pytest.approx(1580.0, abs=2.0)


# --- D: controlled noise, fixed seed -> deterministic -----------------------


def test_d_detection_is_deterministic_across_runs_with_fixed_noise():
    rng = np.random.default_rng(7)
    intensity = 10.0 + _gaussian(RAMAN_SHIFT, 1580.0, 400.0, 4.0) + rng.normal(0, 3.0, size=RAMAN_SHIFT.shape)

    first = detect_raman_peaks(RAMAN_SHIFT, intensity, prominence=100.0)
    second = detect_raman_peaks(RAMAN_SHIFT, intensity, prominence=100.0)

    assert [p.index for p in first] == [p.index for p in second]
    assert [p.raman_shift for p in first] == [p.raman_shift for p in second]


# --- E: closely spaced peaks (distance filtering) ---------------------------


def test_e_closely_spaced_peaks_are_both_found_without_a_distance_constraint():
    # Separation (8) to sigma (2.0) ratio of 4 -- enough for the dip
    # between the two Gaussians to register as two distinct local maxima
    # (mirrors test_xrd_peaks.py's own case E proportions).
    intensity = 10.0 + _gaussian(RAMAN_SHIFT, 1576.0, 400.0, 2.0) + _gaussian(RAMAN_SHIFT, 1584.0, 400.0, 2.0)
    peaks = detect_raman_peaks(RAMAN_SHIFT, intensity, prominence=50.0)
    assert len(peaks) == 2


def test_e_a_large_distance_constraint_merges_closely_spaced_peaks_into_one():
    intensity = 10.0 + _gaussian(RAMAN_SHIFT, 1576.0, 400.0, 2.0) + _gaussian(RAMAN_SHIFT, 1584.0, 400.0, 2.0)
    # 2000 points over 3100 cm^-1 -> ~0.65 points/cm^-1; a 500-sample
    # distance constraint (~325 cm^-1) forces only the taller candidate
    # to survive out of two peaks only 8 cm^-1 apart.
    peaks = detect_raman_peaks(RAMAN_SHIFT, intensity, prominence=10.0, distance=500)
    assert len(peaks) == 1


# --- F: weak peak near a strong peak (prominence filtering) -----------------


def test_f_weak_peak_near_a_strong_peak_is_excluded_by_prominence_but_found_at_a_lower_threshold():
    intensity = 10.0 + _gaussian(RAMAN_SHIFT, 1580.0, 800.0, 6.0) + _gaussian(RAMAN_SHIFT, 1620.0, 40.0, 3.0)

    strict = detect_raman_peaks(RAMAN_SHIFT, intensity, prominence=100.0)
    lenient = detect_raman_peaks(RAMAN_SHIFT, intensity, prominence=10.0)

    assert len(strict) == 1
    assert len(lenient) == 2


# --- optional height/width filters -------------------------------------------


def test_height_filter_excludes_a_peak_below_the_threshold():
    intensity = 10.0 + _gaussian(RAMAN_SHIFT, 1350.0, 30.0, 4.0) + _gaussian(RAMAN_SHIFT, 1580.0, 500.0, 4.0)

    unfiltered = detect_raman_peaks(RAMAN_SHIFT, intensity, prominence=5.0)
    filtered = detect_raman_peaks(RAMAN_SHIFT, intensity, prominence=5.0, height=100.0)

    assert len(unfiltered) == 2
    assert len(filtered) == 1
    assert filtered[0].intensity > 100.0


def test_width_filter_reports_a_detection_diagnostic_not_a_fitted_value():
    intensity = 10.0 + _gaussian(RAMAN_SHIFT, 1580.0, 500.0, 4.0)
    peaks = detect_raman_peaks(RAMAN_SHIFT, intensity, prominence=100.0, width=1.0)
    assert len(peaks) == 1
    # SciPy's find_peaks `width` output is an array-index diagnostic, not
    # a scientific measurement -- see this module's own docstring. Only
    # asserting it is present and numeric here; its value is not, and
    # must never be interpreted as, a fitted linewidth/FWHM.
    assert peaks[0].width_samples is not None
    assert peaks[0].width_samples > 0.0


# --- G: no-peak / flat signal -------------------------------------------------


def test_g_flat_signal_yields_no_peaks():
    intensity = np.full_like(RAMAN_SHIFT, 42.0)
    peaks = detect_raman_peaks(RAMAN_SHIFT, intensity, prominence=1.0)
    assert peaks == []


# --- H: NaN / non-finite input ------------------------------------------------


def test_h_non_finite_intensity_raises():
    intensity = np.full_like(RAMAN_SHIFT, 10.0)
    intensity[500] = float("nan")
    with pytest.raises(InvalidPeakDetectionError):
        detect_raman_peaks(RAMAN_SHIFT, intensity, prominence=1.0)


def test_h_non_finite_raman_shift_raises():
    raman_shift = RAMAN_SHIFT.copy()
    raman_shift[10] = float("inf")
    intensity = np.full_like(RAMAN_SHIFT, 10.0)
    with pytest.raises(InvalidPeakDetectionError):
        detect_raman_peaks(raman_shift, intensity, prominence=1.0)


def test_shape_mismatch_raises():
    with pytest.raises(InvalidPeakDetectionError):
        detect_raman_peaks(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_negative_distance_is_rejected():
    intensity = 10.0 + _gaussian(RAMAN_SHIFT, 1580.0, 400.0, 4.0)
    with pytest.raises(InvalidPeakDetectionError):
        detect_raman_peaks(RAMAN_SHIFT, intensity, distance=-5)


# --- manual seeds / enabled state / identity --------------------------------


def test_manual_seed_has_no_detection_metadata():
    seed = RamanPeakSeed.manual(raman_shift=1580.0, intensity=123.0)
    assert seed.origin == ORIGIN_MANUAL
    assert seed.index is None
    assert seed.prominence is None
    assert seed.enabled is True


def test_each_seed_has_a_stable_unique_id():
    a = RamanPeakSeed.manual(1580.0, 1.0)
    b = RamanPeakSeed.manual(1580.0, 1.0)
    assert a.id != b.id


def test_disabling_a_seed_keeps_it_in_the_list():
    seed = RamanPeakSeed.manual(1580.0, 1.0)
    seed.enabled = False
    assert seed.enabled is False
    assert seed.raman_shift == 1580.0  # still present, not removed


def test_seed_to_dict_from_dict_round_trip():
    seed = RamanPeakSeed(
        raman_shift=1580.0, intensity=200.0, origin=ORIGIN_AUTOMATIC, index=5, prominence=50.0,
        width_samples=3.2, enabled=False,
    )
    restored = RamanPeakSeed.from_dict(seed.to_dict())
    assert restored == seed


def test_seed_is_deepcopy_safe():
    seed = RamanPeakSeed.manual(1580.0, 1.0)
    cloned = copy.deepcopy(seed)
    assert cloned == seed
    assert cloned is not seed
    cloned.enabled = False
    assert seed.enabled is True
