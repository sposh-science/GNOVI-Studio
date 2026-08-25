"""modules.xrd.peaks: scipy.signal.find_peaks wrapper, synthetic-pattern
validation (cases A-H), and the manual/automatic seed model."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from gnovi_plot.modules.xrd.peaks import (
    ORIGIN_AUTOMATIC,
    ORIGIN_MANUAL,
    InvalidPeakDetectionError,
    XRDPeakSeed,
    detect_peaks,
)


def _gaussian(x: np.ndarray, center: float, amplitude: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2))


TWO_THETA = np.linspace(10.0, 90.0, 2000)


# --- A: isolated Gaussian peaks with known centers --------------------------


def test_a_isolated_peaks_are_detected_near_their_known_centers():
    centers = [20.0, 40.0, 60.0, 80.0]
    intensity = np.full_like(TWO_THETA, 10.0)
    for c in centers:
        intensity = intensity + _gaussian(TWO_THETA, c, 500.0, 0.2)

    peaks = detect_peaks(TWO_THETA, intensity, prominence=100.0)

    assert len(peaks) == len(centers)
    detected_centers = sorted(p.two_theta for p in peaks)
    for expected, found in zip(centers, detected_centers):
        # Detection tolerance: within the array's own sample spacing --
        # this is SEED detection, not a fitted center (see this file's
        # own module-under-test docstring), so a coarse index-level
        # tolerance is the correct bar, not a fit-grade one.
        spacing = TWO_THETA[1] - TWO_THETA[0]
        assert abs(found - expected) <= spacing


def test_a_peaks_are_returned_as_seeds_not_raw_indices():
    intensity = 10.0 + _gaussian(TWO_THETA, 40.0, 500.0, 0.2)
    peaks = detect_peaks(TWO_THETA, intensity, prominence=100.0)
    assert len(peaks) == 1
    assert isinstance(peaks[0], XRDPeakSeed)
    assert peaks[0].origin == ORIGIN_AUTOMATIC
    assert peaks[0].index is not None


# --- B: peaks of different intensity ----------------------------------------


def test_b_peaks_of_different_intensity_are_all_detected_with_low_enough_prominence():
    intensity = 10.0 + _gaussian(TWO_THETA, 25.0, 100.0, 0.2) + _gaussian(TWO_THETA, 55.0, 900.0, 0.2)
    peaks = detect_peaks(TWO_THETA, intensity, prominence=50.0)
    assert len(peaks) == 2
    intensities = sorted(p.intensity for p in peaks)
    assert intensities[0] < intensities[1]


# --- C: sloped/curved background --------------------------------------------


def test_c_a_peak_is_still_detected_on_top_of_a_sloped_background():
    background = 5.0 + 0.8 * TWO_THETA
    intensity = background + _gaussian(TWO_THETA, 50.0, 300.0, 0.2)
    peaks = detect_peaks(TWO_THETA, intensity, prominence=100.0)
    assert len(peaks) == 1
    assert peaks[0].two_theta == pytest.approx(50.0, abs=0.1)


# --- D: controlled noise, fixed seed -> deterministic -----------------------


def test_d_detection_is_deterministic_across_runs_with_fixed_noise():
    rng = np.random.default_rng(7)
    intensity = 10.0 + _gaussian(TWO_THETA, 45.0, 400.0, 0.2) + rng.normal(0, 3.0, size=TWO_THETA.shape)

    first = detect_peaks(TWO_THETA, intensity, prominence=100.0)
    second = detect_peaks(TWO_THETA, intensity, prominence=100.0)

    assert [p.index for p in first] == [p.index for p in second]
    assert [p.two_theta for p in first] == [p.two_theta for p in second]


# --- E: closely spaced peaks -------------------------------------------------


def test_e_closely_spaced_peaks_are_both_found_without_a_distance_constraint():
    intensity = 10.0 + _gaussian(TWO_THETA, 49.8, 400.0, 0.1) + _gaussian(TWO_THETA, 50.2, 400.0, 0.1)
    peaks = detect_peaks(TWO_THETA, intensity, prominence=50.0)
    assert len(peaks) == 2


def test_e_a_large_distance_constraint_merges_closely_spaced_peaks_into_one():
    intensity = 10.0 + _gaussian(TWO_THETA, 49.8, 400.0, 0.1) + _gaussian(TWO_THETA, 50.2, 400.0, 0.1)
    # ~2000 points over 80 degrees -> 25 points/degree; 0.4 degree
    # separation is ~10 samples, so a distance of 500 samples (~20
    # degrees) forces only the taller of the two candidates to survive.
    peaks = detect_peaks(TWO_THETA, intensity, prominence=10.0, distance=500)
    assert len(peaks) == 1


# --- F: weak peak near a strong peak -----------------------------------------


def test_f_weak_peak_near_a_strong_peak_is_excluded_by_prominence_but_found_at_a_lower_threshold():
    intensity = 10.0 + _gaussian(TWO_THETA, 50.0, 800.0, 0.3) + _gaussian(TWO_THETA, 51.5, 40.0, 0.15)

    strict = detect_peaks(TWO_THETA, intensity, prominence=100.0)
    lenient = detect_peaks(TWO_THETA, intensity, prominence=10.0)

    assert len(strict) == 1
    assert len(lenient) == 2


# --- G: no-peak / flat signal -------------------------------------------------


def test_g_flat_signal_yields_no_peaks():
    intensity = np.full_like(TWO_THETA, 42.0)
    peaks = detect_peaks(TWO_THETA, intensity, prominence=1.0)
    assert peaks == []


# --- H: NaN / non-finite input ------------------------------------------------


def test_h_non_finite_intensity_raises():
    intensity = np.full_like(TWO_THETA, 10.0)
    intensity[500] = float("nan")
    with pytest.raises(InvalidPeakDetectionError):
        detect_peaks(TWO_THETA, intensity, prominence=1.0)


def test_h_non_finite_two_theta_raises():
    two_theta = TWO_THETA.copy()
    two_theta[10] = float("inf")
    intensity = np.full_like(TWO_THETA, 10.0)
    with pytest.raises(InvalidPeakDetectionError):
        detect_peaks(two_theta, intensity, prominence=1.0)


def test_shape_mismatch_raises():
    with pytest.raises(InvalidPeakDetectionError):
        detect_peaks(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_negative_distance_is_rejected():
    intensity = 10.0 + _gaussian(TWO_THETA, 50.0, 400.0, 0.2)
    with pytest.raises(InvalidPeakDetectionError):
        detect_peaks(TWO_THETA, intensity, distance=-5)


# --- manual seeds / enabled state / identity --------------------------------


def test_manual_seed_has_no_detection_metadata():
    seed = XRDPeakSeed.manual(two_theta=42.0, intensity=123.0)
    assert seed.origin == ORIGIN_MANUAL
    assert seed.index is None
    assert seed.prominence is None
    assert seed.enabled is True


def test_each_seed_has_a_stable_unique_id():
    a = XRDPeakSeed.manual(10.0, 1.0)
    b = XRDPeakSeed.manual(10.0, 1.0)
    assert a.id != b.id


def test_disabling_a_seed_keeps_it_in_the_list():
    seed = XRDPeakSeed.manual(10.0, 1.0)
    seed.enabled = False
    assert seed.enabled is False
    assert seed.two_theta == 10.0  # still present, not removed


def test_seed_to_dict_from_dict_round_trip():
    seed = XRDPeakSeed(
        two_theta=30.0, intensity=200.0, origin=ORIGIN_AUTOMATIC, index=5, prominence=50.0,
        width_samples=3.2, enabled=False,
    )
    restored = XRDPeakSeed.from_dict(seed.to_dict())
    assert restored == seed


def test_seed_is_deepcopy_safe():
    seed = XRDPeakSeed.manual(10.0, 1.0)
    cloned = copy.deepcopy(seed)
    assert cloned == seed
    assert cloned is not seed
    cloned.enabled = False
    assert seed.enabled is True
