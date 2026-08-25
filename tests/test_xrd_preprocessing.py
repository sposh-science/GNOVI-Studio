"""modules.xrd.preprocessing: polynomial baseline, arPLS baseline
(pybaselines), and optional Savitzky-Golay smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from gnovi_plot.modules.xrd import preprocessing as xrd_preprocessing
from gnovi_plot.modules.xrd.preprocessing import (
    InvalidPreprocessingError,
    PybaselinesNotAvailableError,
    arpls_baseline,
    polynomial_baseline,
    savgol_smooth,
)


def _gaussian(x: np.ndarray, center: float, amplitude: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2))


# --- polynomial_baseline -------------------------------------------------


def test_polynomial_baseline_recovers_a_known_linear_background():
    two_theta = np.linspace(10.0, 90.0, 400)
    true_baseline = 20.0 + 0.5 * two_theta  # a known, exact linear truth
    intensity = true_baseline + _gaussian(two_theta, 40.0, 500.0, 0.3)

    # Baseline points from well away from the single peak at 40 degrees.
    baseline_indices = list(range(0, 50)) + list(range(350, 400))
    result = polynomial_baseline(two_theta, intensity, baseline_indices, degree=1)

    np.testing.assert_allclose(result.baseline, true_baseline, atol=1e-8)
    np.testing.assert_allclose(result.corrected, intensity - true_baseline, atol=1e-8)
    assert result.method == "polynomial"


def test_polynomial_baseline_recovers_a_known_quadratic_background():
    two_theta = np.linspace(10.0, 90.0, 400)
    true_baseline = 15.0 + 0.2 * two_theta - 0.001 * two_theta**2
    intensity = true_baseline + _gaussian(two_theta, 55.0, 300.0, 0.25)

    baseline_indices = list(range(0, 60)) + list(range(340, 400))
    result = polynomial_baseline(two_theta, intensity, baseline_indices, degree=2)

    np.testing.assert_allclose(result.baseline, true_baseline, atol=1e-6)


def test_polynomial_baseline_does_not_mutate_inputs():
    two_theta = np.linspace(10.0, 90.0, 200)
    intensity = 10.0 + 0.1 * two_theta
    original_intensity = intensity.copy()

    result = polynomial_baseline(two_theta, intensity, list(range(0, 20)), degree=1)

    np.testing.assert_array_equal(intensity, original_intensity)
    assert result.baseline is not intensity
    assert result.corrected is not intensity


def test_polynomial_baseline_rejects_shape_mismatch():
    with pytest.raises(InvalidPreprocessingError):
        polynomial_baseline(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]), [0, 1])


def test_polynomial_baseline_rejects_non_finite_intensity():
    two_theta = np.array([1.0, 2.0, 3.0, 4.0])
    intensity = np.array([1.0, float("nan"), 3.0, 4.0])
    with pytest.raises(InvalidPreprocessingError):
        polynomial_baseline(two_theta, intensity, [0, 1, 2, 3])


def test_polynomial_baseline_rejects_empty_indices():
    two_theta = np.linspace(0, 10, 20)
    intensity = np.linspace(0, 10, 20)
    with pytest.raises(InvalidPreprocessingError):
        polynomial_baseline(two_theta, intensity, [], degree=1)


def test_polynomial_baseline_rejects_out_of_bounds_indices():
    two_theta = np.linspace(0, 10, 20)
    intensity = np.linspace(0, 10, 20)
    with pytest.raises(InvalidPreprocessingError):
        polynomial_baseline(two_theta, intensity, [0, 5, 100], degree=1)


def test_polynomial_baseline_rejects_too_few_points_for_degree():
    two_theta = np.linspace(0, 10, 20)
    intensity = np.linspace(0, 10, 20)
    # degree 3 needs at least 5 points (degree + 2); only 3 given.
    with pytest.raises(InvalidPreprocessingError):
        polynomial_baseline(two_theta, intensity, [0, 1, 2], degree=3)


def test_polynomial_baseline_rejects_negative_degree():
    two_theta = np.linspace(0, 10, 20)
    intensity = np.linspace(0, 10, 20)
    with pytest.raises(InvalidPreprocessingError):
        polynomial_baseline(two_theta, intensity, [0, 1, 2, 3, 4], degree=-1)


# --- arpls_baseline --------------------------------------------------------


def test_arpls_baseline_estimates_a_curved_background_within_tolerance():
    """arPLS is not exact-pointwise like the polynomial primitive (that's
    the whole point of a data-driven baseline) -- so this validates a
    realistic RMS-error tolerance against a KNOWN synthetic truth, not
    bit-for-bit equality."""
    two_theta = np.linspace(10.0, 90.0, 1000)
    true_baseline = 30.0 + 0.1 * two_theta + 0.002 * (two_theta - 50) ** 2
    intensity = true_baseline.copy()
    for center, amp, sigma in [(25.0, 400.0, 0.15), (45.0, 250.0, 0.12), (70.0, 600.0, 0.2)]:
        intensity = intensity + _gaussian(two_theta, center, amp, sigma)

    result = arpls_baseline(two_theta, intensity, lam=1e6)

    rms_error = np.sqrt(np.mean((result.baseline - true_baseline) ** 2))
    # Background varies by ~tens of intensity units over the range; a few
    # units of RMS error is a reasonable, documented tolerance for a
    # data-driven estimator recovering a smooth background under sharp
    # peaks, not an arbitrary pass threshold.
    assert rms_error < 5.0
    assert result.method == "arpls"


def test_arpls_baseline_does_not_mutate_inputs():
    two_theta = np.linspace(10.0, 90.0, 300)
    intensity = 20.0 + 0.05 * two_theta + _gaussian(two_theta, 50.0, 300.0, 0.2)
    original = intensity.copy()

    arpls_baseline(two_theta, intensity, lam=1e5)

    np.testing.assert_array_equal(intensity, original)


def test_arpls_baseline_is_deterministic():
    two_theta = np.linspace(10.0, 90.0, 300)
    intensity = 20.0 + 0.05 * two_theta + _gaussian(two_theta, 50.0, 300.0, 0.2)

    first = arpls_baseline(two_theta, intensity, lam=1e5)
    second = arpls_baseline(two_theta, intensity, lam=1e5)

    np.testing.assert_array_equal(first.baseline, second.baseline)


def test_arpls_baseline_broad_hump_limitation():
    """Documents a real scientific limitation rather than just coverage: a
    BROAD hump (e.g. an amorphous halo) with SHARP peaks on top is a case
    where an aggressive arPLS lam can eat into the hump. This test does
    not assert perfect recovery -- it demonstrates and records the
    direction of the error (lower lam follows the hump too closely; a
    much higher lam is needed to preserve it), which is exactly the risk
    PROJECT_GUIDE.md's XRD section calls out."""
    two_theta = np.linspace(10.0, 90.0, 1000)
    broad_hump = 200.0 * np.exp(-((two_theta - 50.0) ** 2) / (2 * 15.0**2))
    sharp_peak = _gaussian(two_theta, 50.0, 300.0, 0.15)
    intensity = 10.0 + broad_hump + sharp_peak

    aggressive = arpls_baseline(two_theta, intensity, lam=1e3)
    conservative = arpls_baseline(two_theta, intensity, lam=1e8)

    # A low lam follows the data more closely, so it removes more of the
    # broad hump's peak-region amplitude than a high lam does -- i.e. the
    # aggressive baseline sits closer to the hump's own peak than the
    # conservative one does at the hump center.
    hump_center_index = np.argmin(np.abs(two_theta - 50.0))
    assert aggressive.baseline[hump_center_index] > conservative.baseline[hump_center_index]


def test_arpls_baseline_raises_a_clear_error_without_pybaselines(monkeypatch):
    monkeypatch.setattr(xrd_preprocessing, "_PYBASELINES_AVAILABLE", False)
    two_theta = np.linspace(0, 10, 50)
    intensity = np.linspace(0, 10, 50)
    with pytest.raises(PybaselinesNotAvailableError):
        arpls_baseline(two_theta, intensity)


def test_arpls_baseline_rejects_shape_mismatch():
    with pytest.raises(InvalidPreprocessingError):
        arpls_baseline(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


# --- savgol_smooth -----------------------------------------------------------


def test_savgol_smooth_reduces_noise_on_a_known_flat_signal():
    rng = np.random.default_rng(1234)
    two_theta = np.linspace(0.0, 10.0, 500)
    true_signal = np.full_like(two_theta, 50.0)
    noisy = true_signal + rng.normal(0, 5.0, size=two_theta.shape)

    result = savgol_smooth(two_theta, noisy, window_length=21, polyorder=3)

    noisy_std = np.std(noisy - true_signal)
    smoothed_std = np.std(result.smoothed_intensity - true_signal)
    assert smoothed_std < noisy_std


def test_savgol_smooth_does_not_mutate_inputs():
    two_theta = np.linspace(0, 10, 100)
    intensity = np.sin(two_theta)
    original = intensity.copy()

    savgol_smooth(two_theta, intensity, window_length=9, polyorder=2)

    np.testing.assert_array_equal(intensity, original)


def test_savgol_smooth_is_deterministic():
    two_theta = np.linspace(0, 10, 100)
    intensity = np.sin(two_theta) + 0.1 * np.cos(3 * two_theta)

    first = savgol_smooth(two_theta, intensity, window_length=11, polyorder=2)
    second = savgol_smooth(two_theta, intensity, window_length=11, polyorder=2)

    np.testing.assert_array_equal(first.smoothed_intensity, second.smoothed_intensity)


def test_savgol_smooth_preserves_shape():
    two_theta = np.linspace(0, 10, 77)
    intensity = np.sin(two_theta)
    result = savgol_smooth(two_theta, intensity, window_length=7, polyorder=2)
    assert result.smoothed_intensity.shape == intensity.shape


def test_savgol_smooth_rejects_even_window_length():
    two_theta = np.linspace(0, 10, 50)
    intensity = np.sin(two_theta)
    with pytest.raises(InvalidPreprocessingError):
        savgol_smooth(two_theta, intensity, window_length=10, polyorder=2)


def test_savgol_smooth_rejects_polyorder_not_less_than_window():
    two_theta = np.linspace(0, 10, 50)
    intensity = np.sin(two_theta)
    with pytest.raises(InvalidPreprocessingError):
        savgol_smooth(two_theta, intensity, window_length=5, polyorder=5)


def test_savgol_smooth_rejects_window_larger_than_data():
    two_theta = np.linspace(0, 10, 9)
    intensity = np.sin(two_theta)
    with pytest.raises(InvalidPreprocessingError):
        savgol_smooth(two_theta, intensity, window_length=11, polyorder=2)


def test_savgol_smooth_does_not_silently_correct_an_even_window():
    """No hidden heuristic: an even window is a hard error, never bumped
    to the nearest odd value."""
    two_theta = np.linspace(0, 10, 50)
    intensity = np.sin(two_theta)
    with pytest.raises(InvalidPreprocessingError, match="odd"):
        savgol_smooth(two_theta, intensity, window_length=8, polyorder=2)


def test_smoothing_changes_apparent_peak_width_and_therefore_is_never_a_free_operation():
    """Scientific-consequence demonstration (not just coverage): smoothing
    a sharp peak measurably widens its apparent shape -- exactly why
    XRD-1's FWHM-bearing peak-fitting milestone must know whether smoothed
    data was used, and why this module never smooths automatically."""
    two_theta = np.linspace(40.0, 60.0, 400)
    intensity = _gaussian(two_theta, 50.0, 100.0, 0.15)

    smoothed = savgol_smooth(two_theta, intensity, window_length=41, polyorder=2).smoothed_intensity

    def half_max_width(y: np.ndarray) -> float:
        half = y.max() / 2.0
        above = np.where(y >= half)[0]
        return float(two_theta[above[-1]] - two_theta[above[0]])

    raw_width = half_max_width(intensity)
    smoothed_width = half_max_width(smoothed)
    # We deliberately do NOT assert these are close -- the point is that
    # heavy smoothing measurably changes the apparent width.
    assert smoothed_width != pytest.approx(raw_width, rel=1e-3)
