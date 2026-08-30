"""modules.xrd.fitting: single-peak XRD profile fitting.

Synthetic ground-truth validation for the three area-normalized profiles
(Gaussian / Lorentzian / pseudo-Voigt), the local-baseline term, the fit
window helper, uncertainty behaviour, d-spacing propagation, warnings,
failure handling, deterministic curve regeneration, and serialization.

Tolerances are tied to the method: closed-form profiles are recovered from
noiseless data to ~1e-3 relative (curve_fit precision, not machine
epsilon); noisy recoveries use a stated seed / SNR / point count and
correspondingly looser bounds. Lorentzian normalization is checked with
scipy.integrate.quad over an infinite domain so heavy tails are not
truncated.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from gnovi_plot.analysis.results import ResidualData, result_from_dict
from gnovi_plot.modules.xrd.fitting import (
    BASELINE_CONSTANT,
    BASELINE_LINEAR,
    BASELINE_NONE,
    FWHM_UNITS_TWO_THETA_DEG,
    GAUSSIAN,
    LORENTZIAN,
    OPERATION_PEAK_FIT,
    PSEUDO_VOIGT,
    FitWindow,
    XRDFitError,
    XRDPeakFitResult,
    _standard_errors,
    baseline_values,
    derived_height,
    estimate_seed_fwhm,
    evaluate_baseline,
    evaluate_peak_component,
    evaluate_total,
    fit_xrd_peak,
    gaussian_normalized,
    lorentzian_normalized,
    peak_component,
    propose_fit_window,
    pseudo_voigt_normalized,
    sample_fit_curve,
)
from gnovi_plot.modules.xrd.peaks import XRDPeakSeed, detect_peaks
from gnovi_plot.modules.xrd.radiation import radiation_from_preset

_PROV = dict(source_dataset_id="dataset-1", x_column="2theta", y_column="counts")


def _synthetic(model, *, area, center, fwhm, eta=None, baseline=BASELINE_NONE,
               c0=0.0, c1=0.0, x=None, noise=0.0, seed=0):
    """Exact model signal on grid `x`, optionally + deterministic noise."""
    if x is None:
        x = np.linspace(center - 3.0, center + 3.0, 1200)
    x_ref = 0.5 * (x[0] + x[-1])
    y = peak_component(x, model, area, center, fwhm, eta) + baseline_values(x, baseline, x_ref, c0, c1)
    if noise > 0.0:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(0.0, noise, size=x.shape)
    return x, y


# =====================================================================
# 1. Normalization and FWHM of the area-normalized profiles
# =====================================================================


def _integral_over_real_line(f, center):
    """`quad` split at `center` so the peak is a finite endpoint of each half
    (the standard trick for a sharp feature far from the origin / for the
    Lorentzian's slowly-decaying tails)."""
    left, _e1 = quad(f, -np.inf, center)
    right, _e2 = quad(f, center, np.inf)
    return left + right


def test_gaussian_normalized_integrates_to_one():
    assert _integral_over_real_line(lambda x: gaussian_normalized(x, 12.3, 0.37), 12.3) == pytest.approx(1.0, abs=1e-6)


def test_lorentzian_normalized_integrates_to_one():
    assert _integral_over_real_line(lambda x: lorentzian_normalized(x, 12.3, 0.37), 12.3) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("eta", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_pseudo_voigt_normalized_integrates_to_one_for_every_eta(eta):
    assert _integral_over_real_line(lambda x: pseudo_voigt_normalized(x, 5.0, 0.4, eta), 5.0) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "profile",
    [
        lambda x: gaussian_normalized(x, 0.0, 0.4),
        lambda x: lorentzian_normalized(x, 0.0, 0.4),
        lambda x: pseudo_voigt_normalized(x, 0.0, 0.4, 0.3),
        lambda x: pseudo_voigt_normalized(x, 0.0, 0.4, 0.9),
    ],
)
def test_numerical_fwhm_equals_gamma(profile):
    peak = profile(np.array([0.0]))[0]
    hwhm = brentq(lambda x: profile(np.array([x]))[0] - 0.5 * peak, 1e-6, 5.0)
    assert 2.0 * hwhm == pytest.approx(0.4, rel=1e-6)


def test_pseudo_voigt_eta_0_is_gaussian_and_eta_1_is_lorentzian():
    x = np.linspace(-3, 3, 501)
    assert np.allclose(pseudo_voigt_normalized(x, 0.0, 0.5, 0.0), gaussian_normalized(x, 0.0, 0.5), atol=1e-12)
    assert np.allclose(pseudo_voigt_normalized(x, 0.0, 0.5, 1.0), lorentzian_normalized(x, 0.0, 0.5), atol=1e-12)


@pytest.mark.parametrize("model,eta", [(GAUSSIAN, None), (LORENTZIAN, None), (PSEUDO_VOIGT, 0.4)])
def test_derived_height_matches_the_model_maximum(model, eta):
    area, fwhm = 137.0, 0.31
    at_center = float(peak_component(np.array([2.0]), model, area, 2.0, fwhm, eta)[0])
    assert derived_height(model, area, fwhm, eta) == pytest.approx(at_center, rel=1e-12)


# =====================================================================
# 2. Parameter recovery from noiseless synthetic data
# =====================================================================


@pytest.mark.parametrize("model,eta", [(GAUSSIAN, None), (LORENTZIAN, None), (PSEUDO_VOIGT, 0.35)])
def test_noiseless_recovery_no_baseline(model, eta):
    x, y = _synthetic(model, area=420.0, center=33.0, fwhm=0.28, eta=eta)
    res = fit_xrd_peak(x, y, model, fit_window=(31.0, 35.0), baseline=BASELINE_NONE, **_PROV)
    assert res.area == pytest.approx(420.0, rel=1e-3)
    assert res.center_2theta == pytest.approx(33.0, abs=1e-4)
    assert res.fwhm == pytest.approx(0.28, rel=1e-3)
    if model == PSEUDO_VOIGT:
        assert res.eta == pytest.approx(0.35, abs=1e-3)


@pytest.mark.parametrize("model,eta", [(GAUSSIAN, None), (LORENTZIAN, None), (PSEUDO_VOIGT, 0.6)])
def test_noiseless_recovery_constant_baseline(model, eta):
    x, y = _synthetic(model, area=380.0, center=41.0, fwhm=0.5, eta=eta, baseline=BASELINE_CONSTANT, c0=25.0)
    res = fit_xrd_peak(x, y, model, fit_window=(38.5, 43.5), baseline=BASELINE_CONSTANT, **_PROV)
    assert res.area == pytest.approx(380.0, rel=1e-3)
    assert res.center_2theta == pytest.approx(41.0, abs=1e-4)
    assert res.fwhm == pytest.approx(0.5, rel=1e-3)
    assert res.params["baseline_c0"] == pytest.approx(25.0, rel=1e-3)


@pytest.mark.parametrize("model,eta", [(GAUSSIAN, None), (LORENTZIAN, None), (PSEUDO_VOIGT, 0.5)])
def test_noiseless_recovery_linear_baseline(model, eta):
    x, y = _synthetic(model, area=500.0, center=50.0, fwhm=0.4, eta=eta,
                      baseline=BASELINE_LINEAR, c0=40.0, c1=3.0)
    res = fit_xrd_peak(x, y, model, fit_window=(48.0, 52.0), baseline=BASELINE_LINEAR, **_PROV)
    assert res.area == pytest.approx(500.0, rel=1e-3)
    assert res.center_2theta == pytest.approx(50.0, abs=1e-4)
    assert res.fwhm == pytest.approx(0.4, rel=1e-3)
    assert res.params["baseline_c0"] == pytest.approx(40.0, rel=1e-3)  # baseline at window midpoint
    assert res.params["baseline_c1"] == pytest.approx(3.0, rel=1e-3)


def test_lorentzian_area_is_recovered_even_from_a_narrow_window():
    """Because the profile is area-normalized, the fitted `area` is inferred
    from the peak SHAPE, so a moderately narrow window (which truncates real
    integrated intensity in the tails) still recovers A on clean data."""
    x, y = _synthetic(LORENTZIAN, area=300.0, center=30.0, fwhm=0.3, x=np.linspace(29.1, 30.9, 900))
    res = fit_xrd_peak(x, y, LORENTZIAN, fit_window=(29.1, 30.9), baseline=BASELINE_NONE, **_PROV)
    assert res.area == pytest.approx(300.0, rel=2e-3)


# =====================================================================
# 3. Parameter recovery with controlled deterministic noise
# =====================================================================


@pytest.mark.parametrize("model,eta", [(GAUSSIAN, None), (LORENTZIAN, None), (PSEUDO_VOIGT, 0.4)])
def test_noisy_recovery_within_stated_tolerances(model, eta):
    # ~240 points in-window, peak height ~ 30x the noise sigma, seeded.
    x = np.linspace(25.0, 31.0, 1400)
    x, y = _synthetic(model, area=600.0, center=28.0, fwhm=0.45, eta=eta,
                      baseline=BASELINE_LINEAR, c0=50.0, c1=2.0, x=x, noise=6.0, seed=42)
    res = fit_xrd_peak(x, y, model, fit_window=(26.0, 30.0), baseline=BASELINE_LINEAR, **_PROV)

    assert res.area == pytest.approx(600.0, rel=0.05)
    assert abs(res.center_2theta - 28.0) < 0.05 * 0.45
    assert res.fwhm == pytest.approx(0.45, rel=0.10)
    if model == PSEUDO_VOIGT:
        assert res.eta == pytest.approx(0.4, abs=0.15)

    # covariance-derived standard errors are present, positive, and the
    # truth sits within a few of them.
    assert res.area_error is not None and res.area_error > 0
    assert abs(res.area - 600.0) < 5.0 * res.area_error
    assert res.center_error is not None and res.center_error > 0
    assert res.fwhm_error is not None and res.fwhm_error > 0


def test_standard_errors_grow_with_noise():
    def run(noise, seed):
        x = np.linspace(25.0, 31.0, 1400)
        x, y = _synthetic(GAUSSIAN, area=600.0, center=28.0, fwhm=0.45, x=x, noise=noise, seed=seed)
        return fit_xrd_peak(x, y, GAUSSIAN, fit_window=(26.0, 30.0), baseline=BASELINE_NONE, **_PROV)

    low = run(3.0, 1)
    high = run(15.0, 1)
    assert high.area_error > low.area_error
    assert high.fwhm_error > low.fwhm_error


# =====================================================================
# 4. Derived height + its Jacobian-propagated standard error
# =====================================================================


def test_derived_height_and_propagated_height_error():
    x = np.linspace(25.0, 31.0, 1400)
    x, y = _synthetic(GAUSSIAN, area=600.0, center=28.0, fwhm=0.45, x=x, noise=5.0, seed=7)
    res = fit_xrd_peak(x, y, GAUSSIAN, fit_window=(26.0, 30.0), baseline=BASELINE_NONE, **_PROV)

    assert res.height == pytest.approx(derived_height(GAUSSIAN, res.area, res.fwhm), rel=1e-12)
    assert res.height_error is not None and res.height_error > 0
    # sanity: relative height error is comparable in scale to the area/fwhm ones,
    # never a naive independent-quadrature under/over-estimate by orders of magnitude.
    rel_h = res.height_error / res.height
    rel_a = res.area_error / res.area
    assert 0.1 * rel_a < rel_h < 10.0 * rel_a


def test_pseudo_voigt_gaussian_endpoint_is_a_valid_fit_with_only_eta_se_suppressed():
    # Pure Gaussian data fitted with a pseudo-Voigt: eta converges to its 0
    # bound. That is a scientifically valid endpoint, NOT a failed fit -- the
    # fit stays converged, area/center/FWHM keep their standard errors, only
    # eta's SE (and the eta-dependent height SE) are suppressed, with a
    # neutral "converged to the Gaussian endpoint" message.
    x, y = _synthetic(GAUSSIAN, area=400.0, center=20.0, fwhm=0.3)
    res = fit_xrd_peak(x, y, PSEUDO_VOIGT, fit_window=(18.5, 21.5), baseline=BASELINE_NONE, **_PROV)
    assert res.converged is True
    assert res.eta == pytest.approx(0.0, abs=1e-6)
    assert res.eta_error is None
    assert res.height_error is None
    assert res.area_error is not None and res.center_error is not None and res.fwhm_error is not None
    assert any("Gaussian endpoint" in w and "η" in w for w in res.warnings)
    assert not any("fit bound" in w for w in res.warnings)  # eta is not lumped into the generic message


def test_pseudo_voigt_lorentzian_endpoint_message():
    def bare_lorentzian(xx, x0, fwhm, area):
        hwhm = fwhm / 2.0
        return area * hwhm / (np.pi * ((xx - x0) ** 2 + hwhm**2))

    x = np.linspace(28.0, 32.0, 2000)
    y = bare_lorentzian(x, 30.0, 0.35, 500.0) + 10.0
    res = fit_xrd_peak(x, y, PSEUDO_VOIGT, fit_window=(28.6, 31.4), baseline=BASELINE_CONSTANT, **_PROV)
    assert res.eta == pytest.approx(1.0, abs=1e-6)
    assert res.eta_error is None
    assert any("Lorentzian endpoint" in w for w in res.warnings)


# =====================================================================
# 5. d-spacing and propagated standard error
# =====================================================================


def test_d_spacing_and_propagated_error_with_radiation():
    x = np.linspace(26.0, 30.0, 1200)
    x, y = _synthetic(GAUSSIAN, area=500.0, center=28.0, fwhm=0.4, x=x, noise=4.0, seed=3)
    rad = radiation_from_preset("cu_ka1")
    res = fit_xrd_peak(x, y, GAUSSIAN, fit_window=(26.5, 29.5), radiation=rad, baseline=BASELINE_NONE, **_PROV)

    from gnovi_plot.modules.xrd.bragg import d_spacing

    assert res.d_spacing == pytest.approx(float(d_spacing(res.center_2theta, rad.wavelength_angstrom)), rel=1e-9)
    assert res.d_spacing_error is not None and res.d_spacing_error > 0

    # analytic check: |dd/dcenter| * center_error, dd/dcenter = -d*cot(theta)*(pi/360)
    theta = math.radians(res.center_2theta / 2.0)
    expected = abs(-res.d_spacing / math.tan(theta) * (math.pi / 360.0)) * res.center_error
    assert res.d_spacing_error == pytest.approx(expected, rel=1e-9)


def test_no_radiation_means_no_d_spacing_rather_than_a_guessed_wavelength():
    x, y = _synthetic(GAUSSIAN, area=300.0, center=40.0, fwhm=0.3)
    res = fit_xrd_peak(x, y, GAUSSIAN, fit_window=(38.5, 41.5), baseline=BASELINE_NONE, **_PROV)
    assert res.d_spacing is None
    assert res.d_spacing_error is None


def test_d_spacing_error_none_when_center_error_none(monkeypatch):
    x, y = _synthetic(GAUSSIAN, area=300.0, center=40.0, fwhm=0.3)
    res = fit_xrd_peak(
        x, y, GAUSSIAN, fit_window=(38.5, 41.5), radiation=radiation_from_preset("cu_ka1"),
        baseline=BASELINE_NONE,
        param_bounds={"center": (40.0, 40.5)},  # center pinned at lower bound -> center_error None
        **_PROV,
    )
    assert res.center_error is None
    assert res.d_spacing is not None
    assert res.d_spacing_error is None


# =====================================================================
# 6. Fit window helper
# =====================================================================


def test_propose_fit_window_uses_seed_width_times_local_spacing():
    tt = np.linspace(10.0, 90.0, 4001)  # spacing exactly 0.02
    seed = XRDPeakSeed(two_theta=45.0, intensity=100.0, origin="automatic", index=1750, width_samples=10.0)
    # estimated FWHM = 10 * 0.02 = 0.2 ; window = 45 +/- 4*0.2 = [44.2, 45.8]
    assert estimate_seed_fwhm(tt, seed) == pytest.approx(0.2, rel=1e-9)
    win = propose_fit_window(tt, seed)
    assert win.two_theta_min == pytest.approx(44.2, abs=1e-6)
    assert win.two_theta_max == pytest.approx(45.8, abs=1e-6)


def test_propose_fit_window_clips_to_data_bounds():
    tt = np.linspace(10.0, 20.0, 500)
    seed = XRDPeakSeed(two_theta=10.5, intensity=100.0, origin="automatic", index=25, width_samples=20.0)
    win = propose_fit_window(tt, seed)
    assert win.two_theta_min == pytest.approx(10.0)  # clipped at the data start


def test_propose_fit_window_clips_at_neighbouring_peak_midpoints():
    tt = np.linspace(10.0, 90.0, 4000)
    seed = XRDPeakSeed(two_theta=45.0, intensity=100.0, origin="automatic", index=1750, width_samples=40.0)
    win = propose_fit_window(tt, seed, neighbor_two_thetas=[44.4, 46.2])
    assert win.two_theta_min == pytest.approx(0.5 * (44.4 + 45.0), abs=1e-6)
    assert win.two_theta_max == pytest.approx(0.5 * (45.0 + 46.2), abs=1e-6)


def test_estimate_seed_fwhm_none_for_manual_seed():
    tt = np.linspace(10.0, 90.0, 4000)
    assert estimate_seed_fwhm(tt, XRDPeakSeed.manual(45.0, 100.0)) is None


def test_fit_window_proposal_is_the_same_for_ascending_and_descending_2theta():
    # Same synthetic spacing, represented ascending and descending. `seed.index`
    # belongs to the array passed to the function, so it is mirrored for the
    # descending array. The local-step calculation uses abs(diff), so both give
    # the same estimated FWHM and the same window WIDTH (the descending array's
    # window is the ascending one reflected about the centre).
    n = 4001
    tt_asc = np.linspace(10.0, 90.0, n)  # step 0.02
    tt_desc = tt_asc[::-1].copy()
    idx_asc = 1750
    seed_asc = XRDPeakSeed(two_theta=45.0, intensity=1.0, origin="automatic", index=idx_asc, width_samples=10.0)
    seed_desc = XRDPeakSeed(
        two_theta=45.0, intensity=1.0, origin="automatic", index=n - 1 - idx_asc, width_samples=10.0
    )

    assert estimate_seed_fwhm(tt_asc, seed_asc) == pytest.approx(estimate_seed_fwhm(tt_desc, seed_desc), rel=1e-12)

    w_asc = propose_fit_window(tt_asc, seed_asc)
    w_desc = propose_fit_window(tt_desc, seed_desc)
    width_asc = w_asc.two_theta_max - w_asc.two_theta_min
    width_desc = w_desc.two_theta_max - w_desc.two_theta_min
    assert width_asc == pytest.approx(width_desc, rel=1e-9)
    assert width_asc == pytest.approx(1.6, abs=1e-6)  # 2 * 4 * (10 * 0.02)


def test_irregular_x_spacing_still_recovers_parameters():
    rng = np.random.default_rng(11)
    x = np.sort(rng.uniform(25.0, 31.0, 1600))
    x, y = _synthetic(GAUSSIAN, area=520.0, center=28.0, fwhm=0.4, x=x, noise=4.0, seed=11)
    res = fit_xrd_peak(x, y, GAUSSIAN, fit_window=(26.5, 29.5), baseline=BASELINE_NONE, **_PROV)
    assert res.area == pytest.approx(520.0, rel=0.06)
    assert res.center_2theta == pytest.approx(28.0, abs=0.03)


# =====================================================================
# 7. Edge / failure handling
# =====================================================================


def test_reversed_window_is_rejected():
    x, y = _synthetic(GAUSSIAN, area=100.0, center=20.0, fwhm=0.3)
    with pytest.raises(XRDFitError, match="reversed or zero-width|max > min"):
        fit_xrd_peak(x, y, GAUSSIAN, fit_window=(21.0, 19.0), **_PROV)


def test_zero_width_window_is_rejected():
    x, y = _synthetic(GAUSSIAN, area=100.0, center=20.0, fwhm=0.3)
    with pytest.raises(XRDFitError):
        fit_xrd_peak(x, y, GAUSSIAN, fit_window=(20.0, 20.0), **_PROV)


def test_non_finite_window_is_rejected():
    x, y = _synthetic(GAUSSIAN, area=100.0, center=20.0, fwhm=0.3)
    with pytest.raises(XRDFitError, match="finite"):
        fit_xrd_peak(x, y, GAUSSIAN, fit_window=(np.nan, 21.0), **_PROV)


def test_window_entirely_outside_data_is_rejected():
    x, y = _synthetic(GAUSSIAN, area=100.0, center=20.0, fwhm=0.3)
    with pytest.raises(XRDFitError, match="no finite data points"):
        fit_xrd_peak(x, y, GAUSSIAN, fit_window=(100.0, 110.0), **_PROV)


def test_too_few_points_in_window_is_rejected():
    x = np.linspace(19.0, 21.0, 400)
    x, y = _synthetic(GAUSSIAN, area=100.0, center=20.0, fwhm=0.3, x=x)
    with pytest.raises(XRDFitError, match="Not enough finite points"):
        fit_xrd_peak(x, y, GAUSSIAN, fit_window=(19.98, 20.02), baseline=BASELINE_NONE, **_PROV)


def test_flat_data_is_rejected():
    x = np.linspace(19.0, 21.0, 400)
    y = np.full_like(x, 42.0)
    with pytest.raises(XRDFitError, match="no positive peak"):
        fit_xrd_peak(x, y, GAUSSIAN, fit_window=(19.2, 20.8), baseline=BASELINE_NONE, **_PROV)


def test_negative_only_signal_is_rejected_not_fitted_as_a_negative_peak():
    x = np.linspace(19.0, 21.0, 400)
    y = -50.0 - 100.0 * np.exp(-((x - 20.0) ** 2) / (2 * 0.1**2))  # a dip, no positive peak
    with pytest.raises(XRDFitError, match="no positive peak"):
        fit_xrd_peak(x, y, GAUSSIAN, fit_window=(19.2, 20.8), baseline=BASELINE_NONE, **_PROV)


def test_non_finite_points_are_dropped_with_a_warning():
    x, y = _synthetic(GAUSSIAN, area=400.0, center=20.0, fwhm=0.3, x=np.linspace(18.0, 22.0, 1200))
    y = y.copy()
    y[10] = np.nan
    y[900] = np.inf
    res = fit_xrd_peak(x, y, GAUSSIAN, fit_window=(18.5, 21.5), baseline=BASELINE_NONE, **_PROV)
    assert any("non-finite" in w for w in res.warnings)
    assert res.area == pytest.approx(400.0, rel=1e-2)


def _none_baseline_warning(res):
    return any("return near zero" in w for w in res.warnings)


def test_baseline_none_warns_on_a_positive_pedestal():
    x = np.linspace(18.0, 22.0, 1200)
    x, y = _synthetic(GAUSSIAN, area=400.0, center=20.0, fwhm=0.35, x=x)
    y = y + 120.0  # large positive pedestal, no local baseline requested
    res = fit_xrd_peak(x, y, GAUSSIAN, fit_window=(18.6, 21.4), baseline=BASELINE_NONE, **_PROV)
    assert _none_baseline_warning(res)


def test_baseline_none_warns_on_a_negative_pedestal_under_a_positive_peak():
    x = np.linspace(18.0, 22.0, 1200)
    x, y = _synthetic(GAUSSIAN, area=400.0, center=20.0, fwhm=0.35, x=x)
    # positive peak, but the data sit well below zero at the edges
    y = y - 120.0
    res = fit_xrd_peak(x, y, GAUSSIAN, fit_window=(18.6, 21.4), baseline=BASELINE_NONE, **_PROV)
    assert res.height > 0  # still a valid positive peak
    assert _none_baseline_warning(res)


def test_baseline_none_does_not_warn_for_a_zero_centred_peak():
    x = np.linspace(18.0, 22.0, 1200)
    x, y = _synthetic(GAUSSIAN, area=400.0, center=20.0, fwhm=0.35, x=x)  # returns to ~0 at the edges
    res = fit_xrd_peak(x, y, GAUSSIAN, fit_window=(18.6, 21.4), baseline=BASELINE_NONE, **_PROV)
    assert not _none_baseline_warning(res)


def test_unknown_model_and_baseline_are_rejected():
    x, y = _synthetic(GAUSSIAN, area=100.0, center=20.0, fwhm=0.3)
    with pytest.raises(XRDFitError, match="Unknown profile model"):
        fit_xrd_peak(x, y, "voigt", fit_window=(19.0, 21.0), **_PROV)
    with pytest.raises(XRDFitError, match="Unknown baseline model"):
        fit_xrd_peak(x, y, GAUSSIAN, fit_window=(19.0, 21.0), baseline="spline", **_PROV)


def test_fwhm_lower_bound_flags_the_parameter_when_hit():
    # a peak far narrower than the sample spacing -> fwhm pins at its lower bound.
    x = np.linspace(19.0, 21.0, 300)  # spacing ~6.7e-3
    x, y = _synthetic(GAUSSIAN, area=50.0, center=20.0, fwhm=0.004, x=x)
    y = y + 1.0  # tiny pedestal so there is signal at all
    res = fit_xrd_peak(x, y, GAUSSIAN, fit_window=(19.2, 20.8), baseline=BASELINE_CONSTANT, **_PROV)
    assert res.params["fwhm"] == pytest.approx(res.parameters["param_bounds"]["fwhm"][0], rel=1e-3)
    assert res.fwhm_error is None
    assert any("bound" in w for w in res.warnings)


def test_low_degrees_of_freedom_is_flagged():
    # a window near the numerical minimum-points floor -> small dof -> a
    # "few degrees of freedom" warning (the min-points rule guarantees dof > 0).
    x = np.linspace(18.0, 22.0, 800)
    x, y = _synthetic(GAUSSIAN, area=80.0, center=20.0, fwhm=0.25, x=x)
    step = x[1] - x[0]
    lo = 20.0 - 5 * step
    hi = 20.0 + 5 * step  # ~11 points, Gaussian + none baseline (P=3) -> dof ~ 8
    res = fit_xrd_peak(x, y, GAUSSIAN, fit_window=(lo, hi), baseline=BASELINE_NONE, **_PROV)
    assert 0 < res.dof < 10
    assert any("degrees of freedom" in w for w in res.warnings)


def test_solver_non_convergence_raises_xrdfiterror(monkeypatch):
    import gnovi_plot.modules.xrd.fitting as fitting_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("Optimal parameters not found: maxfev exceeded")

    monkeypatch.setattr(fitting_mod, "curve_fit", _boom)
    x, y = _synthetic(GAUSSIAN, area=100.0, center=20.0, fwhm=0.3)
    with pytest.raises(XRDFitError, match="did not converge"):
        fit_xrd_peak(x, y, GAUSSIAN, fit_window=(19.0, 21.0), **_PROV)


def test_standard_errors_helper_handles_singular_and_negative_covariance():
    assert _standard_errors(None, ["a"]) is None
    assert _standard_errors(np.full((2, 2), np.inf), ["a", "b"]) is None
    assert _standard_errors(np.array([[-1.0, 0.0], [0.0, 1.0]]), ["a", "b"]) is None
    ok = _standard_errors(np.array([[4.0, 0.0], [0.0, 9.0]]), ["a", "b"])
    assert ok == {"a": 2.0, "b": 3.0}


# =====================================================================
# 8. Deterministic curve regeneration + serialization
# =====================================================================


def test_curve_regeneration_is_deterministic_and_components_sum_to_total():
    x, y = _synthetic(PSEUDO_VOIGT, area=500.0, center=30.0, fwhm=0.4, eta=0.3,
                      baseline=BASELINE_LINEAR, c0=20.0, c1=1.5)
    res = fit_xrd_peak(x, y, PSEUDO_VOIGT, fit_window=(28.0, 32.0), baseline=BASELINE_LINEAR, **_PROV)

    xc1, yc1 = sample_fit_curve(res)
    xc2, yc2 = sample_fit_curve(res)
    assert np.array_equal(xc1, xc2) and np.array_equal(yc1, yc2)
    assert len(xc1) == res.curve_num_points

    grid = np.linspace(res.fit_window[0], res.fit_window[1], 77)
    assert np.allclose(
        evaluate_peak_component(res, grid) + evaluate_baseline(res, grid),
        evaluate_total(res, grid),
        atol=1e-12,
    )
    # peak component integrates (numerically, wide grid) to ~ area
    wide = np.linspace(res.center_2theta - 60.0, res.center_2theta + 60.0, 400001)
    assert np.trapezoid(evaluate_peak_component(res, wide), wide) == pytest.approx(res.area, rel=1e-3)


def test_result_serialization_round_trip_through_json():
    x, y = _synthetic(PSEUDO_VOIGT, area=475.0, center=44.0, fwhm=0.33, eta=0.42,
                      baseline=BASELINE_LINEAR, c0=30.0, c1=-2.0, noise=3.0, seed=5,
                      x=np.linspace(42.0, 46.0, 1400))
    res = fit_xrd_peak(
        x, y, PSEUDO_VOIGT, fit_window=(42.5, 45.5), baseline=BASELINE_LINEAR,
        radiation=radiation_from_preset("cu_ka1"), **_PROV,
    )
    blob = json.loads(json.dumps(res.to_dict()))
    back = XRDPeakFitResult.from_dict(blob)

    assert back.kind == "xrd_peak_fit"
    assert back.result_id == res.result_id
    assert back.operation == OPERATION_PEAK_FIT
    assert back.model == res.model
    assert back.baseline_model == res.baseline_model
    assert back.fit_window == res.fit_window
    assert back.fwhm_units == FWHM_UNITS_TWO_THETA_DEG
    assert back.params == pytest.approx(res.params)
    assert back.area == pytest.approx(res.area)
    assert back.area_error == pytest.approx(res.area_error)
    assert back.center_error == pytest.approx(res.center_error)
    assert back.d_spacing == pytest.approx(res.d_spacing)
    assert back.d_spacing_error == pytest.approx(res.d_spacing_error)
    assert back.radiation == res.radiation
    assert back.warnings == res.warnings

    # regenerated curves agree after the round trip
    _, y_before = sample_fit_curve(res)
    _, y_after = sample_fit_curve(back)
    assert np.allclose(y_before, y_after, atol=1e-9)


def test_result_registers_with_the_polymorphic_dispatch():
    x, y = _synthetic(GAUSSIAN, area=300.0, center=25.0, fwhm=0.3)
    res = fit_xrd_peak(x, y, GAUSSIAN, fit_window=(23.5, 26.5), baseline=BASELINE_NONE, **_PROV)
    rebuilt = result_from_dict(res.to_dict())
    assert isinstance(rebuilt, XRDPeakFitResult)


def test_result_supports_residuals_and_computes_them():
    x, y = _synthetic(GAUSSIAN, area=300.0, center=25.0, fwhm=0.3, x=np.linspace(23.0, 27.0, 1000))
    res = fit_xrd_peak(x, y, GAUSSIAN, fit_window=(23.5, 26.5), baseline=BASELINE_NONE, **_PROV)
    assert res.supports_residuals() is True
    rd = res.compute_residuals(x, y)
    assert isinstance(rd, ResidualData)
    assert np.max(np.abs(rd.residuals)) < 1e-3  # exact data


def test_result_summary_and_details_are_bounded():
    x, y = _synthetic(PSEUDO_VOIGT, area=300.0, center=25.0, fwhm=0.3, eta=0.3,
                      baseline=BASELINE_LINEAR, c0=10.0, c1=1.0)
    res = fit_xrd_peak(x, y, PSEUDO_VOIGT, fit_window=(23.5, 26.5), baseline=BASELINE_LINEAR,
                       radiation=radiation_from_preset("cu_ka1"), **_PROV)
    assert "peak fit" in res.summary()
    rows = res.details()
    assert len(rows) < 25  # fixed structure, never one row per anything
    labels = {label for label, _ in rows}
    assert {"Model", "Center (°2θ)", "FWHM (°2θ)", "Area (integrated intensity)", "d-spacing (Å)"} <= labels


# =====================================================================
# 9. Single-peak local-window architecture on a realistic pattern
# =====================================================================


def _multi_peak_pattern(seed=0):
    x = np.linspace(15.0, 75.0, 6000)
    centers = [21.0, 28.5, 36.0, 44.5, 52.0, 61.0, 68.5]
    areas = [900.0, 1500.0, 600.0, 1100.0, 400.0, 800.0, 500.0]
    fwhms = [0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.34]
    y = 40.0 + 0.6 * (x - 15.0) - 0.004 * (x - 15.0) ** 2  # smooth polynomial background
    for c, a, w in zip(centers, areas, fwhms):
        y = y + peak_component(x, PSEUDO_VOIGT, a, c, w, 0.3)
    rng = np.random.default_rng(seed)
    y = y + rng.normal(0.0, 3.0, size=x.shape)
    return x, y, list(zip(centers, areas, fwhms))


def test_single_isolated_peak_fit_through_a_local_window_on_a_multi_peak_pattern():
    x, y, truth = _multi_peak_pattern(seed=1)
    seeds = detect_peaks(x, y, prominence=120.0)
    assert len(seeds) >= 6

    # target the 4th reflection (44.5 deg); it is well separated from its neighbours
    target_c, target_a, target_w = truth[3]
    seed = min(seeds, key=lambda s: abs(s.two_theta - target_c))
    others = [s.two_theta for s in seeds if s is not seed]
    window = propose_fit_window(x, seed, neighbor_two_thetas=others)

    # the local window must not span a neighbouring reflection
    assert not any(window.two_theta_min < c < window.two_theta_max for c in [21.0, 28.5, 36.0, 52.0, 61.0, 68.5])

    res = fit_xrd_peak(
        x, y, PSEUDO_VOIGT, fit_window=window, baseline=BASELINE_LINEAR,
        radiation=radiation_from_preset("cu_ka1"), seed=seed, neighbor_two_thetas=others, **_PROV,
    )
    assert res.center_2theta == pytest.approx(target_c, abs=0.02)
    assert res.fwhm == pytest.approx(target_w, rel=0.15)
    assert res.area == pytest.approx(target_a, rel=0.10)
    assert res.source_peak_id == seed.id


def test_overlap_warning_fires_when_a_neighbouring_peak_sits_inside_the_window():
    x = np.linspace(27.0, 33.0, 2400)
    # two Pseudo-Voigt peaks 0.6*FWHM apart -> unresolved doublet
    y = 20.0 + peak_component(x, GAUSSIAN, 800.0, 29.8, 0.4) + peak_component(x, GAUSSIAN, 700.0, 30.2, 0.4)
    res = fit_xrd_peak(
        x, y, GAUSSIAN, fit_window=(28.5, 31.5), baseline=BASELINE_CONSTANT,
        neighbor_two_thetas=[30.2], **_PROV,
    )
    assert any("isolated reflection" in w for w in res.warnings)


def test_no_overlap_warning_without_neighbour_information():
    # XRD-3A does NOT infer overlap from the fit residuals: an earlier
    # MAD-based residual-run heuristic could report a "possible unresolved
    # reflection" on a mathematically correct, near-noiseless single-peak
    # fit (its residual scale collapsed toward floating-point roundoff).
    # It was removed. A single isolated peak fitted with no
    # `neighbor_two_thetas` must never produce an overlap warning, even for
    # exact (noiseless) synthetic data or a genuine unresolved doublet
    # fitted without a neighbour hint.
    for noise in (0.0, 2.0):
        x, y = _synthetic(GAUSSIAN, area=800.0, center=30.0, fwhm=0.4,
                          baseline=BASELINE_LINEAR, c0=15.0, c1=2.0,
                          x=np.linspace(27.0, 33.0, 3000), noise=noise, seed=1)
        res = fit_xrd_peak(x, y, GAUSSIAN, fit_window=(28.4, 31.6), baseline=BASELINE_LINEAR, **_PROV)
        assert not any("isolated reflection" in w for w in res.warnings)
        assert not any("overlap" in w.lower() or "shoulder" in w.lower() for w in res.warnings)

    # a genuine unresolved doublet, but no neighbour hint supplied -> still silent
    x = np.linspace(27.0, 33.0, 3000)
    y = 10.0 + peak_component(x, GAUSSIAN, 900.0, 29.7, 0.35) + peak_component(x, GAUSSIAN, 500.0, 30.5, 0.35)
    res = fit_xrd_peak(x, y, GAUSSIAN, fit_window=(28.6, 31.6), baseline=BASELINE_CONSTANT, **_PROV)
    assert not any("isolated reflection" in w for w in res.warnings)


# =====================================================================
# 10. Provenance
# =====================================================================


def test_provenance_records_the_operation_configuration():
    x, y = _synthetic(GAUSSIAN, area=300.0, center=25.0, fwhm=0.3)
    seed = XRDPeakSeed(two_theta=25.0, intensity=100.0, origin="automatic", index=0)
    res = fit_xrd_peak(
        x, y, GAUSSIAN, fit_window=(23.5, 26.5), baseline=BASELINE_LINEAR,
        radiation=radiation_from_preset("cu_ka1"), seed=seed, source_result_id="detect-1", **_PROV,
    )
    p = res.parameters
    assert p["model"] == GAUSSIAN
    assert p["baseline_model"] == BASELINE_LINEAR
    assert p["fit_window"] == [23.5, 26.5]
    assert p["fwhm_units"] == FWHM_UNITS_TWO_THETA_DEG
    assert "area_normalized" in p["amplitude_convention"]
    assert "area-normalized Gaussian" in p["profile_convention"]
    assert p["solver"] == "scipy.optimize.curve_fit (trf)"
    assert set(p["initial_params"]) == {"area", "center", "fwhm", "baseline_c0", "baseline_c1"}
    assert set(p["param_bounds"]) == set(p["initial_params"])
    assert p["source_peak_id"] == seed.id
    assert p["source_result_id"] == "detect-1"
    assert p["radiation"]["label"] == "Cu Ka1"
    assert res.engine == "gnovi"
    assert res.engine_version is not None
    # SciPy-version-dependent solver internals are NOT persisted in the
    # reproducibility dict (they would break cross-version result equality).
    assert "solver_message" not in p
    assert "n_function_evaluations" not in p
    assert isinstance(res.solver_message, str) and res.solver_message  # kept as a top-level display field


def test_pseudo_voigt_profile_convention_is_recorded_and_unambiguous():
    x, y = _synthetic(PSEUDO_VOIGT, area=400.0, center=30.0, fwhm=0.35, eta=0.4)
    res = fit_xrd_peak(x, y, PSEUDO_VOIGT, fit_window=(28.6, 31.4), baseline=BASELINE_NONE, **_PROV)
    conv = res.parameters["profile_convention"]
    assert "(1 - eta) * Gaussian + eta * Lorentzian" in conv
    assert "eta is the Lorentzian fraction" in conv
    assert "eta = 0 is a pure Gaussian" in conv and "eta = 1 is a pure Lorentzian" in conv
    assert "shar" in conv and "FWHM" in conv  # shared center and FWHM
    # researcher-visible details label states the direction too
    labels = [label for label, _ in res.details()]
    assert any("Lorentzian fraction" in label and "0=Gaussian" in label for label in labels)


# =====================================================================
# 11. Scientific characterization / calibration
# =====================================================================


def test_wrong_profile_model_can_bias_area_badly_while_r_squared_stays_high():
    """R-squared is a descriptive goodness-of-fit statistic; it is NOT, by
    itself, a profile-model selection criterion. Fitting a Gaussian to a
    genuinely Lorentzian peak can leave R-squared > 0.97 while the
    integrated area is off by tens of percent. This test documents that
    limitation; XRD-3A does not do runtime model selection.

    The Lorentzian is generated from a first-principles expression written
    here, not via `peak_component`, so the area assertion is not
    tautological with the fitter's own forward model.
    """

    def bare_lorentzian(xx, x0, fwhm, area):
        hwhm = fwhm / 2.0
        return area * hwhm / (np.pi * ((xx - x0) ** 2 + hwhm**2))  # analytic integral == area

    true_area, x0, fwhm = 1000.0, 30.0, 0.40
    x = np.linspace(x0 - 8 * fwhm, x0 + 8 * fwhm, 3000)
    y = bare_lorentzian(x, x0, fwhm, true_area) + 50.0

    res = fit_xrd_peak(
        x, y, GAUSSIAN, fit_window=(x0 - 4 * fwhm, x0 + 4 * fwhm), baseline=BASELINE_CONSTANT, **_PROV
    )
    rel_area_error = abs(res.area - true_area) / true_area

    assert res.r_squared > 0.97          # looks like a good fit
    assert rel_area_error > 0.20          # yet the area is badly biased by the wrong shape
    # (measured for this deterministic case: R^2 ~ 0.982, area ~ 674, rel error ~ 0.33)


def test_center_standard_error_is_approximately_calibrated():
    """Deterministic Monte-Carlo check that the covariance-derived CENTRE
    standard error is a sane 1-sigma scale under the controlled synthetic
    model with Gaussian noise -- roughly 68% coverage at 1 SE, 95% at 2 SE.

    This validates only the numerical fit standard error under the model;
    it is not a statement about experimental measurement uncertainty.
    N = 150 keeps runtime near 0.4 s while giving a useful gross-calibration
    signal; the tolerances are broad so it does not flake across
    SciPy/platform versions.
    """
    rng = np.random.default_rng(20240501)
    x = np.linspace(27.0, 29.0, 400)
    true_center = 28.0
    n_reps, within_1se, within_2se, n_used = 150, 0, 0, 0
    for _ in range(n_reps):
        y = (
            peak_component(x, GAUSSIAN, 400.0, true_center, 0.30)
            + 25.0
            + rng.normal(0.0, 6.0, size=x.shape)
        )
        res = fit_xrd_peak(x, y, GAUSSIAN, fit_window=(27.2, 28.8), baseline=BASELINE_CONSTANT, **_PROV)
        if res.center_error is None:
            continue
        n_used += 1
        dev = abs(res.center_2theta - true_center)
        within_1se += dev <= res.center_error
        within_2se += dev <= 2.0 * res.center_error

    assert n_used >= n_reps - 5
    cov_1se = within_1se / n_used
    cov_2se = within_2se / n_used
    assert 0.55 <= cov_1se <= 0.82   # nominal 0.68
    assert 0.88 <= cov_2se <= 1.0    # nominal 0.95
