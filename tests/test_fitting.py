from __future__ import annotations

import numpy as np
import pytest

from gnovi_plot.analysis.fitting import (
    EXPONENTIAL,
    GAUSSIAN,
    LINEAR,
    POLYNOMIAL,
    FitError,
    FitResult,
    evaluate_fit,
    fit_curve,
    sample_fit_curve,
)

_PROVENANCE = dict(
    source_dataset_id="dataset-abc",
    source_series_id="series-xyz",
    x_column="x",
    y_column="y",
)


def test_linear_fit_recovers_known_parameters():
    x = np.linspace(0, 10, 25)
    y = 3.0 * x + 2.0

    result = fit_curve(x, y, LINEAR, **_PROVENANCE)

    assert isinstance(result, FitResult)
    assert result.model == LINEAR
    assert result.params["a"] == pytest.approx(3.0, abs=1e-6)
    assert result.params["b"] == pytest.approx(2.0, abs=1e-6)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)
    assert result.formula == "y = a·x + b"


def test_linear_fit_param_errors_grow_with_noise():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 50)
    y = 3.0 * x + 2.0 + rng.normal(scale=0.5, size=x.shape)

    result = fit_curve(x, y, LINEAR, **_PROVENANCE)

    assert result.param_errors is not None
    assert result.param_errors["a"] > 0
    assert result.param_errors["b"] > 0
    assert result.r_squared < 1.0
    assert result.r_squared > 0.9


def test_polynomial_fit_recovers_known_coefficients():
    x = np.linspace(-5, 5, 30)
    y = 1.0 + 2.0 * x + 0.5 * x**2

    result = fit_curve(x, y, POLYNOMIAL, degree=2, **_PROVENANCE)

    assert result.model == POLYNOMIAL
    assert result.params["c0"] == pytest.approx(1.0, abs=1e-6)
    assert result.params["c1"] == pytest.approx(2.0, abs=1e-6)
    assert result.params["c2"] == pytest.approx(0.5, abs=1e-6)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)
    assert "c0" in result.formula and "c2" in result.formula


def test_polynomial_default_degree_is_quadratic():
    x = np.linspace(-3, 3, 20)
    y = x**2

    result = fit_curve(x, y, POLYNOMIAL, **_PROVENANCE)

    assert set(result.params.keys()) == {"c0", "c1", "c2"}


def test_polynomial_invalid_degree_raises_fit_error():
    x = np.linspace(0, 10, 10)
    y = x.copy()

    with pytest.raises(FitError):
        fit_curve(x, y, POLYNOMIAL, degree=0, **_PROVENANCE)


def test_exponential_fit_recovers_known_parameters():
    x = np.linspace(0, 5, 40)
    y = 2.0 * np.exp(0.7 * x) + 1.0

    result = fit_curve(x, y, EXPONENTIAL, **_PROVENANCE)

    assert result.model == EXPONENTIAL
    assert result.params["a"] == pytest.approx(2.0, rel=1e-3)
    assert result.params["b"] == pytest.approx(0.7, rel=1e-3)
    assert result.params["c"] == pytest.approx(1.0, abs=1e-2)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)


def test_gaussian_fit_recovers_known_parameters():
    x = np.linspace(-10, 10, 200)
    true_amplitude, true_mean, true_sigma, true_offset = 5.0, 1.5, 2.0, 0.3
    y = true_amplitude * np.exp(-((x - true_mean) ** 2) / (2 * true_sigma**2)) + true_offset

    result = fit_curve(x, y, GAUSSIAN, **_PROVENANCE)

    assert result.model == GAUSSIAN
    assert result.params["amplitude"] == pytest.approx(true_amplitude, rel=1e-3)
    assert result.params["mean"] == pytest.approx(true_mean, rel=1e-2)
    assert abs(result.params["sigma"]) == pytest.approx(true_sigma, rel=1e-2)
    assert result.params["offset"] == pytest.approx(true_offset, abs=1e-2)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)


def test_gaussian_fit_accepts_explicit_initial_guess():
    x = np.linspace(-10, 10, 200)
    y = 5.0 * np.exp(-((x - 1.5) ** 2) / (2 * 2.0**2)) + 0.3

    result = fit_curve(
        x, y, GAUSSIAN, initial_guess=[4.0, 1.0, 1.5, 0.0], **_PROVENANCE
    )

    assert result.params["amplitude"] == pytest.approx(5.0, rel=1e-2)


def test_unknown_model_raises_fit_error():
    x = np.linspace(0, 10, 10)
    y = x.copy()

    with pytest.raises(FitError):
        fit_curve(x, y, "not-a-real-model", **_PROVENANCE)


def test_too_few_points_raises_fit_error():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])

    with pytest.raises(FitError):
        fit_curve(x, y, LINEAR, **_PROVENANCE)


def test_mismatched_shapes_raise_fit_error():
    x = np.linspace(0, 10, 10)
    y = np.linspace(0, 10, 5)

    with pytest.raises(FitError):
        fit_curve(x, y, LINEAR, **_PROVENANCE)


def test_non_finite_values_are_dropped_before_fitting():
    x = np.linspace(0, 10, 25)
    y = 3.0 * x + 2.0
    y_with_gaps = y.copy()
    y_with_gaps[[3, 7, 12]] = np.nan
    x_with_gap = x.copy()
    x_with_gap[5] = np.inf

    result = fit_curve(x_with_gap, y_with_gaps, LINEAR, **_PROVENANCE)

    assert result.params["a"] == pytest.approx(3.0, abs=1e-6)
    assert result.params["b"] == pytest.approx(2.0, abs=1e-6)


def test_flat_data_does_not_divide_by_zero_in_r_squared():
    x = np.linspace(0, 10, 10)
    y = np.full_like(x, 5.0)

    result = fit_curve(x, y, LINEAR, **_PROVENANCE)

    assert result.r_squared == pytest.approx(1.0, abs=1e-9)


def test_result_carries_stable_provenance_not_labels():
    x = np.linspace(0, 10, 10)
    y = 2.0 * x

    result = fit_curve(
        x,
        y,
        LINEAR,
        source_dataset_id="dataset-abc",
        source_series_id="series-xyz",
        x_column="voltage",
        y_column="current",
        row_range=(10, 20),
    )

    assert result.source_dataset_id == "dataset-abc"
    assert result.source_series_id == "series-xyz"
    assert result.x_column == "voltage"
    assert result.y_column == "current"
    assert result.row_range == (10, 20)


def test_source_series_id_optional_and_row_range_optional():
    x = np.linspace(0, 10, 10)
    y = 2.0 * x

    result = fit_curve(
        x, y, LINEAR, source_dataset_id="dataset-abc", x_column="x", y_column="y"
    )

    assert result.source_series_id is None
    assert result.row_range is None


def test_to_dict_is_json_safe_round_trip_shape():
    import json

    x = np.linspace(0, 10, 25)
    y = 3.0 * x + 2.0

    result = fit_curve(x, y, LINEAR, row_range=(0, 25), **_PROVENANCE)
    data = result.to_dict()

    assert data["kind"] == "fit"
    assert data["model"] == LINEAR
    assert data["row_range"] == [0, 25]
    assert isinstance(data["params"], dict)
    json.dumps(data)  # must not raise


def test_details_reports_parameter_uncertainty_when_available():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 50)
    y = 3.0 * x + 2.0 + rng.normal(scale=0.5, size=x.shape)

    result = fit_curve(x, y, LINEAR, **_PROVENANCE)
    detail_labels = dict(result.details())

    assert "±" in detail_labels["a"]


def test_summary_and_details_do_not_raise():
    x = np.linspace(0, 10, 10)
    y = 2.0 * x + 1.0

    result = fit_curve(x, y, LINEAR, **_PROVENANCE)

    assert "linear fit" in result.summary()
    assert any(label == "R²" for label, _ in result.details())


# --- evaluate_fit / sample_fit_curve (smooth curve for "Add Fit Curve to Plot") --


def test_evaluate_fit_linear_matches_the_fitted_line():
    x = np.linspace(0, 10, 25)
    result = fit_curve(x, 3.0 * x + 2.0, LINEAR, **_PROVENANCE)

    y = evaluate_fit(result, np.array([0.0, 5.0, 10.0]))

    assert y == pytest.approx([2.0, 17.0, 32.0], abs=1e-6)


def test_evaluate_fit_polynomial_matches_the_fitted_curve():
    x = np.linspace(-5, 5, 30)
    y_true = 1.0 + 2.0 * x + 0.5 * x**2
    result = fit_curve(x, y_true, POLYNOMIAL, degree=2, **_PROVENANCE)

    y = evaluate_fit(result, x)

    assert y == pytest.approx(y_true, abs=1e-6)


def test_evaluate_fit_gaussian_matches_the_fitted_curve():
    x = np.linspace(-10, 10, 200)
    y_true = 5.0 * np.exp(-((x - 1.5) ** 2) / (2 * 2.0**2)) + 0.3
    result = fit_curve(x, y_true, GAUSSIAN, **_PROVENANCE)

    y = evaluate_fit(result, x)

    assert y == pytest.approx(y_true, abs=1e-3)


def test_sample_fit_curve_spans_the_requested_range_with_the_requested_count():
    x = np.linspace(0, 10, 25)
    result = fit_curve(x, 3.0 * x + 2.0, LINEAR, **_PROVENANCE)

    xs, ys = sample_fit_curve(result, 0.0, 10.0, num_points=50)

    assert len(xs) == 50
    assert len(ys) == 50
    assert xs[0] == pytest.approx(0.0)
    assert xs[-1] == pytest.approx(10.0)
    assert ys[0] == pytest.approx(2.0, abs=1e-6)
    assert ys[-1] == pytest.approx(32.0, abs=1e-6)


def test_sample_fit_curve_default_sample_count():
    x = np.linspace(0, 10, 25)
    result = fit_curve(x, 3.0 * x + 2.0, LINEAR, **_PROVENANCE)

    xs, ys = sample_fit_curve(result, 0.0, 10.0)

    assert len(xs) == 200


def test_sample_fit_curve_rejects_too_few_points():
    x = np.linspace(0, 10, 25)
    result = fit_curve(x, 3.0 * x + 2.0, LINEAR, **_PROVENANCE)

    with pytest.raises(FitError):
        sample_fit_curve(result, 0.0, 10.0, num_points=1)
