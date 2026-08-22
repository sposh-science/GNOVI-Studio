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
    ResidualData,
    compute_residuals,
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


def test_result_carries_source_panel_id_alongside_dataset_and_series_ids():
    x = np.linspace(0, 10, 10)
    y = 2.0 * x

    result = fit_curve(
        x,
        y,
        LINEAR,
        source_dataset_id="dataset-abc",
        source_series_id="series-xyz",
        source_panel_id="panel-123",
        x_column="x",
        y_column="y",
    )

    assert result.source_panel_id == "panel-123"
    assert result.source_series_id == "series-xyz"
    assert result.source_dataset_id == "dataset-abc"


def test_source_panel_id_defaults_to_none_when_not_supplied():
    x = np.linspace(0, 10, 10)
    y = 2.0 * x

    result = fit_curve(x, y, LINEAR, source_dataset_id="dataset-abc", x_column="x", y_column="y")

    assert result.source_panel_id is None


def test_to_dict_is_json_safe_round_trip_shape():
    import json

    x = np.linspace(0, 10, 25)
    y = 3.0 * x + 2.0

    result = fit_curve(x, y, LINEAR, row_range=(0, 25), source_panel_id="panel-123", **_PROVENANCE)
    data = result.to_dict()

    assert data["kind"] == "fit"
    assert data["model"] == LINEAR
    assert data["row_range"] == [0, 25]
    assert data["source_panel_id"] == "panel-123"
    assert isinstance(data["params"], dict)
    json.dumps(data)  # must not raise


# --- result_id: stable identity, survives persistence -------------------------


def test_every_fit_gets_a_fresh_result_id():
    x = np.linspace(0, 10, 10)
    y = 2.0 * x

    a = fit_curve(x, y, LINEAR, source_dataset_id="d", x_column="x", y_column="y")
    b = fit_curve(x, y, LINEAR, source_dataset_id="d", x_column="x", y_column="y")

    assert a.result_id
    assert b.result_id
    assert a.result_id != b.result_id


def test_result_id_is_not_a_caller_supplied_parameter():
    """fit_curve() never accepts a result_id kwarg -- every call is a
    genuinely new scientific result, always freshly generated."""
    import inspect

    assert "result_id" not in inspect.signature(fit_curve).parameters


# --- FitResult.to_dict() / from_dict(): polymorphic persistence round trip ---


def test_fit_result_to_dict_from_dict_round_trip_preserves_everything():
    from gnovi_plot.analysis.results import result_from_dict

    x = np.linspace(0, 10, 25)
    y = 3.0 * x + 2.0 + np.array([0.0] * 24 + [0.3])  # tiny noise -> real param_errors
    result = fit_curve(
        x,
        y,
        LINEAR,
        source_dataset_id="dataset-abc",
        source_dataset_name="My Dataset",
        source_series_id="series-xyz",
        source_series_label="My Series",
        x_column="voltage",
        y_column="current",
        row_range=(0, 25),
        source_panel_id="panel-123",
    )

    restored = result_from_dict(result.to_dict())

    assert isinstance(restored, FitResult)
    assert restored.result_id == result.result_id
    assert restored.source_panel_id == "panel-123"
    assert restored.source_dataset_id == "dataset-abc"
    assert restored.source_dataset_name == "My Dataset"
    assert restored.source_series_id == "series-xyz"
    assert restored.source_series_label == "My Series"
    assert restored.x_column == "voltage"
    assert restored.y_column == "current"
    assert restored.row_range == (0, 25)
    assert restored.model == result.model
    assert restored.params == result.params
    assert restored.param_errors == result.param_errors
    assert restored.r_squared == result.r_squared
    assert restored.formula == result.formula
    assert restored.residual_sum_of_squares == result.residual_sum_of_squares
    assert restored.rmse == result.rmse
    assert restored.n_points == result.n_points
    # Adjusted R^2 is derived (n_points/params/r_squared), never a stored
    # field -- reproducing correctly is proof those inputs round-tripped.
    assert restored.adjusted_r_squared() == result.adjusted_r_squared()


def test_fit_curve_stamps_the_curve_sampling_range_from_the_fitted_data():
    """`curve_x_min`/`curve_x_max`/`curve_num_points` are captured at fit
    time, from the exact (x, y) fitted -- so "Add Fit Curve to Plot" can
    later regenerate this exact curve without needing the source data's
    *current* range (see `FitResult`'s own docstring)."""
    from gnovi_plot.analysis.fitting import DEFAULT_CURVE_SAMPLES

    x = np.linspace(-3.0, 7.0, 25)
    y = 3.0 * x + 2.0

    result = fit_curve(x, y, LINEAR, **_PROVENANCE)

    assert result.curve_x_min == pytest.approx(-3.0)
    assert result.curve_x_max == pytest.approx(7.0)
    assert result.curve_num_points == DEFAULT_CURVE_SAMPLES


def test_curve_sampling_range_round_trips_through_to_dict_from_dict():
    from gnovi_plot.analysis.results import result_from_dict

    x = np.linspace(0, 10, 25)
    y = 3.0 * x + 2.0
    result = fit_curve(x, y, LINEAR, **_PROVENANCE)

    restored = result_from_dict(result.to_dict())

    assert restored.curve_x_min == result.curve_x_min
    assert restored.curve_x_max == result.curve_x_max
    assert restored.curve_num_points == result.curve_num_points


def test_from_dict_defaults_curve_sampling_range_to_none_when_absent():
    """A `FitResult` persisted before this field existed lacks the key
    entirely -- must load as `None`, not raise, so callers can fall back
    to resolving the source's current live data range."""
    from gnovi_plot.analysis.results import result_from_dict

    x = np.linspace(0, 10, 10)
    y = 2.0 * x
    result = fit_curve(x, y, LINEAR, source_dataset_id="d", x_column="x", y_column="y")
    data = result.to_dict()
    del data["curve_x_min"]
    del data["curve_x_max"]
    del data["curve_num_points"]

    restored = result_from_dict(data)

    assert restored.curve_x_min is None
    assert restored.curve_x_max is None
    assert restored.curve_num_points is None


def test_fit_result_from_dict_residuals_recompute_from_restored_fields():
    """No residual arrays are ever persisted -- compute_residuals() on a
    restored FitResult must still work, using the caller's live (x, y)."""
    from gnovi_plot.analysis.results import result_from_dict

    x = np.linspace(0, 10, 10)
    y = 2.0 * x + 1.0
    result = fit_curve(x, y, LINEAR, source_dataset_id="d", x_column="x", y_column="y")

    restored = result_from_dict(result.to_dict())
    residuals = restored.compute_residuals(x, y)

    assert np.allclose(residuals.residuals, 0.0, atol=1e-9)


def test_fit_result_from_dict_generates_a_result_id_when_absent():
    """Defensive backward compatibility for a hypothetical file saved
    between this field's introduction and its first real release --
    mirrors Panel.id's own generate-on-load fallback."""
    x = np.linspace(0, 10, 10)
    y = 2.0 * x
    data = fit_curve(x, y, LINEAR, source_dataset_id="d", x_column="x", y_column="y").to_dict()
    del data["result_id"]

    restored = FitResult.from_dict(data)

    assert restored.result_id


def test_fit_result_is_registered_for_polymorphic_dispatch():
    from gnovi_plot.analysis.results import _RESULT_KIND_REGISTRY

    assert _RESULT_KIND_REGISTRY["fit"] is FitResult


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


# --- Descriptive provenance snapshot (dataset name / series label) --------


def test_fit_time_dataset_name_is_retained():
    x = np.linspace(0, 10, 10)
    y = 2.0 * x

    result = fit_curve(
        x, y, LINEAR, source_dataset_name="Ferricyanide 50 mV/s", **_PROVENANCE
    )

    assert result.source_dataset_name == "Ferricyanide 50 mV/s"


def test_fit_time_series_label_is_retained():
    x = np.linspace(0, 10, 10)
    y = 2.0 * x

    result = fit_curve(
        x, y, LINEAR, source_series_label="Current vs Potential", **_PROVENANCE
    )

    assert result.source_series_label == "Current vs Potential"


def test_stable_ids_remain_retained_alongside_names():
    x = np.linspace(0, 10, 10)
    y = 2.0 * x

    result = fit_curve(
        x,
        y,
        LINEAR,
        source_dataset_id="dataset-abc",
        source_dataset_name="Ferricyanide 50 mV/s",
        source_series_id="series-xyz",
        source_series_label="Current vs Potential",
        x_column="x",
        y_column="y",
    )

    assert result.source_dataset_id == "dataset-abc"
    assert result.source_series_id == "series-xyz"
    assert result.source_dataset_name == "Ferricyanide 50 mV/s"
    assert result.source_series_label == "Current vs Potential"


def test_name_fields_default_to_none_when_not_supplied():
    """Older/degenerate call sites that don't pass names must still work
    -- names are optional, ids stay the required, authoritative link."""
    x = np.linspace(0, 10, 10)
    y = 2.0 * x

    result = fit_curve(x, y, LINEAR, **_PROVENANCE)

    assert result.source_dataset_name is None
    assert result.source_series_label is None
    assert result.source_dataset_id == "dataset-abc"


def test_names_are_json_safe_in_to_dict():
    import json

    x = np.linspace(0, 10, 10)
    y = 2.0 * x
    result = fit_curve(
        x, y, LINEAR, source_dataset_name="Ferricyanide 50 mV/s", **_PROVENANCE
    )

    data = result.to_dict()
    assert data["source_dataset_name"] == "Ferricyanide 50 mV/s"
    assert data["source_series_label"] is None
    json.dumps(data)


# --- Fit-quality metrics: RMSE / RSS / n_points / adjusted R² -------------


def test_rmse_and_rss_are_zero_for_an_exact_fit():
    x = np.linspace(0, 10, 25)
    y = 3.0 * x + 2.0

    result = fit_curve(x, y, LINEAR, **_PROVENANCE)

    assert result.residual_sum_of_squares == pytest.approx(0.0, abs=1e-9)
    assert result.rmse == pytest.approx(0.0, abs=1e-9)
    assert result.n_points == 25


def test_rmse_matches_manually_computed_residuals():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 10, 40)
    y = 3.0 * x + 2.0 + rng.normal(scale=0.7, size=x.shape)

    result = fit_curve(x, y, LINEAR, **_PROVENANCE)

    fitted = result.params["a"] * x + result.params["b"]
    expected_rss = float(np.sum((y - fitted) ** 2))
    expected_rmse = float(np.sqrt(expected_rss / len(x)))

    assert result.residual_sum_of_squares == pytest.approx(expected_rss, rel=1e-6)
    assert result.rmse == pytest.approx(expected_rmse, rel=1e-6)
    # RMSE/RSS/n_points must always agree with each other.
    assert result.rmse == pytest.approx(
        (result.residual_sum_of_squares / result.n_points) ** 0.5, rel=1e-9
    )


def test_n_points_reflects_cleaned_data_not_raw_input_length():
    x = np.linspace(0, 10, 25)
    y = 3.0 * x + 2.0
    y_with_gaps = y.copy()
    y_with_gaps[[3, 7, 12]] = np.nan

    result = fit_curve(x, y_with_gaps, LINEAR, **_PROVENANCE)

    assert result.n_points == 22  # 25 - 3 dropped NaNs


def test_adjusted_r_squared_is_defined_with_enough_points():
    x = np.linspace(0, 10, 40)
    rng = np.random.default_rng(3)
    y = 3.0 * x + 2.0 + rng.normal(scale=0.5, size=x.shape)

    result = fit_curve(x, y, LINEAR, **_PROVENANCE)
    adjusted = result.adjusted_r_squared()

    n, p = result.n_points, len(result.params)
    expected = 1.0 - (1.0 - result.r_squared) * (n - 1) / (n - p - 1)
    assert adjusted == pytest.approx(expected)
    assert adjusted <= result.r_squared + 1e-9  # adjustment never inflates R²


def test_adjusted_r_squared_is_none_when_barely_enough_points_to_fit():
    # LINEAR needs >= 3 points to fit at all (_min_points); with exactly 3,
    # n - p - 1 == 0, so adjusted R² is undefined.
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([2.0, 5.0, 8.0])

    result = fit_curve(x, y, LINEAR, **_PROVENANCE)

    assert result.n_points == 3
    assert result.adjusted_r_squared() is None


def test_details_includes_adjusted_r_squared_only_when_defined():
    x = np.linspace(0, 10, 40)
    result = fit_curve(x, 3.0 * x + 2.0, LINEAR, **_PROVENANCE)
    assert any(label == "R² (adjusted)" for label, _ in result.details())

    x_small = np.array([0.0, 1.0, 2.0])
    y_small = np.array([2.0, 5.0, 8.0])
    result_small = fit_curve(x_small, y_small, LINEAR, **_PROVENANCE)
    assert not any(label == "R² (adjusted)" for label, _ in result_small.details())


def test_details_includes_rmse_and_rss_but_not_provenance_ids():
    x = np.linspace(0, 10, 25)
    result = fit_curve(x, 3.0 * x + 2.0, LINEAR, **_PROVENANCE)
    detail_labels = dict(result.details())

    assert "RMSE" in detail_labels
    assert "RSS" in detail_labels
    # Provenance moved to provenance_details() -- details() is metrics-only.
    assert "Source dataset" not in detail_labels
    assert "Source series" not in detail_labels
    assert "Columns" not in detail_labels


def test_provenance_details_has_the_ids_and_columns_instead():
    x = np.linspace(0, 10, 25)
    result = fit_curve(x, 3.0 * x + 2.0, LINEAR, row_range=(0, 25), **_PROVENANCE)
    provenance = dict(result.provenance_details())

    assert provenance["Source dataset ID"] == "dataset-abc"
    assert provenance["Source series ID"] == "series-xyz"
    assert provenance["Columns"] == "x → y"
    assert provenance["Row range"] == "0–25"


def test_to_dict_includes_new_metric_fields():
    import json

    x = np.linspace(0, 10, 25)
    result = fit_curve(x, 3.0 * x + 2.0, LINEAR, **_PROVENANCE)
    data = result.to_dict()

    assert data["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert data["residual_sum_of_squares"] == pytest.approx(0.0, abs=1e-9)
    assert data["n_points"] == 25
    json.dumps(data)  # still JSON-safe (native floats/ints, not numpy scalars)


# --- Residuals -------------------------------------------------------------


def test_fit_result_supports_residuals():
    x = np.linspace(0, 10, 10)
    result = fit_curve(x, 2.0 * x, LINEAR, **_PROVENANCE)
    assert result.supports_residuals() is True


def test_compute_residuals_uses_observed_minus_fitted_sign_convention():
    x = np.linspace(0, 10, 10)
    result = fit_curve(x, 2.0 * x, LINEAR, **_PROVENANCE)

    # Perturb one observed point above the fitted line, one below.
    y_observed = 2.0 * x
    y_observed[0] += 5.0
    y_observed[1] -= 3.0

    residual_data = compute_residuals(result, x, y_observed)

    assert isinstance(residual_data, ResidualData)
    assert residual_data.residuals[0] == pytest.approx(5.0, abs=1e-6)
    assert residual_data.residuals[1] == pytest.approx(-3.0, abs=1e-6)
    assert residual_data.observed[0] == pytest.approx(y_observed[0])
    assert residual_data.fitted[0] == pytest.approx(2.0 * x[0])


def test_compute_residuals_is_near_zero_for_an_exact_fit():
    x = np.linspace(0, 10, 25)
    y = 3.0 * x + 2.0
    result = fit_curve(x, y, LINEAR, **_PROVENANCE)

    residual_data = compute_residuals(result, x, y)

    assert residual_data.residuals == pytest.approx(np.zeros_like(x), abs=1e-6)


def test_result_compute_residuals_method_delegates_to_free_function():
    x = np.linspace(0, 10, 10)
    y = 2.0 * x
    result = fit_curve(x, y, LINEAR, **_PROVENANCE)

    via_method = result.compute_residuals(x, y)
    via_function = compute_residuals(result, x, y)

    assert via_method.residuals == pytest.approx(via_function.residuals)
