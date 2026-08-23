from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.optimize import curve_fit

from gnovi_plot.analysis.results import AnalysisResult, register_result_kind

LINEAR = "linear"
POLYNOMIAL = "polynomial"
EXPONENTIAL = "exponential"
GAUSSIAN = "gaussian"

MODELS: tuple[str, ...] = (LINEAR, POLYNOMIAL, EXPONENTIAL, GAUSSIAN)

# Number of free parameters for the fixed-arity models -- used to size the
# "enough points to fit at all" check. POLYNOMIAL is sized separately from
# `degree` (see `_min_points`).
_PARAM_COUNT = {LINEAR: 2, EXPONENTIAL: 3, GAUSSIAN: 4}

# Default density for a regenerated fit curve's smooth (x, y) samples (see
# `sample_fit_curve`) -- defined here, ahead of `fit_curve`, so a freshly
# computed `FitResult` can stamp its own `curve_num_points` with it at fit
# time (see `FitResult.curve_num_points`).
DEFAULT_CURVE_SAMPLES = 200


class FitError(Exception):
    """Raised when a curve fit cannot be performed: not enough numeric
    data points for the chosen model, or the underlying solver fails to
    converge. Callers should not guess in that case, only surface the
    message -- mirrors `analysis.cycles.CycleDetectionError`."""


@register_result_kind
@dataclass
class FitResult(AnalysisResult):
    """The fitting-specific `AnalysisResult`: a model identifier, its
    fitted parameter values (and uncertainties, where the solver could
    estimate them), goodness-of-fit metrics, and a human-readable formula
    template for the model itself (not the fitted numbers -- those are
    `params`).

    `residual_sum_of_squares`/`rmse`/`n_points` are computed once, at fit
    time, from the exact data that was fit -- small, meaningful summary
    scalars, stored alongside `r_squared` for the same reason it already
    is. Full per-point residuals are deliberately *not* stored here (see
    `compute_residuals`/`ResidualData`): they're exactly as large as the
    source data, fully reconstructible from `model`/`params` plus the
    source series' current data, so storing them would only be a
    redundant, potentially-stale copy.

    `curve_x_min`/`curve_x_max`/`curve_num_points` are the sampling range
    `fit_curve()` was actually run against -- captured once, at fit time,
    from the exact (finite) `x` array fitted, exactly like `n_points`.
    Unlike `n_points` (a count of *source* data points, used for
    goodness-of-fit stats), these describe the smooth *rendered curve*
    `sample_fit_curve` should redraw, independent of whatever the source
    series' live data range happens to be later. "Add Fit Curve to Plot"
    (see `gui.widgets.analysis_panel.AnalysisPanel._resolve_curve_range`)
    prefers this stored range so regenerating a fit curve for a result
    from `analysis.panel_results.PanelResultHistory` -- possibly long
    after its derived `PlotSeries` was removed, possibly after a project
    reload -- reproduces exactly what was actually fitted/plotted at
    analysis time, not whatever the source data looks like *now*. `None`
    for a `FitResult` persisted before this field existed; callers fall
    back to resolving the source's current live data range in that case.
    """

    kind: ClassVar[str] = "fit"

    model: str
    params: dict[str, float]
    param_errors: dict[str, float] | None
    r_squared: float
    formula: str
    residual_sum_of_squares: float
    rmse: float
    n_points: int
    curve_x_min: float | None = None
    curve_x_max: float | None = None
    curve_num_points: int | None = None

    def summary(self) -> str:
        param_str = ", ".join(f"{name}={value:.4g}" for name, value in self.params.items())
        return f"{self.model} fit ({param_str}), R²={self.r_squared:.4f}"

    def details(self) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = [("Model", self.model), ("Formula", self.formula)]
        for name, value in self.params.items():
            error = self.param_errors.get(name) if self.param_errors is not None else None
            value_text = f"{value:.6g} ± {error:.2g}" if error is not None else f"{value:.6g}"
            rows.append((name, value_text))
        rows.append(("R²", f"{self.r_squared:.6f}"))
        adjusted = self.adjusted_r_squared()
        if adjusted is not None:
            rows.append(("R² (adjusted)", f"{adjusted:.6f}"))
        rows.append(("RMSE", f"{self.rmse:.6g}"))
        rows.append(("RSS", f"{self.residual_sum_of_squares:.6g}"))
        return rows

    def adjusted_r_squared(self) -> float | None:
        """R² adjusted for the number of fitted parameters -- `None` when
        undefined: needs at least one residual degree of freedom beyond
        fitting (`n_points > len(params) + 1`), the same condition
        `fit_curve`'s own `_min_points` enforces for the *un-adjusted*
        R²/RMSE to be meaningful in the first place, so this can only be
        `None` for a fit that only just barely cleared that bar."""
        n, p = self.n_points, len(self.params)
        denominator = n - p - 1
        if denominator <= 0:
            return None
        return 1.0 - (1.0 - self.r_squared) * (n - 1) / denominator

    def supports_residuals(self) -> bool:
        return True

    def compute_residuals(self, x, y) -> "ResidualData":
        return compute_residuals(self, x, y)

    def residual_window_subtitle(self) -> str:
        return f"{self.model} fit"

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "model": self.model,
                "params": dict(self.params),
                "param_errors": dict(self.param_errors) if self.param_errors is not None else None,
                "r_squared": self.r_squared,
                "formula": self.formula,
                "residual_sum_of_squares": self.residual_sum_of_squares,
                "rmse": self.rmse,
                "n_points": self.n_points,
                "curve_x_min": self.curve_x_min,
                "curve_x_max": self.curve_x_max,
                "curve_num_points": self.curve_num_points,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "FitResult":
        """Reconstruct from `to_dict()`'s output (see
        `analysis.results.result_from_dict`, the polymorphic dispatcher
        that calls this). Never reconstructs residuals -- those are
        always recomputed on demand from the restored fields plus the
        source series' *current* live data (see `compute_residuals`),
        exactly as for a same-session result. `result_id`/`source_panel_id`
        fall back to a freshly generated id / `None` when absent, for a
        hypothetical file saved between this field's introduction and its
        first real release -- not a case expected in practice, but the
        same defensive "generate on load if missing" `Panel.id` already
        established (see `plotting.figure.Panel.from_dict`)."""
        row_range = data.get("row_range")
        return cls(
            source_dataset_id=data["source_dataset_id"],
            source_dataset_name=data.get("source_dataset_name"),
            source_series_id=data.get("source_series_id"),
            source_series_label=data.get("source_series_label"),
            x_column=data["x_column"],
            y_column=data["y_column"],
            row_range=tuple(row_range) if row_range is not None else None,
            source_panel_id=data.get("source_panel_id"),
            result_id=data.get("result_id") or uuid.uuid4().hex,
            model=data["model"],
            params=dict(data["params"]),
            param_errors=dict(data["param_errors"]) if data.get("param_errors") is not None else None,
            r_squared=data["r_squared"],
            formula=data["formula"],
            residual_sum_of_squares=data["residual_sum_of_squares"],
            rmse=data["rmse"],
            n_points=data["n_points"],
            curve_x_min=data.get("curve_x_min"),
            curve_x_max=data.get("curve_x_max"),
            curve_num_points=data.get("curve_num_points"),
        )


def _min_points(model: str, degree: int) -> int:
    """The fewest finite (x, y) pairs needed to even attempt `model` --
    strictly more than the parameter count, so there's at least one
    residual degree of freedom (an exact fit through every point is a
    curve you drew, not one you fitted)."""
    if model == POLYNOMIAL:
        return degree + 2
    return _PARAM_COUNT[model] + 1


def _fit_quality(y: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """`(r_squared, residual_sum_of_squares, rmse)` for `y` vs. `y_pred` --
    one pass over the residuals for every scalar goodness-of-fit metric
    `FitResult` stores, so they can never silently disagree with each
    other."""
    residual = y - y_pred
    ss_res = float(np.sum(residual**2))
    rmse = float(np.sqrt(ss_res / len(y))) if len(y) else 0.0
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        # Constant y: any nonzero ss_tot-relative residual is a real miss,
        # but a solver's floating-point noise on an exact fit (e.g.
        # ~1e-27 from np.polyfit on flat data) must not read as one.
        scale = float(np.sum(y**2)) or 1.0
        r_squared = 1.0 if ss_res <= 1e-9 * scale else 0.0
    else:
        r_squared = 1.0 - ss_res / ss_tot
    return r_squared, ss_res, rmse


def _errors_from_covariance(cov: np.ndarray, param_names: Sequence[str]) -> dict[str, float] | None:
    """`None` (uncertainties "not available") if the solver couldn't
    estimate a covariance matrix (returned as inf-filled by both
    `numpy.polyfit` and `scipy.optimize.curve_fit` when the fit is
    rank-deficient or otherwise underdetermined) rather than reporting a
    meaningless infinite error."""
    variances = np.diag(cov)
    if not np.all(np.isfinite(variances)) or np.any(variances < 0):
        return None
    return {name: float(np.sqrt(var)) for name, var in zip(param_names, variances)}


def _fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[dict[str, float], dict[str, float] | None, np.ndarray, str]:
    coeffs, cov = np.polyfit(x, y, 1, cov=True)
    a, b = float(coeffs[0]), float(coeffs[1])
    params = {"a": a, "b": b}
    errors = _errors_from_covariance(cov, ("a", "b"))
    y_pred = np.polyval(coeffs, x)
    formula = "y = a·x + b"
    return params, errors, y_pred, formula


def _fit_polynomial(
    x: np.ndarray, y: np.ndarray, degree: int
) -> tuple[dict[str, float], dict[str, float] | None, np.ndarray, str]:
    coeffs, cov = np.polyfit(x, y, degree, cov=True)
    # numpy.polyfit orders coefficients highest power first; expose them
    # ascending (c0 = constant term) so the formula reads left-to-right.
    ascending = coeffs[::-1]
    names = [f"c{i}" for i in range(degree + 1)]
    params = {name: float(value) for name, value in zip(names, ascending)}
    errors = _errors_from_covariance(cov[::-1, ::-1], names)
    y_pred = np.polyval(coeffs, x)
    terms = ["c0"] + [f"c{i}·x{_superscript(i)}" if i > 1 else f"c1·x" for i in range(1, degree + 1)]
    formula = "y = " + " + ".join(terms)
    return params, errors, y_pred, formula


def _superscript(n: int) -> str:
    digits = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    return "".join(digits[int(d)] for d in str(n))


def _exponential_func(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(b * x) + c


def _exponential_guess(x: np.ndarray, y: np.ndarray) -> list[float]:
    y_span = float(np.ptp(y)) or 1.0
    rising = (y[-1] - y[0]) >= 0
    a0 = y_span if rising else -y_span
    c0 = float(np.min(y)) if rising else float(np.max(y))
    x_span = float(np.ptp(x)) or 1.0
    b0 = (1.0 if rising else -1.0) / x_span
    return [a0, b0, c0]


def _gaussian_func(x: np.ndarray, amplitude: float, mean: float, sigma: float, offset: float) -> np.ndarray:
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma**2)) + offset


def _gaussian_guess(x: np.ndarray, y: np.ndarray) -> list[float]:
    offset0 = float(np.min(y))
    amplitude0 = float(np.max(y) - np.min(y)) or 1.0
    mean0 = float(x[np.argmax(y)])
    sigma0 = (float(np.ptp(x)) / 6.0) or 1.0
    return [amplitude0, mean0, sigma0, offset0]


_CURVE_FIT_MODELS = {
    EXPONENTIAL: (_exponential_func, ("a", "b", "c"), _exponential_guess, "y = a·exp(b·x) + c"),
    GAUSSIAN: (
        _gaussian_func,
        ("amplitude", "mean", "sigma", "offset"),
        _gaussian_guess,
        "y = amplitude·exp(-((x - mean)²)/(2·sigma²)) + offset",
    ),
}


def fit_curve(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    model: str,
    *,
    source_dataset_id: str,
    x_column: str,
    y_column: str,
    source_dataset_name: str | None = None,
    source_series_id: str | None = None,
    source_series_label: str | None = None,
    row_range: tuple[int, int] | None = None,
    source_panel_id: str | None = None,
    degree: int = 2,
    initial_guess: Sequence[float] | None = None,
) -> FitResult:
    """Fit `model` to `(x, y)` and return a `FitResult`.

    Pure numerical code: `x`/`y` are plain arrays, and `source_dataset_id`/
    `source_dataset_name`/`source_series_id`/`source_series_label`/
    `x_column`/`y_column`/`row_range`/`source_panel_id` are opaque
    provenance values threaded straight into the returned `FitResult` --
    this function never imports `Dataset`, Qt, or anything from
    `gnovi_plot.plotting`. The caller (the GUI layer) is responsible for
    supplying real ids/names. `source_dataset_name`/`source_series_label`
    are a descriptive snapshot of what was fitted, captured now because
    the caller has it live -- display code prefers this snapshot over
    re-resolving the id later (see `AnalysisResult`'s docstring).
    `source_panel_id` is `Panel.id` of whatever panel this fit was run
    against, if any (see `AnalysisResult.source_panel_id`). The returned
    `FitResult`'s own `result_id` is always freshly generated here --
    never a caller-supplied parameter, since every call to `fit_curve()`
    is a genuinely new scientific result (see `AnalysisResult.result_id`).

    `degree` only applies to `POLYNOMIAL` (default 2, i.e. quadratic).
    `initial_guess` only applies to `EXPONENTIAL`/`GAUSSIAN`, whose solver
    is iterative; omit it to use a heuristic seed derived from `x`/`y`.

    Raises `FitError` if `model` isn't recognized, there isn't enough
    finite numeric data for the model's parameter count, or the solver
    fails to converge.
    """
    if model not in MODELS:
        raise FitError(f"Unknown fit model '{model}'; expected one of {MODELS}")
    if model == POLYNOMIAL and degree < 1:
        raise FitError(f"Polynomial degree must be at least 1 (got {degree})")

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise FitError(f"x and y must have the same shape (got {x_arr.shape} and {y_arr.shape})")

    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite_mask]
    y_arr = y_arr[finite_mask]

    required = _min_points(model, degree)
    if len(x_arr) < required:
        raise FitError(
            f"Not enough numeric (x, y) pairs to fit a {model} model "
            f"(found {len(x_arr)}, need at least {required})."
        )

    try:
        if model == LINEAR:
            params, errors, y_pred, formula = _fit_linear(x_arr, y_arr)
        elif model == POLYNOMIAL:
            params, errors, y_pred, formula = _fit_polynomial(x_arr, y_arr, degree)
        else:
            func, param_names, guess_fn, formula = _CURVE_FIT_MODELS[model]
            p0 = list(initial_guess) if initial_guess is not None else guess_fn(x_arr, y_arr)
            popt, pcov = curve_fit(func, x_arr, y_arr, p0=p0, maxfev=10000)
            params = {name: float(value) for name, value in zip(param_names, popt)}
            errors = _errors_from_covariance(pcov, param_names)
            y_pred = func(x_arr, *popt)
    except FitError:
        raise
    except Exception as exc:  # RuntimeError (no convergence), ValueError, LinAlgError, ...
        raise FitError(f"Fitting a {model} model failed: {exc}") from exc

    r_squared, ss_res, rmse = _fit_quality(y_arr, y_pred)

    return FitResult(
        source_dataset_id=source_dataset_id,
        source_dataset_name=source_dataset_name,
        source_series_id=source_series_id,
        source_series_label=source_series_label,
        x_column=x_column,
        y_column=y_column,
        row_range=row_range,
        source_panel_id=source_panel_id,
        result_id=uuid.uuid4().hex,
        model=model,
        params=params,
        param_errors=errors,
        r_squared=r_squared,
        formula=formula,
        residual_sum_of_squares=ss_res,
        rmse=rmse,
        n_points=len(x_arr),
        curve_x_min=float(x_arr.min()),
        curve_x_max=float(x_arr.max()),
        curve_num_points=DEFAULT_CURVE_SAMPLES,
    )


def evaluate_fit(result: FitResult, x: Sequence[float] | np.ndarray) -> np.ndarray:
    """Evaluate `result`'s fitted model -- `result.model` with
    `result.params` -- at `x`. Pure numerical code, same as `fit_curve`:
    no Qt, no plotting, no Dataset. Reuses the same model functions
    `fit_curve` fit with, so a plotted fit curve is always mathematically
    identical to what was actually fit, never a re-derivation that could
    drift from it.
    """
    x_arr = np.asarray(x, dtype=float)
    if result.model == LINEAR:
        return result.params["a"] * x_arr + result.params["b"]
    if result.model == POLYNOMIAL:
        degree = len(result.params) - 1
        ascending = [result.params[f"c{i}"] for i in range(degree + 1)]
        return np.polyval(ascending[::-1], x_arr)
    if result.model in _CURVE_FIT_MODELS:
        func, param_names, _guess_fn, _formula = _CURVE_FIT_MODELS[result.model]
        return func(x_arr, *(result.params[name] for name in param_names))
    raise FitError(f"Cannot evaluate unknown fit model '{result.model}'")


def sample_fit_curve(
    result: FitResult, x_min: float, x_max: float, num_points: int = DEFAULT_CURVE_SAMPLES
) -> tuple[np.ndarray, np.ndarray]:
    """A smooth `(x, y)` curve for `result`'s fitted model: `num_points`
    evenly spaced samples across `[x_min, x_max]`, evaluated via
    `evaluate_fit`. This is what "Add Fit Curve to Plot" turns into a
    derived Dataset -- resampled independently of however densely the
    original data was measured, so the plotted curve is smooth regardless.

    Raises `FitError` if `num_points` is too small to draw a curve.
    """
    if num_points < 2:
        raise FitError(f"num_points must be at least 2 (got {num_points})")
    x = np.linspace(x_min, x_max, num_points)
    return x, evaluate_fit(result, x)


@dataclass
class ResidualData:
    """Per-point residuals for a fit -- observed minus fitted, at whatever
    `(x, y)` the caller supplies (typically the source series' *current*
    data, re-extracted fresh; see `compute_residuals`). Never persisted:
    fully reconstructible from a `FitResult`'s stored `model`/`params`
    plus the source data, so storing it separately would only be a
    redundant, potentially-stale copy (see `FitResult`'s own docstring).
    """

    x: np.ndarray
    observed: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray  # observed - fitted


def compute_residuals(result: FitResult, x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> ResidualData:
    """`observed - fitted` at `(x, y)`, using `result`'s fitted model via
    `evaluate_fit` -- the same model-evaluation code `sample_fit_curve`
    uses, so residuals can never disagree with what "Add Fit Curve to
    Plot" actually drew. Pure: no Qt, no Dataset -- the caller resolves
    `x`/`y` (e.g. from the live source series' current data).
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    fitted = evaluate_fit(result, x_arr)
    return ResidualData(x=x_arr, observed=y_arr, fitted=fitted, residuals=y_arr - fitted)
