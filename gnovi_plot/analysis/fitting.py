from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.optimize import curve_fit

from gnovi_plot.analysis.results import AnalysisResult

LINEAR = "linear"
POLYNOMIAL = "polynomial"
EXPONENTIAL = "exponential"
GAUSSIAN = "gaussian"

MODELS: tuple[str, ...] = (LINEAR, POLYNOMIAL, EXPONENTIAL, GAUSSIAN)

# Number of free parameters for the fixed-arity models -- used to size the
# "enough points to fit at all" check. POLYNOMIAL is sized separately from
# `degree` (see `_min_points`).
_PARAM_COUNT = {LINEAR: 2, EXPONENTIAL: 3, GAUSSIAN: 4}


class FitError(Exception):
    """Raised when a curve fit cannot be performed: not enough numeric
    data points for the chosen model, or the underlying solver fails to
    converge. Callers should not guess in that case, only surface the
    message -- mirrors `analysis.cycles.CycleDetectionError`."""


@dataclass
class FitResult(AnalysisResult):
    """The fitting-specific `AnalysisResult`: a model identifier, its
    fitted parameter values (and uncertainties, where the solver could
    estimate them), goodness of fit, and a human-readable formula template
    for the model itself (not the fitted numbers -- those are `params`).
    """

    kind: ClassVar[str] = "fit"

    model: str
    params: dict[str, float]
    param_errors: dict[str, float] | None
    r_squared: float
    formula: str

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
        rows.append(("Source dataset", self.source_dataset_id))
        if self.source_series_id is not None:
            rows.append(("Source series", self.source_series_id))
        rows.append(("Columns", f"{self.x_column} → {self.y_column}"))
        if self.row_range is not None:
            rows.append(("Row range", f"{self.row_range[0]}–{self.row_range[1]}"))
        return rows

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "model": self.model,
                "params": dict(self.params),
                "param_errors": dict(self.param_errors) if self.param_errors is not None else None,
                "r_squared": self.r_squared,
                "formula": self.formula,
            }
        )
        return data


def _min_points(model: str, degree: int) -> int:
    """The fewest finite (x, y) pairs needed to even attempt `model` --
    strictly more than the parameter count, so there's at least one
    residual degree of freedom (an exact fit through every point is a
    curve you drew, not one you fitted)."""
    if model == POLYNOMIAL:
        return degree + 2
    return _PARAM_COUNT[model] + 1


def _r_squared(y: np.ndarray, y_pred: np.ndarray) -> float:
    residual = y - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        # Constant y: any nonzero ss_tot-relative residual is a real miss,
        # but a solver's floating-point noise on an exact fit (e.g.
        # ~1e-27 from np.polyfit on flat data) must not read as one.
        scale = float(np.sum(y**2)) or 1.0
        return 1.0 if ss_res <= 1e-9 * scale else 0.0
    return 1.0 - ss_res / ss_tot


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
    source_series_id: str | None = None,
    row_range: tuple[int, int] | None = None,
    degree: int = 2,
    initial_guess: Sequence[float] | None = None,
) -> FitResult:
    """Fit `model` to `(x, y)` and return a `FitResult`.

    Pure numerical code: `x`/`y` are plain arrays, and `source_dataset_id`/
    `source_series_id`/`x_column`/`y_column`/`row_range` are opaque
    provenance values threaded straight into the returned `FitResult` --
    this function never imports `Dataset`, Qt, or anything from
    `gnovi_plot.plotting`. The caller (the GUI layer) is responsible for
    supplying real ids.

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

    r_squared = _r_squared(y_arr, y_pred)

    return FitResult(
        source_dataset_id=source_dataset_id,
        source_series_id=source_series_id,
        x_column=x_column,
        y_column=y_column,
        row_range=row_range,
        model=model,
        params=params,
        param_errors=errors,
        r_squared=r_squared,
        formula=formula,
    )


DEFAULT_CURVE_SAMPLES = 200


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
