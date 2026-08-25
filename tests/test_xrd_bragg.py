"""modules.xrd.bragg: d-spacing via Bragg's law.

Every expected value here is computed independently of `d_spacing` itself,
using a hand-written scalar formula (`_expected_d`, below) built straight
from `d = lambda / (2 * sin(theta))` with `math.sin`/`math.radians` --
never by calling the implementation under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gnovi_plot.modules.xrd.bragg import InvalidBraggInputError, d_spacing


def _expected_d(two_theta_deg: float, wavelength_angstrom: float) -> float:
    theta_rad = math.radians(two_theta_deg / 2.0)
    return wavelength_angstrom / (2.0 * math.sin(theta_rad))


CU_KALPHA1 = 1.540562


@pytest.mark.parametrize("two_theta", [10.0, 28.4, 45.0, 60.0, 90.0, 120.0, 150.0, 179.0])
def test_scalar_d_spacing_matches_independent_calculation(two_theta):
    result = d_spacing(two_theta, CU_KALPHA1)
    assert result == pytest.approx(_expected_d(two_theta, CU_KALPHA1), rel=1e-12)


def test_scalar_input_returns_a_python_float_not_an_ndarray():
    result = d_spacing(30.0, CU_KALPHA1)
    assert isinstance(result, float)


def test_known_combination_cu_ka1_at_30_degrees():
    # theta = 15 deg, sin(15 deg) = 0.258819045102521, so
    # d = 1.540562 / (2 * 0.258819045102521) = 2.976247... (independently
    # computed by hand -- see the docstring above for why this file never
    # asserts against d_spacing's own output).
    expected = 1.540562 / (2 * math.sin(math.radians(15.0)))
    assert d_spacing(30.0, CU_KALPHA1) == pytest.approx(expected, rel=1e-12)


def test_custom_wavelength():
    expected = _expected_d(45.0, 0.71073)
    assert d_spacing(45.0, 0.71073) == pytest.approx(expected, rel=1e-12)


def test_vector_input_matches_independent_calculation_elementwise():
    two_theta = np.array([10.0, 20.0, 30.0, 45.0, 60.0, 90.0])
    result = d_spacing(two_theta, CU_KALPHA1)
    assert isinstance(result, np.ndarray)
    expected = np.array([_expected_d(t, CU_KALPHA1) for t in two_theta])
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_vector_input_preserves_shape():
    two_theta = np.linspace(5.0, 170.0, 50)
    result = d_spacing(two_theta, CU_KALPHA1)
    assert result.shape == two_theta.shape


def test_larger_two_theta_gives_smaller_d_spacing():
    """Physical sanity check, independent of the exact formula: d shrinks
    monotonically as 2*theta grows over (0, 180)."""
    values = d_spacing(np.array([10.0, 30.0, 60.0, 90.0, 150.0]), CU_KALPHA1)
    assert np.all(np.diff(values) < 0)


@pytest.mark.parametrize("wavelength", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_wavelength_raises(wavelength):
    with pytest.raises(InvalidBraggInputError):
        d_spacing(30.0, wavelength)


@pytest.mark.parametrize("two_theta", [0.0, -5.0, 180.0, 200.0, float("nan"), float("inf")])
def test_invalid_two_theta_raises(two_theta):
    with pytest.raises(InvalidBraggInputError):
        d_spacing(two_theta, CU_KALPHA1)


def test_one_invalid_value_in_an_array_raises_for_the_whole_call():
    two_theta = np.array([10.0, 20.0, float("nan"), 40.0])
    with pytest.raises(InvalidBraggInputError):
        d_spacing(two_theta, CU_KALPHA1)


def test_one_out_of_range_value_in_an_array_raises():
    two_theta = np.array([10.0, 20.0, 185.0, 40.0])
    with pytest.raises(InvalidBraggInputError):
        d_spacing(two_theta, CU_KALPHA1)
