"""gnovi_plot.modules.electrochemistry.common: unit helpers, the current
sign convention, ElectrodeContext, deterministic sweep segmentation, and
the charge-integration primitive.

Every numerical test separates ALGORITHM validation (does the code do the
arithmetic it claims?) from ELECTROCHEMICAL ASSUMPTIONS (which are stated
explicitly in each fixture/model), and checks source-array immutability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnovi_plot.modules.electrochemistry.common import (
    CurrentSignConvention,
    ChargeIntegrationError,
    ElectrodeContext,
    InvalidElectrodeContextError,
    SWEEP_FALLING,
    SWEEP_RISING,
    SweepSegmentationError,
    UnknownUnitError,
    convert_units,
    current_to_amperes,
    integrate_current,
    oxidative_sign,
    potential_to_volts,
    scan_rate_to_v_per_s,
    segment_sweeps,
)


# --- unit conversion --------------------------------------------------------


@pytest.mark.parametrize(
    "value, unit, expected",
    [(1000.0, "mV", 1.0), (2.5, "V", 2.5), (-250.0, "mV", -0.25)],
)
def test_potential_to_volts(value, unit, expected):
    assert potential_to_volts(value, unit) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value, unit, expected",
    [(1.0, "A", 1.0), (1.0, "mA", 1e-3), (1.0, "µA", 1e-6), (1.0, "uA", 1e-6), (1.0, "nA", 1e-9)],
)
def test_current_to_amperes_all_units(value, unit, expected):
    assert current_to_amperes(value, unit) == pytest.approx(expected)


@pytest.mark.parametrize("value, unit, expected", [(100.0, "mV/s", 0.1), (0.05, "V/s", 0.05)])
def test_scan_rate_to_v_per_s(value, unit, expected):
    assert scan_rate_to_v_per_s(value, unit) == pytest.approx(expected)


def test_convert_units_round_trip_and_charge():
    assert convert_units(5.0, "mC", "C", "charge") == pytest.approx(5e-3)
    assert convert_units(convert_units(3.3, "V", "mV", "potential"), "mV", "V", "potential") == pytest.approx(3.3)


def test_convert_units_rejects_unknown_unit_and_quantity():
    with pytest.raises(UnknownUnitError):
        convert_units(1.0, "kV", "V", "potential")
    with pytest.raises(UnknownUnitError):
        convert_units(1.0, "V", "V", "voltage")


# --- current sign convention ----------------------------------------------


def test_default_sign_convention_is_anodic_positive():
    from gnovi_plot.modules.electrochemistry.common import DEFAULT_SIGN_CONVENTION

    assert DEFAULT_SIGN_CONVENTION is CurrentSignConvention.ANODIC_POSITIVE


def test_oxidative_sign():
    assert oxidative_sign(CurrentSignConvention.ANODIC_POSITIVE) == 1
    assert oxidative_sign(CurrentSignConvention.CATHODIC_POSITIVE) == -1
    # accepts the raw string value too
    assert oxidative_sign("cathodic_positive") == -1


# --- ElectrodeContext -----------------------------------------------------


def test_electrode_context_is_all_optional_and_empty_by_default():
    ctx = ElectrodeContext()
    assert ctx.is_empty()
    assert ctx.area_cm2 is None and ctx.n is None and ctx.temperature_k is None


def test_electrode_context_round_trips_through_dict():
    ctx = ElectrodeContext(area_cm2=0.071, n=1, concentration_mol_cm3=1e-6, temperature_k=298.15,
                           reference_electrode="Ag/AgCl", electrolyte="0.1 M KCl")
    restored = ElectrodeContext.from_dict(ctx.to_dict())
    assert restored == ctx
    assert not restored.is_empty()


@pytest.mark.parametrize("field", ["area_cm2", "n", "concentration_mol_cm3", "temperature_k"])
def test_electrode_context_rejects_non_positive_supplied_values(field):
    with pytest.raises(InvalidElectrodeContextError):
        ElectrodeContext(**{field: 0.0})
    with pytest.raises(InvalidElectrodeContextError):
        ElectrodeContext(**{field: -1.0})


def test_electrode_context_never_silently_defaults():
    # A missing field stays None -- it does NOT become 1 cm2 / n=1 / 1 mM / 298 K.
    ctx = ElectrodeContext(area_cm2=0.5)
    assert ctx.n is None
    assert ctx.concentration_mol_cm3 is None
    assert ctx.temperature_k is None


# --- sweep segmentation --------------------------------------------------

_STEP = 0.001


def _triangle(cycles: int, positive_first: bool = True, lo=-0.2, hi=0.6) -> np.ndarray:
    n = round((hi - lo) / _STEP)
    rising = np.round(np.linspace(lo, hi, n + 1), 10)
    falling = rising[::-1]
    first, second = (rising, falling) if positive_first else (falling, rising)
    legs = [first]
    for k in range(1, 2 * cycles):
        legs.append((second if k % 2 else first)[1:])
    return np.concatenate(legs)


def test_segment_sweeps_one_rising_then_falling():
    e = _triangle(cycles=1, positive_first=True)
    segs = segment_sweeps(e)
    assert [s.direction for s in segs] == [SWEEP_RISING, SWEEP_FALLING]
    assert segs[0].start == 0
    assert segs[-1].end == len(e)
    # adjacent segments share the vertex row
    assert segs[1].start == segs[0].end - 1


def test_segment_sweeps_one_falling_then_rising():
    e = _triangle(cycles=1, positive_first=False)
    segs = segment_sweeps(e)
    assert [s.direction for s in segs] == [SWEEP_FALLING, SWEEP_RISING]


def test_segment_sweeps_three_cycles_gives_six_segments():
    e = _triangle(cycles=3)
    segs = segment_sweeps(e)
    assert len(segs) == 6
    assert [s.direction for s in segs] == [
        SWEEP_RISING, SWEEP_FALLING, SWEEP_RISING, SWEEP_FALLING, SWEEP_RISING, SWEEP_FALLING,
    ]


def test_segment_sweeps_arbitrary_starting_potential():
    # start mid-range, go up to the vertex, back down, up again
    e = np.concatenate([
        np.arange(0.1, 0.6 + _STEP, _STEP),
        np.arange(0.6 - _STEP, -0.2 - _STEP, -_STEP),
        np.arange(-0.2 + _STEP, 0.3 + _STEP, _STEP),
    ])
    segs = segment_sweeps(e)
    assert [s.direction for s in segs] == [SWEEP_RISING, SWEEP_FALLING, SWEEP_RISING]
    assert segs[0].e_start == pytest.approx(0.1)


def test_segment_sweeps_tolerates_noise_and_plateau_at_vertex():
    e = _triangle(cycles=2)
    rng = np.random.default_rng(1)
    e_noisy = e + rng.uniform(-0.3 * _STEP, 0.3 * _STEP, size=e.shape)
    # duplicate the sample at each vertex (a brief hold)
    is_vertex = np.isclose(e, 0.6) | np.isclose(e, -0.2)
    e_held = np.repeat(e_noisy, np.where(is_vertex, 2, 1))
    segs = segment_sweeps(e_held)
    assert [s.direction for s in segs] == [SWEEP_RISING, SWEEP_FALLING, SWEEP_RISING, SWEEP_FALLING]


def test_segment_sweeps_monotonic_lsv_like_data_is_one_segment_not_an_error():
    e = np.linspace(-0.2, 0.6, 400)
    segs = segment_sweeps(e)
    assert len(segs) == 1
    assert segs[0].direction == SWEEP_RISING


def test_segment_sweeps_rejects_flat_data():
    with pytest.raises(SweepSegmentationError):
        segment_sweeps(np.full(50, 0.3))


def test_segment_sweeps_ignores_nan_rows_but_keeps_positional_ranges():
    e = _triangle(cycles=1).astype(float)
    e[5] = np.nan
    segs = segment_sweeps(e)
    assert [s.direction for s in segs] == [SWEEP_RISING, SWEEP_FALLING]
    # the range still spans the original positions (may contain the NaN row)
    assert segs[0].start == 0


def test_segment_sweeps_does_not_mutate_input():
    e = _triangle(cycles=2)
    original = e.copy()
    segment_sweeps(e)
    np.testing.assert_array_equal(e, original)


def test_segment_sweeps_accepts_a_pandas_series():
    e = pd.Series(_triangle(cycles=1))
    segs = segment_sweeps(e)
    assert len(segs) == 2


# --- charge integration -------------------------------------------------


def test_integrate_current_constant_current_over_time():
    t = np.linspace(0.0, 5.0, 501)
    i = np.full_like(t, 2.0)
    # Q = I * dt = 2 A * 5 s = 10 C
    assert integrate_current(i, time=t) == pytest.approx(10.0)


def test_integrate_current_linear_current_over_time_analytic():
    # i(t) = 3t  ->  Q = ∫_0^4 3t dt = 1.5 * 16 = 24
    t = np.linspace(0.0, 4.0, 4001)
    i = 3.0 * t
    assert integrate_current(i, time=t) == pytest.approx(24.0, rel=1e-4)


def test_integrate_current_sign_is_preserved():
    t = np.linspace(0.0, 2.0, 201)
    assert integrate_current(np.full_like(t, -1.5), time=t) == pytest.approx(-3.0)


def test_integrate_current_e_over_v_matches_time_domain_for_constant_rate_sweep():
    v = 0.1  # V/s
    e = np.linspace(-0.2, 0.6, 2001)  # rising, constant rate
    t = (e - e[0]) / v
    i = 1e-5 + 2e-5 * np.exp(-((e - 0.25) / 0.03) ** 2)
    q_time = integrate_current(i, time=t)
    q_ev = integrate_current(i, potential=e, scan_rate_v_per_s=v)
    assert q_ev == pytest.approx(q_time, rel=1e-6)


def test_integrate_current_e_over_v_direction_agnostic():
    v = 0.05
    e_up = np.linspace(-0.2, 0.6, 1601)
    e_down = e_up[::-1]
    i = np.full_like(e_up, 4e-6)
    q_up = integrate_current(i, potential=e_up, scan_rate_v_per_s=v)
    q_down = integrate_current(i, potential=e_down, scan_rate_v_per_s=v)
    assert q_up == pytest.approx(q_down)
    assert q_up == pytest.approx(4e-6 * 0.8 / v)


def test_integrate_current_rejects_zero_and_negative_scan_rate():
    e = np.linspace(-0.2, 0.6, 100)
    i = np.ones_like(e)
    with pytest.raises(ChargeIntegrationError):
        integrate_current(i, potential=e, scan_rate_v_per_s=0.0)
    with pytest.raises(ChargeIntegrationError):
        integrate_current(i, potential=e, scan_rate_v_per_s=-0.1)


def test_integrate_current_rejects_non_monotonic_potential_for_e_over_v():
    e = np.concatenate([np.linspace(-0.2, 0.6, 200), np.linspace(0.6, -0.2, 200)[1:]])
    i = np.ones_like(e)
    with pytest.raises(ChargeIntegrationError):
        integrate_current(i, potential=e, scan_rate_v_per_s=0.1)


def test_integrate_current_rejects_non_monotonic_time():
    t = np.array([0.0, 1.0, 0.5, 2.0])
    with pytest.raises(ChargeIntegrationError):
        integrate_current(np.ones(4), time=t)


def test_integrate_current_rejects_both_and_neither_domain():
    i = np.ones(10)
    with pytest.raises(ChargeIntegrationError):
        integrate_current(i)
    with pytest.raises(ChargeIntegrationError):
        integrate_current(i, time=np.arange(10.0), potential=np.arange(10.0))


def test_integrate_current_rejects_non_finite_and_shape_mismatch():
    with pytest.raises(ChargeIntegrationError):
        integrate_current(np.array([1.0, np.nan, 2.0]), time=np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ChargeIntegrationError):
        integrate_current(np.ones(5), time=np.arange(4.0))


def test_integrate_current_does_not_mutate_inputs():
    i = np.linspace(1.0, 2.0, 100)
    t = np.linspace(0.0, 1.0, 100)
    e = np.linspace(-0.2, 0.6, 100)
    i0, t0, e0 = i.copy(), t.copy(), e.copy()
    integrate_current(i, time=t)
    integrate_current(i, potential=e, scan_rate_v_per_s=0.1)
    np.testing.assert_array_equal(i, i0)
    np.testing.assert_array_equal(t, t0)
    np.testing.assert_array_equal(e, e0)
