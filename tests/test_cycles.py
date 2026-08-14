import numpy as np
import pandas as pd
import pytest

from gnovi_plot.analysis.cycles import CycleDetectionError, detect_cycles

_LEG = [-1.0, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0]


def _three_cycle_values(positive_first: bool = True) -> list[float]:
    leg = _LEG if positive_first else [-v for v in _LEG]
    return leg + leg[1:] + leg[1:]


def _make_df(x_values, y_values=None, x_col="Potential/V", y_col="Current/A"):
    if y_values is None:
        y_values = [float(i) for i in range(len(x_values))]
    return pd.DataFrame({x_col: x_values, y_col: y_values})


def _sweep_cycle_values(v_min: float, v_max: float, step: float, cycles: int = 3, positive_first: bool = True) -> np.ndarray:
    """Build a fine-grained triangular sweep of `cycles` complete cycles between
    `v_min` and `v_max`, sampled at `step`, mirroring the shape of a real CV
    dataset: consecutive legs share their turning-point vertex (no duplicate
    row at the reversal) and the step between adjacent samples is uniform.
    """
    n_steps = round((v_max - v_min) / step)
    rising = np.round(np.linspace(v_min, v_max, n_steps + 1), 10)
    falling = rising[::-1]
    first, second = (rising, falling) if positive_first else (falling, rising)

    legs = [first]
    for i in range(1, 2 * cycles):
        legs.append((second if i % 2 else first)[1:])
    return np.concatenate(legs)


def test_detects_three_cycles_positive_first_scan():
    df = _make_df(_three_cycle_values(positive_first=True))
    cycles = detect_cycles(df, "Potential/V")
    assert len(cycles) == 3


def test_detects_three_cycles_negative_first_scan():
    df = _make_df(_three_cycle_values(positive_first=False))
    cycles = detect_cycles(df, "Potential/V")
    assert len(cycles) == 3


def test_cycle_row_ranges_are_correct():
    df = _make_df(_three_cycle_values(positive_first=True))
    cycles = detect_cycles(df, "Potential/V")
    assert cycles == [(0, 9), (8, 17), (16, 25)]


def test_cycle_detection_does_not_depend_on_column_name():
    df = _make_df(_three_cycle_values(), x_col="foo", y_col="bar")
    cycles = detect_cycles(df, "foo")
    assert len(cycles) == 3


def test_tolerates_small_noise_and_repeated_turning_point_values():
    values = _three_cycle_values(positive_first=True)
    # Perturb every value slightly and duplicate the value at each vertex,
    # simulating sensor noise and repeated samples near a turning point.
    noisy = []
    for i, v in enumerate(values):
        jitter = 0.005 if i % 2 == 0 else -0.005
        noisy.append(v + jitter)
        if abs(v) == 1.0:
            noisy.append(v + jitter / 2)  # extra near-duplicate sample at the vertex
    df = _make_df(noisy)
    cycles = detect_cycles(df, "Potential/V")
    assert len(cycles) == 3


def test_source_dataframe_is_not_modified():
    df = _make_df(_three_cycle_values())
    original = df.copy(deep=True)
    detect_cycles(df, "Potential/V")
    pd.testing.assert_frame_equal(df, original)


def test_raises_on_insufficient_numeric_data():
    df = _make_df(["a", "b"], y_values=[1.0, 2.0])
    with pytest.raises(CycleDetectionError):
        detect_cycles(df, "Potential/V")


def test_raises_on_monotonic_data_no_reversals():
    df = _make_df([0.0, 1.0, 2.0, 3.0, 4.0])
    with pytest.raises(CycleDetectionError):
        detect_cycles(df, "Potential/V")


def test_raises_on_flat_data():
    df = _make_df([1.0, 1.0, 1.0, 1.0, 1.0])
    with pytest.raises(CycleDetectionError):
        detect_cycles(df, "Potential/V")


def test_missing_column_raises_key_error():
    df = _make_df(_three_cycle_values())
    with pytest.raises(KeyError):
        detect_cycles(df, "does_not_exist")


# -- Regression: real CV data has a fine sampling step relative to the full
# potential range (e.g. a 0.001 V step over a 0.8 V range is 0.125% of the
# range), which a range-fraction tolerance misclassified as noise. These
# tests reproduce that shape directly: -0.200 V -> +0.600 V -> -0.200 V,
# repeated for 3 cycles, at a 0.001 V step -- matching the real dataset that
# exposed the bug (4800 rows, turning points at rows ~800/1600/2400/3200/4000).


def test_real_cv_pattern_regression_detects_three_cycles():
    values = _sweep_cycle_values(-0.200, 0.600, 0.001, cycles=3, positive_first=True)
    df = _make_df(values)

    cycles = detect_cycles(df, "Potential/V")

    # 3 complete cycles == 5 internal direction reversals (turning points at
    # rows ~800, 1600, 2400, 3200, 4000) plus the implicit start/end vertices.
    assert len(cycles) == 3
    (s0, e0), (s1, e1), (s2, e2) = cycles
    assert s0 == 0
    assert e2 == len(values)
    # Cycle boundaries fall at the shared reversal vertices, approximately
    # every 1600 rows (0-1600, 1600-3200, 3200-end), give or take the single
    # shared vertex row between adjacent cycles.
    assert abs(e0 - 1601) <= 1
    assert abs(s1 - 1600) <= 1
    assert abs(e1 - 3201) <= 1
    assert abs(s2 - 3200) <= 1


@pytest.mark.parametrize("positive_first", [True, False])
def test_real_cv_pattern_regression_both_scan_directions(positive_first):
    values = _sweep_cycle_values(-0.200, 0.600, 0.001, cycles=3, positive_first=positive_first)
    df = _make_df(values)
    cycles = detect_cycles(df, "Potential/V")
    assert len(cycles) == 3


def test_tolerates_repeated_values_at_turning_points_fine_step():
    values = _sweep_cycle_values(-0.200, 0.600, 0.001, cycles=3, positive_first=True)
    # Duplicate the sample at each turning point, simulating an instrument
    # that logs an extra reading while the potential holds at the vertex.
    is_vertex = np.isclose(values, 0.600) | np.isclose(values, -0.200)
    with_duplicates = np.repeat(values, np.where(is_vertex, 2, 1))
    df = _make_df(with_duplicates)

    cycles = detect_cycles(df, "Potential/V")
    assert len(cycles) == 3


@pytest.mark.parametrize("step", [0.001, 0.005, 0.01])
def test_tolerates_jitter_smaller_than_sampling_step(step):
    values = _sweep_cycle_values(-0.200, 0.600, step, cycles=3, positive_first=True)
    rng = np.random.default_rng(seed=42)
    # Jitter strictly smaller in magnitude than the genuine sampling step, so
    # it should be absorbed as noise rather than register as a reversal.
    jitter = rng.uniform(-0.4 * step, 0.4 * step, size=values.shape)
    jittered = values + jitter
    df = _make_df(jittered)

    cycles = detect_cycles(df, "Potential/V")
    assert len(cycles) == 3


@pytest.mark.parametrize("step", [0.001, 0.005, 0.01])
@pytest.mark.parametrize("positive_first", [True, False])
def test_various_step_sizes(step, positive_first):
    values = _sweep_cycle_values(-0.200, 0.600, step, cycles=3, positive_first=positive_first)
    df = _make_df(values)
    cycles = detect_cycles(df, "Potential/V")
    assert len(cycles) == 3


@pytest.mark.parametrize(
    "v_min, v_max, step",
    [
        (-0.200, 0.600, 0.001),
        (-1.000, 1.000, 0.002),
        (0.000, 2.500, 0.005),
        (-0.050, 0.050, 0.0005),
    ],
)
def test_various_potential_windows(v_min, v_max, step):
    values = _sweep_cycle_values(v_min, v_max, step, cycles=3, positive_first=True)
    df = _make_df(values)
    cycles = detect_cycles(df, "Potential/V")
    assert len(cycles) == 3


def test_real_cv_pattern_does_not_depend_on_column_name():
    values = _sweep_cycle_values(-0.200, 0.600, 0.001, cycles=3, positive_first=True)
    df = _make_df(values, x_col="foo", y_col="bar")
    cycles = detect_cycles(df, "foo")
    assert len(cycles) == 3
