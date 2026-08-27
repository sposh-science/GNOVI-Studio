"""CV-2A headless helpers: source eligibility, the data-scaled prominence
default, mV<->sample conversion, the ambiguous-segmentation flag, and the
anodic/cathodic couple assignment (+ its ΔEp / E½ / ratio agreement with a
fresh measurement)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.widgets.cv_analysis_section import eligible_cv_series
from gnovi_plot.modules.electrochemistry.common import SWEEP_RISING, segment_sweeps
from gnovi_plot.modules.electrochemistry.cv import (
    DEFAULT_PROMINENCE_MULTIPLIER,
    PROCESS_ANODIC,
    PROCESS_CATHODIC,
    PROCESS_UNASSIGNED,
    ambiguous_segmentation,
    default_prominence,
    mv_to_sample_distance,
    pair_cycles,
)
from gnovi_plot.modules.electrochemistry.results import (
    RATIO_BASIS_RAW,
    CVBaselineInfo,
    CVPeakResult,
    assign_couple,
    couple_from_peak_results,
)
from gnovi_plot.plotting.figure import GnoviFigure, Panel3D
from gnovi_plot.plotting.series import PlotSeries
from gnovi_plot.plotting.series3d import Plot3DType, Series3D
from tests.data import generate_synthetic_cv as model


def _peak(process, e, i, prom=1e-5, enabled=True, origin="automatic") -> CVPeakResult:
    return CVPeakResult(
        peak_id=f"{process}-{e}", sweep=SWEEP_RISING, process=process, origin=origin,
        enabled=enabled, e_peak_v=e, i_peak_raw_a=i, i_peak_corrected_a=None,
        baseline=CVBaselineInfo.none(), prominence=prom,
    )


# --- eligible_cv_series ------------------------------------------------


def test_eligible_cv_series_line_and_scatter_only():
    df = pd.DataFrame({"E": [0.0, 0.1, 0.2], "I": [1e-6, 2e-6, 1e-6]})
    ds = Dataset(name="cv", dataframe=df)
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(ds, "E", "I"))
    figure.add_series(PlotSeries.scatter(ds, "E", "I"))
    figure.add_series(PlotSeries.histogram(ds, "E"))
    assert len(eligible_cv_series(figure)) == 2


def test_eligible_cv_series_empty_for_panel3d():
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]})
    ds = Dataset(name="d", dataframe=df)
    figure = GnoviFigure()
    figure.panels[0] = Panel3D(panel_label="3D")
    figure.panels[0].add_series(Series3D(dataset=ds, x_column="x", y_column="y", z_column="z",
                                         plot_type=Plot3DType.SCATTER, label="s"))
    assert eligible_cv_series(figure) == []


# --- default_prominence ---------------------------------------------


def test_default_prominence_scales_with_noise():
    rng = np.random.default_rng(0)
    quiet = 1e-6 + rng.normal(0, 1e-9, 2000)
    loud = 1e-6 + rng.normal(0, 1e-7, 2000)
    assert default_prominence(loud) > default_prominence(quiet)


def test_default_prominence_floored_at_fraction_of_range():
    # essentially no sample-to-sample noise, but a real 1e-5 A span
    e = np.linspace(-0.2, 0.6, 2000)
    signal = 1e-5 * np.exp(-((e - 0.2) / 0.05) ** 2)
    value = default_prominence(signal)
    assert value >= 0.02 * float(np.ptp(signal)) - 1e-18
    assert value > 0


def test_default_prominence_tiny_input():
    assert default_prominence(np.array([1.0, 2.0])) == 0.0


def test_default_prominence_multiplier_is_conservative_but_below_xrd():
    assert 2.0 < DEFAULT_PROMINENCE_MULTIPLIER < 5.0


# --- mv_to_sample_distance ------------------------------------------


def test_mv_to_sample_distance_uses_median_step():
    e = np.arange(0.0, 1.0, 0.001)  # 1 mV step
    assert mv_to_sample_distance(e, 20.0) == 20
    assert mv_to_sample_distance(e, 0.0) is None
    assert mv_to_sample_distance(e, -5.0) is None
    assert mv_to_sample_distance(e, 0.0001) == 1  # always >= 1 for a positive request


# --- ambiguous_segmentation ---------------------------------------


def _triangle(cycles, lo=-0.2, hi=0.6, step=0.001):
    n = round((hi - lo) / step)
    rising = np.round(np.linspace(lo, hi, n + 1), 10)
    falling = rising[::-1]
    legs = [rising]
    for k in range(1, 2 * cycles):
        legs.append((falling if k % 2 else rising)[1:])
    return np.concatenate(legs)


def test_ambiguous_segmentation_false_for_clean_cycles():
    for n in (1, 2, 3):
        assert ambiguous_segmentation(segment_sweeps(_triangle(n))) is False


def test_ambiguous_segmentation_true_for_odd_sweep_count():
    e = np.concatenate([_triangle(1), _triangle(1)[1:401]])  # + a partial rising leg
    assert ambiguous_segmentation(segment_sweeps(e)) is True


def test_ambiguous_segmentation_true_for_interior_stub():
    # rising, tiny reversal, rising, falling  -> an interior stub
    e = np.concatenate([
        np.linspace(-0.2, 0.6, 801),
        np.linspace(0.6, 0.5, 100)[1:],   # stub falling
        np.linspace(0.5, 0.6, 100)[1:],
        np.linspace(0.6, -0.2, 801)[1:],
    ])
    assert ambiguous_segmentation(segment_sweeps(e)) is True


# --- assign_couple / couple_from_peak_results ----------------------


def test_assign_couple_picks_largest_prominence_enabled():
    peaks = [
        _peak(PROCESS_ANODIC, 0.25, 2e-5, prom=2e-5),
        _peak(PROCESS_ANODIC, 0.10, 3e-6, prom=3e-6),
        _peak(PROCESS_CATHODIC, 0.19, -1.9e-5, prom=1.9e-5),
        _peak(PROCESS_UNASSIGNED, 0.40, 9e-5, prom=9e-5),
    ]
    anodic, cathodic = assign_couple(peaks)
    assert anodic.e_peak_v == 0.25
    assert cathodic.e_peak_v == 0.19  # unassigned peak is never a couple member


def test_assign_couple_respects_enabled():
    peaks = [
        _peak(PROCESS_ANODIC, 0.25, 2e-5, prom=2e-5, enabled=False),
        _peak(PROCESS_ANODIC, 0.10, 3e-6, prom=3e-6),
        _peak(PROCESS_CATHODIC, 0.19, -1.9e-5, prom=1.9e-5),
    ]
    anodic, _ = assign_couple(peaks)
    assert anodic.e_peak_v == 0.10  # the enabled one


def test_assign_couple_none_when_a_process_missing():
    anodic, cathodic = assign_couple([_peak(PROCESS_ANODIC, 0.25, 2e-5)])
    assert anodic is not None and cathodic is None


def test_couple_from_peak_results_metrics_match_the_model():
    df = pd.read_csv(model._HERE / "synthetic_cv_reversible.csv")
    e = df["Potential/V"].to_numpy()
    i = df["Current/A"].to_numpy()
    segs = segment_sweeps(e)
    cycle = pair_cycles(segs)[-1]
    rising = next(s for s in cycle.sweeps if s.direction == SWEEP_RISING)
    falling = next(s for s in cycle.sweeps if s.direction != SWEEP_RISING)
    a_idx = rising.start + int(np.argmax(i[rising.start:rising.end]))
    c_idx = falling.start + int(np.argmin(i[falling.start:falling.end]))
    peaks = [
        _peak(PROCESS_ANODIC, float(e[a_idx]), float(i[a_idx])),
        _peak(PROCESS_CATHODIC, float(e[c_idx]), float(i[c_idx])),
    ]
    metrics, anodic_id, cathodic_id = couple_from_peak_results(peaks)
    assert anodic_id == peaks[0].peak_id
    assert cathodic_id == peaks[1].peak_id
    assert metrics.delta_ep_v == pytest.approx(model.DELTA_EP_TRUE, abs=2 * model.STEP)
    assert metrics.e_half_v == pytest.approx(model.E_HALF_TRUE, abs=model.STEP)
    assert metrics.ratio_basis == RATIO_BASIS_RAW  # no baseline drawn


def test_couple_from_peak_results_none_without_a_full_couple():
    metrics, a, c = couple_from_peak_results([_peak(PROCESS_ANODIC, 0.25, 2e-5)])
    assert metrics is None and a is None and c is None
