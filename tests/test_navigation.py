"""Unit tests for `gnovi_plot.plotting.navigation` -- the pure view-limit
math behind the toolbar "Zoom Out" action. No Qt, no Axes model."""

import math

import pytest
from matplotlib.figure import Figure

from gnovi_plot.plotting.navigation import ZOOM_OUT_FACTOR, expand_interval, zoom_axes_out


def test_expand_interval_widens_about_the_center():
    # The example from the feature spec: (20, 40), center 30, factor 1.25.
    assert expand_interval(20.0, 40.0, 1.25) == (17.5, 42.5)


def test_expand_interval_keeps_the_center_fixed():
    lo, hi = expand_interval(-4.0, 10.0, ZOOM_OUT_FACTOR)
    assert (lo + hi) / 2 == pytest.approx(3.0)


def test_expand_interval_is_progressive():
    a = expand_interval(0.0, 10.0, 1.25)
    b = expand_interval(*a, 1.25)
    assert (b[1] - b[0]) > (a[1] - a[0]) > 10.0


def test_expand_interval_preserves_ascending_order():
    lo, hi = expand_interval(1.0, 3.0, 1.25)
    assert lo < hi


def test_expand_interval_preserves_inverted_order():
    lo, hi = expand_interval(40.0, 20.0, 1.25)
    assert (lo, hi) == (42.5, 17.5)
    assert lo > hi  # still inverted, never silently flipped


def test_expand_interval_log_is_multiplicative():
    lo, hi = expand_interval(1.0, 100.0, 1.25, log=True)
    # Symmetric in log space: the geometric mean (here 10) is the fixed point.
    assert math.sqrt(lo * hi) == pytest.approx(10.0)
    assert lo < 1.0 < 100.0 < hi


def test_expand_interval_log_preserves_inversion():
    lo, hi = expand_interval(100.0, 1.0, 1.25, log=True)
    assert lo > hi
    assert math.sqrt(lo * hi) == pytest.approx(10.0)


def test_expand_interval_log_leaves_a_non_positive_range_untouched():
    assert expand_interval(-5.0, 10.0, 1.25, log=True) == (-5.0, 10.0)
    assert expand_interval(0.0, 10.0, 1.25, log=True) == (0.0, 10.0)


def test_expand_interval_leaves_a_degenerate_or_non_finite_range_untouched():
    assert expand_interval(5.0, 5.0, 1.25) == (5.0, 5.0)
    assert expand_interval(math.inf, 1.0, 1.25) == (math.inf, 1.0)


def test_expand_interval_factor_one_is_a_no_op():
    assert expand_interval(2.0, 8.0, 1.0) == (2.0, 8.0)


def _axes():
    fig = Figure()
    return fig.add_subplot(1, 1, 1)


def test_zoom_axes_out_widens_both_axes_about_their_centers():
    ax = _axes()
    ax.set_xlim(20.0, 40.0)
    ax.set_ylim(0.0, 10.0)

    zoom_axes_out(ax, 1.25)

    assert ax.get_xlim() == pytest.approx((17.5, 42.5))
    assert ax.get_ylim() == pytest.approx((-1.25, 11.25))


def test_zoom_axes_out_keeps_an_inverted_axis_inverted():
    ax = _axes()
    ax.set_xlim(40.0, 20.0)  # inverted X (e.g. binding-energy convention)
    ax.set_ylim(1.0, 5.0)

    zoom_axes_out(ax, 1.25)

    xlo, xhi = ax.get_xlim()
    assert xlo > xhi
    assert (xlo, xhi) == pytest.approx((42.5, 17.5))


def test_zoom_axes_out_is_multiplicative_on_a_log_axis():
    ax = _axes()
    ax.set_xscale("log")
    ax.set_xlim(1.0, 100.0)
    ax.set_ylim(0.0, 10.0)  # linear Y alongside log X

    zoom_axes_out(ax, 1.25)

    xlo, xhi = ax.get_xlim()
    assert math.sqrt(xlo * xhi) == pytest.approx(10.0)
    assert xlo > 0.0
    assert ax.get_ylim() == pytest.approx((-1.25, 11.25))
