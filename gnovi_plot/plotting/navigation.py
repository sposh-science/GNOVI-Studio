"""View-only navigation helpers for the interactive plot canvas.

Nothing here reads or writes the ``GnoviFigure``/``Panel`` model. These
functions operate purely on Matplotlib ``Axes`` view limits, exactly like
the built-in ``NavigationToolbar2`` pan/zoom, so an interactive "Zoom Out"
stays transient view state -- never a model edit, never a project
dirty/undo checkpoint -- the same as interactive pan/zoom already is in
GNOVI (see ``Panel.xlim``/``.ylim``: those are only written by the
explicit Axes-settings controls, never by canvas navigation).
"""

from __future__ import annotations

import math

# One "Zoom Out" click widens each visible axis range to this multiple of
# its current width, about its current center. ~1.25 is a comfortable,
# repeatable step: small enough that several clicks still feel controlled,
# large enough that a single click is clearly visible.
ZOOM_OUT_FACTOR = 1.25


def expand_interval(lo: float, hi: float, factor: float, *, log: bool = False) -> tuple[float, float]:
    """Return ``(lo, hi)`` widened about its center by ``factor``.

    ``factor > 1`` zooms out (a wider view); ``factor == 1`` is a no-op.

    The endpoint ORDER is preserved, so an inverted axis (``lo > hi``)
    stays inverted and is never silently flipped. When ``log`` is true the
    widening is done in log space, i.e. a multiplicative zoom rather than
    an invalid linear expansion of a decade axis; a non-positive endpoint
    (not a valid log view) or a degenerate/non-finite range is returned
    unchanged.
    """
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo == hi:
        return (lo, hi)
    if log:
        if lo <= 0.0 or hi <= 0.0:
            return (lo, hi)
        log_lo, log_hi = math.log(lo), math.log(hi)
        center = (log_lo + log_hi) / 2.0
        half = (log_hi - log_lo) / 2.0 * factor
        return (math.exp(center - half), math.exp(center + half))
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0 * factor
    return (center - half, center + half)


def zoom_axes_out(ax, factor: float = ZOOM_OUT_FACTOR) -> None:
    """Widen ``ax``'s current X and Y view limits about their centers by
    ``factor``. X and Y are handled independently; each respects its own
    scale (``"log"`` -> multiplicative) and its own inversion. View-only:
    this sets nothing but the Axes view limits.
    """
    ax.set_xlim(*expand_interval(*ax.get_xlim(), factor, log=ax.get_xscale() == "log"))
    ax.set_ylim(*expand_interval(*ax.get_ylim(), factor, log=ax.get_yscale() == "log"))
