from __future__ import annotations

import numpy as np
import pandas as pd


class CycleDetectionError(Exception):
    """Raised when repeating sweep cycles cannot be reliably detected."""


def detect_cycles(
    dataframe: pd.DataFrame,
    x_column: str,
    min_points: int = 3,
    noise_tolerance: float | None = None,
    tolerance_fraction: float = 0.25,
) -> list[tuple[int, int]]:
    """Detect repeating sweep cycles in `dataframe[x_column]` from direction
    reversals (turning points), without mutating `dataframe`.

    This is a generic, domain-independent segment detector: it makes no
    assumption about column names or units, only that the column is a
    numeric signal that repeatedly rises to a turning point, reverses,
    falls to an opposite turning point, and repeats -- the shape of a
    cyclic-voltammetry potential sweep, but not limited to it.

    Returns a list of (start, end) *positional* row ranges into `dataframe`
    (end exclusive, usable with `dataframe.iloc[start:end]`), one per
    detected complete cycle, in row order. Rows with non-numeric x values
    are ignored when locating turning points but not removed from the
    ranges, so a returned range may include a few invalid rows a caller's
    own numeric extraction will drop.

    `noise_tolerance` is the minimum change in x required to count as a
    genuine directional step; smaller fluctuations (sensor noise, repeated
    values near a turning point) are ignored and the previous direction is
    carried forward. By default it is *not* a fraction of the column's full
    range -- on a fine, densely sampled sweep (e.g. a 4800-row CV scan over
    0.8 V, where the genuine step between adjacent points is ~0.001 V) a
    range-based tolerance is far larger than the real step and swallows
    every genuine directional move. Instead the characteristic sampling
    step is estimated directly from the data as the median of the non-zero
    absolute differences between consecutive values, and the tolerance is
    set to `tolerance_fraction` of that step (default 25%), comfortably
    below a genuine step but above ordinary measurement jitter.

    Raises CycleDetectionError if the column doesn't have enough numeric
    data, or if no repeating turning-point structure can be found (e.g. the
    column is monotonic or flat) -- callers should not guess in that case,
    only fall back to plotting the entire dataset or surface the message.
    """
    x = pd.to_numeric(dataframe[x_column], errors="coerce")
    valid_positions = np.flatnonzero(x.notna().to_numpy())
    if len(valid_positions) < min_points:
        raise CycleDetectionError(
            f"Not enough numeric data in column '{x_column}' to detect cycles "
            f"(found {len(valid_positions)}, need at least {min_points})."
        )

    values = x.to_numpy()[valid_positions]
    diffs = np.diff(values)

    if noise_tolerance is None:
        abs_diffs = np.abs(diffs)
        genuine_steps = abs_diffs[np.isfinite(abs_diffs) & (abs_diffs > 0)]
        characteristic_step = float(np.median(genuine_steps)) if genuine_steps.size > 0 else 0.0
        noise_tolerance = characteristic_step * tolerance_fraction

    raw_sign = np.where(diffs > noise_tolerance, 1, np.where(diffs < -noise_tolerance, -1, 0))

    # Carry the last genuine direction forward across noise/plateau steps so
    # repeated or near-repeated values near a turning point don't register
    # as spurious reversals.
    sign = raw_sign.copy()
    last = 0
    for i, s in enumerate(raw_sign):
        if s == 0:
            sign[i] = last
        else:
            last = s

    if not np.any(sign != 0):
        raise CycleDetectionError(
            f"No direction reversals found in column '{x_column}'; "
            "the data does not appear to be a repeating sweep."
        )

    # Interior turning points: positions in `values` where the outgoing
    # direction differs from the incoming one.
    turning_points = [i for i in range(1, len(sign)) if sign[i] != sign[i - 1] and sign[i - 1] != 0]

    first_sign = next(s for s in sign if s != 0)
    last_sign = next(s for s in reversed(sign) if s != 0)

    # The first and last sample are themselves turning points in the weak
    # sense (there is only one neighbouring direction to compare against),
    # so a sweep that starts or ends exactly at a vertex is still detected.
    boundaries = [0, *turning_points, len(values) - 1]
    boundary_types = [-first_sign, *(sign[i - 1] for i in turning_points), last_sign]

    type_a = boundary_types[0]
    same_type_positions = [pos for pos, kind in zip(boundaries, boundary_types) if kind == type_a]
    cycles_local = list(zip(same_type_positions, same_type_positions[1:]))

    if not cycles_local:
        raise CycleDetectionError(
            f"No complete cycle found in column '{x_column}'; "
            "the data may be monotonic or too short."
        )

    return [(int(valid_positions[a]), int(valid_positions[b]) + 1) for a, b in cycles_local]
