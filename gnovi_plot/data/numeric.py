from __future__ import annotations

import numpy as np
import pandas as pd


class InsufficientNumericDataError(Exception):
    """Raised when too few numeric (x, y) pairs remain after cleaning for plotting."""


def numeric_xy(
    dataframe: pd.DataFrame, x_col: str, y_col: str, min_points: int = 2
) -> tuple[pd.Series, pd.Series]:
    """Extract numeric x/y series for plotting, without mutating `dataframe`.

    Non-numeric values are coerced to NaN via pandas.to_numeric(errors="coerce")
    and rows where either x or y is NaN are dropped. Raises
    InsufficientNumericDataError if fewer than `min_points` valid pairs remain,
    so callers can show a clear error instead of letting Matplotlib crash on
    mixed string/NaN/number columns.
    """
    x = pd.to_numeric(dataframe[x_col], errors="coerce")
    y = pd.to_numeric(dataframe[y_col], errors="coerce")
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]

    if len(x) < min_points:
        raise InsufficientNumericDataError(
            f"Not enough numeric data points in columns '{x_col}'/'{y_col}' to plot "
            f"(found {len(x)}, need at least {min_points})."
        )
    return x, y


def numeric_xyz(
    dataframe: pd.DataFrame, x_col: str, y_col: str, z_col: str, min_points: int = 2
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Extract numeric x/y/z series for 3D plotting, without mutating
    `dataframe` -- the 3D counterpart of `numeric_xy`, same conventions.

    Non-numeric values are coerced to NaN via pandas.to_numeric(errors=
    "coerce"); a row is dropped as a WHOLE row if x, y, OR z is NaN there
    (one combined `valid` mask applied to all three at once) -- row-wise
    X/Y/Z correspondence is always preserved, never independently filtered
    per column (filtering x/y/z separately would misalign which x goes
    with which y/z). Raises InsufficientNumericDataError if fewer than
    `min_points` valid triples remain, so callers can show a clear error
    instead of letting Matplotlib crash on mixed string/NaN/number columns.
    """
    x = pd.to_numeric(dataframe[x_col], errors="coerce")
    y = pd.to_numeric(dataframe[y_col], errors="coerce")
    z = pd.to_numeric(dataframe[z_col], errors="coerce")
    valid = x.notna() & y.notna() & z.notna()
    x, y, z = x[valid], y[valid], z[valid]

    if len(x) < min_points:
        raise InsufficientNumericDataError(
            f"Not enough numeric data points in columns '{x_col}'/'{y_col}'/'{z_col}' to plot "
            f"(found {len(x)}, need at least {min_points})."
        )
    return x, y, z


def group_row_positions(
    dataframe: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    group_col: str | None = None,
    min_points: int = 2,
) -> dict[object, list[int]]:
    """Partition `dataframe`'s row POSITIONS (0-based, `.iloc` order) by
    `group_col`'s value, restricted to rows where x_col/y_col/z_col are all
    numeric (same combined-mask row-wise correspondence as `numeric_xyz`,
    computed identically here so grouping and numeric-validity filtering
    can never disagree about which rows exist).

    `group_col=None` -> a single group under key `None`, holding every
    numerically-valid row position -- callers building an ungrouped
    `Series3D` should ignore these positions and use `row_indices=None`
    instead (meaning "the whole dataset", matching this function's own
    "no grouping" case exactly); this dict form still exists so
    validation/error handling stays identical whether or not grouping is
    requested.

    Row order WITHIN each group is preserved exactly as it appears in
    `dataframe` -- never sorted by X or any other column. Sweep data can be
    genuinely non-monotonic in X by design (a forward+reverse sweep, a
    cyclic scan); silently sorting would connect points in an order the
    experiment never produced. Groups themselves are returned in ascending
    order of their group value when that value is orderable (`sorted()`
    succeeds), which is what makes automatic color-cycle assignment and
    on-screen series order deterministic and reproducible -- falling back
    to first-encountered order only if the group column holds genuinely
    unorderable mixed types.

    Rows whose `group_col` value is missing/NaN are EXCLUDED entirely --
    never bucketed into a synthetic "missing" group -- the same "drop
    rather than guess" policy already applied to invalid X/Y/Z. `group_col`
    values are used exactly as stored (never coerced to numeric): a
    categorical/string group column partitions exactly like a numeric one.

    Raises `InsufficientNumericDataError` if fewer than `min_points` valid
    (x, y, z) rows exist in total (before any grouping) -- the same
    controlled-error convention `numeric_xyz` already uses.
    """
    x = pd.to_numeric(dataframe[x_col], errors="coerce")
    y = pd.to_numeric(dataframe[y_col], errors="coerce")
    z = pd.to_numeric(dataframe[z_col], errors="coerce")
    valid_mask = (x.notna() & y.notna() & z.notna()).to_numpy()
    positions = np.flatnonzero(valid_mask)

    if len(positions) < min_points:
        raise InsufficientNumericDataError(
            f"Not enough numeric data points in columns '{x_col}'/'{y_col}'/'{z_col}' to plot "
            f"(found {len(positions)}, need at least {min_points})."
        )

    if group_col is None:
        return {None: positions.tolist()}

    group_values = dataframe[group_col].to_numpy()
    groups: dict[object, list[int]] = {}
    for pos in positions:
        value = group_values[pos]
        if pd.isna(value):
            continue
        groups.setdefault(value, []).append(int(pos))

    try:
        return dict(sorted(groups.items(), key=lambda item: item[0]))
    except TypeError:
        # Genuinely unorderable/mixed group value types -- keep whatever
        # order they were first encountered in `dataframe` rather than
        # raising; grouping itself is still fully well-defined.
        return groups


def numeric_column(dataframe: pd.DataFrame, column: str, min_points: int = 1) -> pd.Series:
    """Extract a single numeric column for plotting (e.g. a histogram), without
    mutating `dataframe`. Non-numeric values are coerced to NaN and dropped.
    """
    values = pd.to_numeric(dataframe[column], errors="coerce").dropna()

    if len(values) < min_points:
        raise InsufficientNumericDataError(
            f"Not enough numeric data points in column '{column}' to plot "
            f"(found {len(values)}, need at least {min_points})."
        )
    return values
