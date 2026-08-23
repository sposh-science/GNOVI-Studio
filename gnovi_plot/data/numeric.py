from __future__ import annotations

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
