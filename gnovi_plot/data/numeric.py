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
