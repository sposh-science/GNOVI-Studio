"""Numeric evaluation of parsed calculated-column formulas over a DataFrame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import sympy

from gnovi_plot.equations.parser import FormulaError, parse_formula


def evaluate_formula(dataframe: pd.DataFrame, formula: str) -> tuple[pd.Series, list[str]]:
    """Evaluate `formula` against `dataframe`'s columns.

    Returns the resulting numeric Series (aligned to `dataframe.index`) and
    the list of source column names the formula referenced, without
    mutating `dataframe`.

    Non-numeric values in a referenced column are coerced to NaN, consistent
    with `data.numeric`; a column that contains no numeric data at all
    raises FormulaError rather than silently producing an all-NaN result.
    Division by zero and other domain issues (e.g. sqrt of a negative)
    produce inf/NaN like ordinary NumPy/pandas arithmetic and never raise.
    """
    parsed = parse_formula(formula, list(dataframe.columns))
    placeholder_names = sorted(parsed.column_symbols.keys())
    symbols = [sympy.Symbol(name) for name in placeholder_names]

    args = []
    for name in placeholder_names:
        column = parsed.column_symbols[name]
        raw = dataframe[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        if raw.notna().any() and numeric.notna().sum() == 0:
            raise FormulaError(f"Column '{column}' does not contain numeric data")
        args.append(numeric.to_numpy(dtype=float))

    with np.errstate(divide="ignore", invalid="ignore"):
        if args:
            func = sympy.lambdify(symbols, parsed.expression, modules=["numpy"])
            result = np.asarray(func(*args), dtype=float)
            if result.shape == ():
                result = np.full(len(dataframe), float(result))
        else:
            # A pure-constant formula (no column reference) is legal
            # arithmetic; broadcast it across every row.
            result = np.full(len(dataframe), float(parsed.expression))

    source_columns = [parsed.column_symbols[name] for name in placeholder_names]
    return pd.Series(result, index=dataframe.index), source_columns
