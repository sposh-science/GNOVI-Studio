import numpy as np
import pandas as pd
import pytest

from gnovi_plot.equations.evaluator import evaluate_formula
from gnovi_plot.equations.parser import FormulaError, parse_formula


def _make_dataframe():
    return pd.DataFrame(
        {
            "Potential/V": [-0.2, -0.1, 0.0, 0.1],
            "Current/A": [1e-3, 2e-3, 0.0, -1e-3],
            "Voltage/V": [1.0, 2.0, 3.0, 4.0],
            "ScanRate": [1, 4, 9, 16],
            "2theta": [10.0, 20.0, 30.0, 40.0],
            "label": ["a", "b", "c", "d"],
        }
    )


# --- Arithmetic / multiple columns -----------------------------------------


def test_bracketed_column_arithmetic():
    df = _make_dataframe()
    values, source_columns = evaluate_formula(df, "[Current/A] * 1000")
    assert list(values) == pytest.approx([1.0, 2.0, 0.0, -1.0])
    assert source_columns == ["Current/A"]


def test_bare_identifier_column_reference():
    df = _make_dataframe()
    values, source_columns = evaluate_formula(df, "ScanRate / 60")
    assert list(values) == pytest.approx([1 / 60, 4 / 60, 9 / 60, 16 / 60])
    assert source_columns == ["ScanRate"]


def test_multiple_source_columns():
    df = _make_dataframe()
    values, source_columns = evaluate_formula(df, "[Voltage/V] * [Current/A]")
    assert list(values) == pytest.approx([1e-3, 4e-3, 0.0, -4e-3])
    assert set(source_columns) == {"Voltage/V", "Current/A"}


def test_bracket_syntax_for_non_identifier_column_name():
    df = _make_dataframe()
    values, source_columns = evaluate_formula(df, "[2theta] / 2")
    assert list(values) == pytest.approx([5.0, 10.0, 15.0, 20.0])
    assert source_columns == ["2theta"]


# --- Functions ----------------------------------------------------------


def test_sqrt_function():
    df = _make_dataframe()
    values, _ = evaluate_formula(df, "sqrt(ScanRate)")
    assert list(values) == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_log_is_base_ten_and_ln_is_natural_log():
    df = _make_dataframe()
    values_log, _ = evaluate_formula(df, "log(ScanRate)")
    values_ln, _ = evaluate_formula(df, "ln(ScanRate)")
    assert list(values_log) == pytest.approx(np.log10([1, 4, 9, 16]))
    assert list(values_ln) == pytest.approx(np.log([1, 4, 9, 16]))


def test_exp_function():
    df = _make_dataframe()
    values, _ = evaluate_formula(df, "exp(0)")
    assert list(values) == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_abs_function():
    df = _make_dataframe()
    values, _ = evaluate_formula(df, "abs([Current/A])")
    assert list(values) == pytest.approx([1e-3, 2e-3, 0.0, 1e-3])


# --- Error handling -------------------------------------------------------


def test_invalid_formula_syntax_raises():
    df = _make_dataframe()
    with pytest.raises(FormulaError):
        evaluate_formula(df, "1 + ")


def test_unknown_column_raises():
    df = _make_dataframe()
    with pytest.raises(FormulaError):
        evaluate_formula(df, "[NotAColumn] * 2")


def test_unknown_bare_name_raises():
    df = _make_dataframe()
    with pytest.raises(FormulaError):
        evaluate_formula(df, "ScanRate + mystery")


def test_divide_by_zero_produces_inf_not_an_exception():
    df = _make_dataframe()
    values, _ = evaluate_formula(df, "[Voltage/V] / [Current/A]")
    assert np.isinf(values.iloc[2])


def test_non_numeric_source_column_raises():
    df = _make_dataframe()
    with pytest.raises(FormulaError):
        evaluate_formula(df, "label * 2")


def test_partially_numeric_column_propagates_nan_like_numeric_module():
    df = pd.DataFrame({"mixed": ["1", "bad", "3"]})
    values, _ = evaluate_formula(df, "mixed * 2")
    assert values.iloc[0] == pytest.approx(2.0)
    assert np.isnan(values.iloc[1])
    assert values.iloc[2] == pytest.approx(6.0)


# --- Safety: no arbitrary Python execution ---------------------------------


@pytest.mark.parametrize(
    "formula",
    [
        "__import__('os')",
        "__import__('os').system('echo hi')",
        "open('/etc/passwd')",
        "os.system('echo hi')",
        "().__class__",
        "getattr(ScanRate, '__class__')",
        "eval('1')",
        "exec('1')",
        "[Current/A].values",
        "ScanRate; DROP TABLE x",
    ],
)
def test_arbitrary_python_execution_is_rejected(formula):
    df = _make_dataframe()
    with pytest.raises(FormulaError):
        evaluate_formula(df, formula)


def test_parse_formula_rejects_stray_characters():
    with pytest.raises(FormulaError):
        parse_formula("ScanRate = 5", ["ScanRate"])


def test_parse_formula_rejects_empty_formula():
    with pytest.raises(FormulaError):
        parse_formula("", ["ScanRate"])
    with pytest.raises(FormulaError):
        parse_formula("   ", ["ScanRate"])


# --- Raw data integrity / metadata -----------------------------------------


def test_evaluate_formula_does_not_mutate_source_dataframe():
    df = _make_dataframe()
    original = df.copy(deep=True)
    evaluate_formula(df, "[Current/A] * 1000")
    pd.testing.assert_frame_equal(df, original)
