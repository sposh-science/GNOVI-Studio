from pathlib import Path

import pandas as pd
import pytest

from gnovi_plot.data.importers.text_importer import (
    WHITESPACE,
    DataImportError,
    detect_table_start,
    import_table,
    parse_table,
    read_raw_lines,
    resolve_delimiter,
)
from gnovi_plot.data.numeric import numeric_xy

DATA_DIR = Path(__file__).parent / "data"


def test_auto_detects_header_below_instrument_metadata():
    result = import_table(DATA_DIR / "cv_export.csv")

    assert list(result.dataframe.columns) == ["Potential/V", "Current/A"]
    assert len(result.dataframe) == 6
    assert result.delimiter == ","

    # Metadata rows above the real header are preserved, not lost.
    assert any("Instrument" in line for line in result.raw_header_lines)
    assert any(line.startswith("Segment 6") for line in result.raw_header_lines)

    # And they never leak into the numeric table.
    assert "Instrument" not in result.dataframe.to_string()
    assert "Segment" not in result.dataframe.to_string()


def test_auto_detected_columns_are_purely_numeric():
    result = import_table(DATA_DIR / "cv_export.csv")
    numeric = result.dataframe.apply(pd.to_numeric, errors="coerce")
    assert numeric.notna().all().all()


def test_cv_export_columns_parse_without_trailing_comma():
    """Regression test: the comma delimiter must fully separate
    'Potential/V,Current/A' into two clean columns, not leave a trailing
    comma stuck on the first column's header/values (which previously made
    the first column numerically unconvertible and numeric_xy() drop every
    row)."""
    result = import_table(DATA_DIR / "cv_export.csv")
    df = result.dataframe

    assert list(df.columns) == ["Potential/V", "Current/A"]

    first_value = df["Potential/V"].iloc[0]
    assert float(first_value) == pytest.approx(-0.200)
    assert "," not in str(first_value)

    x, y = numeric_xy(df, "Potential/V", "Current/A")
    assert len(x) == 6
    assert len(y) == 6
    assert x.iloc[0] == pytest.approx(-0.200)
    assert y.iloc[0] == pytest.approx(3.312e-05)


def test_real_cv_export_with_blank_separator_and_spaced_header():
    """Regression test for a real CHI660E export: CRLF line endings, a
    "Header:, Current/A" comma header with a space after the comma, and a
    *blank line* between the header row and the first data row.

    Previously, `_tokenize("", ",")` returned `['']` instead of `[]`, so the
    blank separator line (not the real header) was mistaken for a 1-column
    header candidate. That disqualified the comma delimiter (needs >= 2
    columns), auto-detection fell back to whitespace splitting, and
    whitespace-splitting "-0.200, 3.335e-5" glued the comma onto the first
    column as "-0.200," -- an unparseable trailing-comma artifact that made
    numeric_xy() drop every row.
    """
    path = DATA_DIR / "real_cv_export.csv"
    lines = read_raw_lines(path)

    delimiter = resolve_delimiter(lines, "auto")
    assert delimiter == ","

    header_row, data_start_row = detect_table_start(lines, delimiter)
    assert header_row == 34
    assert lines[header_row] == "Potential/V, Current/A"
    assert data_start_row == 35
    assert lines[data_start_row] == ""

    result = import_table(path)
    df = result.dataframe

    assert list(df.columns) == ["Potential/V", "Current/A"]
    assert df["Potential/V"].dtype.kind == "f"
    assert df["Current/A"].dtype.kind == "f"

    first_value = df["Potential/V"].iloc[0]
    assert "," not in str(first_value)
    assert float(first_value) == pytest.approx(-0.200)

    x, y = numeric_xy(df, "Potential/V", "Current/A")
    assert len(x) == 4800
    assert len(y) == 4800
    assert x.iloc[0] == pytest.approx(-0.200)
    assert y.iloc[0] == pytest.approx(3.335e-5)


def test_manual_header_and_data_start_override(tmp_path):
    path = tmp_path / "cv.txt"
    path.write_text(
        "Instrument: X\n"
        "Operator: Y\n"
        "Potential/V\tCurrent/A\n"
        "-0.200\t3.312e-5\n"
        "-0.199\t3.240e-5\n"
    )
    lines = read_raw_lines(path)

    result = import_table(path, delimiter_option="tab", header_row=2, data_start_row=3)

    assert list(result.dataframe.columns) == ["Potential/V", "Current/A"]
    assert len(result.dataframe) == 2
    assert result.raw_header_lines == lines[:2]


def test_manual_data_start_can_skip_a_stray_row(tmp_path):
    path = tmp_path / "cv.txt"
    path.write_text(
        "Instrument: X\n"
        "Potential/V\tCurrent/A\n"
        "garbage row not part of the table\n"
        "-0.200\t3.312e-5\n"
        "-0.199\t3.240e-5\n"
    )
    # User overrides the "row immediately after header" default to skip a stray row.
    result = import_table(path, delimiter_option="tab", header_row=1, data_start_row=3)
    assert len(result.dataframe) == 2


def test_whitespace_delimited_auto_detection(tmp_path):
    path = tmp_path / "cv.txt"
    path.write_text(
        "Results:\n"
        "Segment 1:\n"
        "Potential/V    Current/A\n"
        "-0.200         3.312e-5\n"
        "-0.199         3.240e-5\n"
        "-0.198         3.198e-5\n"
    )
    result = import_table(path, delimiter_option="auto")
    assert result.delimiter == WHITESPACE
    assert list(result.dataframe.columns) == ["Potential/V", "Current/A"]
    assert len(result.dataframe) == 3


def test_detect_table_start_finds_header_before_numeric_rows():
    lines = [
        "Instrument: X",
        "Segment 1:",
        "Segment 2:",
        "Potential/V,Current/A",
        "-0.2,1e-5",
        "-0.1,2e-5",
        "0.0,3e-5",
    ]
    header_row, data_start_row = detect_table_start(lines, ",")
    assert header_row == 3
    assert data_start_row == 4


def test_resolve_delimiter_auto_detects_comma():
    lines = ["a,b,c", "1,2,3"]
    assert resolve_delimiter(lines, "auto") == ","


def test_resolve_delimiter_explicit_semicolon():
    lines = ["a;b", "1;2"]
    assert resolve_delimiter(lines, "semicolon") == ";"


def test_parse_table_out_of_range_header_raises():
    lines = ["a,b", "1,2"]
    with pytest.raises(DataImportError):
        parse_table(lines, ",", header_row=5, data_start_row=6)


def test_parse_table_data_start_before_header_raises():
    lines = ["a,b", "1,2", "3,4"]
    with pytest.raises(DataImportError):
        parse_table(lines, ",", header_row=1, data_start_row=0)
