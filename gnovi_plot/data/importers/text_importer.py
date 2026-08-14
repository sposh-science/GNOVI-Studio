from __future__ import annotations

import csv
import io
import itertools
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".txt", ".tsv", ".dat"}

_EXPLICIT_DELIMITERS = {
    ".csv": ",",
    ".tsv": "\t",
}

# Sentinel delimiter value meaning "split on runs of whitespace" rather than a
# single literal character.
WHITESPACE = "whitespace"

DELIMITER_OPTIONS = ["Auto", "Comma", "Tab", "Semicolon", "Whitespace"]

_DELIMITER_CHARS = {"comma": ",", "tab": "\t", "semicolon": ";"}


class DataImportError(Exception):
    """Raised when a tabular data file cannot be imported."""


def load_text_file(path: str | Path) -> pd.DataFrame:
    """Load a delimited text file into a DataFrame, unmodified from what pandas parses."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise DataImportError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if not path.exists():
        raise DataImportError(f"File not found: {path}")

    delimiter = _EXPLICIT_DELIMITERS.get(suffix)
    if delimiter is None:
        delimiter = _detect_delimiter(path)

    try:
        if delimiter is None:
            df = pd.read_csv(path, sep=r"\s+", engine="python")
        else:
            df = pd.read_csv(path, sep=delimiter, skipinitialspace=True)
    except DataImportError:
        raise
    except Exception as exc:
        raise DataImportError(f"Failed to import '{path.name}': {exc}") from exc

    if df.shape[1] == 0 or df.shape[0] == 0:
        raise DataImportError(f"No tabular data found in '{path.name}'.")

    return df


def _detect_delimiter(path: Path) -> str | None:
    """Sniff a practical delimiter from a sample of the file. None means whitespace."""
    try:
        with open(path, "r", newline="", errors="replace") as f:
            sample = f.read(4096)
    except OSError as exc:
        raise DataImportError(f"Could not read '{path.name}': {exc}") from exc

    if not sample.strip():
        raise DataImportError(f"File '{path.name}' is empty.")

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return None


# --- Preview-driven import (Import Data dialog) --------------------------
#
# Scientific instrument exports (e.g. cyclic voltammetry CSVs) often place
# free-form metadata *above* the real column-header row. The functions below
# let the raw file be previewed and the header/data-start rows be detected
# or explicitly chosen, instead of assuming row 0 is always the header.


@dataclass
class ImportResult:
    """Outcome of a preview-driven import: the parsed table plus what was
    skipped above it, so pre-table metadata is never silently discarded."""

    dataframe: pd.DataFrame
    raw_header_lines: list[str]
    header_row: int
    data_start_row: int
    delimiter: str


def read_raw_lines(path: str | Path, max_lines: int | None = None) -> list[str]:
    """Read a file as plain text lines (no parsing), for raw preview/detection."""
    path = Path(path)
    if not path.exists():
        raise DataImportError(f"File not found: {path}")
    try:
        with open(path, "r", errors="replace") as f:
            raw_lines = f.readlines() if max_lines is None else list(itertools.islice(f, max_lines))
    except OSError as exc:
        raise DataImportError(f"Could not read '{path.name}': {exc}") from exc
    return [line.rstrip("\n").rstrip("\r") for line in raw_lines]


def _tokenize(line: str, delimiter: str) -> list[str]:
    if not line.strip():
        return []
    if delimiter == WHITESPACE:
        return line.split()
    return [cell.strip() for cell in line.split(delimiter)]


def _is_number(token: str) -> bool:
    token = token.strip()
    if not token:
        return False
    try:
        float(token)
        return True
    except ValueError:
        return False


def _row_is_mostly_numeric(cells: list[str], threshold: float = 0.6) -> bool:
    cells = [c for c in cells if c.strip()]
    if not cells:
        return False
    numeric_count = sum(1 for c in cells if _is_number(c))
    return (numeric_count / len(cells)) >= threshold


def detect_table_start(
    lines: list[str],
    delimiter: str,
    window: int = 5,
    numeric_ratio_threshold: float = 0.8,
) -> tuple[int, int]:
    """Best-effort guess at (header_row, data_start_row), both 0-based.

    Looks for a non-numeric row immediately followed by several predominantly
    numeric rows -- a practical signature of "column header, then data table"
    even when preceded by free-form instrument metadata. This is only a
    convenience; callers (the Import Data dialog) must let the user override
    both rows explicitly.
    """
    n = len(lines)
    for i in range(n):
        row = _tokenize(lines[i], delimiter)
        if not row:
            continue
        following = [_tokenize(lines[j], delimiter) for j in range(i + 1, min(i + 1 + window, n))]
        following = [cells for cells in following if cells]
        if len(following) < 2:
            continue
        # The row right after the candidate header must itself be numeric --
        # otherwise a metadata line just before the real header (e.g. the last
        # "Segment N:" line) can look like a match purely from window-average
        # density, even though the very next row is still metadata/header text.
        if not _row_is_mostly_numeric(following[0]):
            continue
        numeric_ratio = sum(_row_is_mostly_numeric(cells) for cells in following) / len(following)
        if numeric_ratio >= numeric_ratio_threshold and not _row_is_mostly_numeric(row):
            return i, i + 1

    for i in range(n):
        if lines[i].strip():
            return i, min(i + 1, n)
    return 0, 0


_AUTO_DELIMITER_CANDIDATES = [",", "\t", ";", WHITESPACE]


def resolve_delimiter(lines: list[str], delimiter_option: str = "auto") -> str:
    """Resolve a delimiter option name to an actual delimiter.

    Returns a single-character delimiter, or the WHITESPACE sentinel meaning
    "split on runs of whitespace". `delimiter_option` is case-insensitive and
    should be one of "auto", "comma", "tab", "semicolon", "whitespace".

    For "auto", each candidate delimiter is scored by how well it lets
    `detect_table_start` find a header row followed by predominantly numeric
    data (and produces more than one column) -- this is more robust for real
    instrument exports than sniffing the whole file, since free-form metadata
    lines (e.g. "Instrument: X") can otherwise fool a naive delimiter sniffer.
    """
    key = delimiter_option.strip().lower()
    if key in _DELIMITER_CHARS:
        return _DELIMITER_CHARS[key]
    if key == "whitespace":
        return WHITESPACE
    if key != "auto":
        raise DataImportError(f"Unknown delimiter option: {delimiter_option}")

    best_delimiter = WHITESPACE
    best_score = -1.0
    for candidate in _AUTO_DELIMITER_CANDIDATES:
        header_row, data_start_row = detect_table_start(lines, candidate)
        header_cols = len(_tokenize(lines[header_row], candidate)) if lines else 0
        following = [
            _tokenize(lines[j], candidate) for j in range(data_start_row, min(data_start_row + 5, len(lines)))
        ]
        following = [cells for cells in following if cells]
        if not following or header_cols < 2:
            continue
        ratio = sum(_row_is_mostly_numeric(cells) for cells in following) / len(following)
        if ratio <= 0:
            continue
        score = ratio + 0.01 * header_cols
        if score > best_score:
            best_score = score
            best_delimiter = candidate
    return best_delimiter


def parse_table(lines: list[str], delimiter: str, header_row: int, data_start_row: int) -> pd.DataFrame:
    """Parse only the selected header row + data rows into a DataFrame.

    Rows before `header_row` (instrument metadata, etc.) are never included.
    """
    if not lines:
        raise DataImportError("No lines to parse.")
    if header_row < 0 or header_row >= len(lines):
        raise DataImportError(f"Header row {header_row} is out of range for this file.")
    if data_start_row <= header_row:
        raise DataImportError("Data start row must be after the header row.")
    if data_start_row > len(lines):
        raise DataImportError(f"Data start row {data_start_row} is out of range for this file.")

    data_lines = [line for line in lines[data_start_row:] if line.strip()]
    if not data_lines:
        raise DataImportError("No data rows found after the selected data start row.")

    text = "\n".join([lines[header_row], *data_lines])

    try:
        if delimiter == WHITESPACE:
            df = pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python")
        else:
            df = pd.read_csv(io.StringIO(text), sep=delimiter, skipinitialspace=True)
    except Exception as exc:
        raise DataImportError(f"Failed to parse table: {exc}") from exc

    df = df.dropna(how="all").reset_index(drop=True)
    if df.shape[1] == 0 or df.shape[0] == 0:
        raise DataImportError("No tabular data found for the selected header/data rows.")
    return df


def import_table(
    path: str | Path,
    delimiter_option: str = "auto",
    header_row: int | None = None,
    data_start_row: int | None = None,
) -> ImportResult:
    """Full preview-driven import: read, resolve delimiter, detect (or accept
    explicit) header/data rows, and parse -- preserving skipped rows as
    `raw_header_lines` rather than discarding them.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DataImportError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    lines = read_raw_lines(path)
    if not any(line.strip() for line in lines):
        raise DataImportError(f"File '{path.name}' is empty.")

    delimiter = resolve_delimiter(lines, delimiter_option)

    if header_row is None or data_start_row is None:
        detected_header, detected_data_start = detect_table_start(lines, delimiter)
        if header_row is None:
            header_row = detected_header
        if data_start_row is None:
            data_start_row = detected_data_start

    dataframe = parse_table(lines, delimiter, header_row, data_start_row)
    raw_header_lines = list(lines[:header_row])

    return ImportResult(
        dataframe=dataframe,
        raw_header_lines=raw_header_lines,
        header_row=header_row,
        data_start_row=data_start_row,
        delimiter=delimiter,
    )
