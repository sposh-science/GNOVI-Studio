from __future__ import annotations

from collections.abc import Sequence


class InvalidRowRangeError(Exception):
    """Raised when a row selection can't be turned into a valid row range."""


class OverlappingRowRangeError(Exception):
    """Raised when a row range overlaps a range already in the collection."""


def contiguous_row_range(row_positions: Sequence[int]) -> tuple[int, int]:
    """Convert a set of selected positional row indices (e.g. from a table
    selection) into a single contiguous (start, end) range, end exclusive.

    This is the generic building block behind manual row-range selection --
    it knows nothing about cycles, and is meant to be reused later for
    cropping, exclusion, and other range-based selections over the same
    DataFrame preview table.

    Raises InvalidRowRangeError if fewer than 2 rows are given, or the rows
    are not contiguous (no gaps).
    """
    positions = sorted({int(p) for p in row_positions})
    if len(positions) < 2:
        raise InvalidRowRangeError(
            f"Select at least 2 rows to define a range (got {len(positions)})."
        )
    for a, b in zip(positions, positions[1:]):
        if b != a + 1:
            raise InvalidRowRangeError("Selected rows must be contiguous, with no gaps.")
    return positions[0], positions[-1] + 1


class RowRangeCollection:
    """An ordered set of non-overlapping, in-bounds (start, end) row ranges
    (end exclusive, `DataFrame.iloc`-style) against a dataset of `row_count`
    rows.

    Generic and domain-independent -- today it backs manual cycle selection,
    but the same mechanism is meant to be reusable later for cropping,
    exclusion, and other range-based selections. Never touches or copies the
    source DataFrame; it only tracks integer positions.
    """

    def __init__(self, row_count: int) -> None:
        if row_count < 0:
            raise ValueError("row_count must not be negative")
        self._row_count = row_count
        self._ranges: list[tuple[int, int]] = []

    @property
    def ranges(self) -> list[tuple[int, int]]:
        """Ranges sorted by start position. Never overlapping, so this is
        also their row order."""
        return list(self._ranges)

    def __len__(self) -> int:
        return len(self._ranges)

    def add(self, start: int, end: int) -> None:
        if end - start < 2:
            raise InvalidRowRangeError(f"A range must contain at least 2 rows (got {end - start}).")
        if not (0 <= start < end <= self._row_count):
            raise InvalidRowRangeError(
                f"Row range ({start}, {end}) is out of bounds for {self._row_count} rows."
            )
        for existing_start, existing_end in self._ranges:
            if start < existing_end and existing_start < end:
                raise OverlappingRowRangeError(
                    f"Row range ({start}, {end}) overlaps existing range "
                    f"({existing_start}, {existing_end})."
                )
        self._ranges.append((start, end))
        self._ranges.sort(key=lambda r: r[0])

    def add_from_positions(self, row_positions: Sequence[int]) -> tuple[int, int]:
        """Validate `row_positions` as a contiguous selection and record it.
        Returns the resulting (start, end) range."""
        start, end = contiguous_row_range(row_positions)
        self.add(start, end)
        return start, end

    def remove_at(self, index: int) -> None:
        del self._ranges[index]

    def clear(self) -> None:
        self._ranges = []
