from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from gnovi_plot.data.transforms import CalculatedColumnInfo, Transformation, describe_row_positions
from gnovi_plot.equations.evaluator import evaluate_formula


@dataclass
class Dataset:
    """A pandas DataFrame plus the metadata needed to track it in the app.

    `dataframe` is the *working* data -- what plotting, column selectors, and
    the Data Preview operate on. The original import is preserved separately
    in `raw_dataframe`, set once and never reassigned by any transformation
    method below, so the raw imported data is always recoverable via
    `reset_working_data()` regardless of what's been done to `dataframe`.
    """

    name: str
    dataframe: pd.DataFrame
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    raw_dataframe: pd.DataFrame = field(init=False, repr=False)
    calculated_columns: dict[str, CalculatedColumnInfo] = field(default_factory=dict, init=False)
    transformations: list[Transformation] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.dataframe, pd.DataFrame):
            raise TypeError("Dataset.dataframe must be a pandas DataFrame")
        if not self.name:
            raise ValueError("Dataset.name must not be empty")
        self.raw_dataframe = self.dataframe

    def to_dict(self) -> dict:
        """Metadata for project save -- never includes `dataframe`/
        `raw_dataframe`; those are written as separate CSV files alongside
        `project.json` (see `core.project_io`)."""
        return {
            "id": self.id,
            "name": self.name,
            "source_path": self.source_path,
            "metadata": self.metadata,
            "calculated_columns": {
                name: info.to_dict() for name, info in self.calculated_columns.items()
            },
            "transformations": [t.to_dict() for t in self.transformations],
        }

    @classmethod
    def from_project_data(
        cls,
        data: dict,
        raw_dataframe: pd.DataFrame,
        working_dataframe: pd.DataFrame,
    ) -> "Dataset":
        """Reconstruct a Dataset from a project save's metadata dict (see
        `to_dict`) plus its separately-loaded raw/working CSV data.

        Unlike the normal constructor -- which assumes a fresh import where
        `raw_dataframe` and `dataframe` start identical (see
        `__post_init__`) -- a saved project's raw and working data have
        already diverged by however many transformations were applied
        before it was saved, so both are supplied explicitly and
        `raw_dataframe` is set directly rather than derived from
        `dataframe`. `id` is taken from `data` rather than freshly
        generated, so `PlotSeries.dataset_id` references captured before
        save still resolve correctly after load.
        """
        dataset = cls(
            id=data["id"],
            name=data["name"],
            dataframe=raw_dataframe,
            source_path=data.get("source_path"),
            metadata=dict(data.get("metadata", {})),
        )
        dataset.raw_dataframe = raw_dataframe
        dataset.dataframe = working_dataframe
        dataset.calculated_columns = {
            name: CalculatedColumnInfo.from_dict(info)
            for name, info in data.get("calculated_columns", {}).items()
        }
        dataset.transformations = [
            Transformation.from_dict(t) for t in data.get("transformations", [])
        ]
        return dataset

    @property
    def columns(self) -> list[str]:
        return list(self.dataframe.columns)

    @property
    def row_count(self) -> int:
        return len(self.dataframe.index)

    @property
    def raw_row_count(self) -> int:
        return len(self.raw_dataframe.index)

    # --- Calculated columns -------------------------------------------------

    def add_calculated_column(self, name: str, formula: str) -> CalculatedColumnInfo:
        """Create a new column on the working data from `formula`, evaluated
        safely via `equations.evaluator` (SymPy-based, never `eval()`).

        Raises ValueError if `name` is empty or already a column (raw or
        calculated); raises equations.parser.FormulaError for an unsafe or
        invalid formula. The raw imported DataFrame is never touched.
        """
        name = name.strip()
        if not name:
            raise ValueError("Calculated column name must not be empty")
        if name in self.dataframe.columns:
            raise ValueError(f"Column '{name}' already exists")

        values, source_columns = evaluate_formula(self.dataframe, formula)

        new_df = self.dataframe.copy()
        new_df[name] = values
        info = CalculatedColumnInfo(name=name, formula=formula, source_columns=source_columns)

        self.dataframe = new_df
        self.calculated_columns[name] = info
        self.transformations.append(
            Transformation(
                kind="calculated_column",
                description=f"Created calculated column: {name} = {formula}",
                detail={"name": name, "formula": formula, "source_columns": source_columns},
            )
        )
        return info

    # --- Row selection operations --------------------------------------------

    def exclude_rows(self, row_positions: Sequence[int]) -> None:
        """Remove `row_positions` (positional, into the current working
        data) from the working data. Non-destructive: the raw DataFrame is
        never touched, and the surviving rows keep their original index
        labels (not renumbered), so the Data Preview's row numbers keep
        meaning relative to the raw import.
        """
        positions = sorted({int(p) for p in row_positions})
        self._validate_positions(positions)

        keep_mask = pd.Series(True, index=self.dataframe.index)
        keep_mask.iloc[positions] = False
        new_df = self.dataframe.loc[keep_mask]

        self.dataframe = new_df
        self.transformations.append(
            Transformation(
                kind="exclude_rows",
                description=f"Excluded rows: {describe_row_positions(positions)}",
                detail={"row_positions": positions},
            )
        )

    def keep_rows(self, row_positions: Sequence[int]) -> None:
        """Restrict the working data to only `row_positions` (positional,
        into the current working data). Non-destructive: see `exclude_rows`.
        """
        positions = sorted({int(p) for p in row_positions})
        self._validate_positions(positions)

        new_df = self.dataframe.iloc[positions]

        self.dataframe = new_df
        self.transformations.append(
            Transformation(
                kind="keep_rows",
                description=f"Kept rows: {describe_row_positions(positions)}",
                detail={"row_positions": positions},
            )
        )

    def _validate_positions(self, positions: list[int]) -> None:
        if not positions:
            raise ValueError("No rows selected")
        if positions[0] < 0 or positions[-1] >= self.row_count:
            raise ValueError(
                f"Row selection {positions[0]}-{positions[-1]} is out of bounds for "
                f"{self.row_count} working rows"
            )

    def reset_working_data(self) -> None:
        """Restore the working data to the imported raw state: all row
        filtering is undone and all calculated columns are dropped, since
        neither exists in the raw import. The transformation history is
        never cleared -- a "reset" entry is appended -- so the ordered log
        stays intact even though working data no longer reflects the
        earlier entries.
        """
        self.dataframe = self.raw_dataframe
        self.calculated_columns = {}
        self.transformations.append(
            Transformation(kind="reset", description="Reset working data to raw import", detail={})
        )
