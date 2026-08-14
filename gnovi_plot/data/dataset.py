from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Dataset:
    """A pandas DataFrame plus the metadata needed to track it in the app."""

    name: str
    dataframe: pd.DataFrame
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if not isinstance(self.dataframe, pd.DataFrame):
            raise TypeError("Dataset.dataframe must be a pandas DataFrame")
        if not self.name:
            raise ValueError("Dataset.name must not be empty")

    @property
    def columns(self) -> list[str]:
        return list(self.dataframe.columns)

    @property
    def row_count(self) -> int:
        return len(self.dataframe.index)
