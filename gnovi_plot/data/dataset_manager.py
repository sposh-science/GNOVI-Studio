from __future__ import annotations

from gnovi_plot.data.dataset import Dataset


class DatasetManager:
    """Holds multiple coexisting datasets. Has no Qt/GUI dependencies."""

    def __init__(self) -> None:
        self._datasets: dict[str, Dataset] = {}

    def add(self, dataset: Dataset) -> None:
        self._datasets[dataset.id] = dataset

    def remove(self, dataset_id: str) -> None:
        self._datasets.pop(dataset_id, None)

    def get(self, dataset_id: str) -> Dataset | None:
        return self._datasets.get(dataset_id)

    def clear(self) -> None:
        self._datasets.clear()

    @property
    def datasets(self) -> list[Dataset]:
        return list(self._datasets.values())

    def __len__(self) -> int:
        return len(self._datasets)

    def __iter__(self):
        return iter(self._datasets.values())

    def __contains__(self, dataset_id: str) -> bool:
        return dataset_id in self._datasets
