import pandas as pd

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    return Dataset(name=name, dataframe=df)


def test_add_and_retrieve_dataset():
    manager = DatasetManager()
    dataset = _make_dataset()
    manager.add(dataset)

    assert len(manager) == 1
    assert manager.get(dataset.id) is dataset
    assert dataset in list(manager)


def test_multiple_datasets_coexist():
    manager = DatasetManager()
    a, b = _make_dataset("a"), _make_dataset("b")
    manager.add(a)
    manager.add(b)

    assert len(manager) == 2
    assert {d.id for d in manager.datasets} == {a.id, b.id}


def test_remove_dataset():
    manager = DatasetManager()
    dataset = _make_dataset()
    manager.add(dataset)

    manager.remove(dataset.id)

    assert len(manager) == 0
    assert manager.get(dataset.id) is None


def test_remove_unknown_dataset_is_a_no_op():
    manager = DatasetManager()
    manager.remove("does-not-exist")
    assert len(manager) == 0


def test_clear_removes_all_datasets():
    manager = DatasetManager()
    manager.add(_make_dataset("a"))
    manager.add(_make_dataset("b"))

    manager.clear()

    assert len(manager) == 0
