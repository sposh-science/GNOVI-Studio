from pathlib import Path

import pytest

from gnovi_plot.data.importers.text_importer import DataImportError, load_text_file

DATA_DIR = Path(__file__).parent / "data"


def test_load_csv():
    df = load_text_file(DATA_DIR / "sample.csv")
    assert list(df.columns) == ["x", "y", "z"]
    assert len(df) == 4


def test_load_tsv():
    df = load_text_file(DATA_DIR / "sample.tsv")
    assert list(df.columns) == ["x", "y", "z"]
    assert len(df) == 3


def test_load_whitespace_delimited_txt(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("x y z\n1 2 3\n4 5 6\n")

    df = load_text_file(path)

    assert list(df.columns) == ["x", "y", "z"]
    assert len(df) == 2


def test_load_dat_with_semicolons(tmp_path):
    path = tmp_path / "sample.dat"
    path.write_text("x;y;z\n1;2;3\n4;5;6\n")

    df = load_text_file(path)

    assert list(df.columns) == ["x", "y", "z"]
    assert len(df) == 2


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text("{}")

    with pytest.raises(DataImportError):
        load_text_file(path)


def test_missing_file_raises():
    with pytest.raises(DataImportError):
        load_text_file(DATA_DIR / "does_not_exist.csv")


def test_empty_file_raises_cleanly():
    with pytest.raises(DataImportError):
        load_text_file(DATA_DIR / "empty.txt")
