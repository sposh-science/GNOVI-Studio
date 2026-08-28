"""`data.numeric.group_row_positions` -- the data-layer partitioning
function behind "Group by" 3D curve families. Pure pandas/numpy, no Qt --
this is where the row-alignment/ordering/missing-value guarantees this
milestone depends on are actually verified; `test_plot3d_panel.py` only
re-confirms the GUI wires this function correctly, not its own semantics.
"""

import pandas as pd
import pytest

from gnovi_plot.data.numeric import InsufficientNumericDataError, group_row_positions


def test_group_by_none_creates_a_single_group_with_every_valid_position():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "z": [1.0, 2.0, 3.0]})
    groups = group_row_positions(df, "x", "y", "z", group_col=None)

    assert list(groups.keys()) == [None]
    assert groups[None] == [0, 1, 2]


def test_group_by_a_column_creates_the_correct_number_of_groups():
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0], "z": [1.0, 2.0, 3.0, 4.0], "g": [25.0, 35.0, 25.0, 35.0]}
    )
    groups = group_row_positions(df, "x", "y", "z", group_col="g")

    assert len(groups) == 2
    assert set(groups.keys()) == {25.0, 35.0}


def test_each_group_contains_only_its_own_rows():
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0], "z": [1.0, 2.0, 3.0, 4.0], "g": [25.0, 35.0, 25.0, 35.0]}
    )
    groups = group_row_positions(df, "x", "y", "z", group_col="g")

    assert groups[25.0] == [0, 2]
    assert groups[35.0] == [1, 3]


def test_source_row_order_is_preserved_within_each_group_never_sorted_by_x():
    """A non-monotonic sweep (X goes up then down within one group) must
    keep its recorded order -- grouping must never reorder by X."""
    df = pd.DataFrame(
        {
            "x": [3.0, 1.0, 2.0, 5.0, 4.0, 6.0],
            "y": [0.0] * 6,
            "z": [0.0] * 6,
            "g": ["a", "b", "a", "b", "a", "b"],
        }
    )
    groups = group_row_positions(df, "x", "y", "z", group_col="g")

    assert groups["a"] == [0, 2, 4]  # positions, in source order -- NOT sorted by x value
    assert groups["b"] == [1, 3, 5]


def test_no_cross_group_row_positions_overlap():
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [0.0] * 5, "z": [0.0] * 5, "g": ["a", "b", "a", "c", "b"]}
    )
    groups = group_row_positions(df, "x", "y", "z", group_col="g")

    all_positions = [pos for positions in groups.values() for pos in positions]
    assert sorted(all_positions) == [0, 1, 2, 3, 4]
    assert len(all_positions) == len(set(all_positions))  # no duplicates/overlap


def test_invalid_x_rows_are_excluded_from_every_group():
    df = pd.DataFrame({"x": [1.0, None, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0], "z": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "b", "b"]})
    groups = group_row_positions(df, "x", "y", "z", group_col="g")

    assert groups["a"] == [0]  # row 1 dropped -- invalid x
    assert groups["b"] == [2, 3]


def test_invalid_y_rows_are_excluded_from_every_group():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, None, 3.0, 4.0], "z": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "b", "b"]})
    groups = group_row_positions(df, "x", "y", "z", group_col="g")

    assert groups["a"] == [0]
    assert groups["b"] == [2, 3]


def test_invalid_z_rows_are_excluded_from_every_group():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0], "z": [1.0, None, 3.0, 4.0], "g": ["a", "a", "b", "b"]})
    groups = group_row_positions(df, "x", "y", "z", group_col="g")

    assert groups["a"] == [0]
    assert groups["b"] == [2, 3]


def test_group_values_remain_correctly_aligned_after_xyz_filtering():
    """The critical alignment guarantee: dropping an invalid XYZ row must
    never shift which group value belongs to which surviving row."""
    df = pd.DataFrame(
        {
            "x": [1.0, "bad", 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "z": [1.0, 2.0, 3.0, 4.0, 5.0],
            "g": ["A", "B", "A", "B", "A"],
        }
    )
    groups = group_row_positions(df, "x", "y", "z", group_col="g")

    # Row 1 (g="B", x="bad") is dropped; row 3 (g="B", x=4.0) survives alone.
    assert groups["B"] == [3]
    assert groups["A"] == [0, 2, 4]


def test_missing_group_value_rows_are_excluded_not_bucketed():
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "z": [1.0, 2.0, 3.0], "g": ["a", None, "a"]}
    )
    groups = group_row_positions(df, "x", "y", "z", group_col="g")

    assert set(groups.keys()) == {"a"}
    assert groups["a"] == [0, 2]
    assert 1 not in [pos for positions in groups.values() for pos in positions]


def test_string_categorical_group_column_works_without_numeric_coercion():
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "z": [1.0, 2.0, 3.0], "material": ["Si", "Ge", "Si"]}
    )
    groups = group_row_positions(df, "x", "y", "z", group_col="material")

    assert set(groups.keys()) == {"Si", "Ge"}
    assert groups["Si"] == [0, 2]
    assert groups["Ge"] == [1]


def test_groups_are_returned_in_ascending_order_of_group_value_when_orderable():
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [0.0] * 4, "z": [0.0] * 4, "g": [85.0, 25.0, 55.0, 35.0]}
    )
    groups = group_row_positions(df, "x", "y", "z", group_col="g")

    assert list(groups.keys()) == [25.0, 35.0, 55.0, 85.0]


def test_insufficient_numeric_data_raises_before_any_grouping():
    df = pd.DataFrame({"x": ["a", "b"], "y": [1.0, 2.0], "z": [1.0, 2.0], "g": ["a", "b"]})
    with pytest.raises(InsufficientNumericDataError):
        group_row_positions(df, "x", "y", "z", group_col="g")


def test_missing_group_column_raises_key_error():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "z": [1.0, 2.0]})
    with pytest.raises(KeyError):
        group_row_positions(df, "x", "y", "z", group_col="does_not_exist")
