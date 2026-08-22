from __future__ import annotations

import uuid

import pytest

from gnovi_plot.analysis.panel_results import PanelResultHistory
from gnovi_plot.analysis.results import AnalysisResult


def _result(**overrides) -> AnalysisResult:
    defaults = dict(
        source_dataset_id="dataset-1",
        source_dataset_name=None,
        source_series_id=None,
        source_series_label=None,
        x_column="x",
        y_column="y",
        row_range=None,
        source_panel_id=None,
        result_id=uuid.uuid4().hex,
    )
    defaults.update(overrides)
    return AnalysisResult(**defaults)


# --- add / current -----------------------------------------------------------


def test_fresh_history_has_no_current_result_for_any_panel():
    history = PanelResultHistory()
    assert history.current("panel-1") is None


def test_add_makes_the_result_current_for_that_panel():
    history = PanelResultHistory()
    result = _result()

    history.add("panel-1", result)

    assert history.current("panel-1") is result


def test_the_latest_added_result_becomes_current():
    history = PanelResultHistory()
    first = _result()
    second = _result()

    history.add("panel-1", first)
    history.add("panel-1", second)

    assert history.current("panel-1") is second


# --- all -----------------------------------------------------------------------


def test_all_returns_full_ordered_history_oldest_first():
    history = PanelResultHistory()
    first, second, third = _result(), _result(), _result()
    history.add("panel-1", first)
    history.add("panel-1", second)
    history.add("panel-1", third)

    assert history.all("panel-1") == [first, second, third]


def test_all_for_an_unknown_panel_is_an_empty_list():
    history = PanelResultHistory()
    assert history.all("does-not-exist") == []


def test_all_returns_a_copy_not_internal_state():
    history = PanelResultHistory()
    history.add("panel-1", _result())

    returned = history.all("panel-1")
    returned.append(_result())

    assert len(history.all("panel-1")) == 1  # mutation of the returned list didn't leak in


def test_running_a_new_fit_never_overwrites_or_discards_earlier_results():
    history = PanelResultHistory()
    linear = _result()
    gaussian = _result()
    polynomial = _result()
    another_linear = _result()

    for r in (linear, gaussian, polynomial, another_linear):
        history.add("panel-3", r)

    assert history.all("panel-3") == [linear, gaussian, polynomial, another_linear]
    assert history.current("panel-3") is another_linear


# --- set_current -----------------------------------------------------------


def test_set_current_selects_an_older_result_without_reordering():
    history = PanelResultHistory()
    first, second = _result(), _result()
    history.add("panel-1", first)
    history.add("panel-1", second)

    history.set_current("panel-1", first.result_id)

    assert history.current("panel-1") is first
    assert history.all("panel-1") == [first, second]  # order unchanged


def test_a_later_add_overrides_an_explicit_older_selection():
    history = PanelResultHistory()
    first, second = _result(), _result()
    history.add("panel-1", first)
    history.set_current("panel-1", first.result_id)

    history.add("panel-1", second)

    assert history.current("panel-1") is second  # latest always wins on add


def test_set_current_with_an_unknown_result_id_raises():
    history = PanelResultHistory()
    history.add("panel-1", _result())

    with pytest.raises(ValueError):
        history.set_current("panel-1", "not-a-real-result-id")


def test_set_current_for_a_result_belonging_to_a_different_panel_raises():
    history = PanelResultHistory()
    other_panel_result = _result()
    history.add("panel-1", _result())
    history.add("panel-2", other_panel_result)

    with pytest.raises(ValueError):
        history.set_current("panel-1", other_panel_result.result_id)


def test_clear_panel_also_drops_the_current_selection_marker():
    history = PanelResultHistory()
    first, second = _result(), _result()
    history.add("panel-1", first)
    history.add("panel-1", second)
    history.set_current("panel-1", first.result_id)

    history.clear_panel("panel-1")
    history.add("panel-1", second)  # re-added fresh -- must become current on its own

    assert history.current("panel-1") is second


def test_prune_to_also_drops_the_current_selection_marker_for_removed_panels():
    history = PanelResultHistory()
    first, second = _result(), _result()
    history.add("panel-removed", first)
    history.add("panel-removed", second)
    history.set_current("panel-removed", first.result_id)

    history.prune_to(set())  # panel-removed no longer exists

    assert history.current("panel-removed") is None


# --- panel isolation -----------------------------------------------------------


def test_different_panels_have_completely_independent_history():
    history = PanelResultHistory()
    result_a = _result()
    result_b = _result()

    history.add("panel-1", result_a)
    history.add("panel-2", result_b)

    assert history.current("panel-1") is result_a
    assert history.current("panel-2") is result_b
    assert history.all("panel-1") == [result_a]
    assert history.all("panel-2") == [result_b]


# --- clear_panel -----------------------------------------------------------------


def test_clear_panel_removes_that_panels_history_only():
    history = PanelResultHistory()
    result_a = _result()
    result_b = _result()
    history.add("panel-1", result_a)
    history.add("panel-2", result_b)

    history.clear_panel("panel-1")

    assert history.current("panel-1") is None
    assert history.all("panel-1") == []
    assert history.current("panel-2") is result_b


def test_clear_panel_on_an_unknown_panel_is_a_no_op():
    history = PanelResultHistory()
    history.clear_panel("does-not-exist")  # must not raise


# --- prune_to --------------------------------------------------------------------


def test_prune_to_drops_history_for_panels_not_in_the_valid_set():
    history = PanelResultHistory()
    result_a = _result()
    result_b = _result()
    history.add("panel-1", result_a)
    history.add("panel-2", result_b)

    history.prune_to({"panel-1"})

    assert history.current("panel-1") is result_a
    assert history.current("panel-2") is None
    assert history.all("panel-2") == []


def test_prune_to_leaves_surviving_panel_histories_completely_untouched():
    history = PanelResultHistory()
    first = _result()
    second = _result()
    history.add("panel-1", first)
    history.add("panel-1", second)
    history.add("panel-2", _result())

    history.prune_to({"panel-1"})

    assert history.all("panel-1") == [first, second]


def test_prune_to_never_migrates_an_orphaned_result_to_another_panel():
    history = PanelResultHistory()
    orphaned = _result()
    history.add("panel-removed", orphaned)
    history.add("panel-survives", _result())

    history.prune_to({"panel-survives"})

    assert history.current("panel-removed") is None
    for result in history.all("panel-survives"):
        assert result is not orphaned


def test_prune_to_with_all_current_ids_is_a_no_op():
    history = PanelResultHistory()
    result = _result()
    history.add("panel-1", result)

    history.prune_to({"panel-1", "panel-2"})

    assert history.current("panel-1") is result


# --- to_dict / from_dict: persistence round trip ------------------------------


def _fit_result(**overrides):
    import numpy as np

    from gnovi_plot.analysis.fitting import LINEAR, fit_curve

    x = np.linspace(0, 10, 10)
    y = 2.0 * x + 1.0
    defaults = dict(source_dataset_id="dataset-1", x_column="x", y_column="y")
    defaults.update(overrides)
    return fit_curve(x, y, LINEAR, **defaults)


def test_to_dict_from_dict_round_trip_preserves_multiple_panels_and_order():
    history = PanelResultHistory()
    a1 = _fit_result(source_panel_id="panel-1")
    a2 = _fit_result(source_panel_id="panel-1")
    b1 = _fit_result(source_panel_id="panel-2")
    history.add("panel-1", a1)
    history.add("panel-1", a2)
    history.add("panel-2", b1)

    restored = PanelResultHistory.from_dict(history.to_dict())

    assert [r.result_id for r in restored.all("panel-1")] == [a1.result_id, a2.result_id]
    assert restored.current("panel-1").result_id == a2.result_id  # order preserved -> current preserved
    assert restored.current("panel-2").result_id == b1.result_id


def test_to_dict_from_dict_round_trip_preserves_result_type_and_fields():
    history = PanelResultHistory()
    history.add("panel-1", _fit_result(source_panel_id="panel-1"))

    restored = PanelResultHistory.from_dict(history.to_dict())

    from gnovi_plot.analysis.fitting import FitResult

    current = restored.current("panel-1")
    assert isinstance(current, FitResult)
    assert current.model == "linear"
    assert current.source_panel_id == "panel-1"


def test_empty_history_round_trips_to_empty():
    restored = PanelResultHistory.from_dict(PanelResultHistory().to_dict())
    assert restored.current("anything") is None


def test_from_dict_skips_an_unrecognized_kind_without_raising(caplog):
    """A result of a kind this app doesn't know (e.g. saved by a newer
    version) is dropped with a log warning, not fatal -- one bad entry
    must not sink the whole project load."""
    data = {"panel-1": [{"kind": "some-future-analysis-tool"}]}

    restored = PanelResultHistory.from_dict(data)

    assert restored.current("panel-1") is None
    assert restored.all("panel-1") == []


def test_from_dict_keeps_recognized_entries_alongside_a_skipped_one():
    good = _fit_result(source_panel_id="panel-1")
    data = {
        "panel-1": [
            {"kind": "some-future-analysis-tool"},
            good.to_dict(),
        ]
    }

    restored = PanelResultHistory.from_dict(data)

    assert restored.current("panel-1").result_id == good.result_id


# --- to_dict / from_dict: explicit current-selection round trip --------------


def test_to_dict_shape_carries_history_and_current_result_id_per_panel():
    history = PanelResultHistory()
    first = _fit_result(source_panel_id="panel-1")
    second = _fit_result(source_panel_id="panel-1")
    history.add("panel-1", first)
    history.add("panel-1", second)
    history.set_current("panel-1", first.result_id)

    data = history.to_dict()

    assert data["panel-1"]["current_result_id"] == first.result_id
    assert [entry["result_id"] for entry in data["panel-1"]["history"]] == [first.result_id, second.result_id]


def test_round_trip_preserves_an_explicit_selection_of_an_older_result():
    history = PanelResultHistory()
    gaussian = _fit_result(source_panel_id="panel-1")
    linear = _fit_result(source_panel_id="panel-1")
    history.add("panel-1", gaussian)
    history.add("panel-1", linear)
    history.set_current("panel-1", gaussian.result_id)  # explicitly select the older one

    restored = PanelResultHistory.from_dict(history.to_dict())

    assert restored.current("panel-1").result_id == gaussian.result_id
    assert [r.result_id for r in restored.all("panel-1")] == [gaussian.result_id, linear.result_id]


def test_legacy_plain_list_shape_still_defaults_current_to_the_latest_result():
    """A project saved before `set_current`/`current_result_id` existed
    stores each panel as a bare list (no wrapper dict) -- must still load
    and default to the latest result, exactly like before this feature."""
    first = _fit_result(source_panel_id="panel-1")
    second = _fit_result(source_panel_id="panel-1")
    data = {"panel-1": [first.to_dict(), second.to_dict()]}

    restored = PanelResultHistory.from_dict(data)

    assert restored.current("panel-1").result_id == second.result_id
    assert [r.result_id for r in restored.all("panel-1")] == [first.result_id, second.result_id]


def test_a_current_result_id_naming_a_dropped_unknown_kind_falls_back_to_latest():
    good = _fit_result(source_panel_id="panel-1")
    data = {
        "panel-1": {
            "history": [{"kind": "some-future-analysis-tool", "result_id": "ghost"}, good.to_dict()],
            "current_result_id": "ghost",
        }
    }

    restored = PanelResultHistory.from_dict(data)

    assert restored.current("panel-1").result_id == good.result_id
