import pytest

from gnovi_plot.plotting.units import (
    ASPECT_RATIO_PRESETS,
    PUBLICATION_PRESETS_MM,
    from_inches,
    to_inches,
)


def test_mm_round_trips_through_inches():
    assert to_inches(25.4, "mm") == pytest.approx(1.0)
    assert from_inches(1.0, "mm") == pytest.approx(25.4)


def test_cm_round_trips_through_inches():
    assert to_inches(2.54, "cm") == pytest.approx(1.0)
    assert from_inches(1.0, "cm") == pytest.approx(2.54)


def test_inches_are_identity():
    assert to_inches(6.4, "in") == pytest.approx(6.4)
    assert from_inches(6.4, "in") == pytest.approx(6.4)


def test_unknown_unit_is_rejected():
    with pytest.raises(ValueError):
        to_inches(1.0, "parsecs")
    with pytest.raises(ValueError):
        from_inches(1.0, "parsecs")


def test_aspect_ratio_presets_cover_the_required_set():
    expected = {"Auto / Fit workspace", "1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "Custom"}
    assert expected == set(ASPECT_RATIO_PRESETS)
    assert ASPECT_RATIO_PRESETS["1:1"] == pytest.approx(1.0)
    assert ASPECT_RATIO_PRESETS["4:3"] == pytest.approx(4 / 3)
    assert ASPECT_RATIO_PRESETS["Auto / Fit workspace"] is None


def test_publication_presets_are_generic_and_configurable():
    expected = {"Square figure", "Presentation 16:9", "Journal single column", "Journal double column"}
    assert expected == set(PUBLICATION_PRESETS_MM)
    width_mm, height_mm = PUBLICATION_PRESETS_MM["Journal single column"]
    assert width_mm == pytest.approx(85.0)
    assert width_mm > 0 and height_mm > 0

    double_width_mm, _ = PUBLICATION_PRESETS_MM["Journal double column"]
    assert 175.0 <= double_width_mm <= 180.0
