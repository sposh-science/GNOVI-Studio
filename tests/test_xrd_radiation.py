"""modules.xrd.radiation: presets, custom wavelength, validation, and
serialization."""

from __future__ import annotations

import copy

import pytest

from gnovi_plot.modules.xrd.radiation import (
    CO_KALPHA1_ANGSTROM,
    CU_KALPHA1_ANGSTROM,
    CU_KALPHA2_ANGSTROM,
    CU_KALPHA_ANGSTROM,
    MO_KALPHA1_ANGSTROM,
    RADIATION_PRESETS,
    InvalidRadiationError,
    Radiation,
    radiation_from_preset,
)


def test_cu_ka1_preset_uses_the_single_line_value():
    radiation = radiation_from_preset("cu_ka1")
    assert radiation.wavelength_angstrom == pytest.approx(1.540562, abs=1e-6)


def test_cu_ka_weighted_average_differs_from_ka1():
    """Cu Ka1 and the weighted Cu Ka average are NOT the same number --
    this is the exact distinction the XRD design explicitly requires."""
    ka1 = radiation_from_preset("cu_ka1")
    ka_weighted = radiation_from_preset("cu_ka")
    assert ka1.wavelength_angstrom != ka_weighted.wavelength_angstrom
    assert ka_weighted.wavelength_angstrom == pytest.approx(1.54184, abs=1e-4)


def test_cu_ka_weighted_average_is_the_2to1_intensity_weighting_of_ka1_ka2():
    expected = (2 * CU_KALPHA1_ANGSTROM + CU_KALPHA2_ANGSTROM) / 3
    assert CU_KALPHA_ANGSTROM == pytest.approx(expected, abs=1e-12)


def test_all_documented_presets_are_present_and_positive():
    expected_ids = {"cu_ka1", "cu_ka", "co_ka1", "co_ka", "mo_ka1", "mo_ka"}
    assert set(RADIATION_PRESETS) == expected_ids
    for radiation in RADIATION_PRESETS.values():
        assert radiation.wavelength_angstrom > 0


def test_co_and_mo_ka1_values_are_the_standard_tabulated_ones():
    assert radiation_from_preset("co_ka1").wavelength_angstrom == pytest.approx(
        CO_KALPHA1_ANGSTROM, abs=1e-6
    )
    assert radiation_from_preset("mo_ka1").wavelength_angstrom == pytest.approx(
        MO_KALPHA1_ANGSTROM, abs=1e-6
    )


def test_unknown_preset_id_raises():
    with pytest.raises(InvalidRadiationError):
        radiation_from_preset("does-not-exist")


def test_custom_wavelength_is_labeled_custom():
    radiation = Radiation.custom(0.98)
    assert radiation.label == "Custom"
    assert radiation.wavelength_angstrom == 0.98


@pytest.mark.parametrize("wavelength", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_wavelengths_are_rejected(wavelength):
    with pytest.raises(InvalidRadiationError):
        Radiation("Test", wavelength)


def test_empty_label_is_rejected():
    with pytest.raises(InvalidRadiationError):
        Radiation("", 1.5)


def test_to_dict_from_dict_round_trip():
    original = radiation_from_preset("mo_ka")
    restored = Radiation.from_dict(original.to_dict())
    assert restored == original


def test_radiation_is_deepcopy_safe():
    original = Radiation("Cu Ka1", 1.540562)
    cloned = copy.deepcopy(original)
    assert cloned == original
    assert cloned is not original
