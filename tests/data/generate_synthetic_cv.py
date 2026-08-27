"""Deterministic synthetic Cyclic Voltammetry fixtures for CV-1 tests.

Run ``python tests/data/generate_synthetic_cv.py`` to (re)write:

* ``synthetic_cv_reversible.csv``      -- clean, 2 complete cycles
* ``synthetic_cv_sloped_background.csv`` -- 1 anodic peak on a linear
  charging background + deterministic noise

The MODEL is fully explicit here so the tests can independently derive
their expected values (peak potentials, ΔEp, E½, baseline-corrected peak
current) from these constants rather than hard-coding textbook numbers.
This is a synthetic ALGORITHM-validation fixture: the peak shapes are
Gaussians, not a Nicholson-Shain current function. The peak SEPARATION is
set to the common reversible one-electron 25 °C approximation (59 mV) by
construction -- named accordingly, not asserted as a universal constant.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent

# --- shared sweep geometry -------------------------------------------------
E_LOW = -0.20  # V
E_HIGH = 0.60  # V
STEP = 0.001  # V per sample  -> uniform, constant-rate
SCAN_RATE_V_PER_S = 0.10  # V/s  (documented; not stored in the CSV)

# --- reversible-couple model (25 °C, 1 e-, reversible approximation) ------
E_HALF_TRUE = 0.2205  # V  -- midpoint potential
DELTA_EP_TRUE = 0.059  # V  -- 59 mV, the common reversible 25 °C approximation
EPA_TRUE = E_HALF_TRUE + DELTA_EP_TRUE / 2  # 0.2500 V, on the rising sweep
EPC_TRUE = E_HALF_TRUE - DELTA_EP_TRUE / 2  # 0.1910 V, on the falling sweep
PEAK_SIGMA = 0.030  # V  -- Gaussian width of each wave
PEAK_AMPLITUDE_A = 1.00e-5  # A  -- true faradaic peak current magnitude
FLAT_BACKGROUND_A = 2.0e-7  # A  -- small constant (double-layer) offset

# --- sloped-background fixture -------------------------------------------
# The slope is chosen so the RAW current extremum still lands near Epa (the
# faradaic peak is the global max of the rising sweep) while the background
# under the peak is a large fraction of the true peak current -- so raw !=
# baseline-corrected by a wide margin, but the raw extremum is not simply
# "the last sample".
SLOPE_A_PER_V = 1.2e-5  # A/V  -- linear charging current vs potential
BG_NOISE_A = 1.0e-7  # A  -- deterministic Gaussian noise (seeded)
NOISE_SEED = 20260827


def _triangular_sweep(cycles: int, positive_first: bool = True) -> np.ndarray:
    """`cycles` complete cycles between E_LOW and E_HIGH at STEP, adjacent
    legs sharing the turning-point vertex (no duplicated row)."""
    n = round((E_HIGH - E_LOW) / STEP)
    rising = np.round(np.linspace(E_LOW, E_HIGH, n + 1), 10)
    falling = rising[::-1]
    first, second = (rising, falling) if positive_first else (falling, rising)
    legs = [first]
    for k in range(1, 2 * cycles):
        legs.append((second if k % 2 else first)[1:])
    return np.concatenate(legs)


def _gaussian(e: np.ndarray, center: float, amp: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-((e - center) ** 2) / (2.0 * sigma**2))


def _rising_mask(e: np.ndarray) -> np.ndarray:
    d = np.gradient(e)
    return d > 0


def build_reversible() -> pd.DataFrame:
    e = _triangular_sweep(cycles=2, positive_first=True)
    rising = _rising_mask(e)
    current = np.full_like(e, FLAT_BACKGROUND_A)
    current += _gaussian(e, EPA_TRUE, PEAK_AMPLITUDE_A, PEAK_SIGMA) * rising
    current -= _gaussian(e, EPC_TRUE, PEAK_AMPLITUDE_A, PEAK_SIGMA) * (~rising)
    return pd.DataFrame({"Potential/V": e, "Current/A": current})


def build_sloped_background() -> pd.DataFrame:
    e = _triangular_sweep(cycles=1, positive_first=True)
    rising = _rising_mask(e)
    rng = np.random.default_rng(NOISE_SEED)
    background = SLOPE_A_PER_V * (e - E_LOW)
    faradaic = _gaussian(e, EPA_TRUE, PEAK_AMPLITUDE_A, PEAK_SIGMA) * rising
    noise = rng.normal(0.0, BG_NOISE_A, size=e.shape)
    return pd.DataFrame({"Potential/V": e, "Current/A": background + faradaic + noise})


def main() -> None:
    build_reversible().to_csv(_HERE / "synthetic_cv_reversible.csv", index=False)
    build_sloped_background().to_csv(_HERE / "synthetic_cv_sloped_background.csv", index=False)
    print("wrote synthetic_cv_reversible.csv, synthetic_cv_sloped_background.csv")


if __name__ == "__main__":
    main()
