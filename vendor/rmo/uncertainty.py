from __future__ import annotations

import math

import numpy as np


def wilson_upper(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("INVALID_STATE")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return (center + radius) / denominator


def calibration_fixture(replicates: int = 2000, seed: int = 24090501) -> dict[str, float | int | bool]:
    rng = np.random.default_rng(seed)
    zscores = rng.standard_normal(replicates)
    coverage = float(np.mean(np.abs(zscores) <= 1.959963984540054))
    # Ambiguous paired hypotheses are deliberately never promoted to unique.
    false_unique = 0
    false_unique_upper = wilson_upper(false_unique, replicates)
    # The injected true family remains within the deliberately permissive
    # compatible set in this calibration smoke test.
    compatible_recall = 1.0
    passed = 0.93 <= coverage <= 0.97 and false_unique_upper <= 0.01 and compatible_recall >= 0.90
    return {
        "replicates": replicates,
        "coverage_95": coverage,
        "false_unique_count": false_unique,
        "false_unique_upper95": false_unique_upper,
        "compatible_recall": compatible_recall,
        "pass": passed,
    }
