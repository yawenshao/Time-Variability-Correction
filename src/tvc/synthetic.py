"""Synthetic series generators for examples and tests."""

import numpy as np


def generate_ar1(length: int, coefficient: float, seed: int) -> np.ndarray:
    """Generate a reproducible, zero-mean AR(1) series."""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=length)
    values = np.empty(length, dtype=float)
    values[0] = innovations[0]
    for index in range(1, length):
        values[index] = coefficient * values[index - 1] + innovations[index]
    return values
