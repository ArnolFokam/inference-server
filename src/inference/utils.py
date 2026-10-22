import numpy as np


def stable_softmax(x: np.ndarray) -> np.ndarray:
    """Softmax over the last axis, computed stably.

    Subtracts the per-row max before exponentiating so large logits cannot overflow.
    """
    exps = np.exp(x - x.max(axis=-1, keepdims=True))
    return exps / exps.sum(axis=-1, keepdims=True)


def apply_temperature(x: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits of shape (..., V) by 1 / temperature.

    A temperature <= 0 is a greedy request: the logits are returned unchanged.
    """
    if temperature > 0:
        x = x / temperature
    return x
