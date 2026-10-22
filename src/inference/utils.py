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


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest logits per row and set the rest to -inf.

    Accepts shape (V,) or (B, V) and returns the same shape and dtype. If k >= V,
    the logits are returned unchanged. Kept entries retain their original values.
    """
    logits = logits.copy()

    if k >= logits.shape[-1]:
        return logits

    idx = np.argsort(logits, axis=-1)[..., :-k]
    np.put_along_axis(logits, idx, -np.inf, axis=-1)
    return logits
