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

    kth_largest = np.partition(logits, -k, axis=-1)[..., [-k]]
    logits = np.where(logits >= kth_largest, logits, -np.inf)
    return logits


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus filter: keep the smallest set of tokens whose probability mass is >= p.

    Accepts shape (V,) or (B, V) and filters along the last axis. Tokens are
    ranked by probability (highest first); the smallest prefix whose cumulative
    probability reaches p is kept and every other entry is set to -inf. Kept
    entries retain their original logit values. p is in (0, 1]; p == 1 keeps
    everything, and the top token is always kept.
    """
    probs = stable_softmax(logits)
    sorted_idx = np.argsort(-probs, axis=-1, kind="stable")
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=-1)

    # Keep a token while the mass ranked above it is still below p.
    mass_before = np.cumsum(sorted_probs, axis=-1) - sorted_probs
    sorted_keep = mass_before < p

    keep = np.zeros_like(sorted_keep)
    np.put_along_axis(keep, sorted_idx, sorted_keep, axis=-1)
    return np.where(keep, logits, -np.inf)


def sample_from_probs(probs: np.ndarray, rng: np.random.Generator) -> int:
    """Draw one token id from the categorical distribution probs of shape (V,)."""
    return int(rng.choice(len(probs), p=probs))


def greedy_select(logits: np.ndarray) -> np.ndarray:
    """Return the index of the largest logit along the last axis.

    Accepts shape (V,) or (B, V). Ties resolve to the lowest index.
    """
    return np.argmax(logits, axis=-1)