import numpy as np
import pytest

from inference.utils import apply_temperature, stable_softmax


def test_rows_sum_to_one(rng):
    """Probabilities along the last axis sum to 1 for a random 3-D batch."""
    x = rng.random((4, 5, 6))
    probs = stable_softmax(x)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0)


def test_logits_do_not_overflow():
    """Extreme float32 logits give finite probabilities that still sum to 1."""
    large = np.finfo(np.float32).max / 2

    # One huge logit takes all the probability mass.
    x = np.array([1.0, 2.0, large], dtype=np.float32)
    probs = stable_softmax(x)
    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs, [0.0, 0.0, 1.0])

    # One hugely negative logit gets zero probability; the rest split the mass.
    x = np.array([1.0, 2.0, -large], dtype=np.float32)
    probs = stable_softmax(x)
    assert np.isfinite(probs).all()
    expected = np.array([np.exp(-1), 1.0, 0.0]) / (1 + np.exp(-1))
    np.testing.assert_allclose(probs, expected, rtol=1e-6)


@pytest.mark.parametrize("c", [-100.0, 0.5, 100.0])
def test_shift_invariant(rng, c):
    """softmax(x) equals softmax(x + c) for any constant c."""
    x = rng.random((4, 5, 6))
    x_plus_c = x + c
    np.testing.assert_allclose(stable_softmax(x), stable_softmax(x_plus_c))


def test_apply_temperature(rng):
    """Non-positive temperatures leave logits unchanged; large positive ones rescale them."""
    logits = rng.random((4, 5, 6))
    np.testing.assert_almost_equal(apply_temperature(logits, -np.finfo(np.float32).max), logits)
    np.testing.assert_almost_equal(apply_temperature(logits, 0), logits)
    assert not np.allclose(apply_temperature(logits, np.finfo(np.float32).max), logits)
