import numpy as np
import pytest

from inference.utils import apply_temperature, stable_softmax, top_k_filter, top_p_filter


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


def test_top_k_filter_1d_keeps_k_largest():
    """On a 1D input, only the k largest logits survive and the rest are -inf."""
    logits = np.array([0.1, 3.0, -1.0, 2.0, 0.5])
    out = top_k_filter(logits, 2)
    np.testing.assert_array_equal(out, [-np.inf, 3.0, -np.inf, 2.0, -np.inf])


def test_top_k_filter_2d_filters_each_row_independently(rng):
    """On a 2D input, each row keeps its own k largest logits."""
    logits = rng.random((4, 10))
    k = 3
    out = top_k_filter(logits, k)
    for row_in, row_out in zip(logits, out):
        kept = np.isfinite(row_out)
        assert set(np.flatnonzero(kept)) == set(np.argsort(row_in)[-k:])


@pytest.mark.parametrize("k", [10, 11])
def test_top_k_filter_k_at_least_vocab_size_is_noop(rng, k):
    """When k >= V, the logits are returned unchanged."""
    logits = rng.random((4, 10))
    np.testing.assert_array_equal(top_k_filter(logits, k), logits)


def test_top_k_filter_keeps_all_values_tied_at_threshold():
    """Values tied with the k-th largest logit are all kept, even if that exceeds k."""
    logits = np.array([2.0, 2.0, 1.0, 2.0])
    out = top_k_filter(logits, 2)
    np.testing.assert_array_equal(out, [2.0, 2.0, -np.inf, 2.0])



def test_top_p_filter_1d_keeps_smallest_set_reaching_p_in_original_order():
    """On an unsorted 1D input, the smallest set with mass >= p survives in place."""
    logits = np.log(np.array([0.15, 0.5, 0.05, 0.3]))
    out = top_p_filter(logits, 0.75)
    np.testing.assert_array_equal(out, [-np.inf, logits[1], -np.inf, logits[3]])


def test_top_p_filter_always_keeps_top_token_when_p_is_tiny():
    """When the top token alone already exceeds p, only that token is kept."""
    logits = np.log(np.array([0.05, 0.15, 0.5, 0.3]))
    out = top_p_filter(logits, 0.01)
    np.testing.assert_array_equal(out, [-np.inf, -np.inf, logits[2], -np.inf])


def test_top_p_filter_p_one_keeps_everything():
    """With p == 1 no token is masked, despite floating-point error in the cumsum."""
    logits = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
    np.testing.assert_array_equal(top_p_filter(logits, 1.0), logits)


def test_top_p_filter_2d_filters_each_row_independently():
    """On a 2D input, each row reaches p with its own number of tokens."""
    logits = np.log(np.array([[0.9, 0.05, 0.03, 0.02], [0.4, 0.3, 0.2, 0.1]]))
    out = top_p_filter(logits, 0.8)
    expected = np.array(
        [
            [logits[0, 0], -np.inf, -np.inf, -np.inf],
            [logits[1, 0], logits[1, 1], logits[1, 2], -np.inf],
        ]
    )
    np.testing.assert_array_equal(out, expected)
