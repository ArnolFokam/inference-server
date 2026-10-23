import numpy as np
import pytest

from inference.model import (
    append_kv,
    causal_attention,
    embed_tokens,
    init_kv_cache,
    linear_projection,
    model_decode_step,
    model_prefill,
)


@pytest.fixture
def params() -> dict:
    """Tiny model with V=3, D=2: identity Q/K/O, Wv doubles, W_out = [h0, h1, h0+h1]."""
    return {
        "embedding": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        "Wq": np.eye(2),
        "Wk": np.eye(2),
        "Wv": 2.0 * np.eye(2),
        "Wo": np.eye(2),
        "W_out": np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        "max_seq_len": 4,
    }


@pytest.fixture
def cache_with_token_zero() -> dict:
    """Cache of capacity 4 holding one position: K = [1, 0], V = [2, 0] (token 0)."""
    cache = {
        "K": np.zeros((4, 2), dtype=np.float32),
        "V": np.zeros((4, 2), dtype=np.float32),
        "length": 1,
    }
    cache["K"][0] = [1.0, 0.0]
    cache["V"][0] = [2.0, 0.0]
    return cache


# embed_tokens


def test_embed_tokens_returns_the_row_of_each_id_in_order():
    """Each id selects its row of the matrix, repeats included, giving shape (T, D)."""
    embedding_matrix = np.array([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]])
    out = embed_tokens(np.array([2, 0, 2]), embedding_matrix)
    np.testing.assert_array_equal(out, [[20.0, 21.0], [0.0, 1.0], [20.0, 21.0]])


# linear_projection


def test_linear_projection_without_bias_is_a_matrix_product():
    """With bias=None the result is exactly x @ weight."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    weight = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]])
    out = linear_projection(x, weight)
    np.testing.assert_array_equal(out, [[1.0, 2.0, 8.0], [3.0, 4.0, 18.0]])


def test_linear_projection_adds_bias_to_every_row():
    """The (D_out,) bias is added to each row of x @ weight."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    weight = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]])
    bias = np.array([10.0, 20.0, 30.0])
    out = linear_projection(x, weight, bias)
    np.testing.assert_array_equal(out, [[11.0, 22.0, 38.0], [13.0, 24.0, 48.0]])


def test_linear_projection_preserves_leading_dimensions():
    """Extra leading axes pass through: an identity weight returns x unchanged."""
    x = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    out = linear_projection(x, np.eye(2))
    np.testing.assert_array_equal(out, x)


# init_kv_cache


def test_init_kv_cache_allocates_float32_buffers_of_the_requested_shape():
    """K and V are float32 arrays of shape (max_seq_len, d_model)."""
    cache = init_kv_cache(4, 3)
    assert cache["K"].shape == cache["V"].shape == (4, 3)
    assert cache["K"].dtype == cache["V"].dtype == np.float32


def test_init_kv_cache_starts_with_length_zero():
    """A fresh cache has no positions written yet."""
    assert init_kv_cache(4, 3)["length"] == 0


def test_init_kv_cache_buffers_are_contiguous():
    """K and V are C-contiguous so rows can be appended and sliced without copies."""
    cache = init_kv_cache(4, 3)
    assert cache["K"].flags.c_contiguous
    assert cache["V"].flags.c_contiguous


def test_init_kv_cache_k_and_v_do_not_share_memory():
    """Writing to K must not change V."""
    cache = init_kv_cache(4, 3)
    assert not np.shares_memory(cache["K"], cache["V"])


# append_kv


def test_append_kv_writes_rows_at_the_start_of_an_empty_cache():
    """The first append fills rows 0..t-1 of K and V and leaves the rest zero."""
    cache = init_kv_cache(4, 2)
    k_new = np.array([[1.0, 2.0], [3.0, 4.0]])
    v_new = np.array([[5.0, 6.0], [7.0, 8.0]])
    append_kv(cache, k_new, v_new)
    np.testing.assert_array_equal(
        cache["K"], [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]]
    )
    np.testing.assert_array_equal(
        cache["V"], [[5.0, 6.0], [7.0, 8.0], [0.0, 0.0], [0.0, 0.0]]
    )


def test_append_kv_advances_length_by_the_number_of_rows():
    """After appending t rows, the length grows by t."""
    cache = init_kv_cache(4, 2)
    append_kv(cache, np.ones((2, 2)), np.ones((2, 2)))
    assert cache["length"] == 2


def test_append_kv_second_call_writes_after_the_existing_rows():
    """A later append lands at the current length and leaves earlier rows intact."""
    cache = init_kv_cache(4, 2)
    append_kv(cache, np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]))
    append_kv(cache, np.array([[5.0, 6.0]]), np.array([[7.0, 8.0]]))
    np.testing.assert_array_equal(cache["K"][:2], [[1.0, 2.0], [5.0, 6.0]])
    np.testing.assert_array_equal(cache["V"][:2], [[3.0, 4.0], [7.0, 8.0]])
    assert cache["length"] == 2


def test_append_kv_mutates_and_returns_the_same_cache():
    """The cache is updated in place and the very same object is returned."""
    cache = init_kv_cache(4, 2)
    out = append_kv(cache, np.ones((1, 2)), np.ones((1, 2)))
    assert out is cache


# causal_attention


def test_causal_attention_first_query_sees_only_the_first_key():
    """Under the causal mask, query 0 attends to key 0 alone, so it returns v[0]."""
    q = k = np.array([[1.0, 0.0], [0.0, 1.0]])
    v = np.array([[10.0, 20.0], [30.0, 40.0]])
    out = causal_attention(q, k, v)
    np.testing.assert_allclose(out[0], [10.0, 20.0])


def test_causal_attention_masks_future_keys_when_scores_are_equal():
    """With zero scores, each query averages only the values it is allowed to see."""
    q = np.zeros((2, 2))
    k = np.zeros((2, 2))
    v = np.array([[10.0, 20.0], [30.0, 40.0]])
    out = causal_attention(q, k, v)
    np.testing.assert_allclose(out, [[10.0, 20.0], [20.0, 30.0]])


def test_causal_attention_non_causal_attends_to_every_key():
    """With is_causal=False and zero scores, every query returns the mean of v."""
    q = np.zeros((2, 2))
    k = np.zeros((2, 2))
    v = np.array([[10.0, 20.0], [30.0, 40.0]])
    out = causal_attention(q, k, v, is_causal=False)
    np.testing.assert_allclose(out, [[20.0, 30.0], [20.0, 30.0]])


def test_causal_attention_single_query_attends_to_all_keys():
    """A decode query (Tq=1) sits after every key, so it averages all Tk values."""
    q = np.zeros((1, 2))
    k = np.zeros((3, 2))
    v = np.array([[0.0, 0.0], [3.0, 6.0], [6.0, 12.0]])
    out = causal_attention(q, k, v)
    np.testing.assert_allclose(out, [[3.0, 6.0]])


def test_causal_attention_offsets_the_mask_when_there_are_more_keys_than_queries():
    """With Tq=2, Tk=3, query i sees keys 0..i+1: row 0 sees 2 keys, row 1 all 3."""
    q = np.zeros((2, 1))
    k = np.zeros((3, 1))
    v = np.array([[0.0], [6.0], [12.0]])
    out = causal_attention(q, k, v)
    np.testing.assert_allclose(out, [[3.0], [6.0]])


def test_causal_attention_scales_scores_by_sqrt_of_d():
    """Scores are divided by sqrt(D): with D=4 the raw score 2 becomes 1."""
    q = np.array([[2.0, 0.0, 0.0, 0.0]])
    k = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    v = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    out = causal_attention(q, k, v, is_causal=False)
    weight_on_first_key = np.exp(1.0) / (1.0 + np.exp(1.0))
    np.testing.assert_allclose(out, [[weight_on_first_key] * 4])


# model_prefill


def test_model_prefill_single_token_logits(params):
    """One token attends only to itself: hidden = Wv-scaled embedding [2, 0]."""
    logits, _ = model_prefill(np.array([0]), params)
    np.testing.assert_allclose(logits, [2.0, 0.0, 2.0])


def test_model_prefill_returns_the_logits_of_the_last_position(params):
    """For tokens [0, 1] the logits come from position 1, which attends to both."""
    logits, _ = model_prefill(np.array([0, 1]), params)
    weight_on_second = 1.0 / (1.0 + np.exp(-1.0 / np.sqrt(2.0)))
    weight_on_first = 1.0 - weight_on_second
    np.testing.assert_allclose(
        logits, [2 * weight_on_first, 2 * weight_on_second, 2.0]
    )


def test_model_prefill_fills_the_cache_with_projected_keys_and_values(params):
    """The cache holds K = x @ Wk and V = x @ Wv for every prompt position."""
    _, cache = model_prefill(np.array([0, 1]), params)
    assert cache["length"] == 2
    np.testing.assert_allclose(cache["K"][:2], [[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(cache["V"][:2], [[2.0, 0.0], [0.0, 2.0]])


def test_model_prefill_sizes_the_cache_from_max_seq_len(params):
    """The returned cache has room for params["max_seq_len"] positions."""
    _, cache = model_prefill(np.array([0, 1]), params)
    assert cache["K"].shape == cache["V"].shape == (4, 2)


# model_decode_step


def test_model_decode_step_attends_over_filled_positions_only(
    params, cache_with_token_zero
):
    """Token 1 attends to the cached position and itself, ignoring unwritten rows."""
    logits, _ = model_decode_step(1, params, cache_with_token_zero)
    weight_on_new = 1.0 / (1.0 + np.exp(-1.0 / np.sqrt(2.0)))
    weight_on_cached = 1.0 - weight_on_new
    np.testing.assert_allclose(logits, [2 * weight_on_cached, 2 * weight_on_new, 2.0])


def test_model_decode_step_appends_the_new_key_and_value(
    params, cache_with_token_zero
):
    """The new token's K and V are written right after the existing rows."""
    _, cache = model_decode_step(1, params, cache_with_token_zero)
    assert cache["length"] == 2
    np.testing.assert_allclose(cache["K"][1], [0.0, 1.0])
    np.testing.assert_allclose(cache["V"][1], [0.0, 2.0])


def test_model_decode_step_updates_the_given_cache_in_place(
    params, cache_with_token_zero
):
    """The cache passed in is the one returned, not a copy."""
    _, cache = model_decode_step(1, params, cache_with_token_zero)
    assert cache is cache_with_token_zero
