import numpy as np

from inference.utils import stable_softmax

# KV cache: {"K": (max_seq_len, d_model), "V": (max_seq_len, d_model), "length": int}
KVCache = dict
# Model params: embedding, Wq, Wk, Wv, Wo, W_out (arrays) and max_seq_len (int)
Params = dict


def embed_tokens(token_ids: np.ndarray, embedding_matrix: np.ndarray) -> np.ndarray:
    """Look up the embedding vector of each token id.

    token_ids has shape (T,) and embedding_matrix has shape (V, D). Returns the
    (T, D) array whose row t is embedding_matrix[token_ids[t]].
    """
    return np.take(embedding_matrix, token_ids, axis=0)


def linear_projection(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None
) -> np.ndarray:
    """Apply the affine map y = x @ weight + bias over the last axis.

    x has shape (..., D_in), weight has shape (D_in, D_out) and bias, when given,
    has shape (D_out,). Returns shape (..., D_out). A bias of None skips the term.
    """
    y = x @ weight
    if bias is not None:
        y = y + bias
    return y


def init_kv_cache(max_seq_len: int, d_model: int) -> KVCache:
    """Allocate an empty KV cache for a single sequence.

    Returns a dict with float32 buffers "K" and "V" of shape (max_seq_len, d_model)
    and "length", the number of positions written so far, starting at 0.
    """
    return {
        "K": np.zeros((max_seq_len, d_model), dtype=np.float32),
        "V": np.zeros((max_seq_len, d_model), dtype=np.float32),
        "length": 0,
    }


def append_kv(cache: KVCache, k_new: np.ndarray, v_new: np.ndarray) -> KVCache:
    """Write new key and value rows into the cache and advance its length.

    k_new and v_new have shape (t, d_model). They are placed in cache["K"] and
    cache["V"] starting at cache["length"], and the length grows by t. The cache is
    mutated in place and returned.
    """
    length = cache["length"]
    t = k_new.shape[0]
    capacity = cache["K"].shape[0]

    if length + t > capacity:
        shift = capacity - length - t
        cache["K"] = np.roll(cache["K"], shift=shift, axis=0)
        cache["V"] = np.roll(cache["V"], shift=shift, axis=0)

    cache["K"][length : length + t] = k_new
    cache["V"][length : length + t] = v_new
    cache["length"] = min(length + t, capacity)
    return cache


def causal_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, is_causal: bool = True
) -> np.ndarray:
    """Scaled dot-product attention.

    q has shape (Tq, D); k and v have shape (Tk, D). Returns shape (Tq, D). When
    is_causal is True, query i may not attend to keys at positions j > i + (Tk - Tq),
    so the usual causal mask applies when Tq == Tk, and a single decode query
    (Tq == 1) attends to all Tk keys.
    """
    scores = q @ k.T / np.sqrt(q.shape[-1])

    if is_causal:
        Tq, Tk = scores.shape
        allowed = np.tril(np.ones((Tq, Tk), dtype=bool), k=max(Tk - Tq, 0))
        scores = np.where(allowed, scores, -np.inf)

    return stable_softmax(scores) @ v


def model_prefill(token_ids: np.ndarray, params: Params) -> tuple[np.ndarray, KVCache]:
    """Run the prompt-encoding pass of a tiny single-layer transformer.

    token_ids has shape (T,). Embeds the tokens, projects to Q, K, V with Wq, Wk,
    Wv, writes K and V into a fresh KV cache, runs one causal self-attention block,
    projects its output with Wo, and projects the LAST position through W_out.
    Returns (logits, cache), with logits of shape (V,).
    """
    x = embed_tokens(token_ids, params["embedding"])
    q = linear_projection(x, params["Wq"])
    k = linear_projection(x, params["Wk"])
    v = linear_projection(x, params["Wv"])

    cache = init_kv_cache(params["max_seq_len"], d_model=k.shape[-1])
    append_kv(cache, k, v)

    length = cache["length"]
    attn = causal_attention(q, cache["K"][:length], cache["V"][:length])
    hidden = linear_projection(attn, params["Wo"])
    return linear_projection(hidden[-1], params["W_out"]), cache


def model_decode_step(
    token_id: int, params: Params, cache: KVCache
) -> tuple[np.ndarray, KVCache]:
    """Advance generation by one token using the existing KV cache.

    Embeds the single token_id, projects it to Q, K, V, appends the new key/value
    row to the cache in place, attends over every stored entry, and projects the
    result to next-token logits. Returns (logits, cache), with logits of shape (V,).
    """
    x = embed_tokens(np.array([token_id]), params["embedding"])
    q = linear_projection(x, params["Wq"])
    k = linear_projection(x, params["Wk"])
    v = linear_projection(x, params["Wv"])

    append_kv(cache, k, v)

    length = cache["length"]
    attn = causal_attention(q, cache["K"][:length], cache["V"][:length])
    hidden = linear_projection(attn, params["Wo"])
    return linear_projection(hidden[-1], params["W_out"]), cache
