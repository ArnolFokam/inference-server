import pytest

from inference.vocab import Vocab


@pytest.fixture
def vocab() -> Vocab:
    return Vocab(["a", "b", "c", " "], special_tokens=["<unk>", "<bos>", "<eos>"])


def test_vocab_len_counts_special_and_regular_tokens(vocab):
    """The length counts the special tokens and the regular tokens together."""
    assert len(vocab) == 7


def test_vocab_decode_joins_regular_tokens_into_a_string(vocab):
    """Regular ids decode to their characters, concatenated in input order."""
    assert vocab.decode([5, 3, 4, 6, 5]) == "cab c"


def test_vocab_decode_skips_special_tokens_by_default(vocab):
    """With the default skip_special=True, every special id is left out."""
    assert vocab.decode([0, 1, 5, 3, 2]) == "ca"


def test_vocab_decode_keeps_special_tokens_when_skip_special_is_false(vocab):
    """With skip_special=False, special ids appear as their literal token text."""
    assert vocab.decode([1, 5, 2], skip_special=False) == "<bos>c<eos>"


def test_vocab_encode_maps_characters_to_ids_after_the_specials(vocab):
    """Each character encodes to its id, regular tokens sitting after the specials."""
    assert vocab.encode("cab c") == [5, 3, 4, 6, 5]


def test_vocab_encode_add_bos_prepends_the_bos_id(vocab):
    """With add_bos=True the encoding starts with the id of <bos>."""
    assert vocab.encode("ab", add_bos=True) == [1, 3, 4]


def test_vocab_encode_empty_string_is_empty(vocab):
    """An empty string encodes to an empty list when no <bos> is requested."""
    assert vocab.encode("") == []


def test_vocab_encode_empty_string_with_add_bos_is_only_bos(vocab):
    """An empty string with add_bos=True still encodes to just the <bos> id."""
    assert vocab.encode("", add_bos=True) == [1]


def test_vocab_encode_unknown_character_maps_to_unk_id(vocab):
    """A character that is not in the vocabulary encodes to the id of <unk>."""
    assert vocab.encode("azb") == [3, 0, 4]


def test_vocab_encode_drops_unknown_character_without_unk_token():
    """With no <unk> token, characters absent from the vocabulary are dropped."""
    vocab = Vocab(["a", "b"], special_tokens=["<bos>"])
    assert vocab.encode("azb") == [1, 2]


def test_vocab_duplicate_regular_tokens_are_rejected():
    """Building a vocabulary with a repeated regular token raises ValueError."""
    with pytest.raises(ValueError):
        Vocab(["a", "a"], special_tokens=["<unk>"])


def test_vocab_token_listed_as_special_and_regular_is_rejected():
    """A token present in both special_tokens and tokens raises ValueError."""
    with pytest.raises(ValueError):
        Vocab(["a", "<eos>"], special_tokens=["<unk>", "<eos>"])
