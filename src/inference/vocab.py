class Vocab:
    """A bijection between token strings and integer ids.

    Special tokens take the first ids, in the order given; the regular tokens
    follow in the order given. Text is encoded character by character; characters
    missing from the vocabulary map to "<unk>" when it is a special token and are
    dropped otherwise.
    """

    _BOS_TOKEN = "<bos>"
    _UNK_TOKEN = "<unk>"

    def __init__(self, tokens: list[str], special_tokens: list[str]) -> None:
        if len(set(tokens)) != len(tokens):
            raise ValueError(f"tokens must be unique, got {tokens}")
        if set(tokens) & set(special_tokens):
            raise ValueError(f"tokens {tokens} overlap special_tokens {special_tokens}")

        all_tokens = special_tokens + tokens
        self._special_tokens = set(special_tokens)
        self._token_to_id = {token: idx for idx, token in enumerate(all_tokens)}
        self._id_to_token = dict(enumerate(all_tokens))

    def __len__(self) -> int:
        """Return the number of tokens, special tokens included."""
        return len(self._token_to_id)

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        """Turn a raw string into a flat list of token ids, one per character.

        When add_bos is True the result starts with the id of "<bos>". Characters
        absent from the vocabulary become the id of "<unk>" if that special token
        exists and are dropped otherwise.
        """
        unk_id = self._token_to_id.get(self._UNK_TOKEN)
        ids = [self._token_to_id[self._BOS_TOKEN]] if add_bos else []

        for char in text:
            idx = self._token_to_id.get(char, unk_id)
            if idx is not None:
                ids.append(idx)

        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Turn token ids back into a string by concatenating their tokens.

        When skip_special is True the special tokens are left out; otherwise they
        appear in the output as their literal text, such as "<bos>".
        """
        tokens = [self._id_to_token.get(idx, self._UNK_TOKEN) for idx in ids]
        if skip_special:
            tokens = [token for token in tokens if token not in self._special_tokens]
        return "".join(tokens)
