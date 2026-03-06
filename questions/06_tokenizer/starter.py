"""
Question 06: Tokenizer with Round-Trip Fidelity

Implement tokenize and detokenize functions with greedy longest-match
tokenization and round-trip fidelity for unknown characters.

See QUESTION.md for full problem description.
"""

from __future__ import annotations


def tokenize(text: str, vocab: dict[str, int]) -> list[int]:
    """
    Tokenize the input text using greedy longest-match against the vocabulary.

    At each position, find the longest substring starting at that position
    that exists in the vocabulary, emit its token ID, and advance past it.
    If no vocabulary entry matches, handle the character as unknown and
    advance by one character.

    The output must support round-trip fidelity:
        detokenize(tokenize(text, vocab), vocab) == text

    Args:
        text: The input string to tokenize.
        vocab: A mapping from token strings to positive integer IDs.
               Does not contain the empty string.

    Returns:
        A list of integer token IDs representing the tokenized text.
        Unknown characters must be encoded in a way that allows
        detokenize to recover the original character.
    """
    # TODO: Implement this function
    pass


def detokenize(token_ids: list[int], vocab: dict[str, int]) -> str:
    """
    Convert a list of token IDs back into the original text string.

    Must correctly handle both known tokens (positive IDs from vocab)
    and unknown character representations.

    Args:
        token_ids: A list of token IDs produced by tokenize().
        vocab: The same vocabulary used during tokenization.

    Returns:
        The reconstructed text string.
    """
    # TODO: Implement this function
    pass


# =============================================================================
# Usage Example
# =============================================================================
if __name__ == "__main__":
    vocab = {"hello": 1, "world": 2, " ": 3, "!": 4}
    tokens = tokenize("hello world!", vocab)
    print(f"Tokens: {tokens}")
    text = detokenize(tokens, vocab)
    print(f"Text: {text}")
