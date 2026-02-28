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


# ---------------------------------------------------------------------------
# TEST CASES
# ---------------------------------------------------------------------------

def run_tests() -> None:
    """Run test cases to verify tokenize and detokenize implementations."""

    vocab = {
        "hello": 1,
        "world": 2,
        " ": 3,
        "he": 4,
        "ll": 5,
        "o": 6,
        "!": 7,
    }

    print("=" * 60)
    print("Tokenizer Tests")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Test 1: Basic tokenization with greedy longest-match
    # ------------------------------------------------------------------
    print("Test 1: Greedy longest-match")
    tokens = tokenize("hello world!", vocab)
    text_back = detokenize(tokens, vocab)
    print(f"  Input:       'hello world!'")
    print(f"  Tokens:      {tokens}")
    print(f"  Detokenized: '{text_back}'")
    # "hello" should match as a single token (not "he" + "ll" + "o")
    assert text_back == "hello world!", f"Round-trip failed: got '{text_back}'"
    # Check greedy: "hello"(1) should be chosen over "he"(4)
    assert tokens[0] == 1, f"Expected 'hello'(1) not 'he'(4), got token {tokens[0]}"
    print("  PASSED")
    print()

    # ------------------------------------------------------------------
    # Test 2: Unknown characters
    # ------------------------------------------------------------------
    print("Test 2: Unknown characters")
    tokens = tokenize("hello world?", vocab)
    text_back = detokenize(tokens, vocab)
    print(f"  Input:       'hello world?'")
    print(f"  Tokens:      {tokens}")
    print(f"  Detokenized: '{text_back}'")
    assert text_back == "hello world?", f"Round-trip failed: got '{text_back}'"
    print("  PASSED")
    print()

    # ------------------------------------------------------------------
    # Test 3: Multiple unknown characters
    # ------------------------------------------------------------------
    print("Test 3: Multiple unknowns")
    tokens = tokenize("hello @world#", vocab)
    text_back = detokenize(tokens, vocab)
    print(f"  Input:       'hello @world#'")
    print(f"  Tokens:      {tokens}")
    print(f"  Detokenized: '{text_back}'")
    assert text_back == "hello @world#", f"Round-trip failed: got '{text_back}'"
    print("  PASSED")
    print()

    # ------------------------------------------------------------------
    # Test 4: Empty string
    # ------------------------------------------------------------------
    print("Test 4: Empty string")
    tokens = tokenize("", vocab)
    text_back = detokenize(tokens, vocab)
    print(f"  Input:       ''")
    print(f"  Tokens:      {tokens}")
    print(f"  Detokenized: '{text_back}'")
    assert tokens == [], f"Expected empty list, got {tokens}"
    assert text_back == "", f"Expected empty string, got '{text_back}'"
    print("  PASSED")
    print()

    # ------------------------------------------------------------------
    # Test 5: All unknown characters
    # ------------------------------------------------------------------
    print("Test 5: All unknown characters")
    tokens = tokenize("xyz", vocab)
    text_back = detokenize(tokens, vocab)
    print(f"  Input:       'xyz'")
    print(f"  Tokens:      {tokens}")
    print(f"  Detokenized: '{text_back}'")
    assert text_back == "xyz", f"Round-trip failed: got '{text_back}'"
    print("  PASSED")
    print()

    # ------------------------------------------------------------------
    # Test 6: Overlapping prefixes
    # ------------------------------------------------------------------
    print("Test 6: Overlapping prefixes (greedy match)")
    tokens = tokenize("hell", vocab)
    text_back = detokenize(tokens, vocab)
    print(f"  Input:       'hell'")
    print(f"  Tokens:      {tokens}")
    print(f"  Detokenized: '{text_back}'")
    # "he" (4) + "ll" (5) -- greedy should match "he" first, then "ll"
    assert text_back == "hell", f"Round-trip failed: got '{text_back}'"
    print("  PASSED")
    print()

    # ------------------------------------------------------------------
    # Test 7: Unicode characters
    # ------------------------------------------------------------------
    print("Test 7: Unicode characters")
    tokens = tokenize("hello world\u2603", vocab)  # snowman
    text_back = detokenize(tokens, vocab)
    print(f"  Input:       'hello world\\u2603'")
    print(f"  Tokens:      {tokens}")
    print(f"  Detokenized: '{text_back}'")
    assert text_back == "hello world\u2603", f"Round-trip failed: got '{text_back}'"
    print("  PASSED")
    print()

    # ------------------------------------------------------------------
    # Test 8: Single character tokens used when longer match available
    # ------------------------------------------------------------------
    print("Test 8: Longest match preferred")
    # "hello" should be one token, not "he" + "ll" + "o"
    tokens = tokenize("hello", vocab)
    print(f"  Input:       'hello'")
    print(f"  Tokens:      {tokens}")
    assert len(tokens) == 1, (
        f"Expected 1 token for 'hello' (greedy), got {len(tokens)}: {tokens}"
    )
    assert tokens[0] == 1, f"Expected token ID 1 for 'hello', got {tokens[0]}"
    print("  PASSED")
    print()

    # ------------------------------------------------------------------
    # Test 9: Round-trip fidelity stress test
    # ------------------------------------------------------------------
    print("Test 9: Round-trip stress test")
    test_strings = [
        "hello",
        "world",
        "hello world",
        "hello world!",
        "",
        "!!!",
        "he",
        "hell",
        "hello hello hello",
        "abc def ghi",
        "hello\nworld",
        "\t\thello",
        "hello world! 123 @#$ xyz",
    ]
    all_passed = True
    for s in test_strings:
        tokens = tokenize(s, vocab)
        result = detokenize(tokens, vocab)
        if result != s:
            print(f"  FAILED for '{s}': got '{result}'")
            all_passed = False
    if all_passed:
        print(f"  All {len(test_strings)} strings passed round-trip")
    print("  PASSED" if all_passed else "  FAILED")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
