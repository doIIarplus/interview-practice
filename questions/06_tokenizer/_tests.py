"""Hidden tests for Question 06: Tokenizer with Round-Trip Fidelity
Run: python questions/06_tokenizer/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from starter import tokenize, detokenize


VOCAB = {
    "hello": 1,
    "world": 2,
    " ": 3,
    "he": 4,
    "ll": 5,
    "o": 6,
    "!": 7,
}


def test_greedy_longest_match():
    """Test greedy longest-match tokenization."""
    tokens = tokenize("hello world!", VOCAB)
    text_back = detokenize(tokens, VOCAB)
    assert text_back == "hello world!", f"Round-trip failed: got '{text_back}'"
    assert tokens[0] == 1, f"Expected 'hello'(1) not 'he'(4), got token {tokens[0]}"
    print("  [PASS] test_greedy_longest_match")


def test_unknown_characters():
    """Test handling of unknown characters."""
    tokens = tokenize("hello world?", VOCAB)
    text_back = detokenize(tokens, VOCAB)
    assert text_back == "hello world?", f"Round-trip failed: got '{text_back}'"
    print("  [PASS] test_unknown_characters")


def test_multiple_unknowns():
    """Test multiple unknown characters."""
    tokens = tokenize("hello @world#", VOCAB)
    text_back = detokenize(tokens, VOCAB)
    assert text_back == "hello @world#", f"Round-trip failed: got '{text_back}'"
    print("  [PASS] test_multiple_unknowns")


def test_empty_string():
    """Test empty string."""
    tokens = tokenize("", VOCAB)
    text_back = detokenize(tokens, VOCAB)
    assert tokens == [], f"Expected empty list, got {tokens}"
    assert text_back == "", f"Expected empty string, got '{text_back}'"
    print("  [PASS] test_empty_string")


def test_all_unknown():
    """Test all unknown characters."""
    tokens = tokenize("xyz", VOCAB)
    text_back = detokenize(tokens, VOCAB)
    assert text_back == "xyz", f"Round-trip failed: got '{text_back}'"
    print("  [PASS] test_all_unknown")


def test_overlapping_prefixes():
    """Test overlapping prefixes (greedy match)."""
    tokens = tokenize("hell", VOCAB)
    text_back = detokenize(tokens, VOCAB)
    assert text_back == "hell", f"Round-trip failed: got '{text_back}'"
    print("  [PASS] test_overlapping_prefixes")


def test_unicode():
    """Test unicode characters."""
    tokens = tokenize("hello world\u2603", VOCAB)
    text_back = detokenize(tokens, VOCAB)
    assert text_back == "hello world\u2603", f"Round-trip failed: got '{text_back}'"
    print("  [PASS] test_unicode")


def test_longest_match_preferred():
    """Test that longest match is preferred."""
    tokens = tokenize("hello", VOCAB)
    assert len(tokens) == 1, f"Expected 1 token for 'hello' (greedy), got {len(tokens)}: {tokens}"
    assert tokens[0] == 1, f"Expected token ID 1 for 'hello', got {tokens[0]}"
    print("  [PASS] test_longest_match_preferred")


def test_round_trip_stress():
    """Round-trip stress test with many strings."""
    test_strings = [
        "hello", "world", "hello world", "hello world!", "",
        "!!!", "he", "hell", "hello hello hello", "abc def ghi",
        "hello\nworld", "\t\thello", "hello world! 123 @#$ xyz",
    ]
    all_passed = True
    for s in test_strings:
        tokens = tokenize(s, VOCAB)
        result = detokenize(tokens, VOCAB)
        if result != s:
            print(f"  FAILED for '{s}': got '{result}'")
            all_passed = False
    assert all_passed, "Some round-trip tests failed"
    print(f"  [PASS] test_round_trip_stress ({len(test_strings)} strings)")


def run_tests():
    print("=" * 60)
    print("Tokenizer Tests")
    print("=" * 60 + "\n")

    test_greedy_longest_match()
    test_unknown_characters()
    test_multiple_unknowns()
    test_empty_string()
    test_all_unknown()
    test_overlapping_prefixes()
    test_unicode()
    test_longest_match_preferred()
    test_round_trip_stress()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
