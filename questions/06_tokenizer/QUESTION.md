# Question 06: Tokenizer with Round-Trip Fidelity

## Problem Statement

You are building a simple tokenizer for a text processing system. You are given a
**vocabulary** -- a mapping from token strings to integer IDs.

Implement two functions:

### 1. `tokenize(text: str, vocab: dict[str, int]) -> list[int]`

Tokenize the input text using **greedy longest-match** from left to right:
- At each position in the text, find the longest string starting at that position
  that exists in the vocabulary.
- Emit that token's ID and advance past it.
- If **no** token in the vocabulary matches the current position (not even a single
  character), emit a special representation for the **unknown character** and advance
  by one character.

### 2. `detokenize(token_ids: list[int], vocab: dict[str, int]) -> str`

Convert a list of token IDs back into the original text string.

### Critical Requirement: Round-Trip Fidelity

The tokenizer must satisfy this invariant for **all** inputs:

```python
detokenize(tokenize(text, vocab), vocab) == text
```

This means unknown characters must be **preserved** through the tokenize/detokenize
round-trip. The challenge is designing a representation that encodes the unknown
character's identity within the token ID sequence.

## Examples

```python
vocab = {
    "hello": 1,
    "world": 2,
    " ": 3,
    "he": 4,
    "ll": 5,
    "o": 6,
    "!": 7,
}

# Example 1: All tokens found via greedy longest-match
tokenize("hello world!", vocab)
# -> [1, 3, 2, 7]
# Explanation: "hello" (1), " " (3), "world" (2), "!" (7)
# Note: "hello" is preferred over "he" + "ll" + "o" because it's longer

detokenize([1, 3, 2, 7], vocab)
# -> "hello world!"

# Example 2: Unknown characters
tokenize("hello world?", vocab)
# -> [1, 3, 2, <unknown '?'>]
# The '?' is not in the vocabulary, so it must be encoded somehow
# Your representation must allow detokenize to recover '?'

# Example 3: Multiple unknowns
tokenize("hello @world#", vocab)
# -> [1, 3, <unknown '@'>, 2, <unknown '#'>]

# Round-trip invariant holds:
text = "hello @world#"
assert detokenize(tokenize(text, vocab), vocab) == text
```

## Design Decision

You need to decide how to represent unknown characters in the token ID list.
Some options to consider:
- Use negative IDs to encode the character (e.g., `-ord('?')` for `'?'`)
- Return a richer type than `list[int]` (e.g., list of token objects)
- Maintain a side channel for unknown character data

Each approach has trade-offs in terms of simplicity, compatibility, and generality.
Choose one and justify your decision.

## Constraints

- The vocabulary will not contain the empty string.
- Token IDs in the vocabulary are positive integers (>= 1).
- The input text can contain any Unicode character.
- Empty text should tokenize to an empty list.

## Getting Started

See `starter.py` for function signatures and test cases.
