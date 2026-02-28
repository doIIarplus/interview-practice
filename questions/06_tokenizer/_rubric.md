# Rubric: Tokenizer with Round-Trip Fidelity

**Total: 100 points**

---

## 1. Correct Greedy Longest-Match Tokenization (25 points)

| Points | Criteria |
|--------|----------|
| 25 | Correctly implements greedy longest-match from left to right; "hello" matches as one token, not "he"+"ll"+"o" |
| 20 | Greedy matching works for most cases but has edge case issues |
| 10 | Attempts longest match but implementation is incorrect (e.g., shortest match) |
| 5  | Simple character-by-character matching without greedy optimization |
| 0  | Does not tokenize correctly |

**Key implementation detail:**
At each position `i`, try all possible lengths from longest to shortest (or build a trie):
```python
pos = 0
while pos < len(text):
    matched = False
    for length in range(max_token_len, 0, -1):
        candidate = text[pos:pos + length]
        if candidate in vocab:
            tokens.append(vocab[candidate])
            pos += length
            matched = True
            break
    if not matched:
        # handle unknown character
        pos += 1
```

**Alternative (more efficient):** Build a trie from the vocabulary and walk it at each position.

**Common mistakes:**
- Matching shortest first instead of longest
- Not advancing by the full length of the matched token
- Off-by-one errors in substring extraction

---

## 2. Correct Detokenization (20 points)

| Points | Criteria |
|--------|----------|
| 20 | Correctly converts all token IDs (both known and unknown) back to text |
| 15 | Detokenizes known tokens correctly but partially handles unknowns |
| 10 | Basic detokenization works but breaks on edge cases |
| 0  | Does not detokenize correctly |

**Key requirement:** Build a reverse mapping from ID -> token string:
```python
id_to_token = {v: k for k, v in vocab.items()}
```

Must also handle whatever unknown character encoding was chosen in tokenize.

---

## 3. Round-Trip Fidelity for Unknown Characters (25 points)

This is the core challenge of the problem. The candidate must choose a strategy for
encoding unknown characters so they survive the round-trip.

| Points | Criteria |
|--------|----------|
| 25 | Perfect round-trip: `detokenize(tokenize(text, vocab), vocab) == text` for ALL inputs including unknown chars, Unicode, whitespace |
| 20 | Round-trip works for ASCII unknowns but fails for some Unicode |
| 15 | Has a strategy for unknowns but implementation has bugs |
| 10 | Acknowledges the problem but solution is incomplete |
| 5  | Uses a single UNK token that loses character identity |
| 0  | No handling of unknown characters |

### Encoding Strategies (from best to acceptable):

**Strategy A: Negative ordinal encoding (recommended)**
```python
# In tokenize: encode unknown char as -ord(char)
tokens.append(-ord(char))

# In detokenize: decode negative IDs back to characters
if token_id < 0:
    result += chr(-token_id)
```
- **Pros**: Simple, compact, handles all Unicode, return type is still `list[int]`
- **Cons**: Assumes vocab IDs are positive (stated in constraints)
- **Rating**: Excellent -- clean, minimal, leverages the constraint

**Strategy B: Rich token type**
```python
@dataclass
class Token:
    id: int
    text: str | None = None  # populated for unknown tokens

# tokenize returns list[Token]
```
- **Pros**: Explicit, self-documenting, extensible
- **Cons**: Changes the return type, more complex API
- **Rating**: Good -- solid engineering, but heavier

**Strategy C: Special UNK ID + side channel**
```python
# Return (token_ids, unknown_chars) tuple
def tokenize(text, vocab) -> tuple[list[int], dict[int, str]]:
    ...
```
- **Pros**: Keeps token IDs as plain ints
- **Cons**: More complex API, two things to pass around
- **Rating**: Acceptable but awkward

**Strategy D: Offset encoding**
```python
# Use IDs above max_vocab_id for unknown chars
UNK_OFFSET = max(vocab.values()) + 1
tokens.append(UNK_OFFSET + ord(char))
```
- **Pros**: All IDs are positive ints
- **Cons**: Requires knowing max vocab ID, potential collisions with future vocab additions
- **Rating**: Acceptable

---

## 4. Handling Edge Cases (15 points)

| Points | Criteria |
|--------|----------|
| 15 | Correctly handles all edge cases |
| 10 | Handles most edge cases |
| 5  | Handles basic cases only |
| 0  | Fails on common edge cases |

**Edge cases to verify:**
- **Empty string**: `tokenize("", vocab)` returns `[]`
- **All unknown characters**: `tokenize("xyz", vocab)` with none in vocab
- **All known characters**: standard case
- **Unicode**: snowman, emoji, CJK characters, combining characters
- **Newlines and whitespace**: tabs, newlines, carriage returns
- **Overlapping prefixes**: "he" vs "hello" -- must pick longest
- **Single character at end**: "hello world?" where "?" is unknown
- **Consecutive unknowns**: "hello @#$ world"

---

## 5. Code Clarity and Efficiency (15 points)

| Points | Criteria |
|--------|----------|
| 15 | Clean, well-structured code with good variable names; O(n * max_token_len) or better |
| 12 | Clean code but naive efficiency (e.g., tries all possible lengths up to len(text)) |
| 8  | Working code but hard to follow |
| 4  | Messy code with unclear logic |
| 0  | Incomprehensible |

**Efficiency analysis:**
- **Naive**: O(n * V) where V is vocab size (check every vocab entry at every position)
- **Better**: O(n * L) where L is max token length (try all lengths at each position)
- **Best**: O(n) amortized with a trie (Aho-Corasick for parallel matching)

**Code quality indicators:**
- Uses a reverse vocab mapping (built once, not per-token)
- Precomputes max token length to bound the inner loop
- Clear separation of tokenize and detokenize logic
- Handles the unknown encoding strategy consistently in both functions
- Good variable names: `pos`, `token_ids`, `id_to_token`, etc.

---

## Ideal Solution

```python
def tokenize(text: str, vocab: dict[str, int]) -> list[int]:
    if not text:
        return []

    max_len = max(len(token) for token in vocab)
    tokens: list[int] = []
    pos = 0

    while pos < len(text):
        # Greedy longest-match
        best_length = 0
        best_id = None
        for length in range(min(max_len, len(text) - pos), 0, -1):
            candidate = text[pos:pos + length]
            if candidate in vocab:
                best_id = vocab[candidate]
                best_length = length
                break

        if best_id is not None:
            tokens.append(best_id)
            pos += best_length
        else:
            # Unknown character: encode as negative ordinal
            tokens.append(-ord(text[pos]))
            pos += 1

    return tokens


def detokenize(token_ids: list[int], vocab: dict[str, int]) -> str:
    id_to_token = {v: k for k, v in vocab.items()}
    parts: list[str] = []

    for token_id in token_ids:
        if token_id < 0:
            # Unknown character encoded as negative ordinal
            parts.append(chr(-token_id))
        else:
            parts.append(id_to_token[token_id])

    return "".join(parts)
```

---

## Red Flags

- Uses a single `UNK = 0` token that loses the original character (no round-trip fidelity)
- Doesn't implement greedy longest-match (uses first match or shortest match)
- Doesn't build a reverse vocab mapping (scans entire vocab on each detokenize step)
- Modifies the input vocab dict
- Doesn't handle empty string
- Uses regex for tokenization without understanding why greedy matters

## Green Flags

- Immediately identifies the round-trip fidelity challenge as the core difficulty
- Chooses negative ordinal encoding quickly with clear justification
- Precomputes max token length for efficiency
- Mentions tries as a potential optimization
- Discusses connection to real tokenizers (BPE, tiktoken)
- Considers the time complexity and discusses optimization paths
- Tests their solution mentally with edge cases before writing code
