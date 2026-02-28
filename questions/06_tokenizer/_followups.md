# Follow-up Questions: Tokenizer

---

## 1. What's the time complexity of your tokenize function? How would you optimize it?

**Expected discussion:**
- **Current (naive greedy)**: O(n * L) where n = text length, L = max token length
  - At each position, try up to L substrings, each requiring a hash lookup in the vocab dict
  - String slicing is O(L) itself, so worst case is O(n * L^2) with slicing
- **Optimization with a set of lengths**: Only try lengths that actually exist in the vocab
  - Precompute `valid_lengths = {len(token) for token in vocab}` and only check those
- **Optimization with a trie**: Build a trie from the vocabulary
  - Walk the trie character by character from each position
  - Find the longest match in O(L) time without creating substrings
  - Total: O(n * L) but with much smaller constant factor (no hashing, no string creation)
- **Aho-Corasick**: Finds all matches in O(n + total_matches) but overkill for greedy left-to-right

**Strong answer:** Describes the trie optimization with concrete implementation ideas.
Mentions that in practice, L is small (typical max token length is ~20-30 characters),
so the naive approach is often fast enough.

---

## 2. How would you implement this using a trie for the vocabulary lookup?

**Expected discussion:**
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.token_id = None  # set if this node is end of a token

def build_trie(vocab):
    root = TrieNode()
    for token, token_id in vocab.items():
        node = root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.token_id = token_id
    return root

def tokenize_with_trie(text, root):
    tokens = []
    pos = 0
    while pos < len(text):
        node = root
        best_match = None
        best_length = 0
        for i in range(pos, len(text)):
            char = text[i]
            if char not in node.children:
                break
            node = node.children[char]
            if node.token_id is not None:
                best_match = node.token_id
                best_length = i - pos + 1
        if best_match is not None:
            tokens.append(best_match)
            pos += best_length
        else:
            tokens.append(-ord(text[pos]))
            pos += 1
    return tokens
```

**Key points:**
- Trie avoids redundant string comparisons -- shared prefixes are traversed once
- Memory trade-off: trie uses more memory than a flat dict for the vocab
- For large vocabularies (100K+ tokens), trie can be significantly faster
- Alternative: DAFSA (Directed Acyclic Finite State Automaton) for compressed tries

---

## 3. What if the vocabulary is very large (100K+ tokens)? How does this affect performance?

**Expected discussion:**
- **Dict approach**: Hash lookups are O(1) amortized, so vocab size doesn't matter directly.
  But more tokens means more potential lengths to try, so L (max token length) matters more.
- **Trie approach**: Build time is O(sum of all token lengths). Lookup is bounded by max
  token length, independent of vocab size. Memory scales with total characters in vocab.
- **Real-world consideration**: tiktoken uses ~100K tokens. The vocabulary is stored as a
  compiled structure (like a perfect hash or sorted array) for fast lookup.
- **Memory pressure**: 100K entries with average length 5 = ~500KB for the vocab alone.
  Trie with per-character nodes could be several MB. Consider compressed tries.
- **Batch tokenization**: For large texts, consider processing in chunks or using SIMD
  for character comparisons.

**Strong answer:** Distinguishes between lookup time (independent of vocab size with hashing)
and the number of candidate lengths to try. Mentions that real tokenizers use compiled/native
implementations for performance.

---

## 4. How would you handle Unicode characters that are multiple bytes?

**Expected discussion:**
- In Python, strings are sequences of Unicode code points, so `text[i]` gives a full
  character regardless of its byte representation. The naive approach "just works."
- However, some Unicode "characters" are composed of multiple code points:
  - Combining characters: `e` + `\u0301` = `e'` (accented e, two code points)
  - Emoji with modifiers: `\U0001F468\U0001F3FB` (person with skin tone, multiple code points)
  - ZWJ sequences: family emoji = multiple emoji joined by zero-width joiners
- If the vocabulary contains multi-codepoint entries, greedy matching handles them naturally
- For unknown characters: `ord(char)` works for single code points, but a multi-codepoint
  grapheme cluster would need special handling
- **Real tokenizers (tiktoken, sentencepiece)** work at the **byte level**, not the character
  level. Every possible byte sequence can be tokenized. This avoids all Unicode issues.
- `unicodedata.normalize()` can be used to normalize text before tokenization for consistency

**Strong answer:** Distinguishes code points from grapheme clusters. Mentions byte-level
tokenization as the industry solution.

---

## 5. How would BPE (Byte-Pair Encoding) differ from this greedy approach?

**Expected discussion:**
- **Greedy longest-match** (this problem): fixed vocabulary, always match the longest token
- **BPE**: vocabulary is learned from data through iterative merging
  1. Start with individual bytes (or characters) as the initial vocabulary
  2. Count all adjacent pairs in the training corpus
  3. Merge the most frequent pair into a new token
  4. Repeat until desired vocabulary size is reached
- **Key difference in tokenization**:
  - BPE applies merge rules in a specific priority order, which may not produce
    the longest match at each position
  - Example: if "ab" was merged before "abc", BPE would tokenize "abcd" as
    ["ab", "cd"], not ["abc", "d"], even if "abc" is in the vocabulary
  - Our greedy approach would pick "abc" if it's the longest match
- **BPE is deterministic** given the merge rules order
- **Advantages of BPE**: learns a vocabulary that captures statistical patterns in the data
  (common words, subwords, morphemes)
- **Disadvantage of greedy**: requires a hand-crafted vocabulary

**Strong answer:** Explains the merge-rule ordering distinction and why BPE tokenization
order differs from greedy longest-match. Mentions that BPE produces subword units
that generalize better to unseen words.

---

## 6. Real tokenizers like tiktoken use byte-level BPE. What are the advantages?

**Expected discussion:**
- **Byte-level BPE** operates on raw bytes (0-255), not characters
  - Initial vocabulary: 256 byte tokens
  - Every possible input can be tokenized (no unknown characters!)
  - Unicode handling is "free" -- just tokenize the UTF-8 byte sequence
- **Advantages**:
  - No UNK tokens needed -- every byte sequence is representable
  - Language-agnostic -- works for any language/script
  - Compact vocabulary for common sequences, graceful degradation for rare ones
  - Can represent any binary data, not just text
- **tiktoken specifics**:
  - Uses a precomputed BPE merge table (not trained at runtime)
  - Implemented in Rust for performance
  - Vocabulary size: ~100K tokens for cl100k_base
  - Special tokens for control (e.g., `<|endoftext|>`)
- **Comparison to our tokenizer**:
  - Our tokenizer has UNK characters; byte-level BPE doesn't
  - Our tokenizer works at character level; byte-level BPE is lower level
  - Our tokenizer uses a fixed vocab; BPE vocabulary is learned from data
  - Our tokenizer is O(n * L); tiktoken uses optimized Rust code

**Strong answer:** Explains the "no UNK" advantage of byte-level approaches and
connects it to the round-trip fidelity challenge from this problem. Mentions that
byte-level BPE solves the unknown character problem by design.
