# Question 12: Low-Latency Token Sampler

## Difficulty: Hard
## Time: 60 minutes
## Category: ML Inference / Performance Optimization

---

## Background

You are implementing a token sampler for a language model inference engine. During
autoregressive text generation, the model produces a vector of **logits** (raw,
unnormalized scores) over the entire vocabulary at each step. A **sampling strategy**
converts these logits into a probability distribution and selects the next token.

The choice of sampling strategy has a dramatic effect on text quality:
- **Greedy**: Always picks the most likely token. Deterministic but repetitive.
- **Temperature**: Controls randomness. Low temperature = more deterministic.
- **Top-k**: Only considers the k most likely tokens. Prevents rare garbage tokens.
- **Top-p (Nucleus)**: Dynamically sizes the candidate set based on cumulative probability.
- **Min-p**: Filters based on probability relative to the most likely token.

Sampling happens on **every token generation step**, so it is on the critical path
of inference latency. For a 128K vocabulary, sampling must be fast.

---

## Task

Implement a `TokenSampler` class with the following methods. Use **only the Python
standard library** (no numpy, no torch). Then optimize for speed.

### `softmax(logits: list[float]) -> list[float]`

Convert raw logits to a probability distribution.

**Critical**: Must be numerically stable. Naive `exp(x) / sum(exp(x))` overflows
for large logits. Subtract the maximum logit before exponentiating.

**Example:**
```python
sampler = TokenSampler()
probs = sampler.softmax([2.0, 1.0, 0.1])
# probs ~= [0.659, 0.242, 0.099]
# sum(probs) == 1.0
```

### `sample_greedy(logits: list[float]) -> int`

Return the index of the token with the highest logit.

**Example:**
```python
sampler.sample_greedy([1.0, 3.0, 2.0])  # Returns 1
```

### `sample_temperature(logits: list[float], temperature: float) -> int`

Apply temperature scaling: divide all logits by `temperature`, then sample from
the resulting distribution.

- `temperature > 1.0`: More random (flattens the distribution)
- `temperature < 1.0`: More deterministic (sharpens the distribution)
- `temperature == 0`: Should behave like greedy (do NOT divide by zero)

**Example:**
```python
# High temperature — more random
sampler.sample_temperature([2.0, 1.0, 0.1], temperature=2.0)
# Could return 0, 1, or 2 with more even probabilities

# temperature=0 — greedy
sampler.sample_temperature([2.0, 1.0, 0.1], temperature=0.0)
# Always returns 0
```

### `sample_top_k(logits: list[float], k: int, temperature: float = 1.0) -> int`

Top-k sampling:
1. Find the k tokens with the highest logits
2. Set all other logits to negative infinity
3. Apply temperature scaling to the remaining logits
4. Convert to probabilities (softmax) and sample

**Example:**
```python
sampler.sample_top_k([5.0, 3.0, 1.0, 0.5, 0.1], k=2, temperature=1.0)
# Can only return 0 or 1 (the top-2 tokens)
```

### `sample_top_p(logits: list[float], p: float, temperature: float = 1.0) -> int`

Nucleus (top-p) sampling:
1. Convert logits to probabilities (apply temperature first)
2. Sort tokens by probability descending
3. Include tokens until cumulative probability exceeds `p`
4. Sample from this reduced set (renormalize probabilities)

At least one token (the highest probability token) must always be included, even
if its probability alone exceeds `p`.

**Example:**
```python
# If probs are [0.5, 0.3, 0.15, 0.05] and p=0.8:
# Include tokens until cumulative > 0.8: [0.5, 0.3] (cumulative = 0.8)
# Include one more since we need to exceed p: [0.5, 0.3, 0.15]
# Actually: standard top-p includes tokens until cumulative >= p
# So [0.5, 0.3] with cumulative 0.8 >= 0.8 => stop here
# Sample from token 0 and token 1 only
sampler.sample_top_p([2.0, 1.0, -1.0, -3.0], p=0.9, temperature=1.0)
```

### `sample_min_p(logits: list[float], min_p: float, temperature: float = 1.0) -> int`

Min-p sampling:
1. Convert logits to probabilities
2. Find the maximum probability (`max_prob`)
3. Filter out tokens whose probability is less than `min_p * max_prob`
4. Sample from the remaining tokens (renormalize)

At least one token (the highest probability token) must always be included.

**Example:**
```python
# If probs are [0.5, 0.3, 0.15, 0.05] and min_p=0.2:
# Threshold = 0.2 * 0.5 = 0.1
# Keep tokens with prob >= 0.1: [0.5, 0.3, 0.15]
# Filter out token 3 (prob 0.05 < 0.1)
sampler.sample_min_p([2.0, 1.0, -1.0, -3.0], min_p=0.2, temperature=1.0)
```

---

## Performance Requirement

All sampling methods should work on vocabulary sizes up to **128,000 tokens**.
The benchmark in `starter.py` will test with 50,000 tokens.

Targets (on a modern machine):
- Greedy: < 0.05 ms/call
- Temperature: < 0.5 ms/call
- Top-k: < 1.0 ms/call
- Top-p: < 2.0 ms/call
- Min-p: < 1.0 ms/call

---

## Starter Code

See `starter.py` for the class skeleton, a logit distribution generator, and a
benchmarking harness.

---

## Evaluation Criteria

- Numerically stable softmax implementation
- Correct greedy sampling
- Correct temperature sampling including the temperature=0 edge case
- Correct top-k: proper filtering, probability redistribution
- Correct top-p: cumulative probability threshold, correct boundary handling
- Correct min-p: threshold relative to max probability
- Performance optimization and avoiding redundant computation
- Code clarity
