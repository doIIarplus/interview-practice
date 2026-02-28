# Rubric: Low-Latency Token Sampler

**Total: 100 points**

---

## 1. Numerically Stable Softmax (10 points)

### Full marks (10):
- Subtracts the maximum logit before exponentiating to prevent overflow
- Correctly computes `exp(x - max) / sum(exp(x - max))`
- Result sums to 1.0 (within floating point tolerance)
- Handles edge cases: single element, all same values, very large/small logits

### Partial credit (5-8):
- Correct formula but doesn't subtract max (works for small logits, fails for large)
- Subtracts max but has an off-by-one or other minor bug

### Minimal credit (1-4):
- Naive implementation without numerical stability consideration

### Reference implementation:
```python
def softmax(self, logits):
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

### Key test:
```python
# This MUST work without overflow:
probs = sampler.softmax([1000.0, 999.0, 998.0])
assert abs(sum(probs) - 1.0) < 1e-6
```

---

## 2. Correct Greedy Sampling (5 points)

### Full marks (5):
- Returns index of maximum logit value
- Handles ties consistently (first occurrence is fine)
- O(n) implementation (single pass)

### Partial credit (2-3):
- Correct but uses sort (O(n log n) instead of O(n))

### Minimal credit (1):
- Incorrect or doesn't handle ties

### Reference:
```python
def sample_greedy(self, logits):
    return max(range(len(logits)), key=lambda i: logits[i])
```

---

## 3. Correct Temperature Sampling (15 points)

### Full marks (15):
- Divides logits by temperature before softmax
- Handles temperature=0 by returning greedy result (no division by zero)
- Handles temperature=1 as identity (no scaling needed)
- Correctly samples from the resulting distribution

### Partial credit (8-12):
- Correct temperature scaling but temperature=0 causes crash
- Correct logic but doesn't optimize the temperature=1 case

### Minimal credit (1-7):
- Multiplies instead of divides by temperature
- Doesn't handle edge cases

### Reference:
```python
def sample_temperature(self, logits, temperature):
    if temperature == 0:
        return self.sample_greedy(logits)
    scaled = [x / temperature for x in logits]
    probs = self.softmax(scaled)
    return self._weighted_sample(probs)
```

### Key edge case:
Temperature=0 should NOT raise ZeroDivisionError. The candidate must either
check for it explicitly or use a very small epsilon. Checking explicitly and
returning greedy is the correct approach.

---

## 4. Correct Top-k Sampling (15 points)

### Full marks (15):
- Identifies the top-k logits correctly
- Sets non-top-k logits to negative infinity (or equivalent filtering)
- Applies temperature to the remaining logits
- Converts to probabilities and samples
- Returns a valid token index (not a position in the sorted array)

### Partial credit (8-12):
- Correct filtering but returns position in sorted array instead of original index
- Correct but uses full sort instead of partial sort

### Minimal credit (1-7):
- Incorrect filtering or probability computation

### Common bug:
Returning the position within the top-k list instead of the original token index:
```python
# WRONG: returns position in top-k
return random.choices(range(k), weights=top_probs)[0]

# CORRECT: returns original token index
return random.choices(top_indices, weights=top_probs)[0]
```

### Optimization opportunity:
Using `heapq.nlargest(k, range(len(logits)), key=lambda i: logits[i])` gives
O(n log k) instead of O(n log n) for the full sort. Worth bonus discussion points.

---

## 5. Correct Top-p (Nucleus) Sampling (20 points)

### Full marks (20):
- Applies temperature first, then converts to probabilities
- Sorts tokens by probability descending
- Accumulates probabilities until cumulative sum >= p
- Always includes at least one token (highest probability)
- Samples from the reduced set after renormalizing
- Returns original token index, not sorted position

### Partial credit (10-15):
- Correct logic but boundary handling is off (off-by-one in cumulative sum)
- Doesn't renormalize after filtering
- Sorts by logit instead of probability (correct if temperature=1, wrong otherwise)

### Minimal credit (1-9):
- Fundamentally incorrect cumulative probability computation
- Doesn't handle the case where the top token alone exceeds p

### Reference:
```python
def sample_top_p(self, logits, p, temperature=1.0):
    if temperature == 0:
        return self.sample_greedy(logits)
    scaled = [x / temperature for x in logits]
    probs = self.softmax(scaled)

    # Sort indices by probability descending
    indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)

    cumulative = 0.0
    selected_indices = []
    selected_probs = []
    for idx, prob in indexed:
        selected_indices.append(idx)
        selected_probs.append(prob)
        cumulative += prob
        if cumulative >= p:
            break

    # Renormalize
    total = sum(selected_probs)
    selected_probs = [p / total for p in selected_probs]

    return random.choices(selected_indices, weights=selected_probs, k=1)[0]
```

### Key boundary question:
When cumulative probability exactly equals p, should we include the next token?
The standard convention is to stop when cumulative >= p. The candidate should
state their choice and be consistent.

---

## 6. Correct Min-p Sampling (15 points)

### Full marks (15):
- Converts to probabilities first
- Finds max probability
- Computes threshold = min_p * max_probability
- Filters tokens with probability >= threshold
- Always includes at least one token
- Renormalizes and samples

### Partial credit (8-12):
- Correct threshold computation but doesn't renormalize
- Off-by-one in filtering (uses > instead of >=)

### Minimal credit (1-7):
- Incorrect threshold computation
- Uses min_p as an absolute threshold instead of relative to max

### Key distinction from top-p:
Min-p is simpler — it's a flat cutoff relative to the max, not cumulative.
This makes it O(n) instead of O(n log n) since no sorting is needed.

```python
def sample_min_p(self, logits, min_p, temperature=1.0):
    if temperature == 0:
        return self.sample_greedy(logits)
    scaled = [x / temperature for x in logits]
    probs = self.softmax(scaled)
    max_prob = max(probs)
    threshold = min_p * max_prob

    selected_indices = []
    selected_probs = []
    for i, p in enumerate(probs):
        if p >= threshold:
            selected_indices.append(i)
            selected_probs.append(p)

    # Renormalize
    total = sum(selected_probs)
    selected_probs = [p / total for p in selected_probs]

    return random.choices(selected_indices, weights=selected_probs, k=1)[0]
```

---

## 7. Performance Optimization (10 points)

### Full marks (10):
- Avoids computing full softmax when not needed (greedy doesn't need it)
- For top-k: uses partial sort (heapq) instead of full sort
- Avoids redundant list copies
- Uses `random.choices` with weights (efficient weighted sampling)
- For min-p: single pass filter without sorting

### Partial credit (5-8):
- Some optimizations but still computes unnecessary softmax in some paths
- Reasonable performance but not optimized

### Minimal credit (1-4):
- No performance consideration
- O(n^2) algorithms where O(n log n) or O(n) suffices

### Advanced optimizations (bonus discussion):
- Using `math.fsum` for more accurate summation
- Pre-computing log probabilities to avoid repeated exp/log
- For top-k: using `heapq.nlargest` which is O(n log k) vs O(n log n)
- Caching intermediate results across multiple samples from the same logits
- Using `array` module instead of lists for better cache performance

---

## 8. Code Clarity (10 points)

### Full marks (10):
- Clear, self-documenting function names and variable names
- Consistent style across all methods
- Helper methods factored out to avoid repetition (e.g., `_weighted_sample`)
- Type hints and docstrings
- Logical flow easy to follow

### Partial credit (5-8):
- Readable but some repetition across methods
- Minor style inconsistencies

### Minimal credit (1-4):
- Hard to follow, inconsistent naming, no factoring

---

## Red Flags (Automatic Deductions)

- **-10 points**: Softmax without max subtraction (numerical instability)
- **-10 points**: Temperature=0 causes ZeroDivisionError
- **-5 points**: Top-p doesn't include at least one token
- **-5 points**: Returns index within filtered array instead of original token index
- **-5 points**: Using numpy or torch (instructions say standard library only)
- **-5 points**: Probabilities don't renormalize to 1 after filtering

---

## Exceptional Answers (Bonus Discussion Points)

- Discusses how in practice this would use CUDA kernels (GPU-side sampling)
- Mentions that top-p sorting can be done with a GPU-friendly parallel prefix sum
- Proposes using log-domain arithmetic to avoid repeated exp/log conversions
- Discusses Gumbel-max trick as an alternative sampling approach
- Mentions that real systems (vLLM, TensorRT-LLM) use custom CUDA kernels for sampling
- Discusses how speculative decoding changes the sampling requirements
- Proposes batch sampling for multiple sequences simultaneously
