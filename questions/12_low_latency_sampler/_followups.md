# Follow-up Questions: Low-Latency Token Sampler

---

## 1. Your softmax computes exp() for all tokens. For top-k, can you avoid this?

**What we're looking for:**
- Yes! For top-k, you only need the k largest logits. You can find those BEFORE
  computing softmax, since the ordering of logits is the same as the ordering of
  probabilities (exp is monotonic)
- Use `heapq.nlargest(k, ...)` to find top-k in O(n log k) without full sort
- Then only compute softmax over the k selected logits
- This reduces exp() calls from vocab_size (128K) to k (typically 50)
- For min-p this is harder: you need the probabilities to know the threshold,
  so you typically need full softmax first (though you could use a two-pass approach)

**Strong answer includes:**
- Recognizes that logit ordering = probability ordering for identifying top tokens
- Proposes a lazy/partial softmax approach
- Discusses that for top-p, you could sort by logit, compute softmax incrementally
  until cumulative probability exceeds p, and stop early

---

## 2. What is the time complexity of each method? Can you do top-k without full sort?

**What we're looking for:**

| Method      | Naive         | Optimized       |
|-------------|---------------|-----------------|
| Greedy      | O(n)          | O(n)            |
| Temperature | O(n)          | O(n)            |
| Top-k       | O(n log n)    | O(n log k)      |
| Top-p       | O(n log n)    | O(n log n)*     |
| Min-p       | O(n)          | O(n)            |

*Top-p requires sorting by probability, so O(n log n) is hard to avoid unless
you use approximate methods or the distribution is very peaked.

**Strong answer includes:**
- `heapq.nlargest` for O(n log k) top-k selection
- QuickSelect (introselect) for O(n) expected time top-k partition
- For top-p: could use a partial sort approach — sort until cumulative prob exceeds p
- Radix sort for integer-quantized logits (O(n) but with high constant)

---

## 3. How would you implement repetition penalty? Frequency/presence penalties?

**What we're looking for:**
- **Repetition penalty**: Multiply logits of previously generated tokens by a
  penalty factor (divide if logit > 0, multiply if < 0)
- **Frequency penalty**: Subtract `frequency_penalty * count(token)` from logit
  for each token that appeared `count` times in generated text
- **Presence penalty**: Subtract `presence_penalty * (1 if token appeared else 0)`
- These are applied BEFORE temperature/sampling

**Strong answer includes:**
- Implementation detail: maintain a frequency counter of generated tokens
- The penalty is applied to logits, not probabilities
- The repetition penalty from the original paper (Keskar et al., 2019) divides
  by the penalty if logit > 0 and multiplies if logit < 0
- Discussion of how different penalties affect generation quality:
  - Repetition penalty prevents loops
  - Frequency penalty encourages vocabulary diversity
  - Presence penalty encourages topic diversity

---

## 4. What would you use instead of Python lists in a real inference engine?

**What we're looking for:**
- **numpy arrays**: Vectorized operations, BLAS-accelerated
- **PyTorch tensors**: GPU-native, can stay on GPU without data transfer
- **CUDA kernels**: Custom kernels for fused sampling operations
- **Triton**: Write GPU kernels in Python-like syntax

**Strong answer includes:**
- The key insight: in a real system, logits are already on the GPU as a tensor.
  Copying them to CPU for sampling and back adds latency.
- Best approach: do sampling entirely on GPU with a custom CUDA kernel
- vLLM and TensorRT-LLM both have custom CUDA sampling kernels
- Even for CPU sampling, numpy vectorization gives 10-100x speedup over Python lists
- `torch.multinomial` is the standard GPU-side weighted sampling function

---

## 5. How does KV-cache affect sampling performance in autoregressive generation?

**What we're looking for:**
- KV-cache stores the key/value tensors from previous tokens so they don't
  need to be recomputed at each step
- Without KV-cache: each step recomputes attention over all previous tokens (O(n^2))
- With KV-cache: each step only computes attention for the new token (O(n) per step)
- The KV-cache itself consumes GPU memory and grows linearly with sequence length
- Memory pressure from KV-cache can affect sampling by limiting batch size

**Strong answer includes:**
- KV-cache memory = 2 * num_layers * num_heads * head_dim * seq_len * batch_size * sizeof(dtype)
- For large models (e.g., 70B), KV-cache can consume tens of GB for long sequences
- Paged attention (vLLM) manages KV-cache like virtual memory pages to reduce waste
- Multi-query attention (MQA) and grouped-query attention (GQA) reduce KV-cache size
- The sampling step itself is not directly affected by KV-cache, but KV-cache
  management determines how many sequences can be batched together

---

## 6. What are the trade-offs between top-k, top-p, and min-p in practice?

**What we're looking for:**
- **Top-k** is simple but the fixed k doesn't adapt to confidence — when the model
  is very confident, k tokens is too many; when uncertain, k might be too few
- **Top-p** adapts dynamically: when the model is confident, fewer tokens are included;
  when uncertain, more tokens are included. This is generally preferred over top-k.
- **Min-p** is similar to top-p but based on individual token probabilities rather
  than cumulative. It's more intuitive: "include tokens that are at least X% as
  likely as the top token"
- In practice, combinations are common: top-k with top-p, or top-k with temperature

**Strong answer includes:**
- Top-p was introduced by Holtzman et al. (2020) "The Curious Case of Neural Text
  Degeneration" — showed that top-k produces poor results for peaked distributions
- Min-p is newer and gaining popularity in open-source LLM inference
- In production systems (like Anthropic's), the sampling strategy is a
  hyperparameter tuned for the application
- Temperature typically applied FIRST, then top-k/top-p/min-p filtering

---

## 7. How would you implement speculative decoding to reduce latency?

**What we're looking for:**
- **Core idea**: Use a small, fast "draft" model to generate k candidate tokens,
  then verify them all at once with the large "target" model
- The target model runs a single forward pass on all k candidates (which is
  efficient due to parallelism) and accepts/rejects each one
- If all k are accepted, you get k tokens for the cost of one target forward pass
- Acceptance rate depends on how well the draft model approximates the target

**Strong answer includes:**
- The acceptance criterion uses the target model's probabilities to decide
  which draft tokens to accept, maintaining the exact same distribution as
  sampling from the target alone
- Speedup = k * acceptance_rate, where k is the speculation length
- The draft model must be much faster than the target (e.g., 10x fewer parameters)
- Medusa approach: instead of a separate draft model, add extra heads to the
  target model that predict future tokens
- The sampling algorithm becomes more complex: need to handle partial acceptance
  and re-sampling from a modified distribution

---

## 8. How would you batch multiple sampling requests efficiently?

**What we're looking for:**
- In serving systems, multiple requests are processed simultaneously
- Logits for different sequences can be stacked into a batch tensor
- Sampling can be parallelized across the batch on GPU
- Challenge: different sequences may use different sampling parameters
  (temperature, top-k, etc.)

**Strong answer includes:**
- **Continuous batching** (vLLM, Orca): new requests join the batch as old ones
  finish, maximizing GPU utilization
- Per-sequence sampling parameters: need to handle different temperatures, top-k
  values, etc. within the same batch. Can group by parameters or use a kernel that
  takes per-sequence parameters.
- Memory layout: batch of logits is a 2D tensor [batch_size, vocab_size]; sampling
  operates independently per row but can be parallelized across rows
- `torch.multinomial` supports batched sampling natively
