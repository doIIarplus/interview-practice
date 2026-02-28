# Follow-Up Questions: KV-Cache Manager

## 1. Why is paged attention such a big deal for LLM serving throughput?

**What to look for:**
- Without paging, each request pre-allocates memory for the maximum possible generation length. Since most requests generate far fewer tokens than the maximum, this leads to massive **memory fragmentation** and **waste**.
- vLLM showed that naive memory management wastes 60-80% of KV-cache memory. PagedAttention reduced waste to <4%.
- More efficient memory usage means more requests can be served concurrently (higher **batch size**), which directly increases throughput (tokens/second across all requests).
- The throughput improvement is often 2-4x over naive implementations.
- Strong candidates understand that LLM serving is **memory-bandwidth bound** during the decode phase, so fitting more requests in memory (larger batch) amortizes the cost of loading model weights.

## 2. How does vLLM implement PagedAttention? What are the GPU kernel implications?

**What to look for:**
- vLLM uses a block-based approach where each "block" holds a fixed number of tokens' KV vectors.
- A **block table** (similar to an OS page table) maps logical block indices to physical block addresses in GPU memory.
- The attention kernel must be modified to perform **indirect lookups**: instead of accessing KV-cache at contiguous addresses, it looks up the block table to find the physical address of each block.
- This adds some overhead per attention computation (pointer chasing), but the memory savings far outweigh the cost.
- The GPU kernel uses the block table to gather KV vectors from non-contiguous physical blocks.
- Strong candidates mention that this requires custom CUDA kernels (vLLM provides them) because standard attention implementations assume contiguous KV storage.

## 3. What is Grouped Query Attention (GQA) and Multi-Query Attention (MQA)? How do they reduce KV-cache size?

**What to look for:**
- **Multi-Head Attention (MHA)**: each attention head has its own K, V projections. KV-cache size is proportional to `num_heads`.
- **Multi-Query Attention (MQA)**: all attention heads share a SINGLE K, V head. Reduces KV-cache by `num_heads`x. Used in PaLM, Falcon.
- **Grouped Query Attention (GQA)**: a compromise -- `num_kv_heads` groups, each shared by `num_heads / num_kv_heads` query heads. Example: Llama 2 70B uses 8 KV heads and 64 query heads.
- GQA reduces KV-cache by `num_heads / num_kv_heads` factor with minimal quality loss.
- For our `ModelConfig(num_kv_heads=8)`: if the model has 64 query heads, GQA reduces KV-cache by 8x compared to MHA.
- Strong candidates compute concrete numbers: MHA with 64 heads would need `64 * 128 * 2 = 16,384` bytes per token per layer; GQA with 8 KV heads needs `8 * 128 * 2 = 2,048` bytes.

## 4. How does the KV-cache size scale with sequence length? What happens at 100K+ context?

**What to look for:**
- KV-cache size scales **linearly** with sequence length: `bytes = seq_len * bytes_per_token`.
- For our config: `bytes_per_token = 128 KB`. At 100K context: `100,000 * 128 KB = 12.5 GB` per request.
- An A100 80GB GPU, after loading model weights (~70GB for a 70B model in FP16), has only ~10GB left for KV-cache -- not enough for even one 100K request.
- Solutions: KV-cache compression (quantization to FP8 or INT4), multi-GPU KV-cache sharding, attention sinks (StreamingLLM), sliding window attention, KV-cache eviction policies.
- Strong candidates mention that the attention computation itself is O(N^2), so long contexts are expensive in both memory and compute.

## 5. What is KV-cache quantization (e.g., FP8 KV-cache)? What are the accuracy trade-offs?

**What to look for:**
- KV-cache quantization stores K and V vectors in lower precision (FP8, INT8, or even INT4) instead of FP16.
- This reduces cache size by 2-4x with minimal accuracy impact (because the cached values are just intermediate activations, not weights).
- Techniques: per-tensor quantization, per-channel quantization, per-token quantization.
- FP8 KV-cache (used in some production systems) halves memory usage with negligible perplexity degradation.
- INT4 KV-cache (e.g., KIVI, Atom) achieves 4x compression but requires more careful calibration.
- The key insight is that KV values have lower dynamic range than weights, so aggressive quantization is feasible.
- Trade-offs: very long sequences accumulate quantization error; some tasks (math, code) are more sensitive.

## 6. How would you implement beam search with a shared KV-cache (copy-on-write)?

**What to look for:**
- In beam search, multiple beams share a common prefix and diverge at different tokens.
- Naive approach: copy the entire KV-cache for each beam -- O(beam_width * seq_len) memory.
- Efficient approach: **copy-on-write** (COW) with a tree structure. Shared prefix pages have reference counts. When a beam diverges, only the new pages are allocated; shared pages are referenced, not copied.
- This is exactly like OS copy-on-write for forked processes.
- With paged attention, COW is natural: just add the shared pages to the new beam's page table and increment reference counts.
- Strong candidates note that this is one of vLLM's key innovations beyond just paging.

## 7. How does speculative decoding interact with the KV-cache?

**What to look for:**
- Speculative decoding uses a small "draft" model to generate K candidate tokens, then the large "target" model verifies them in parallel.
- The KV-cache must handle **tentative** entries: the draft model's tokens are added to the cache, but if the target model rejects them, those entries must be **rolled back**.
- This requires either: (a) maintaining a checkpoint of the KV-cache state before speculation, or (b) supporting efficient deletion of the last N entries.
- With paged attention, rollback is efficient: just deallocate the most recently allocated pages (or adjust the token count within the last page).
- Strong candidates note that speculative decoding also needs a separate KV-cache for the draft model (which is much smaller).

## 8. What is continuous batching and how does it relate to KV-cache management?

**What to look for:**
- **Static batching**: all requests in a batch start and end together. Short requests waste GPU time waiting for long ones.
- **Continuous batching** (aka "iteration-level batching"): requests can join and leave the batch at each decoding step. When one request finishes, a new one takes its slot immediately.
- This requires **dynamic KV-cache management**: memory must be allocated for new requests and freed for completed ones on every iteration.
- Paged attention enables continuous batching because pages can be allocated and freed at fine granularity without memory fragmentation.
- Without paging, continuous batching is impractical because contiguous memory allocation leads to fragmentation as requests come and go.
- Strong candidates mention that continuous batching was introduced by Orca (Yu et al., OSDI 2022) and is now standard in all production LLM serving systems.
