# Transformer Inference Optimization — From Architecture to Production

This guide explains the transformer architecture and the key optimizations used
to serve large language models efficiently. It covers KV-caching, paged
attention, quantization, sampling, Flash Attention, and speculative decoding —
all topics relevant to Q12, Q14, Q16, and Q17.

---

## Table of Contents

1. [The Transformer Block](#the-transformer-block)
2. [Self-Attention Mechanism](#self-attention-mechanism)
3. [Autoregressive Generation](#autoregressive-generation)
4. [The KV-Cache](#the-kv-cache)
5. [Memory Cost of KV-Cache](#memory-cost-of-kv-cache)
6. [Paged Attention (vLLM)](#paged-attention-vllm)
7. [Grouped Query Attention (GQA) and Multi-Query Attention (MQA)](#grouped-query-attention-gqa-and-multi-query-attention-mqa)
8. [Flash Attention](#flash-attention)
9. [Quantization for Inference](#quantization-for-inference)
10. [Token Sampling Strategies](#token-sampling-strategies)
11. [Speculative Decoding](#speculative-decoding)
12. [Continuous Batching](#continuous-batching)
13. [Concrete Numbers: Llama 70B Example](#concrete-numbers-llama-70b-example)
14. [Key Takeaways](#key-takeaways)

---

## The Transformer Block

A transformer model is a stack of identical **transformer blocks** (also called
layers). Each block has two main sub-layers:

```
Input (sequence of token embeddings)
  │
  ▼
┌─────────────────────────────────┐
│         Layer Norm              │
├─────────────────────────────────┤
│     Multi-Head Self-Attention   │ ◄── Attends to all positions
├────────────────┬────────────────┤
│                │ + Residual     │ ◄── Skip connection: output += input
├────────────────┴────────────────┤
│         Layer Norm              │
├─────────────────────────────────┤
│    Feed-Forward Network (FFN)   │ ◄── Two linear layers with activation
│    (also called MLP)            │     FFN(x) = W2 * activation(W1 * x)
├────────────────┬────────────────┤
│                │ + Residual     │ ◄── Skip connection: output += input
└────────────────┴────────────────┘
  │
  ▼
Output (same shape as input)

Repeat this block L times (e.g., L=80 for a 70B model).
```

### Dimensions

- **d_model** (hidden dimension): e.g., 8192 for a 70B model
- **n_heads** (number of attention heads): e.g., 64
- **d_head** (dimension per head): d_model / n_heads = 128
- **d_ff** (FFN intermediate dimension): typically 4 * d_model = 32768
  (or with SwiGLU: 8/3 * d_model, rounded)
- **n_layers**: e.g., 80

---

## Self-Attention Mechanism

Self-attention lets each token "look at" all other tokens in the sequence to
decide what information to gather.

### Step by step

Given input X of shape (seq_len, d_model):

```
1. Project into Q, K, V for each head:
   Q = X @ W_Q    shape: (seq_len, n_heads, d_head)
   K = X @ W_K    shape: (seq_len, n_kv_heads, d_head)
   V = X @ W_V    shape: (seq_len, n_kv_heads, d_head)

2. Compute attention scores (per head):
   scores = Q @ K^T / sqrt(d_head)     shape: (seq_len, seq_len)

       Q (query)      K^T (key)        scores
   ┌───────────┐   ┌───────────┐    ┌───────────┐
   │ q0        │   │ k0 k1 k2  │    │ s00 s01 s02│
   │ q1        │ @ │           │ =  │ s10 s11 s12│
   │ q2        │   │           │    │ s20 s21 s22│
   └───────────┘   └───────────┘    └───────────┘
   (seq, d_head)   (d_head, seq)    (seq, seq)

3. Apply causal mask (for autoregressive models):
   scores[i][j] = -inf  if j > i   (can't attend to future tokens)

   ┌───────────┐
   │ s00 -inf -inf│   Token 0 can only see token 0
   │ s10  s11 -inf│   Token 1 can see tokens 0-1
   │ s20  s21  s22│   Token 2 can see tokens 0-2
   └───────────┘

4. Softmax (per row):
   weights = softmax(scores, dim=-1)   shape: (seq_len, seq_len)

5. Weighted sum of V:
   output = weights @ V                shape: (seq_len, d_head)

6. Concatenate heads and project:
   output = concat(head_0, ..., head_n) @ W_O   shape: (seq_len, d_model)
```

### Complexity

- **Compute:** O(seq_len^2 * d_model) per layer — quadratic in sequence length
- **Memory:** O(seq_len^2) for the attention matrix — this is the bottleneck
  for long sequences

---

## Autoregressive Generation

LLMs generate text one token at a time:

```
Prompt: "The cat sat on the"
                                       ┌──────────────────────┐
Step 1: Process prompt (prefill)       │ Forward pass with all │
        Input: [The, cat, sat, on, the]│ 5 tokens at once     │
        Output: logits for next token  │ → predict "mat"      │
                                       └──────────────────────┘

Step 2: Generate token 6 (decode)      ┌──────────────────────┐
        Input: [mat]                   │ Forward pass with     │
        (+ cached K,V from step 1)     │ 1 new token          │
        Output: logits → "."           │ Attend to all 6 prev │
                                       └──────────────────────┘

Step 3: Generate token 7 (decode)
        Input: [.]
        (+ cached K,V from steps 1-2)
        Output: logits → <end>

Each decode step is O(seq_len * d_model) — linear in sequence length.
But we need the K,V from ALL previous tokens for attention.
```

### Two phases

1. **Prefill (prompt processing):** Process all prompt tokens in parallel.
   Compute-bound (large matrix multiplies). High GPU utilization.

2. **Decode (token generation):** Generate one token at a time. Memory-bound
   (tiny matrix multiplies, loading model weights for each token). Low GPU
   utilization. **This is the bottleneck for latency.**

---

## The KV-Cache

### The problem

At each decode step, self-attention requires Q, K, V for ALL previous tokens.
Without caching, we would recompute K and V for every previous token at every
step — O(seq_len^2) total work for generating seq_len tokens.

### The solution

Cache the K and V tensors from all previous tokens. At each new step, only
compute K and V for the new token, and concatenate with the cache.

```
Step 1 (prefill): Compute K0, K1, K2, K3, K4 for the prompt
                  Store in KV-cache

Step 2 (decode):  Compute K5 for new token
                  KV-cache = [K0, K1, K2, K3, K4, K5]
                  Attention: Q5 @ [K0...K5]^T → weights → @ [V0...V5]

Step 3 (decode):  Compute K6 for new token
                  KV-cache = [K0, K1, K2, K3, K4, K5, K6]
                  Attention: Q6 @ [K0...K6]^T → weights → @ [V0...V6]

Without cache: recompute all K,V at each step → O(n^2) total
With cache:    compute only new K,V → O(n) total
```

### Savings

| Metric | Without KV-Cache | With KV-Cache |
|--------|-------------------|---------------|
| FLOPs per decode step | O(seq_len * d_model) for K,V projection | O(d_model) for new token only |
| Total FLOPs for n tokens | O(n^2 * d_model) | O(n * d_model) |
| Memory cost | None | O(n * d_head * n_kv_heads * n_layers * 2) |

The trade-off: we save compute but use more GPU memory.

---

## Memory Cost of KV-Cache

### Formula

```
KV-cache memory per token per layer:
  = 2 * n_kv_heads * d_head * bytes_per_element
  (2 for K and V)

KV-cache memory per token (all layers):
  = 2 * n_layers * n_kv_heads * d_head * bytes_per_element

KV-cache memory for a sequence:
  = 2 * n_layers * n_kv_heads * d_head * seq_len * bytes_per_element
```

### Example: Llama 2 70B (with GQA)

```
n_layers = 80
n_kv_heads = 8  (GQA: 8 KV heads shared across 64 query heads)
d_head = 128
bytes_per_element = 2 (FP16)

Per token:
  = 2 * 80 * 8 * 128 * 2 = 327,680 bytes = 320 KB per token

For 4096-token sequence:
  = 320 KB * 4096 = 1.28 GB per request

For 100 concurrent requests at 4096 tokens:
  = 128 GB of KV-cache memory!  (more than A100's 80 GB HBM)
```

This is why KV-cache management is critical and why Q17 exists.

---

## Paged Attention (vLLM)

### The problem with contiguous KV-cache

Traditional KV-cache allocates a contiguous block of memory for each request,
sized for the maximum possible sequence length. This leads to:

- **Internal fragmentation:** Request uses only 200 tokens but allocated for
  4096.
- **External fragmentation:** Free memory is scattered in small chunks.
- **Cannot share prefixes:** Two requests with the same system prompt each store
  their own copy of the KV-cache for the prompt.

### The virtual memory analogy

Paged Attention borrows the idea of **virtual memory** from operating systems:

```
Traditional (contiguous):
  Request A: [████████████████████░░░░░░░░░░░░░]  ← wasted space
  Request B: [████████░░░░░░░░░░░░░░░░░░░░░░░░░]  ← wasted space
  Free:      [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]

Paged Attention:
  Physical blocks:  [B0][B1][B2][B3][B4][B5][B6][B7]...

  Request A page table:          Request B page table:
    slot 0 → B0                    slot 0 → B3
    slot 1 → B1                    slot 1 → B5
    slot 2 → B4                    (only 2 blocks allocated!)
    (3 blocks allocated)

  Block B2, B6, B7: FREE (available for new requests)
```

### How it works

1. KV-cache is divided into fixed-size **blocks** (e.g., 16 tokens per block).
2. Each request has a **page table** mapping logical block indices to physical
   blocks.
3. Blocks are allocated on demand as the sequence grows.
4. When a request finishes, its blocks are freed immediately.
5. Blocks can be **shared** between requests (e.g., for shared system prompts).

### Benefits

- Near-zero memory waste (only last block can be partially empty)
- Efficient prefix sharing
- Dynamic memory allocation — no need to pre-allocate for max length
- Better batching — can fit more concurrent requests

---

## Grouped Query Attention (GQA) and Multi-Query Attention (MQA)

These are architectural changes to reduce KV-cache size.

### Standard Multi-Head Attention (MHA)

```
n_heads = 64 query heads, 64 KV heads
Each head has its own Q, K, V projections
KV-cache per token = 2 * 64 * 128 * 2 bytes = 32 KB
```

### Multi-Query Attention (MQA)

```
n_heads = 64 query heads, 1 KV head (shared!)
All query heads attend to the same K, V
KV-cache per token = 2 * 1 * 128 * 2 bytes = 512 bytes  ← 64x smaller!
Downside: some quality loss
```

### Grouped Query Attention (GQA)

```
n_heads = 64 query heads, 8 KV heads (groups of 8 share K,V)
Each group of 8 query heads shares one K,V pair
KV-cache per token = 2 * 8 * 128 * 2 bytes = 4 KB  ← 8x smaller than MHA

Trade-off: minimal quality loss, significant memory savings
```

```
MHA:  Q0→KV0  Q1→KV1  Q2→KV2  ... Q63→KV63   (64 KV pairs)
GQA:  Q0→KV0  Q1→KV0  ... Q7→KV0              (8 Q share 1 KV)
      Q8→KV1  Q9→KV1  ... Q15→KV1             (8 groups total)
MQA:  Q0→KV0  Q1→KV0  Q2→KV0  ... Q63→KV0     (1 KV pair for all)
```

Llama 2 70B uses GQA with 8 KV heads.
Llama 3 models also use GQA.

---

## Flash Attention

### The problem

Standard attention computes the full seq_len x seq_len attention matrix,
which requires O(seq_len^2) memory and O(seq_len^2) HBM reads/writes.

For seq_len = 32,768 (a common context length), the attention matrix is
32K x 32K x 2 bytes = 2 GB per head per layer. This is too large for on-chip
SRAM.

### The idea

Flash Attention **tiles** the attention computation so that it never
materializes the full attention matrix. Instead, it computes attention in blocks
that fit in SRAM (shared memory on GPU).

```
Standard Attention:
  1. Compute S = Q @ K^T        → write (seq x seq) to HBM
  2. Compute P = softmax(S)     → read/write (seq x seq) from/to HBM
  3. Compute O = P @ V          → read (seq x seq) from HBM
  Total HBM access: O(seq^2) reads + O(seq^2) writes

Flash Attention:
  For each tile of Q (block of rows):
    For each tile of K, V (block of columns):
      1. Load Q_tile, K_tile, V_tile into SRAM
      2. Compute S_tile = Q_tile @ K_tile^T    (in SRAM)
      3. Compute local softmax and update running statistics
      4. Accumulate O_tile = local_weights @ V_tile  (in SRAM)
    Write final O_tile to HBM (once per Q_tile)

  Total HBM access: O(seq * d_head)  — much less!
```

### The softmax trick

The challenge is that softmax requires the full row to normalize. Flash
Attention uses the **online softmax** trick:

```
For standard softmax over [x1, x2, x3]:
  m = max(x1, x2, x3)
  sum = exp(x1-m) + exp(x2-m) + exp(x3-m)
  softmax(xi) = exp(xi-m) / sum

For online softmax (process in chunks):
  Process [x1, x2]:
    m_old = max(x1, x2)
    sum_old = exp(x1-m_old) + exp(x2-m_old)

  Process [x3] and update:
    m_new = max(m_old, x3)
    sum_new = sum_old * exp(m_old - m_new) + exp(x3 - m_new)
              ^                ^
              rescale old sum    add new term

  This gives the same result as computing softmax over [x1, x2, x3] at once.
```

### Impact

- **2-4x speedup** over standard attention (less HBM traffic)
- **O(seq_len) memory** instead of O(seq_len^2)
- Enables much longer context lengths (32K, 128K, 1M tokens)
- Used by virtually every modern LLM inference system

---

## Quantization for Inference

Quantization reduces the precision of model weights (and sometimes activations)
to use less memory and compute faster. See `study_materials/06_quantization.md`
for a deep dive.

### Quick summary

```
FP32 (32-bit float):  4 bytes per parameter
FP16 (16-bit float):  2 bytes per parameter  — standard training/inference
INT8 (8-bit integer): 1 byte per parameter   — 2x memory reduction from FP16
FP8 (8-bit float):    1 byte per parameter   — similar reduction, better for matmul
INT4 (4-bit integer): 0.5 bytes per parameter — 4x reduction from FP16
```

### Why quantization helps

1. **Less memory:** Smaller model fits on fewer GPUs. Smaller KV-cache.
2. **Faster matmul:** INT8/FP8 Tensor Cores are 2x faster than FP16.
3. **Less memory traffic:** Loading 1-byte weights instead of 2-byte weights
   doubles effective bandwidth.

### Quality impact

- FP16 → INT8: Minimal quality loss with proper calibration (< 1% accuracy drop)
- FP16 → INT4: Noticeable quality loss, acceptable for many applications with
  careful quantization (GPTQ, AWQ)

---

## Token Sampling Strategies

After the model produces logits (one score per vocabulary token), we need to
select the next token. This is tested directly in Q12.

### Greedy Decoding

Pick the highest-probability token every time.

```python
next_token = argmax(logits)
```

- Deterministic
- Often repetitive and boring
- Used when you want the most likely completion

### Temperature

Scale logits before softmax to control randomness:

```python
scaled_logits = logits / temperature
probs = softmax(scaled_logits)
next_token = sample(probs)
```

- **temperature = 1.0:** Standard softmax, normal randomness
- **temperature < 1.0:** Sharper distribution, more deterministic (peakier)
- **temperature > 1.0:** Flatter distribution, more random
- **temperature → 0:** Approaches greedy decoding
- **temperature → inf:** Approaches uniform random

```
Example logits: [2.0, 1.0, 0.5, 0.1]

temp=1.0: probs = [0.45, 0.17, 0.10, 0.07, ...]   (normal)
temp=0.5: probs = [0.72, 0.10, 0.04, 0.01, ...]   (concentrated)
temp=2.0: probs = [0.30, 0.22, 0.18, 0.15, ...]   (spread out)
```

### Top-k Sampling

Only consider the k most likely tokens:

```python
top_k_logits = top_k(logits, k=50)  # Zero out all but top 50
probs = softmax(top_k_logits)
next_token = sample(probs)
```

- Prevents sampling very unlikely tokens
- Fixed k regardless of distribution shape
- k=1 is greedy decoding

### Top-p (Nucleus) Sampling

Keep the smallest set of tokens whose cumulative probability exceeds p:

```python
sorted_probs = sort(softmax(logits), descending=True)
cumsum = cumulative_sum(sorted_probs)
# Keep tokens until cumsum > p
cutoff_idx = first_index_where(cumsum > p)
# Zero out everything after cutoff
probs[cutoff_idx:] = 0
next_token = sample(normalize(probs))
```

- Adapts to the distribution: keeps few tokens when model is confident, many
  when uncertain
- p=0.9 (nucleus sampling) is very common
- More principled than top-k

```
Confident prediction (logits: [10.0, 2.0, 1.0, 0.5]):
  probs = [0.95, 0.03, 0.01, 0.005]
  top_p=0.9 keeps only 1 token   (adaptive — model is sure)

Uncertain prediction (logits: [1.0, 0.9, 0.8, 0.7]):
  probs = [0.28, 0.26, 0.24, 0.22]
  top_p=0.9 keeps 4 tokens       (adaptive — model is unsure)
```

### Min-p Sampling

Keep tokens whose probability is at least min_p times the maximum probability:

```python
max_prob = max(softmax(logits))
threshold = min_p * max_prob
# Keep tokens with probability >= threshold
mask = probs >= threshold
next_token = sample(normalize(probs * mask))
```

- Simpler than top-p, similar adaptive behavior
- min_p = 0.1 means keep tokens at least 10% as likely as the best token

### Combined strategy (common in practice)

```python
# 1. Apply temperature
logits = logits / temperature

# 2. Apply top-k (optional)
logits = top_k_filter(logits, k=50)

# 3. Apply top-p
logits = top_p_filter(logits, p=0.9)

# 4. Sample from remaining distribution
probs = softmax(logits)
next_token = multinomial_sample(probs)
```

---

## Speculative Decoding

### The problem

Decode is memory-bound: loading model weights to generate one token at a time
wastes GPU compute. The GPU spends most of its time loading weights from HBM,
not computing.

### The idea

Use a **small, fast draft model** to generate several candidate tokens, then
**verify** them all at once with the large model.

```
Draft model (7B):  Generate 5 tokens quickly
  "The" → "cat" → "sat" → "on" → "the" → "mat"

Large model (70B): Verify all 5 in ONE forward pass
  Input: ["The", "cat", "sat", "on", "the"]
  Check: Does the large model agree with each token?

  "cat" ✓  (accepted)
  "sat" ✓  (accepted)
  "on"  ✓  (accepted)
  "the" ✓  (accepted)
  "mat" ✗  (rejected — large model prefers "rug")

Result: Accepted 4 tokens + sample 1 new token = 5 tokens from one big-model pass
  (instead of 5 separate big-model passes)
```

### Why this works

- The draft model is fast (small, few layers)
- The large model can verify multiple tokens in parallel (just one forward pass
  with the full sequence — the prefill phase naturally scores all positions)
- For "easy" tokens (common words, predictable text), the draft model agrees
  with the large model most of the time
- Mathematically, speculative decoding produces the **exact same distribution**
  as the large model alone (using a clever rejection sampling scheme)

### Speedup

- Typical acceptance rate: 70-90% (depends on model pair and text difficulty)
- Typical speedup: 2-3x for token generation latency
- No quality loss (mathematically equivalent output distribution)

---

## Continuous Batching

### Static batching (traditional)

Wait for a batch of requests, process them all together, return results when
ALL requests finish.

```
Static batch of 4 requests:
  Req A: [======] done at step 10
  Req B: [===============] done at step 20
  Req C: [===] done at step 5
  Req D: [========] done at step 12

  All wait until step 20 (longest request).
  GPU utilization drops as requests finish.

  Timeline:
  Step:  1  2  3  4  5  6  7  8  9  10 11 12 ... 20
  Req A: ■  ■  ■  ■  ■  ■  ■  ■  ■  ■
  Req B: ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ... ■
  Req C: ■  ■  ■  ■  ■
  Req D: ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■
  Active: 4  4  4  4  4  3  3  3  3  3  2  2  ... 1  ← underutilized!
```

### Continuous batching

New requests join the batch as soon as old ones finish.

```
  Timeline:
  Step:  1  2  3  4  5  6  7  8  9  10 11 12 ... 20
  Req A: ■  ■  ■  ■  ■  ■  ■  ■  ■  ■
  Req B: ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ... ■
  Req C: ■  ■  ■  ■  ■
  Req D: ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■
  Req E:                ■  ■  ■  ■  ■  ■  ■       ← joins at step 5
  Req F:                            ■  ■  ■  ... ■ ← joins at step 10
  Active: 4  4  4  4  4  4  4  4  4  4  4  4  ... 4  ← full utilization!
```

### Benefits

- Much higher GPU utilization
- Lower average latency (new requests do not wait for current batch to finish)
- Better throughput
- Used by vLLM, TensorRT-LLM, and all modern serving systems

---

## Concrete Numbers: Llama 70B Example

Let us work through the concrete numbers for Llama 2 70B on A100-80GB GPUs.

### Model size

```
Parameters: 70 billion
Bytes per param (FP16): 2
Model size: 70B * 2 = 140 GB  → needs at least 2 A100-80GB GPUs

With INT8 quantization: 70B * 1 = 70 GB → fits on 1 A100 (barely)
With INT4 quantization: 70B * 0.5 = 35 GB → fits on 1 A100 easily
```

### KV-cache (GQA with 8 KV heads)

```
n_layers = 80
n_kv_heads = 8
d_head = 128
bytes_per_element = 2 (FP16)

Per token (all layers): 2 * 80 * 8 * 128 * 2 = 327,680 bytes ≈ 320 KB

Sequence length 4096: 320 KB * 4096 = 1.28 GB per request
Sequence length 32K:  320 KB * 32768 = 10.24 GB per request

On 80 GB GPU with 35 GB model (INT4):
  Free memory: 80 - 35 = 45 GB for KV-cache
  Max concurrent 4K requests: 45 / 1.28 ≈ 35 requests
  Max concurrent 32K requests: 45 / 10.24 ≈ 4 requests
```

### Decode throughput (memory-bound)

```
Per decode step, load ALL model weights once:
  70 GB (INT8) / 2 TB/s (A100 bandwidth) = 35 ms per token

With 2 GPUs (tensor parallel, combined 4 TB/s):
  70 GB / 4 TB/s = 17.5 ms per token ≈ 57 tokens/sec

Batched: since we load weights once for the whole batch:
  With batch size 32: 32 tokens per 35 ms = 914 tokens/sec (on 1 GPU)
  (Assuming KV-cache fits in memory and is not the bottleneck)
```

### Prefill throughput (compute-bound)

```
FLOPs per token (forward pass, approximate): 2 * params = 140 GFLOPS
Prompt of 1024 tokens: 140 TFLOPS total

A100 FP16 Tensor: 312 TFLOPS
Time: 140 / 312 = 0.45 seconds for 1024 tokens
  ≈ 2300 prompt tokens/sec per GPU (at full utilization)
```

---

## Key Takeaways

1. **Two phases:** Prefill (compute-bound, parallel) and Decode (memory-bound,
   sequential). Optimizing decode latency is the main challenge.

2. **KV-cache is essential** to avoid recomputing Q,K,V for all previous tokens.
   It trades memory for compute.

3. **KV-cache memory is the scaling bottleneck.** At long context lengths, it
   dominates GPU memory and limits concurrent requests.

4. **Paged Attention** eliminates memory fragmentation in KV-cache by using
   virtual memory concepts (block allocation, page tables).

5. **GQA reduces KV-cache size** by sharing KV heads across groups of query
   heads (e.g., 8 KV heads for 64 query heads = 8x reduction).

6. **Flash Attention** reduces HBM traffic by tiling the attention computation
   in on-chip SRAM. It reduces memory from O(n^2) to O(n) and provides 2-4x
   speedup.

7. **Quantization** (INT8/FP8/INT4) reduces model size and speeds up matmul.
   The decode phase benefits most because it is memory-bandwidth-bound.

8. **Speculative decoding** uses a small draft model to amortize the cost of
   the large model. 2-3x speedup with no quality loss.

9. **Continuous batching** keeps GPU utilization high by dynamically adding new
   requests as old ones finish.

10. **Know the numbers:** Model size, KV-cache per token, GPU memory, bandwidth.
    These let you reason about feasibility and bottlenecks during interviews.
