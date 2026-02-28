# Follow-Up Questions: Performance Engineering Take-Home

These follow-ups are for discussion after the candidate completes the take-home, typically in the hiring manager call or onsite.

## Understanding Questions

### 1. Walk me through your optimization strategy
**What to look for:**
- Did they profile/analyze before optimizing?
- Can they articulate the bottleneck at each stage?
- Do they understand the machine model well?

### 2. What was the biggest speedup you achieved and why?
**What to look for:**
- SIMD vectorization typically gives the largest single speedup (~8x theoretical)
- VLIW packing is the next biggest win
- Can they quantify the impact of each optimization?

### 3. What optimization did you try that didn't work? Why?
**What to look for:**
- Intellectual honesty about failed attempts
- Understanding of why certain optimizations don't help in this context
- Debugging methodology

## Technical Deep-Dives

### 4. How does this simulated architecture compare to real GPU architectures?
**What to look for:**
- VLIW is similar to older GPU architectures (AMD VLIW4/5)
- Modern GPUs use SIMT (warps/wavefronts) rather than VLIW
- The concept of instruction-level parallelism is universal
- Scratch space is analogous to shared memory / registers

### 5. If this kernel ran on actual hardware, what additional optimizations would be possible?
**What to look for:**
- Hardware prefetching
- Cache hierarchy exploitation
- Out-of-order execution (the sim is in-order)
- Branch prediction
- SIMD wider than 8 (AVX-512 = 16 floats, GPU warps = 32 threads)

### 6. The hash function dominates compute. How would you optimize it on a real GPU?
**What to look for:**
- Fused multiply-add instructions
- Bit manipulation units
- Look-up tables in shared memory
- Pipeline the hash stages across different data elements

### 7. How would you handle the tree traversal if the tree didn't fit in memory?
**What to look for:**
- Tiling / blocking strategies
- Prefetching based on predicted traversal paths
- The tree structure means locality is poor (random access pattern)
- Batch-level sorting to improve coherence

## Scaling Questions

### 8. How would you parallelize this across multiple GPUs?
**What to look for:**
- Data parallelism: split batch across GPUs
- The tree is read-only — can be replicated
- Communication patterns: gather results
- Load balancing considerations

### 9. If the batch size was 1 million, what would change in your approach?
**What to look for:**
- Memory constraints
- Streaming / double-buffering
- The hash computation is the bottleneck regardless
- May need to tile the batch to fit in scratch/shared memory
