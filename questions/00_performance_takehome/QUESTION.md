# Question 00: Anthropic Performance Engineering Take-Home (Official)

**Source:** [github.com/anthropics/original_performance_takehome](https://github.com/anthropics/original_performance_takehome)

**Time limit:** Originally 4 hours (later 2 hours). Now open-ended for practice.

**Difficulty:** Very High — this is the actual take-home used by Anthropic for Performance Engineer candidates.

---

## Problem

You are given a simulator for a custom **VLIW SIMD architecture** (Very Large Instruction Word, Single Instruction Multiple Data). Your task is to **optimize the kernel** in `perf_takehome.py` (`KernelBuilder.build_kernel`) to minimize the number of clock cycles.

### The Architecture

The simulated machine has:
- **Engines**: `alu` (12 slots), `valu` (6 slots), `load` (2 slots), `store` (2 slots), `flow` (1 slot)
- **VLIW execution**: All engine slots execute in parallel within a single cycle. Effects don't take place until end of cycle.
- **SIMD vectors**: `VLEN = 8` — vector instructions operate on 8 elements at once
- **Scratch space**: 1536 words acting as registers/cache
- **Memory**: Flat address space with load/store access

### The Kernel

The kernel performs a **parallel tree traversal** on a perfect balanced binary tree:
- For each item in a batch, at each node: `val = myhash(val ^ node_val)`
- Then branch left if `val` is even, right if odd
- If reaching the bottom, wrap to the top
- This repeats for multiple rounds

The reference implementation is a naive scalar version that processes one batch element at a time with one instruction per cycle.

### Your Goal

Optimize `build_kernel` to reduce clock cycles. The baseline naive implementation runs at **147,734 cycles**.

### Benchmarks

| Cycles | Who |
|--------|-----|
| 147,734 | Naive baseline |
| 18,532 | Updated starting point (given to candidates) |
| 2,164 | Claude Opus 4 (many hours) |
| 1,790 | Claude Opus 4.5 (casual, ~best human 2hr) |
| 1,579 | Claude Opus 4.5 (2hr harness) |
| 1,487 | Target to impress |
| 1,363 | Claude Opus 4.5 (improved harness) |

### Files

- `problem.py` — The machine simulator and reference kernel. **Read this carefully.**
- `perf_takehome.py` — The `KernelBuilder` class. **This is what you modify.**
- `watch_trace.py` / `watch_trace.html` — Debugging/visualization tools
- `tests/submission_tests.py` — Validation harness. **Do NOT modify.**

### Validation

```bash
# Ensure tests/ is unmodified
git diff origin/main tests/

# Run your solution
python tests/submission_tests.py
```

### Getting Started

1. Read `problem.py` to understand the machine ISA (instruction set architecture)
2. Read the reference `build_kernel` to understand the naive implementation
3. Identify optimization opportunities:
   - **VLIW packing**: Multiple operations per cycle across engines
   - **SIMD vectorization**: Process VLEN=8 batch elements at once with `valu` instructions
   - **Loop structure**: The naive version fully unrolls — can you use loops?
   - **Instruction scheduling**: Minimize pipeline stalls by reordering operations
   - **Memory access patterns**: Optimize load/store usage
4. Use the trace visualization to understand where cycles are spent
5. Iterate: optimize, measure, repeat

### Rules

- Only modify `perf_takehome.py` (specifically `KernelBuilder.build_kernel` and related methods)
- Do **NOT** modify anything in `tests/`
- Do **NOT** modify `problem.py` (the submission tests use a frozen copy)
- The `debug` engine is ignored by submission tests — use it freely for debugging
- `pause` instructions are also ignored by submission tests
