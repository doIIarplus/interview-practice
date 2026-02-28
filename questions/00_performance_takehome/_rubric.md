# Rubric: Performance Engineering Take-Home

## Scoring (by cycle count)

### Tier 1: Basic Understanding (> 18,532 cycles)
- Candidate read the code but didn't make significant optimizations
- May have attempted VLIW packing but with errors
- **Assessment: No Hire** — insufficient optimization skill for the role

### Tier 2: Good Fundamentals (10,000 - 18,532 cycles)
- Successfully packed multiple instructions per cycle (VLIW)
- May have started vectorization but incomplete
- Shows understanding of instruction-level parallelism
- **Assessment: Lean No Hire** — on the right track but needs more depth

### Tier 3: Strong Performance (3,000 - 10,000 cycles)
- Successful SIMD vectorization processing VLEN=8 elements per cycle
- Good instruction scheduling across engines
- Proper use of valu, vload, vstore instructions
- **Assessment: Lean Hire** — demonstrates solid systems optimization skills

### Tier 4: Excellent (1,800 - 3,000 cycles)
- Full vectorization with optimized instruction scheduling
- Efficient loop structure (not fully unrolled)
- Minimized pipeline bubbles
- Good use of scratch space as cache
- **Assessment: Hire** — strong performance engineering candidate

### Tier 5: Exceptional (< 1,800 cycles)
- All of the above plus:
  - Advanced scheduling optimizations (software pipelining, loop body interleaving)
  - Optimal VLIW packing (maximize filled slots per cycle)
  - Creative use of the ISA (e.g., multiply_add fused ops)
  - Memory access pattern optimization
- **Assessment: Strong Hire** — exceptional candidate

### Tier 6: Outstanding (< 1,487 cycles)
- Beating this threshold earns recruiting consideration
- Likely involves novel optimization insights beyond standard techniques
- **Assessment: Strong Hire** — top-tier candidate

## Key Optimization Techniques to Look For

### 1. VLIW Instruction Packing
The naive implementation puts one operation per instruction bundle. The machine can execute:
- 12 ALU ops + 6 VALU ops + 2 loads + 2 stores + 1 flow op **per cycle**
- Packing independent operations into the same cycle is the first major optimization

### 2. SIMD Vectorization
- Batch size is 256, VLEN is 8 → process 8 batch elements simultaneously
- Convert scalar alu/load/store to valu/vload/vstore
- Use vbroadcast for scalar-to-vector promotion
- The hash function operates independently per batch element — perfect for SIMD

### 3. Loop Structure
- Naive version fully unrolls rounds × batch_size
- Can use `cond_jump` / `jump` for looping
- Reduces instruction count dramatically
- Trade-off: loop overhead vs code size

### 4. Software Pipelining
- Overlap computation of one iteration with loads for the next
- The load-use latency means starting loads early is critical

### 5. Memory Access Optimization
- Preload tree values into scratch space if reused
- Minimize redundant loads of the same addresses
- Use vector loads for contiguous batch elements

## Red Flags
- Modifying tests/ folder (automatic disqualification)
- Modifying problem.py (won't work with frozen copy in submission)
- Not understanding the VLIW execution model (putting dependent ops in same cycle thinking they execute sequentially)
- Ignoring correctness — fast but wrong answers are worthless
- Not using the trace visualization for debugging

## Green Flags
- Methodical approach: profile first, then optimize hotspots
- Good use of the trace/debug tools
- Incremental optimization with correctness checks between steps
- Clear understanding of why each optimization helps
- Ability to reason about instruction-level parallelism
- Creative approaches (e.g., precomputing hash stages, exploiting algebraic properties)
