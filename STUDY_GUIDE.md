# Anthropic Performance Engineer — Study Guide & Roadmap

This guide maps every topic you need to the interview questions that test it,
suggests a study order, and links to external resources. If you "don't even know
where to begin," start at Tier 1 and work forward.

---

## How to Use This Guide

1. **Read the tier descriptions** below — they are ordered from foundational to
   advanced. Each tier builds on the ones before it.
2. **Check the Question Difficulty Map** at the bottom to pick which questions to
   tackle first.
3. **Use the study materials** in `study_materials/` for deep dives on specific
   topics. Each file is self-contained with explanations, diagrams, and runnable
   code examples.
4. **Practice the questions** in `questions/` after you feel comfortable with the
   relevant topic area.

---

## Tier 1: Foundations (Study First)

These topics underpin almost every question. Get comfortable here before moving
on.

### Python Data Structures & Algorithms

**Why:** Every question requires you to choose the right data structure and
reason about time/space complexity. Anthropic interviews are in Python.

**What to study:**

- `collections` module: `defaultdict`, `OrderedDict`, `deque`, `Counter`
- `heapq` for priority queues (min-heap by default; negate values for max-heap)
- Hash maps (`dict`) and hash sets (`set`) for O(1) lookup
- Sorting: `sorted()` and `list.sort()`, custom key functions, stability
- Binary search: `bisect` module, manual implementation
- BFS / DFS graph traversal (iterative and recursive)
- Stack-based algorithms (matching parentheses, expression evaluation,
  call-stack simulation)
- Two-pointer and sliding-window patterns
- Trie (prefix tree) for string matching

**External resources:**

- [Python `collections` docs](https://docs.python.org/3/library/collections.html)
- [Python `heapq` docs](https://docs.python.org/3/library/heapq.html)
- "Problem Solving with Algorithms and Data Structures using Python" (Miller &
  Ranum) — free online
- LeetCode "Top Interview 150" problem set for drill

**Relevant to:** Q01 (In-Memory Database), Q02 (Web Crawler), Q03 (Profiler
Trace), Q04 (Distributed Mode), Q05 (LRU Cache), Q06 (Tokenizer), Q19 (Web
Crawler Multithreaded), Q20 (Duplicate Files), Q21 (Exclusive Time)

---

### Python Concurrency

**Why:** Several questions require concurrent or parallel implementations.
Understanding the GIL and when threading helps is essential.

**What to study:**

- Threading vs multiprocessing vs asyncio — when to use each
- The GIL (Global Interpreter Lock): what it is, why it exists, when threading
  still helps (I/O-bound tasks)
- `threading.Lock`, `threading.RLock`, `threading.Event`, `threading.Condition`
- `concurrent.futures.ThreadPoolExecutor` and `ProcessPoolExecutor`
- `asyncio` basics: event loop, coroutines, `await`, `asyncio.gather`
- Thread-safe data structures: `queue.Queue`, `collections.deque` (with locks)
- Common pitfalls: deadlocks (lock ordering), race conditions, starvation

**Study material:** `study_materials/01_python_concurrency.md`

**External resources:**

- [Python `threading` docs](https://docs.python.org/3/library/threading.html)
- [Python `concurrent.futures` docs](https://docs.python.org/3/library/concurrent.futures.html)
- "Python Concurrency with asyncio" by Matthew Fowler (Manning)
- Real Python: "An Intro to Threading in Python"

**Relevant to:** Q02 (Web Crawler), Q09 (Concurrent Task Scheduler), Q15
(Collective Communication), Q19 (Web Crawler Multithreaded)

---

## Tier 2: Systems Fundamentals (Study Second)

These topics appear in the performance-focused questions and are key to the
Performance Engineer role.

### Memory Hierarchy & Cache Optimization

**Why:** Multiple questions test whether you understand *why* certain access
patterns are fast and others are slow. This is at the heart of performance
engineering.

**What to study:**

- CPU caches: L1 (32-64 KB, ~4 cycles), L2 (256 KB-1 MB, ~12 cycles), L3
  (several MB, ~40 cycles), DRAM (~200 cycles)
- Cache lines: typically 64 bytes — the unit of data transfer between cache
  levels
- Spatial locality: accessing nearby memory addresses is fast because the whole
  cache line is loaded
- Temporal locality: accessing recently-used data is fast because it is still in
  cache
- Why naive matrix multiplication is slow: column access pattern causes cache
  misses
- Tiled/blocked algorithms: how they keep data in cache
- Memory alignment and padding
- False sharing in multi-threaded code

**Study material:** `study_materials/02_memory_and_caching.md`

**External resources:**

- "What Every Programmer Should Know About Memory" by Ulrich Drepper (free PDF)
- "Computer Architecture: A Quantitative Approach" by Hennessy & Patterson
  (chapters on memory hierarchy)
- Brendan Gregg's [performance page](https://www.brendangregg.com)

**Relevant to:** Q07 (Matrix Tiling Optimizer), Q11 (Memory Pool Allocator),
Q13 (GPU Kernel Sim)

---

### Distributed Systems Basics

**Why:** Several questions involve distributing work across nodes, handling
failures, and understanding network costs.

**What to study:**

- Network communication costs: latency (round-trip time) vs bandwidth
  (throughput)
- Why network is ~1000x slower than local memory access
- Consistency models: strong consistency, eventual consistency
- Fault tolerance: replication, heartbeats, retry strategies
- MapReduce pattern: map phase (parallel), shuffle, reduce phase
- Consistent hashing for load balancing
- Leader election basics

**External resources:**

- "Designing Data-Intensive Applications" by Martin Kleppmann (the bible of
  distributed systems)
- MIT 6.824 Distributed Systems lecture notes (free online)

**Relevant to:** Q04 (Distributed Mode), Q08 (Load Balancer), Q09 (Concurrent
Task Scheduler), Q10 (Network Latency Analyzer), Q15 (Collective
Communication), Q18 (GPU Cluster Scheduler)

---

### Performance Analysis & Profiling

**Why:** You need to find bottlenecks before you can fix them. Several questions
ask you to analyze performance data.

**What to study:**

- Big-O analysis: time and space complexity for all common operations
- Profiling tools: `cProfile`, `line_profiler`, Py-Spy (sampling profiler),
  `perf` (Linux)
- Flame graphs: how to read them (x-axis = proportion of time, y-axis = call
  stack depth)
- Trace visualization: Chrome Trace Format, Perfetto
- Bottleneck identification: is it CPU-bound, memory-bound, or I/O-bound?
- Amdahl's law: speedup = 1 / (s + p/N) where s = serial fraction, p =
  parallel fraction, N = processors
- Roofline model: peak performance limited by compute or memory bandwidth

**External resources:**

- Brendan Gregg's [Systems Performance](https://www.brendangregg.com/systems-performance-2nd-edition-book.html)
- [Py-Spy documentation](https://github.com/benfred/py-spy)
- "The Art of Performance Analysis" — various blog posts and resources

**Relevant to:** Q03 (Profiler Trace), Q10 (Network Latency Analyzer), Q21
(Exclusive Time)

---

## Tier 3: GPU & ML Specifics (Study Third)

These topics are critical for the GPU Performance Engineer variant of the role
and appear in questions 12-18.

### GPU Architecture

**Why:** Questions 13, 16, and 18 directly test GPU knowledge. Even non-GPU
questions benefit from understanding parallel hardware.

**What to study:**

- CPU vs GPU: few powerful cores vs thousands of simple cores
- CUDA programming model: grid -> blocks -> threads
- Warps: 32 threads executing in lockstep (SIMT — Single Instruction, Multiple
  Threads)
- GPU memory hierarchy: registers (fastest, per-thread), shared memory
  (per-block, programmable L1), L1/L2 cache, global memory (slowest, on HBM)
- Key performance concepts:
  - Memory coalescing: adjacent threads access adjacent memory for full
    bandwidth
  - Bank conflicts: shared memory has 32 banks; conflicts serialize access
  - Occupancy: ratio of active warps to max warps per SM; higher is generally
    better but not always
  - Warp divergence: if/else causes threads in a warp to serialize

**Study material:** `study_materials/03_gpu_architecture.md`

**External resources:**

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- "Programming Massively Parallel Processors" by Hwu, Kirk, and El Hajj
- [NVIDIA Nsight Compute docs](https://docs.nvidia.com/nsight-compute/)
- YouTube: "CUDA Crash Course" by CoffeeBeforeArch

**Relevant to:** Q00 (Take-Home), Q07 (Matrix Tiling), Q13 (GPU Kernel Sim),
Q16 (Kernel Fusion)

---

### ML Inference Optimization

**Why:** Anthropic runs large language models in production. Understanding how
inference works and how to make it fast is directly relevant.

**What to study:**

- Transformer architecture: self-attention, FFN, layer norm, residual
  connections
- Autoregressive generation: producing one token at a time
- KV-cache: storing past Key and Value tensors to avoid recomputation
- Memory cost of KV-cache and why it dominates at long sequence lengths
- Paged Attention (vLLM): virtual-memory-style management of KV blocks
- Grouped Query Attention (GQA) and Multi-Query Attention (MQA)
- Quantization for inference: reducing precision to save memory and compute
- Token sampling: greedy, temperature, top-k, top-p (nucleus), min-p
- Flash Attention: tiling attention computation to reduce HBM accesses
- Speculative decoding: draft with a small model, verify with the large model
- Continuous batching: dynamically adding/removing requests to maximize
  throughput

**Study material:** `study_materials/04_transformer_inference.md`

**External resources:**

- ["The Illustrated Transformer"](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- [vLLM paper](https://arxiv.org/abs/2309.06180): "Efficient Memory Management
  for Large Language Model Serving with PagedAttention"
- [Flash Attention paper](https://arxiv.org/abs/2205.14135)
- [Speculative Decoding paper](https://arxiv.org/abs/2211.17192)

**Relevant to:** Q12 (Low-Latency Sampler), Q14 (Quantization Engine), Q16
(Kernel Fusion), Q17 (KV-Cache Manager)

---

### Distributed Training & Inference

**Why:** Large models require multiple GPUs. Understanding how to split work
across GPUs and the communication costs is key.

**What to study:**

- Data parallelism: replicate the model, split the data, AllReduce gradients
- Tensor parallelism: split individual layers across GPUs (Megatron-style column
  and row parallel linear layers)
- Pipeline parallelism: split model into stages, pipeline micro-batches
- 3D parallelism: combining data, tensor, and pipeline parallelism
- Communication primitives: AllReduce, AllGather, ReduceScatter, Broadcast
- Ring AllReduce: how it works, why it is bandwidth-optimal
- NVLink (~600 GB/s bidirectional) vs InfiniBand (~50 GB/s) vs PCIe (~64 GB/s)
- NCCL: NVIDIA's communication library
- ZeRO optimizer stages: partitioning optimizer states (Stage 1), gradients
  (Stage 2), and parameters (Stage 3)

**Study material:** `study_materials/05_distributed_training.md`

**External resources:**

- [Megatron-LM paper](https://arxiv.org/abs/1909.08053)
- "Efficient Large-Scale Language Model Training on GPU Clusters" (Narayanan et
  al., 2021)
- [DeepSpeed ZeRO paper](https://arxiv.org/abs/1910.02054)
- [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/)

**Relevant to:** Q15 (Collective Communication), Q18 (GPU Cluster Scheduler)

---

## Tier 4: Advanced / The Take-Home (Study Last or As Needed)

### VLIW & Instruction-Level Parallelism

**Why:** Question 00 (the actual Anthropic take-home) is a VLIW SIMD kernel
optimization challenge. This is specialized knowledge.

**What to study:**

- Instruction pipelining: fetch, decode, execute, memory, writeback
- Pipeline hazards: data hazards (RAW, WAR, WAW), control hazards (branches),
  structural hazards
- VLIW (Very Long Instruction Word): compiler schedules multiple operations per
  cycle into "bundles"
- SIMD (Single Instruction, Multiple Data): one instruction operates on a vector
  of values
- Loop optimization: unrolling, software pipelining, strength reduction
- Instruction scheduling: reorder instructions to avoid stalls
- Register pressure: more unrolling requires more registers

**External resources:**

- "Computer Organization and Design" by Patterson & Hennessy
- "Engineering a Compiler" by Cooper & Torczon (chapter on instruction
  scheduling)
- The take-home's own README and documentation (in `questions/00_performance_takehome/`)

**Relevant to:** Q00 (Performance Take-Home)

---

### Quantization Deep Dive

**Why:** Question 14 directly implements a quantization engine. Understanding
the math and formats in detail is necessary.

**What to study:**

- Symmetric vs asymmetric quantization: formulas, trade-offs
- Per-tensor vs per-channel vs per-group quantization
- INT8 quantization: scale factor computation, clipping
- FP8 formats: E4M3 (range [-448, 448]) vs E5M2 (range [-57344, 57344])
- Calibration: min-max, percentile, entropy-based (KL divergence)
- GPTQ, AWQ, SmoothQuant: weight-only and weight-activation quantization
  methods
- How tensor cores accelerate INT8 and FP8 matmul

**Study material:** `study_materials/06_quantization.md`

**Relevant to:** Q14 (Quantization Engine)

---

## Suggested Study Schedule

This schedule assumes ~2-3 hours of study per day. Adjust based on your
available time and existing knowledge.

### Week 1: Python Foundations

| Day | Topic | Activity |
|-----|-------|----------|
| 1-2 | Python data structures | Review `collections`, `heapq`, `bisect`. Solve 5-10 easy LeetCode problems using these. |
| 3-4 | Python concurrency | Read `study_materials/01_python_concurrency.md`. Write a concurrent URL fetcher. |
| 5 | Practice | Attempt Q05 (LRU Cache) and Q06 (Tokenizer) — these are pure data structures. |
| 6-7 | Practice | Attempt Q01 (In-Memory Database) levels 1-3. Attempt Q21 (Exclusive Time). |

### Week 2: Systems & Performance

| Day | Topic | Activity |
|-----|-------|----------|
| 1-2 | Memory hierarchy | Read `study_materials/02_memory_and_caching.md`. Understand cache lines and tiling. |
| 3 | Profiling | Learn `cProfile` and flame graphs. Attempt Q03 (Profiler Trace). |
| 4-5 | Distributed systems | Read Kleppmann Chapter 1-2. Understand network costs. Attempt Q08 (Load Balancer). |
| 6-7 | Practice | Attempt Q02 (Web Crawler), Q09 (Concurrent Task Scheduler), Q10 (Network Latency). |

### Week 3: GPU & ML Inference

| Day | Topic | Activity |
|-----|-------|----------|
| 1-2 | GPU architecture | Read `study_materials/03_gpu_architecture.md`. Understand warps, coalescing, bank conflicts. |
| 3-4 | Transformer inference | Read `study_materials/04_transformer_inference.md`. Understand KV-cache, Flash Attention. |
| 5 | Quantization | Read `study_materials/06_quantization.md`. Attempt Q14 (Quantization Engine). |
| 6-7 | Practice | Attempt Q12 (Low-Latency Sampler), Q13 (GPU Kernel Sim), Q17 (KV-Cache Manager). |

### Week 4: Distributed & Advanced

| Day | Topic | Activity |
|-----|-------|----------|
| 1-2 | Distributed training | Read `study_materials/05_distributed_training.md`. Understand ring allreduce. |
| 3-4 | Practice | Attempt Q15 (Collective Communication), Q16 (Kernel Fusion), Q18 (GPU Cluster Scheduler). |
| 5-7 | Take-home | Work through Q00 (the actual Anthropic take-home). This is the most important single exercise. |

### Final Week: Review & Polish

- Re-attempt any questions you struggled with
- Time yourself on 2-3 questions to simulate interview conditions
- Review your solutions against the follow-up questions
- Practice explaining your approach out loud

---

## Question Difficulty Map

| # | Question | Difficulty | Category | Prerequisites | Est. Time |
|---|----------|-----------|----------|---------------|-----------|
| 00 | Performance Take-Home (VLIW/SIMD) | Very Hard | ILP / Low-Level | VLIW, SIMD, instruction scheduling | 4-8 hrs |
| 01 | In-Memory Database | Medium | Data Structures | dict, sorted containers, indexing | 60-90 min |
| 02 | Web Crawler | Medium | Concurrency + Graphs | BFS, URL parsing, threading | 45-60 min |
| 03 | Profiler Trace | Medium | Performance Analysis | Stack-based parsing, tree construction | 45-60 min |
| 04 | Distributed Mode | Medium | Distributed + Stats | Hash maps, statistical mode, MapReduce | 45-60 min |
| 05 | LRU Cache | Easy-Medium | Data Structures | OrderedDict or dict + doubly-linked list | 30-45 min |
| 06 | Tokenizer | Easy-Medium | String Processing | Trie or regex, BPE understanding | 30-45 min |
| 07 | Matrix Tiling Optimizer | Hard | Memory / Caching | Cache hierarchy, tiling, blocking | 60-90 min |
| 08 | Load Balancer | Medium | Distributed Systems | Consistent hashing, routing strategies | 45-60 min |
| 09 | Concurrent Task Scheduler | Hard | Concurrency | Thread pools, DAGs, fault tolerance | 60-90 min |
| 10 | Network Latency Analyzer | Medium | Networking / Perf | Statistics, percentiles, anomaly detection | 45-60 min |
| 11 | Memory Pool Allocator | Hard | Systems / Memory | Free lists, fragmentation, buddy system | 60-90 min |
| 12 | Low-Latency Sampler | Medium-Hard | ML Inference | Probability, sampling algorithms, numerics | 45-60 min |
| 13 | GPU Kernel Sim | Hard | GPU Architecture | Warps, coalescing, bank conflicts, shared mem | 60-90 min |
| 14 | Quantization Engine | Hard | ML / Numerics | INT8/FP8 formats, scale factors, matmul | 60-90 min |
| 15 | Collective Communication | Hard | Distributed / GPU | Ring allreduce, NCCL patterns, bandwidth | 60-90 min |
| 16 | Kernel Fusion | Very Hard | GPU / ML | Operation graphs, memory traffic, transformers | 90-120 min |
| 17 | KV-Cache Manager | Hard | ML Inference | Paged attention, memory budgeting, eviction | 60-90 min |
| 18 | GPU Cluster Scheduler | Very Hard | Distributed / GPU | Topology, 3D parallelism, fault tolerance | 90-120 min |
| 19 | Web Crawler (Multithreaded) | Medium | Concurrency | Threading, queues, URL dedup | 45-60 min |
| 20 | Duplicate Files | Easy-Medium | Data Structures | Hashing, file I/O, grouping | 30-45 min |
| 21 | Exclusive Time | Medium | Stack Algorithms | Stack-based simulation, interval tracking | 45-60 min |

### Difficulty Legend

- **Easy-Medium:** Good warm-up. Should be completable with standard Python
  knowledge and basic data structures.
- **Medium:** Core interview difficulty. Requires solid data structures and some
  domain knowledge.
- **Hard:** Requires domain-specific knowledge (GPU, distributed systems, memory
  hierarchy) plus strong implementation skills.
- **Very Hard:** Requires deep domain expertise and significant implementation
  effort. These are differentiators.

---

## Suggested Question Order (By Increasing Difficulty)

If you are starting from scratch, tackle questions in roughly this order:

1. **Q05** (LRU Cache) — classic data structure question, good warm-up
2. **Q06** (Tokenizer) — string processing, approachable
3. **Q20** (Duplicate Files) — file hashing, straightforward
4. **Q21** (Exclusive Time) — stack-based algorithm, classic pattern
5. **Q01** (In-Memory Database) — more complex data structure design
6. **Q02** (Web Crawler) — introduces concurrency
7. **Q03** (Profiler Trace) — trace parsing and analysis
8. **Q04** (Distributed Mode) — distributed computing concepts
9. **Q08** (Load Balancer) — distributed systems design
10. **Q10** (Network Latency Analyzer) — performance analysis
11. **Q19** (Web Crawler Multithreaded) — concurrency deep dive
12. **Q12** (Low-Latency Sampler) — ML inference, numerics
13. **Q09** (Concurrent Task Scheduler) — advanced concurrency
14. **Q07** (Matrix Tiling Optimizer) — cache optimization
15. **Q11** (Memory Pool Allocator) — systems programming
16. **Q14** (Quantization Engine) — ML numerics
17. **Q13** (GPU Kernel Sim) — GPU architecture
18. **Q17** (KV-Cache Manager) — ML inference systems
19. **Q15** (Collective Communication) — distributed GPU
20. **Q16** (Kernel Fusion) — GPU optimization
21. **Q18** (GPU Cluster Scheduler) — distributed GPU systems
22. **Q00** (Performance Take-Home) — save this for last as a capstone
