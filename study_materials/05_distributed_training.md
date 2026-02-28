# Distributed Training & Inference — Multi-GPU Systems

This guide covers how large models are split across multiple GPUs for training
and inference. It explains data parallelism, tensor parallelism, pipeline
parallelism, communication primitives, and the hardware interconnects that
make it all work. Relevant to Q15 (Collective Communication) and Q18 (GPU
Cluster Scheduler).

---

## Table of Contents

1. [Why Distributed?](#why-distributed)
2. [Data Parallelism](#data-parallelism)
3. [Tensor Parallelism](#tensor-parallelism)
4. [Pipeline Parallelism](#pipeline-parallelism)
5. [3D Parallelism](#3d-parallelism)
6. [Communication Primitives](#communication-primitives)
7. [Ring AllReduce — Step by Step](#ring-allreduce--step-by-step)
8. [Hardware: NVLink, InfiniBand, PCIe](#hardware-nvlink-infiniband-pcie)
9. [ZeRO Optimizer](#zero-optimizer)
10. [Pipeline Bubbles and Micro-Batching](#pipeline-bubbles-and-micro-batching)
11. [Topology-Aware Placement](#topology-aware-placement)
12. [Practical Numbers](#practical-numbers)
13. [Key Takeaways](#key-takeaways)

---

## Why Distributed?

Large language models do not fit on a single GPU:

```
Model Size vs GPU Memory:

Model        Parameters    FP16 Size    GPUs Needed (80 GB each)
─────────    ──────────    ─────────    ────────────────────────
GPT-2        1.5B          3 GB         1
LLaMA 7B     7B            14 GB        1
LLaMA 70B    70B           140 GB       2
GPT-3        175B          350 GB       5
GPT-4*       ~1.8T?        ~3.6 TB      45+

* Estimated; actual architecture is undisclosed
```

Even for models that fit on one GPU, training requires:
- **Gradients:** Same size as model (another 140 GB for 70B)
- **Optimizer states:** 2-4x model size for Adam (momentum + variance)
- **Activations:** Proportional to batch size and sequence length

Total training memory for 70B in FP16 with Adam:
```
Model:       140 GB
Gradients:   140 GB
Adam states: 560 GB  (FP32 copy + momentum + variance = 4 * 140 GB)
Activations: Variable (depends on batch size, can be 100+ GB)
Total:       ~1 TB → needs 8-16 GPUs minimum
```

---

## Data Parallelism

**The simplest form of distributed training.** Replicate the entire model on
each GPU, split the data.

```
                    ┌─────────────────────────┐
                    │   Data Batch (size B)    │
                    └────────┬────────────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
     ┌─────┴─────┐    ┌─────┴─────┐    ┌─────┴─────┐
     │  GPU 0    │    │  GPU 1    │    │  GPU 2    │
     │  Model    │    │  Model    │    │  Model    │
     │  (copy)   │    │  (copy)   │    │  (copy)   │
     │           │    │           │    │           │
     │ B/3 data  │    │ B/3 data  │    │ B/3 data  │
     │           │    │           │    │           │
     │ Gradients │    │ Gradients │    │ Gradients │
     └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                      ┌──────┴──────┐
                      │  AllReduce  │  Average gradients
                      │  Gradients  │  across all GPUs
                      └──────┬──────┘
                             │
                    All GPUs update weights
                    with the same averaged gradients
                    → models stay in sync
```

### Steps

1. Each GPU holds a complete copy of the model
2. Each GPU processes a different mini-batch of data
3. Each GPU computes gradients independently (forward + backward pass)
4. **AllReduce:** All GPUs exchange and average their gradients
5. Each GPU updates its weights using the averaged gradients
6. Since all GPUs start with the same weights and apply the same update, they
   remain synchronized

### Pros and cons

| Pros | Cons |
|------|------|
| Simple to implement | Model must fit on one GPU |
| Scales well for throughput | Communication cost: AllReduce of full gradient tensor |
| No model changes needed | Memory inefficient (model replicated N times) |

### Communication cost

```
AllReduce cost for data parallelism:
  Data volume: 2 * model_size * (N-1) / N  ← ring allreduce formula
  For 70B FP16 on 8 GPUs: 2 * 140 GB * 7/8 = 245 GB transferred

  At 600 GB/s NVLink: 245 / 600 = 0.41 seconds per step
  At 50 GB/s InfiniBand: 245 / 50 = 4.9 seconds per step

  If forward+backward takes 2 seconds, NVLink adds 20% overhead (ok).
  InfiniBand adds 245% overhead (terrible for data parallelism alone).
```

---

## Tensor Parallelism

**Split individual layers across GPUs.** Each GPU holds a fraction of each
layer's weights.

### Column-parallel linear layer

Split the weight matrix by columns, each GPU computes part of the output:

```
Standard:  Y = X @ W     where W is (d_in, d_out)

Tensor parallel (2 GPUs):
  GPU 0: Y_0 = X @ W_0   where W_0 is (d_in, d_out/2) — left half
  GPU 1: Y_1 = X @ W_1   where W_1 is (d_in, d_out/2) — right half

  Y = [Y_0 | Y_1]  ← concatenate (AllGather)

  ┌────────────┐     ┌─────────┐     ┌────────────┐
  │    X       │     │  W_0    │     │    Y_0     │
  │ (B, d_in)  │  @  │(d_in,  │  =  │ (B, d/2)  │   GPU 0
  │            │     │  d/2)   │     │            │
  └────────────┘     └─────────┘     └────────────┘

  ┌────────────┐     ┌─────────┐     ┌────────────┐
  │    X       │     │  W_1    │     │    Y_1     │
  │ (B, d_in)  │  @  │(d_in,  │  =  │ (B, d/2)  │   GPU 1
  │            │     │  d/2)   │     │            │
  └────────────┘     └─────────┘     └────────────┘
```

### Row-parallel linear layer

Split the weight matrix by rows, each GPU receives part of the input:

```
Standard:  Y = X @ W     where W is (d_in, d_out)

Tensor parallel (2 GPUs):
  GPU 0: Y_0 = X_0 @ W_0   where W_0 is (d_in/2, d_out)
  GPU 1: Y_1 = X_1 @ W_1   where W_1 is (d_in/2, d_out)

  Y = Y_0 + Y_1  ← sum (AllReduce)
```

### In a transformer layer (Megatron-LM style)

```
Attention:
  Q, K, V projections: column-parallel (split heads across GPUs)
  Output projection:   row-parallel (reduces across GPUs)
  → 1 AllReduce per attention sublayer

FFN:
  First linear (W1):  column-parallel
  Second linear (W2): row-parallel
  → 1 AllReduce per FFN sublayer

Total: 2 AllReduce operations per transformer layer
  Each AllReduce transmits ~2 * batch * seq_len * d_model bytes
```

### Pros and cons

| Pros | Cons |
|------|------|
| Model does not need to fit on one GPU | High communication (2 AllReduce per layer) |
| Each GPU uses less memory | Best within a node (needs NVLink bandwidth) |
| Reduces per-GPU compute | Increases GPU count needed for efficiency |

---

## Pipeline Parallelism

**Split the model into stages (groups of layers), assign each stage to a
different GPU.**

```
Model: 80 layers split into 4 stages of 20 layers each

  GPU 0: Layers 0-19   (Stage 0)
  GPU 1: Layers 20-39  (Stage 1)
  GPU 2: Layers 40-59  (Stage 2)
  GPU 3: Layers 60-79  (Stage 3)

Data flows through the pipeline:

  GPU 0 ──activations──▶ GPU 1 ──activations──▶ GPU 2 ──activations──▶ GPU 3
         ◀──gradients──         ◀──gradients──         ◀──gradients──
```

### The bubble problem

With simple pipelining, only one GPU is active at a time:

```
Naive pipeline (1 micro-batch):

Time → →→ → → → → → → → → → → → → → → → →
GPU 0: [Forward 0][                              ][Backward 0]
GPU 1:            [Forward 1][              ][Backward 1]
GPU 2:                      [Forward 2][Backward 2]
GPU 3:                                [Fwd 3][Bwd 3]

  ░░░ = idle (pipeline bubble)
  Utilization: ~25% (only 1 of 4 GPUs active at a time)
```

### Micro-batching reduces bubbles

Split the batch into M micro-batches and pipeline them:

```
Pipeline with 4 micro-batches (m0, m1, m2, m3):

Time → → → → → → → → → → → → → → → → → → → → → → → →
GPU 0: [F_m0][F_m1][F_m2][F_m3]          [B_m3][B_m2][B_m1][B_m0]
GPU 1:       [F_m0][F_m1][F_m2][F_m3][B_m3][B_m2][B_m1][B_m0]
GPU 2:             [F_m0][F_m1][F_m2][F_m3][B_m2][B_m1][B_m0]
GPU 3:                   [F_m0][F_m1][F_m2][B_m1][B_m0]

                         ↑ pipeline fill      pipeline drain ↑

Bubble fraction = (P - 1) / (M + P - 1)
  P = number of pipeline stages
  M = number of micro-batches

For P=4, M=32: bubble = 3/35 ≈ 8.6% (acceptable)
For P=4, M=4:  bubble = 3/7  ≈ 43%  (too high)
```

### Communication cost

Pipeline parallelism only sends **activations** between adjacent stages (not
full model gradients). The volume is:

```
batch_size * seq_len * d_model * bytes_per_element

For batch=32, seq=2048, d_model=8192, FP16:
  32 * 2048 * 8192 * 2 = 1 GB per stage boundary

Much less than data parallelism's full-model AllReduce.
Suitable for inter-node communication (InfiniBand).
```

---

## 3D Parallelism

**Combine all three types** for maximum scalability. This is how systems like
Megatron-LM train 100B+ models.

```
Example: 64 GPUs organized as:
  - 8-way tensor parallelism (within each node of 8 GPUs)
  - 4-way pipeline parallelism (across 4 groups of nodes)
  - 2-way data parallelism (2 replicas of the entire pipeline)

  ┌──────────────── Data Parallel Replica 0 ────────────────┐
  │                                                          │
  │  Node 0 (8 GPUs)     Node 1 (8 GPUs)                    │
  │  TP group 0          TP group 1                          │
  │  Pipeline Stage 0    Pipeline Stage 1                    │
  │  Layers 0-19         Layers 20-39                        │
  │  ┌──┬──┬──┬──┬──┬──┬──┬──┐  ┌──┬──┬──┬──┬──┬──┬──┬──┐  │
  │  │G0│G1│G2│G3│G4│G5│G6│G7│  │G8│G9│..│..│..│..│..│15│  │
  │  └──┴──┴──┴──┴──┴──┴──┴──┘  └──┴──┴──┴──┴──┴──┴──┴──┘  │
  │       ↑ NVLink ↑                  ↑ NVLink ↑             │
  │                ├── InfiniBand ────┘                       │
  │                                                          │
  │  Node 2 (8 GPUs)     Node 3 (8 GPUs)                    │
  │  TP group 2          TP group 3                          │
  │  Pipeline Stage 2    Pipeline Stage 3                    │
  │  Layers 40-59         Layers 60-79                       │
  └──────────────────────────────────────────────────────────┘

  ┌──────────────── Data Parallel Replica 1 ────────────────┐
  │  (Same structure, different data)                        │
  │  Node 4-7                                                │
  └──────────────────────────────────────────────────────────┘
```

### Rules of thumb for placement

1. **Tensor parallelism within a node** (NVLink: 600 GB/s). High communication
   frequency (2x per layer), needs maximum bandwidth.

2. **Pipeline parallelism across nodes** (InfiniBand: 50 GB/s). Low
   communication frequency (1x per micro-batch), point-to-point.

3. **Data parallelism across replicas** (InfiniBand). Communication once per
   training step (AllReduce of gradients), can overlap with backward pass.

---

## Communication Primitives

These are the building blocks of distributed computing. NCCL (NVIDIA Collective
Communications Library) implements them efficiently on GPUs.

### AllReduce

Reduce (sum/avg) data across all GPUs, result on all GPUs.

```
Before:                    After (sum):
GPU 0: [1, 2, 3]          GPU 0: [10, 20, 30]
GPU 1: [4, 5, 6]    →     GPU 1: [10, 20, 30]
GPU 2: [5, 13, 21]        GPU 2: [10, 20, 30]

Used for: gradient averaging in data parallelism, layer outputs in tensor
parallelism
```

### AllGather

Each GPU sends its data to all GPUs. Result: all GPUs have the full
concatenated data.

```
Before:                    After:
GPU 0: [A]                 GPU 0: [A, B, C]
GPU 1: [B]           →    GPU 1: [A, B, C]
GPU 2: [C]                 GPU 2: [A, B, C]

Used for: gathering split activations in tensor parallelism, ZeRO Stage 3
parameter gathering
```

### ReduceScatter

Reduce data across GPUs, then scatter the result (each GPU gets a different
portion).

```
Before:                    After:
GPU 0: [1, 2, 3]          GPU 0: [10]      (sum of position 0)
GPU 1: [4, 5, 6]    →     GPU 1: [20]      (sum of position 1)
GPU 2: [5, 13, 21]        GPU 2: [30]      (sum of position 2)

Used for: gradient reduction in ZeRO, efficient AllReduce decomposition
Note: AllReduce = ReduceScatter + AllGather
```

### Broadcast

One GPU sends its data to all other GPUs.

```
Before:                    After:
GPU 0: [data]              GPU 0: [data]
GPU 1: [  ?  ]       →    GPU 1: [data]
GPU 2: [  ?  ]             GPU 2: [data]

Used for: distributing model weights, broadcasting parameters
```

### Reduce

All GPUs contribute data, result on ONE GPU.

```
Before:                    After (sum on GPU 0):
GPU 0: [1, 2, 3]          GPU 0: [10, 20, 30]
GPU 1: [4, 5, 6]    →     GPU 1: (unchanged)
GPU 2: [5, 13, 21]        GPU 2: (unchanged)
```

### Send / Recv (Point-to-Point)

One GPU sends to one other GPU. Used in pipeline parallelism.

---

## Ring AllReduce — Step by Step

Ring AllReduce is the most important collective algorithm. It is
**bandwidth-optimal**: it makes full use of all links simultaneously.

### Setup

N GPUs arranged in a ring. Each GPU has a vector of size S.

```
Goal: Compute the sum of all N vectors and distribute to all GPUs.

Algorithm has 2 phases:
  Phase 1: ReduceScatter — each GPU ends up with 1/N of the final sum
  Phase 2: AllGather — distribute the partial sums to all GPUs
```

### Example: 4 GPUs, vector of 4 chunks

```
Initial state:
  GPU 0: [A0, A1, A2, A3]
  GPU 1: [B0, B1, B2, B3]
  GPU 2: [C0, C1, C2, C3]
  GPU 3: [D0, D1, D2, D3]

═══════════════════════════════════════════════════
Phase 1: ReduceScatter (3 steps for 4 GPUs)
═══════════════════════════════════════════════════

Each GPU sends one chunk to its right neighbor, and receives one chunk
from its left neighbor. The received chunk is ADDED to the local chunk.

Step 1: GPU i sends chunk i to GPU (i+1)%4
  GPU 0 sends A0 to GPU 1;  receives D3 from GPU 3
  GPU 1 sends B1 to GPU 2;  receives A0 from GPU 0
  GPU 2 sends C2 to GPU 3;  receives B1 from GPU 1
  GPU 3 sends D3 to GPU 0;  receives C2 from GPU 2

  After adding received to local:
  GPU 0: [A0,    A1,    A2,    A3+D3  ]
  GPU 1: [A0+B0, B1,    B2,    B3     ]
  GPU 2: [C0,    B1+C1, C2,    C3     ]
  GPU 3: [D0,    D1,    C2+D2, D3     ]

Step 2: GPU i sends the chunk that was just updated
  GPU 0 sends (A3+D3) to GPU 1;  receives (C2+D2) from GPU 3
  GPU 1 sends (A0+B0) to GPU 2;  receives (A3+D3) from GPU 0
  GPU 2 sends (B1+C1) to GPU 3;  receives (A0+B0) from GPU 1
  GPU 3 sends (C2+D2) to GPU 0;  receives (B1+C1) from GPU 2

  After adding:
  GPU 0: [A0,       A1,       A2+C2+D2, A3+D3     ]
  GPU 1: [A0+B0,    B1+A3+D3, B2,       B3        ]
  GPU 2: [A0+B0+C0, B1+C1,    C2,       C3        ]
  GPU 3: [D0,       B1+C1+D1, C2+D2,    D3        ]

Step 3: One more step...
  After adding:
  GPU 0: [A0,             A1,             A2+B2+C2+D2, A3+D3        ]
  GPU 1: [A0+B0,          A1+B1+C1+D1,   B2,          B3           ]
  GPU 2: [A0+B0+C0+D0,   B1+C1,         C2,          C3           ]
  GPU 3: [D0,             B1+C1+D1,      A2+B2+C2+D2, A3+B3+C3+D3 ]

  Wait — let me redo this correctly:

  After Phase 1 (ReduceScatter), each GPU has ONE chunk of the full sum:
  GPU 0: chunk 0 = A0+B0+C0+D0
  GPU 1: chunk 1 = A1+B1+C1+D1
  GPU 2: chunk 2 = A2+B2+C2+D2
  GPU 3: chunk 3 = A3+B3+C3+D3

═══════════════════════════════════════════════════
Phase 2: AllGather (3 steps for 4 GPUs)
═══════════════════════════════════════════════════

Each GPU sends its completed chunk to the right neighbor.
The received chunk is STORED (not added).

Step 1:
  GPU 0 sends sum_0 to GPU 1;  receives sum_3 from GPU 3
  GPU 1 sends sum_1 to GPU 2;  receives sum_0 from GPU 0
  GPU 2 sends sum_2 to GPU 3;  receives sum_1 from GPU 1
  GPU 3 sends sum_3 to GPU 0;  receives sum_2 from GPU 2

Step 2: Forward the received chunk
Step 3: Forward again

After Phase 2:
  GPU 0: [sum_0, sum_1, sum_2, sum_3]  ← full reduced vector!
  GPU 1: [sum_0, sum_1, sum_2, sum_3]
  GPU 2: [sum_0, sum_1, sum_2, sum_3]
  GPU 3: [sum_0, sum_1, sum_2, sum_3]
```

### Bandwidth analysis

```
Phase 1 (ReduceScatter): N-1 steps, each GPU sends S/N data per step
  Total data sent per GPU: (N-1) * S/N = S * (N-1)/N

Phase 2 (AllGather): N-1 steps, each GPU sends S/N data per step
  Total data sent per GPU: (N-1) * S/N = S * (N-1)/N

Grand total per GPU: 2 * S * (N-1)/N

Time = 2 * S * (N-1)/N / bandwidth_per_link

This is BANDWIDTH-OPTIMAL: you cannot do better because every GPU
must receive S * (N-1)/N data (it needs data from N-1 other GPUs,
each contributing S/N of the final result).
```

### Why ring allreduce scales well

```
For N GPUs with data size S:
  Time = 2 * S * (N-1)/N / BW ≈ 2S/BW   (for large N)

The time is INDEPENDENT of N (for large N)!

This is because every link is fully utilized simultaneously.
Adding more GPUs does not increase the time (much).
```

---

## Hardware: NVLink, InfiniBand, PCIe

### NVLink

- **Purpose:** High-bandwidth GPU-to-GPU connection within a node
- **Bandwidth:** 600 GB/s bidirectional (A100), 900 GB/s (H100)
- **Latency:** ~1 microsecond
- **Topology:** NVSwitch connects all 8 GPUs in a fully-connected graph
- **Use:** Tensor parallelism, fast AllReduce within a node

### InfiniBand (IB)

- **Purpose:** GPU-to-GPU across nodes (inter-node network)
- **Bandwidth:** ~50 GB/s per port (HDR), ~100 GB/s (NDR)
- **Latency:** ~1-5 microseconds
- **Topology:** Fat tree, Dragonfly
- **Use:** Pipeline parallelism between nodes, data parallelism AllReduce

### PCIe

- **Purpose:** CPU-GPU connection, GPU connection without NVLink
- **Bandwidth:** ~32 GB/s (Gen 4 x16), ~64 GB/s (Gen 5 x16)
- **Latency:** ~1 microsecond
- **Use:** CPU-GPU data transfer, less optimal GPU-GPU communication

### Comparison

```
Bandwidth comparison:

HBM (GPU memory):   ████████████████████████████████████ 2000 GB/s
NVLink (intra-node): ███████████                          600 GB/s
PCIe Gen5:           █                                     64 GB/s
InfiniBand NDR:      █                                    100 GB/s

Key ratio: NVLink is 6-12x faster than InfiniBand.
This is why tensor parallelism goes within nodes (NVLink)
and pipeline parallelism goes across nodes (InfiniBand).
```

---

## ZeRO Optimizer

**ZeRO** (Zero Redundancy Optimizer) reduces memory redundancy in data
parallelism by partitioning optimizer states, gradients, and parameters
across GPUs.

### Standard Data Parallelism Memory

```
Each GPU stores (for a 10B model in FP16 with Adam):
  Model parameters:   20 GB (FP16)
  Gradients:          20 GB (FP16)
  Optimizer states:   80 GB (FP32 copy + momentum + variance)
  Total per GPU:     120 GB  ← replicated on ALL GPUs!
```

### ZeRO Stages

```
                     Per-GPU Memory (8 GPUs, 10B params)
                     ═══════════════════════════════════
                     Params  Grads   Opt     Total
Standard DP:         20 GB   20 GB   80 GB   120 GB

ZeRO Stage 1:       20 GB   20 GB   10 GB    50 GB
  (partition optimizer states)

ZeRO Stage 2:       20 GB   2.5 GB  10 GB   32.5 GB
  (+ partition gradients)

ZeRO Stage 3:       2.5 GB  2.5 GB  10 GB    15 GB
  (+ partition parameters)
```

### ZeRO Stage 3 details

Each GPU holds only 1/N of the parameters, gradients, and optimizer states.
When a layer's parameters are needed for forward/backward, they are gathered
from all GPUs (AllGather), used, then discarded.

```
Forward pass with ZeRO-3:
  For each layer:
    AllGather parameters from all GPUs    ← communication cost
    Compute forward pass with full parameters
    Discard non-local parameters          ← save memory

Backward pass:
  For each layer (reverse):
    AllGather parameters from all GPUs
    Compute gradients
    ReduceScatter gradients               ← each GPU gets its shard
    Discard non-local parameters

Trade-off: Less memory, more communication.
```

---

## Pipeline Bubbles and Micro-Batching

### The bubble problem quantified

```
Pipeline bubble fraction = (P - 1) / (P - 1 + M)

where P = pipeline stages, M = micro-batches

P=4, M=4:   bubble = 3/7 = 43%    ← bad
P=4, M=16:  bubble = 3/19 = 16%   ← acceptable
P=4, M=64:  bubble = 3/67 = 4.5%  ← good
P=8, M=64:  bubble = 7/71 = 9.9%  ← acceptable with more stages
```

### Interleaved pipeline schedule (1F1B)

The **1F1B** (one forward, one backward) schedule reduces memory requirements
by starting backward passes before all forward passes complete:

```
Standard schedule (all forward then all backward):
  GPU 0: [F0][F1][F2][F3]                    [B3][B2][B1][B0]
  GPU 1:     [F0][F1][F2][F3]            [B3][B2][B1][B0]
  GPU 2:         [F0][F1][F2][F3]    [B3][B2][B1][B0]
  GPU 3:             [F0][F1][F2][F3][B3][B2][B1][B0]

  Memory: Must store activations for ALL 4 micro-batches simultaneously.

1F1B schedule:
  GPU 0: [F0][F1][F2][F3][B0][F ][B1][F ][B2][  ][B3]
  GPU 1:     [F0][F1][F2][B0][F3][B1][  ][B2][  ][B3]
  GPU 2:         [F0][F1][B0][F2][B1][F3][B2][  ][B3]
  GPU 3:             [F0][B0][F1][B1][F2][B2][F3][B3]

  Memory: At most P micro-batches of activations stored simultaneously
  (instead of M).
```

---

## Topology-Aware Placement

When scheduling jobs on a GPU cluster, the physical topology matters:

```
Typical 8-GPU node (DGX A100):
  ┌────────────────────────────────────────┐
  │              NVSwitch                  │  All-to-all 600 GB/s
  │  ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐
  │  │G0│ │G1│ │G2│ │G3│ │G4│ │G5│ │G6│ │G7│
  │  └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘
  └────────────────────────────────────────┘
               │ InfiniBand │
               └──────┬─────┘
                      │
  ┌───────────────────┼───────────────────┐
  │    Network Switch (Fat Tree)          │
  │                                       │
  │  Node 0  Node 1  Node 2  Node 3 ...  │
  └───────────────────────────────────────┘
```

### Placement rules

1. **Tensor parallel groups on the same node** (need NVLink)
2. **Pipeline parallel stages on adjacent nodes** (need only point-to-point)
3. **Data parallel groups can span far nodes** (AllReduce can overlap with
   compute)
4. **Avoid fragmenting nodes** — a job needing 4 GPUs should get 4 GPUs from
   the same node, not 4 GPUs from 4 different nodes

### Fragmentation problem

```
Bad placement:
  Node 0: [Job A][Job A][ free ][ free ][Job B][Job B][ free ][ free ]
  Node 1: [Job A][Job A][ free ][ free ][Job B][Job B][ free ][ free ]

  Job C needs 4 GPUs with tensor parallelism (same node).
  4 free GPUs exist, but split across nodes! Cannot schedule.

Good placement:
  Node 0: [Job A][Job A][Job A][Job A][ free ][ free ][ free ][ free ]
  Node 1: [Job B][Job B][Job B][Job B][ free ][ free ][ free ][ free ]

  Job C can use the 4 free GPUs on Node 0 (or Node 1).
```

---

## Practical Numbers

### Communication times (approximate)

| Operation | Data Size | Network | Time |
|-----------|-----------|---------|------|
| AllReduce 70B gradients | 140 GB | 8x NVLink (600 GB/s) | 0.23 s |
| AllReduce 70B gradients | 140 GB | InfiniBand (50 GB/s) | 2.8 s |
| Pipeline activation transfer | 1 GB | InfiniBand (50 GB/s) | 20 ms |
| Tensor parallel AllReduce | 16 MB | NVLink (600 GB/s) | 0.027 ms |

### Scaling efficiency

```
Ideal: N GPUs → N times throughput

Reality (data parallelism, large model):
  2 GPUs:  ~1.9x  (95% efficiency)
  8 GPUs:  ~7.2x  (90% efficiency)
  64 GPUs: ~51x   (80% efficiency)
  256 GPUs: ~180x (70% efficiency)

Efficiency drops due to:
  - Communication overhead (AllReduce)
  - Pipeline bubbles
  - Load imbalance
  - Synchronization overhead
```

---

## Key Takeaways

1. **Data parallelism** is simplest but requires model to fit on one GPU and
   has AllReduce communication cost proportional to model size.

2. **Tensor parallelism** splits layers across GPUs. High communication
   frequency — use within a node (NVLink).

3. **Pipeline parallelism** splits model into stages. Low communication
   frequency but introduces bubbles. Use across nodes (InfiniBand).

4. **3D parallelism** combines all three to scale to thousands of GPUs.
   Tensor parallel within node, pipeline across nodes, data parallel across
   replicas.

5. **Ring AllReduce** is bandwidth-optimal: time is approximately 2S/BW,
   independent of the number of GPUs for large N.

6. **NVLink (600 GB/s) vs InfiniBand (50 GB/s)**: this 12x gap determines
   which parallelism strategy goes where.

7. **ZeRO** reduces memory redundancy in data parallelism at the cost of
   more communication (AllGather parameters on demand).

8. **Pipeline bubbles** are minimized by using many micro-batches (M >> P)
   and interleaved schedules (1F1B).

9. **Topology-aware placement** is critical: putting tensor-parallel groups
   on the same node (NVLink) vs across nodes (InfiniBand) can mean 12x
   difference in communication bandwidth.

10. **Know the trade-offs:** Every parallelism strategy trades memory for
    communication or vice versa. The right combination depends on model size,
    cluster topology, and latency requirements.
