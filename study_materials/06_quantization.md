# Quantization for ML — From Theory to Practice

This guide explains quantization from first principles: why it works, the math
behind it, the different formats and schemes, and how it enables faster
inference on modern GPUs. Directly relevant to Q14 (Quantization Engine).

---

## Table of Contents

1. [What Is Quantization?](#what-is-quantization)
2. [Why Quantize?](#why-quantize)
3. [Floating Point Review](#floating-point-review)
4. [Symmetric Quantization](#symmetric-quantization)
5. [Asymmetric Quantization](#asymmetric-quantization)
6. [Per-Tensor vs Per-Channel vs Per-Group](#per-tensor-vs-per-channel-vs-per-group)
7. [Calibration](#calibration)
8. [Dynamic vs Static Quantization](#dynamic-vs-static-quantization)
9. [INT8 Quantization In Detail](#int8-quantization-in-detail)
10. [FP8 Formats: E4M3 vs E5M2](#fp8-formats-e4m3-vs-e5m2)
11. [Quantized Matrix Multiplication](#quantized-matrix-multiplication)
12. [Error Metrics](#error-metrics)
13. [Advanced Methods: GPTQ, AWQ, SmoothQuant](#advanced-methods-gptq-awq-smoothquant)
14. [Tensor Cores and Quantized Operations](#tensor-cores-and-quantized-operations)
15. [Worked Examples](#worked-examples)
16. [Key Takeaways](#key-takeaways)

---

## What Is Quantization?

Quantization is the process of mapping values from a high-precision format to a
lower-precision format.

```
Original (FP32):     [0.0312, -1.4821, 0.7654, -0.0028, 2.1003]
                          ↓ quantize (to INT8)
Quantized (INT8):    [   3,     -127,    65,      0,      127  ]
Scale factor:        0.01654   (to convert back)
                          ↓ dequantize
Reconstructed:       [0.0496, -2.1006, 1.0751, 0.0000, 2.1006]
                          ↑
                     Not exact! Quantization introduces error.
```

The goal is to minimize this error while achieving significant memory and
compute savings.

---

## Why Quantize?

### Memory reduction

```
70B parameter model:
  FP32: 70B * 4 bytes = 280 GB  (needs 4 A100-80GB GPUs)
  FP16: 70B * 2 bytes = 140 GB  (needs 2 A100-80GB GPUs)
  INT8: 70B * 1 byte  =  70 GB  (fits on 1 A100-80GB GPU!)
  INT4: 70B * 0.5 B   =  35 GB  (fits on 1 A100 with room for KV-cache)
```

### Faster compute

Modern GPU Tensor Cores have specialized hardware for lower-precision arithmetic:

```
NVIDIA A100 Tensor Core throughput:
  FP32:  19.5 TFLOPS
  FP16: 312   TFLOPS  (16x faster than FP32!)
  INT8: 624   TOPS    (32x faster than FP32!)

NVIDIA H100 Tensor Core throughput:
  FP32:  67   TFLOPS
  FP16: 990   TFLOPS
  FP8: 1979   TFLOPS  (30x faster than FP32!)
  INT8: 1979  TOPS
```

### Less memory bandwidth

For memory-bound operations (like decode in LLM inference), quantization
directly translates to speed because you load fewer bytes:

```
Loading 70B model weights for one decode step:
  FP16: 140 GB / 2 TB/s = 70 ms   → ~14 tokens/sec
  INT8:  70 GB / 2 TB/s = 35 ms   → ~28 tokens/sec  (2x faster!)
  INT4:  35 GB / 2 TB/s = 17.5 ms → ~57 tokens/sec  (4x faster!)
```

---

## Floating Point Review

Before diving into quantization formats, let us review how floating point
numbers work.

### FP32 (IEEE 754 single precision)

```
┌─┬──────────┬───────────────────────┐
│S│ Exponent │       Mantissa        │
│1│  8 bits  │       23 bits         │
└─┴──────────┴───────────────────────┘

Value = (-1)^S * 2^(E-127) * (1 + M/2^23)

Range: +-3.4 * 10^38
Precision: ~7 decimal digits
```

### FP16 (IEEE 754 half precision)

```
┌─┬─────┬──────────┐
│S│  E  │    M     │
│1│5 bit│  10 bit  │
└─┴─────┴──────────┘

Range: +-65504
Precision: ~3 decimal digits
```

### BF16 (Brain Float 16)

```
┌─┬──────────┬───────┐
│S│    E     │   M   │
│1│  8 bits  │ 7 bit │
└─┴──────────┴───────┘

Same exponent range as FP32 (so same dynamic range: +-3.4 * 10^38)
Less precision than FP16 (~2 decimal digits)
Very popular for training because it does not overflow like FP16
```

---

## Symmetric Quantization

In symmetric quantization, zero maps to zero, and the scale is determined by
the maximum absolute value.

### Formula

```
Quantize:
  scale = max(|x|) / Q_max
  x_q = round(x / scale)
  x_q = clamp(x_q, -Q_max, Q_max)

Dequantize:
  x_approx = x_q * scale

For INT8 (signed): Q_max = 127, range = [-127, 127]
  (We use 127 instead of 128 to keep symmetry around zero)
```

### Example

```python
import numpy as np

def symmetric_quantize(x, num_bits=8):
    """Symmetric quantization to signed integer."""
    Q_max = (1 << (num_bits - 1)) - 1  # 127 for INT8

    # Compute scale
    x_max = np.max(np.abs(x))
    scale = x_max / Q_max

    # Quantize
    x_q = np.round(x / scale).astype(np.int8)
    x_q = np.clip(x_q, -Q_max, Q_max)

    return x_q, scale

def symmetric_dequantize(x_q, scale):
    """Dequantize back to float."""
    return x_q.astype(np.float32) * scale

# Example
x = np.array([0.5, -1.2, 0.3, -0.8, 1.5], dtype=np.float32)
x_q, scale = symmetric_quantize(x)

print(f"Original:     {x}")
print(f"Scale:        {scale:.6f}")
print(f"Quantized:    {x_q}")
print(f"Dequantized:  {symmetric_dequantize(x_q, scale)}")
print(f"Error:        {x - symmetric_dequantize(x_q, scale)}")
```

Output:
```
Original:     [ 0.5  -1.2   0.3  -0.8   1.5 ]
Scale:        0.011811
Quantized:    [  42  -102    25   -68   127]
Dequantized:  [ 0.496063 -1.204724  0.295276 -0.803150  1.500000]
Error:        [ 0.003937 0.004724 0.004724 0.003150 0.000000]
```

### Properties of symmetric quantization

- **Zero is exactly representable** (0 quantizes to 0)
- **Simple:** only one parameter (scale)
- **Wastes range if distribution is asymmetric:** e.g., ReLU outputs are all
  non-negative, so half the quantization range (negative values) is unused

---

## Asymmetric Quantization

Asymmetric quantization uses both a scale and a **zero point** to map an
arbitrary range [x_min, x_max] to the full quantized range.

### Formula

```
Quantize:
  scale = (x_max - x_min) / (Q_max - Q_min)
  zero_point = round(Q_min - x_min / scale)
  x_q = round(x / scale) + zero_point
  x_q = clamp(x_q, Q_min, Q_max)

Dequantize:
  x_approx = (x_q - zero_point) * scale

For unsigned INT8: Q_min = 0, Q_max = 255
For signed INT8:   Q_min = -128, Q_max = 127
```

### Example

```python
def asymmetric_quantize(x, num_bits=8):
    """Asymmetric quantization to unsigned integer."""
    Q_min = 0
    Q_max = (1 << num_bits) - 1  # 255 for 8-bit

    x_min = np.min(x)
    x_max = np.max(x)

    scale = (x_max - x_min) / (Q_max - Q_min)
    zero_point = round(-x_min / scale)  # Simplified; clamped to Q range
    zero_point = int(np.clip(zero_point, Q_min, Q_max))

    x_q = np.round(x / scale + zero_point).astype(np.uint8)
    x_q = np.clip(x_q, Q_min, Q_max)

    return x_q, scale, zero_point

def asymmetric_dequantize(x_q, scale, zero_point):
    return (x_q.astype(np.float32) - zero_point) * scale

# Example with non-negative values (e.g., after ReLU)
x = np.array([0.0, 0.5, 1.2, 0.3, 2.0], dtype=np.float32)
x_q, scale, zp = asymmetric_quantize(x)

print(f"Original:     {x}")
print(f"Scale:        {scale:.6f}, Zero Point: {zp}")
print(f"Quantized:    {x_q}")
print(f"Dequantized:  {asymmetric_dequantize(x_q, scale, zp)}")
```

### Symmetric vs Asymmetric

| Property | Symmetric | Asymmetric |
|----------|-----------|------------|
| Parameters | scale only | scale + zero_point |
| Zero exact? | Yes (always) | Yes (if zero_point is integer) |
| Best for | Weights (often symmetric around 0) | Activations (often non-negative after ReLU) |
| Matmul cost | Simpler (no zero_point offset) | Extra addition needed |

---

## Per-Tensor vs Per-Channel vs Per-Group

The **granularity** of quantization significantly affects accuracy.

### Per-tensor

One scale for the entire tensor:

```
Weight matrix W (4x4):
┌──────┬──────┬──────┬──────┐
│  0.1 │  0.5 │  0.2 │  0.3 │   All 16 values share
│  0.8 │ -1.2 │  0.4 │ -0.1 │   ONE scale factor.
│  0.3 │  0.2 │ -0.5 │  0.7 │
│  2.0 │ -0.3 │  0.1 │  0.6 │   scale = max(|W|) / 127 = 2.0/127
└──────┴──────┴──────┴──────┘

Problem: The outlier (2.0) determines the scale.
Small values (0.1) map to round(0.1/0.01575) = 6 out of 127.
Poor resolution for small values!
```

### Per-channel (per output channel)

One scale per row (output channel) of the weight matrix:

```
Weight matrix W (4x4):
                                        Scale
┌──────┬──────┬──────┬──────┐
│  0.1 │  0.5 │  0.2 │  0.3 │   scale_0 = 0.5/127 = 0.00394
│  0.8 │ -1.2 │  0.4 │ -0.1 │   scale_1 = 1.2/127 = 0.00945
│  0.3 │  0.2 │ -0.5 │  0.7 │   scale_2 = 0.7/127 = 0.00551
│  2.0 │ -0.3 │  0.1 │  0.6 │   scale_3 = 2.0/127 = 0.01575
└──────┴──────┴──────┴──────┘

Now 0.1 in row 0 maps to round(0.1/0.00394) = 25 out of 127.
Much better resolution!
```

### Per-group

One scale per group of G consecutive values (e.g., G=128):

```
For a row of 4096 values with group_size=128:
  Groups: [0:128], [128:256], [256:384], ..., [3968:4096]
  Each group has its own scale → 32 scales per row
```

### Comparison

| Granularity | Scales per matrix (MxN) | Accuracy | Compute overhead |
|-------------|------------------------|----------|-----------------|
| Per-tensor | 1 | Lowest | None |
| Per-channel | M (one per row) | Good | Minimal |
| Per-group (G=128) | M * N/128 | Best | Some |

**Per-channel** is the standard for weight quantization. It adds negligible
overhead (one extra multiply per output channel in dequantization) and
significantly improves accuracy.

---

## Calibration

Calibration determines the optimal scale factors by analyzing representative
data.

### For weights

Weights are fixed after training, so calibration is straightforward:
- **Min-max:** scale = max(|w|) / Q_max. Simple but sensitive to outliers.
- **Percentile:** Use the 99.9th percentile instead of max. Clips outliers.

### For activations

Activations depend on the input data. Run a calibration dataset through the
model and collect activation statistics:

```
For each layer's activations:
  1. Run 100-1000 calibration samples through the model
  2. Collect min/max (or histogram) of activations at each layer
  3. Choose scale based on collected statistics

Methods:
  - Min-max: scale = max(observed_max) / Q_max
  - Percentile: Clip to p-th percentile (e.g., 99.99%)
  - MSE: Choose scale that minimizes mean squared error between
         original and quantized values
  - Entropy / KL divergence: Choose scale that minimizes information
         loss (used by TensorRT)
```

### Example: percentile calibration

```python
def calibrate_percentile(activations, percentile=99.99, num_bits=8):
    """Find optimal scale using percentile clipping."""
    Q_max = (1 << (num_bits - 1)) - 1

    # Clip outliers
    threshold = np.percentile(np.abs(activations), percentile)

    scale = threshold / Q_max
    return scale

# Collect activations from calibration data
all_activations = []
for batch in calibration_dataset:
    output = model.layer_42(batch)
    all_activations.append(output.flatten())

activations = np.concatenate(all_activations)
scale = calibrate_percentile(activations)
```

---

## Dynamic vs Static Quantization

### Static quantization

- Scale factors determined **during calibration** (before deployment)
- Scales are fixed at inference time
- Faster inference (no runtime scale computation)
- Requires representative calibration data

### Dynamic quantization

- Scale factors computed **at runtime** for each input
- No calibration needed
- Slightly slower (must compute scale per forward pass)
- Better accuracy for inputs that differ from calibration data

```
Static:  scale = precomputed_from_calibration
         x_q = round(x / scale)

Dynamic: scale = max(|x|) / Q_max    ← computed on the fly
         x_q = round(x / scale)
```

### In practice

- **Weights:** Always static (weights do not change at inference time)
- **Activations:** Static for maximum speed, dynamic for better accuracy
- **Most production systems** use static quantization with good calibration

---

## INT8 Quantization In Detail

### INT8 signed: range [-128, 127]

```
Bit pattern: SMMM MMMM

S = sign bit (0 = positive, 1 = negative)
M = magnitude (7 bits)

Range: -128 to +127 (256 distinct values)
For symmetric quantization: use [-127, 127] (so zero maps to zero)
```

### INT8 quantized linear layer

```
Original:  Y = X @ W + bias     (all FP32 or FP16)

Quantized:
  W_q = quantize(W, scale_w)           INT8 weights (offline)
  X_q = quantize(X, scale_x)           INT8 activations (runtime)
  Y_q = X_q @ W_q                      INT32 accumulation (hardware)
  Y   = dequantize(Y_q, scale_x * scale_w) + bias
      = Y_q * (scale_x * scale_w) + bias

  The matmul is done in INT8 on Tensor Cores (2x faster than FP16).
  The accumulation and dequantization are in INT32/FP32 (negligible cost).
```

### INT4 quantization

```
4 bits → 16 distinct values
Signed: [-8, 7]
Usually with per-group scaling (group_size=128)

Two INT4 values packed into one byte:
  ┌────────┐
  │ val1 val2 │
  │ 4bit 4bit │
  └────────┘

Requires unpacking before matmul, but 2x more compressed than INT8.
```

---

## FP8 Formats: E4M3 vs E5M2

FP8 (8-bit floating point) provides a middle ground between INT8 (fixed point)
and FP16 (floating point). Two formats exist:

### E4M3 (4-bit exponent, 3-bit mantissa)

```
┌─┬────┬───┐
│S│ E  │ M │
│1│4bit│3bt│
└─┴────┴───┘

Exponent range: 2^(-6) to 2^8 → values from ~2^-9 to 448
Precision: 3 mantissa bits → ~1 decimal digit
Normal range: [-448, 448]

Better precision than E5M2 (more mantissa bits)
Narrower range than E5M2 (fewer exponent bits)

Best for: forward pass, weights and activations
```

### E5M2 (5-bit exponent, 2-bit mantissa)

```
┌─┬─────┬──┐
│S│  E  │M │
│1│5 bit│2b│
└─┴─────┴──┘

Exponent range: 2^(-14) to 2^15 → values from ~2^-16 to 57344
Precision: 2 mantissa bits → <1 decimal digit
Normal range: [-57344, 57344]

Wider range than E4M3 (more exponent bits)
Less precise than E4M3 (fewer mantissa bits)

Best for: backward pass (gradients have wider dynamic range)
```

### Comparison

```
Format    Range           Precision    Use Case
──────    ─────           ─────────    ────────
FP32      +-3.4e38        ~7 digits    Training (master weights)
FP16      +-65504         ~3 digits    Training/inference
BF16      +-3.4e38        ~2 digits    Training
FP8 E4M3  +-448           ~1 digit     Inference, forward pass
FP8 E5M2  +-57344         <1 digit     Backward pass (gradients)
INT8      [-128, 127]     N/A (fixed)  Inference (with scale factor)
INT4      [-8, 7]         N/A (fixed)  Inference (weight-only)
```

### Why FP8 vs INT8?

```
INT8 requires explicit scale management:
  - Scale must be computed (calibration or dynamic)
  - Zero point may be needed (asymmetric)
  - Per-channel or per-group scales add complexity

FP8 has implicit scale (the exponent):
  - Dynamic range is built into the format
  - Simpler workflow (just cast, possibly with per-tensor scale)
  - Hardware support on H100+ (FP8 Tensor Cores)

INT8 gives more precision (8 bits of mantissa after scaling)
FP8 E4M3 gives less precision (3 bits of mantissa) but handles range better
```

---

## Quantized Matrix Multiplication

The key operation that quantization accelerates.

### Standard FP16 matmul

```
C = A @ B

A: (M, K) in FP16
B: (K, N) in FP16
C: (M, N) in FP16

FLOPs: 2 * M * K * N
Memory: (M*K + K*N + M*N) * 2 bytes
```

### INT8 quantized matmul

```
Step 1: Quantize (offline for weights, online for activations)
  A_q = round(A / scale_a).to(int8)     scale_a: per-tensor or per-channel
  B_q = round(B / scale_b).to(int8)     scale_b: per-channel (for weights)

Step 2: Integer matmul (on INT8 Tensor Cores)
  C_q = A_q @ B_q   → result in INT32 (to avoid overflow)

Step 3: Dequantize (back to float)
  C = C_q * (scale_a * scale_b)

  If per-channel on B (axis 1): scale_b is a vector of length N
  C[i][j] = C_q[i][j] * scale_a * scale_b[j]

           ┌──────────────────────┐
           │  A_q (INT8)          │
           │  (M x K)             │
           └──────────┬───────────┘
                      │  INT8 matmul
           ┌──────────┴───────────┐
           │  B_q (INT8)          │
           │  (K x N)             │
           └──────────┬───────────┘
                      │  accumulate in INT32
           ┌──────────┴───────────┐
           │  C_q (INT32)         │  ← intermediate
           │  (M x N)             │
           └──────────┬───────────┘
                      │  * scale_a * scale_b  (FP32)
           ┌──────────┴───────────┐
           │  C (FP16/FP32)       │  ← final result
           │  (M x N)             │
           └──────────────────────┘
```

### Why INT32 accumulation?

```
Each element of C_q is a dot product of K INT8 values:
  C_q[i][j] = sum(A_q[i][k] * B_q[k][j] for k in range(K))

Max value of each product: 127 * 127 = 16,129
Max value of sum (K terms): K * 16,129

For K=4096: max value = 66,048,769 — fits in INT32 (max ~2 billion)
For K=32768: max value = 528,424,833 — still fits in INT32

INT16 would overflow at K >= 2 (max 32,767).
```

---

## Error Metrics

How to measure quantization quality.

### Mean Squared Error (MSE)

```
MSE = mean((x - x_reconstructed)^2)
RMSE = sqrt(MSE)

Lower is better. Most common metric.
```

### Maximum Absolute Error

```
max_error = max(|x - x_reconstructed|)

Important for worst-case guarantees.
```

### Signal-to-Noise Ratio (SNR)

```
SNR = 10 * log10(mean(x^2) / MSE)

Measured in dB. Higher is better.
For INT8: typically 40-60 dB (excellent)
For INT4: typically 20-35 dB (acceptable)
```

### Cosine Similarity

```
cos_sim = dot(x, x_reconstructed) / (|x| * |x_reconstructed|)

Range: [-1, 1]. Higher is better.
> 0.999 is typically acceptable for INT8.
> 0.99 is acceptable for INT4.
```

### Example: measuring quantization error

```python
def evaluate_quantization(original, reconstructed):
    """Compute quantization error metrics."""
    diff = original - reconstructed

    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    max_err = np.max(np.abs(diff))

    signal_power = np.mean(original ** 2)
    snr_db = 10 * np.log10(signal_power / mse) if mse > 0 else float('inf')

    cos_sim = np.dot(original.flatten(), reconstructed.flatten()) / (
        np.linalg.norm(original) * np.linalg.norm(reconstructed) + 1e-10
    )

    print(f"MSE:         {mse:.8f}")
    print(f"RMSE:        {rmse:.6f}")
    print(f"Max Error:   {max_err:.6f}")
    print(f"SNR:         {snr_db:.1f} dB")
    print(f"Cos Sim:     {cos_sim:.6f}")

    return {"mse": mse, "rmse": rmse, "max_error": max_err,
            "snr_db": snr_db, "cosine_similarity": cos_sim}
```

---

## Advanced Methods: GPTQ, AWQ, SmoothQuant

These are state-of-the-art methods for INT4 and INT8 quantization of LLMs.

### GPTQ (Generalized Post-Training Quantization)

- **Weight-only** quantization (INT4 or INT3)
- Quantizes one weight column at a time
- Uses second-order information (Hessian) to minimize output error
- Adjusts remaining (unquantized) weights to compensate for quantization error
- Result: very high accuracy with INT4 weights

```
For each column j of weight matrix W:
  1. Quantize column j: w_q = quantize(w_j)
  2. Compute quantization error: delta = w_j - dequantize(w_q)
  3. Distribute error to remaining columns using Hessian information:
     W[:, j+1:] -= delta * H_inv[j, j+1:] / H_inv[j, j]
  4. This compensates for the error in column j
```

### AWQ (Activation-Aware Weight Quantization)

- **Key insight:** Not all weights are equally important. Weights connected to
  large activations matter more.
- Scale weights per-channel based on activation magnitudes before quantization
- Search for optimal per-channel scales
- Result: Better accuracy than naive INT4, simpler than GPTQ

```
For each channel:
  1. Measure average activation magnitude
  2. Scale up "important" weights (connected to large activations)
     → these get more precision after quantization
  3. Scale down "unimportant" weights
  4. Quantize the scaled weights
  5. At inference: apply inverse scale to dequantized weights
```

### SmoothQuant

- For **weight AND activation** INT8 quantization (W8A8)
- Problem: Activations often have outlier channels (values 10-100x larger)
  that make quantization difficult
- Solution: Migrate the quantization difficulty from activations to weights
  by per-channel scaling

```
Original:  Y = X @ W

SmoothQuant:
  Y = (X * diag(1/s)) @ (diag(s) * W)
  Y =  X_smooth      @   W_smooth

  Choose s to equalize the difficulty:
  s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
  (alpha = 0.5 typically)

  X_smooth has smaller outliers → easier to quantize
  W_smooth has slightly larger values → still easy to quantize (weights are nice)
```

---

## Tensor Cores and Quantized Operations

### How Tensor Cores work

Tensor Cores perform small matrix multiply-accumulate operations:

```
A100 Tensor Core operation (INT8):
  D = A @ B + C

  A: 16x16 INT8
  B: 16x16 INT8
  C: 16x16 INT32  (accumulator)
  D: 16x16 INT32  (result)

  One Tensor Core does this in a single operation.
  With 432 Tensor Cores on A100: massive throughput.
```

### Mixed-precision patterns

```
Common inference patterns:

W8A8 (weight INT8, activation INT8):
  - INT8 Tensor Core matmul
  - INT32 accumulation
  - FP16 dequantization and bias add
  - Used by: TensorRT, SmoothQuant

W4A16 (weight INT4, activation FP16):
  - Dequantize INT4 weights to FP16 on the fly
  - FP16 Tensor Core matmul
  - Used by: GPTQ, AWQ
  - Speed gain from reduced memory (not faster compute)

W8A16 (weight FP8, activation FP16):
  - Cast FP8 weights to FP16
  - FP16 Tensor Core matmul
  - Simpler than INT8 (no scale management)

FP8xFP8 (H100+):
  - FP8 Tensor Core matmul
  - FP16 or FP32 accumulation
  - Fastest: 2x throughput vs FP16 Tensor Cores
```

---

## Worked Examples

### Example 1: Symmetric INT8 quantization of a weight matrix

```python
import numpy as np

# Weight matrix (simulating one layer of a neural network)
np.random.seed(42)
W = np.random.randn(4, 8).astype(np.float32) * 0.5

print("Original weights:")
print(W.round(3))
print()

# Per-channel symmetric quantization
Q_max = 127
scales = np.max(np.abs(W), axis=1) / Q_max  # One scale per row

print("Per-channel scales:", scales.round(6))
print()

# Quantize
W_q = np.zeros_like(W, dtype=np.int8)
for i in range(W.shape[0]):
    W_q[i] = np.round(W[i] / scales[i]).clip(-127, 127).astype(np.int8)

print("Quantized weights (INT8):")
print(W_q)
print()

# Dequantize
W_recon = np.zeros_like(W)
for i in range(W.shape[0]):
    W_recon[i] = W_q[i].astype(np.float32) * scales[i]

print("Reconstructed weights:")
print(W_recon.round(3))
print()

# Error
error = W - W_recon
mse = np.mean(error**2)
max_err = np.max(np.abs(error))
print(f"MSE: {mse:.8f}")
print(f"Max error: {max_err:.6f}")
print(f"Relative max error: {max_err / np.max(np.abs(W)) * 100:.2f}%")
```

### Example 2: Quantized matrix multiplication

```python
def quantized_matmul(X, W, scale_x, scale_w):
    """INT8 quantized matrix multiply with per-channel weight scales.

    Args:
        X: float input (M, K)
        W: float weight (K, N)
        scale_x: float scalar (per-tensor scale for X)
        scale_w: float array of shape (N,) (per-channel scales for W)

    Returns:
        Y: float output (M, N)
    """
    # Quantize X (per-tensor, symmetric)
    X_q = np.round(X / scale_x).clip(-127, 127).astype(np.int8)

    # Quantize W (per-channel, symmetric) - in practice done offline
    W_q = np.zeros_like(W, dtype=np.int8)
    for j in range(W.shape[1]):
        W_q[:, j] = np.round(W[:, j] / scale_w[j]).clip(-127, 127).astype(np.int8)

    # Integer matmul (accumulate in INT32 to avoid overflow)
    Y_q = X_q.astype(np.int32) @ W_q.astype(np.int32)

    # Dequantize: Y = Y_q * scale_x * scale_w
    Y = Y_q.astype(np.float32) * scale_x * scale_w[np.newaxis, :]

    return Y

# Test
M, K, N = 4, 128, 8
X = np.random.randn(M, K).astype(np.float32)
W = np.random.randn(K, N).astype(np.float32) * 0.1

# Compute scales
scale_x = np.max(np.abs(X)) / 127
scale_w = np.max(np.abs(W), axis=0) / 127

# Compare
Y_fp = X @ W
Y_quant = quantized_matmul(X, W, scale_x, scale_w)

error = Y_fp - Y_quant
print(f"FP32 matmul result (first row):  {Y_fp[0].round(3)}")
print(f"INT8 matmul result (first row):  {Y_quant[0].round(3)}")
print(f"MSE: {np.mean(error**2):.8f}")
print(f"Cosine similarity: {np.dot(Y_fp.flatten(), Y_quant.flatten()) / (np.linalg.norm(Y_fp) * np.linalg.norm(Y_quant)):.6f}")
```

---

## Key Takeaways

1. **Quantization reduces precision to save memory and compute.** INT8 is 2x
   smaller and 2x faster than FP16. INT4 is 4x smaller.

2. **Symmetric quantization** is simpler (one scale factor). Use for weights.
   **Asymmetric** is better for activations with non-zero mean.

3. **Per-channel quantization** is much more accurate than per-tensor and is
   standard practice. It adds negligible overhead.

4. **Calibration** is needed for static quantization of activations. Use
   representative data and percentile clipping.

5. **FP8 E4M3** has better precision, **E5M2** has wider range. E4M3 for
   forward pass, E5M2 for backward pass.

6. **INT8 matmul** accumulates in INT32 to avoid overflow. The Tensor Core
   does the INT8 multiply, INT32 accumulate, and you dequantize at the end.

7. **Advanced methods** (GPTQ, AWQ, SmoothQuant) achieve near-lossless INT4
   or W8A8 quantization by being smarter about which weights to prioritize
   and how to handle activation outliers.

8. **For interviews:** Be ready to implement symmetric and asymmetric
   quantization, explain per-channel vs per-tensor, compute scales from data,
   and walk through a quantized matmul step by step.

9. **The memory-bound insight:** In LLM inference (decode phase), quantization
   speeds things up primarily by reducing the amount of data loaded from HBM,
   not by faster arithmetic. Loading 70 GB (INT8) instead of 140 GB (FP16) is
   2x faster on memory-bound workloads.

10. **Quality-speed trade-off:** INT8 is nearly lossless. INT4 requires careful
    methods (GPTQ, AWQ) but achieves acceptable quality. Below INT4, quality
    degrades significantly for most models.
