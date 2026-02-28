# Question 14: Quantization Engine

## Overview

You are building a quantization engine for deploying large language models at lower precision. Quantization reduces model size and increases inference throughput by representing weights and activations in fewer bits (e.g., INT8 instead of FP32).

This is a core technique for production ML inference: a model quantized to INT8 uses 4x less memory and can leverage specialized integer tensor cores for higher throughput.

---

## Background

### Why Quantize?

A 70B parameter model in FP16 requires ~140 GB of memory -- more than fits on a single GPU. Quantizing to INT8 halves this to ~70 GB. Quantizing to INT4 quarters it to ~35 GB. Beyond memory savings, integer arithmetic is faster and more energy-efficient on modern hardware.

### Symmetric vs Asymmetric Quantization

**Symmetric quantization** maps the floating-point range `[-max_abs, +max_abs]` to the integer range `[-128, 127]` (for INT8). The zero point is always 0. This works well for distributions centered around zero (e.g., model weights).

**Asymmetric quantization** maps an arbitrary range `[min_val, max_val]` to `[0, 255]` (for unsigned INT8). It introduces a zero point to handle distributions not centered at zero (e.g., activations after ReLU, which are always non-negative).

### Per-Tensor vs Per-Channel

**Per-tensor**: One scale factor for the entire tensor. Simple but loses precision if different channels have very different value ranges.

**Per-channel**: One scale factor per row/column (channel). More precise but requires more bookkeeping.

---

## Task

Implement the following functions. Use only the Python standard library (no NumPy).

### Part 1: Symmetric Quantization (INT8)

```
scale = max(abs(tensor)) / (2^(bits-1) - 1)
quantized_value = round(clamp(value / scale, -(2^(bits-1)), 2^(bits-1) - 1))
```

**1.** `quantize_symmetric(tensor: list[float], bits: int = 8) -> tuple[list[int], float]`

Quantize a floating-point tensor to signed integer representation using symmetric quantization.

- Compute the scale factor: `scale = max(|x| for x in tensor) / (2^(bits-1) - 1)`
- Quantize each value: `q = round(clamp(value / scale, -2^(bits-1), 2^(bits-1) - 1))`
- Return `(quantized_values, scale)`.
- Handle the edge case where all values are zero (scale should be 1.0 or similar to avoid division by zero).

**2.** `dequantize_symmetric(quantized: list[int], scale: float) -> list[float]`

Convert quantized values back to floating point: `value = quantized * scale`. Note that this is lossy -- the dequantized values will not exactly match the originals.

### Part 2: Asymmetric Quantization (INT8)

```
scale = (max_val - min_val) / (2^bits - 1)
zero_point = round(-min_val / scale)
quantized_value = round(clamp(value / scale + zero_point, 0, 2^bits - 1))
```

**3.** `quantize_asymmetric(tensor: list[float], bits: int = 8) -> tuple[list[int], float, int]`

Quantize using asymmetric quantization (unsigned integer output).

- `scale = (max(tensor) - min(tensor)) / (2^bits - 1)`
- `zero_point = round(-min(tensor) / scale)`, clamped to `[0, 2^bits - 1]`
- Quantize: `q = round(clamp(value / scale + zero_point, 0, 2^bits - 1))`
- Return `(quantized_values, scale, zero_point)`.
- Handle edge case: all values identical (scale = 1.0, zero_point = 0 or appropriate).

**4.** `dequantize_asymmetric(quantized: list[int], scale: float, zero_point: int) -> list[float]`

Dequantize: `value = (quantized - zero_point) * scale`

### Part 3: Per-Channel Quantization

**5.** `quantize_per_channel(weight_matrix: list[list[float]], axis: int = 0, bits: int = 8) -> tuple[list[list[int]], list[float]]`

Quantize each channel (row if axis=0, column if axis=1) independently using symmetric quantization with its own scale factor.

- For axis=0: each row gets its own scale.
- For axis=1: each column gets its own scale.
- Return `(quantized_matrix, scales)` where `scales[i]` is the scale for channel `i`.

### Part 4: Quantized Matrix Multiply

**6.** `quantized_matmul(A_quant: list[list[int]], B_quant: list[list[int]], scale_a: float, scale_b: float) -> list[list[float]]`

Perform matrix multiplication in integer arithmetic, then rescale.

The key mathematical insight:
```
C_float = A_float @ B_float
       = (A_int * scale_a) @ (B_int * scale_b)
       = scale_a * scale_b * (A_int @ B_int)
```

Implementation:
1. Perform `A_int @ B_int` using only integer arithmetic (no floats in the inner loop).
2. Multiply the result by `scale_a * scale_b` to get the final float result.

This is how quantized inference works on real hardware: the expensive matmul is done in INT8 on tensor cores, and only the final rescaling uses floating point.

### Part 5: Quantization Error Analysis

**7.** `quantization_error(original: list[float], quantized: list[int], scale: float, zero_point: int = 0, symmetric: bool = True) -> dict`

Analyze the quantization error.

- First, dequantize the quantized values (using symmetric or asymmetric based on the flag).
- Compute and return:
  - `"mse"`: Mean Squared Error between original and dequantized values.
  - `"max_error"`: Maximum absolute error across all elements.
  - `"snr_db"`: Signal-to-Noise Ratio in decibels.
    - `signal_power = mean(x^2 for x in original)`
    - `noise_power = mean((x - x_hat)^2 for x in original, x_hat in dequantized)` (this is the MSE)
    - `snr_db = 10 * log10(signal_power / noise_power)`
    - Handle edge cases (zero signal power or zero noise power).

---

## Examples

```python
# --- Symmetric Quantization ---
tensor = [1.0, -0.5, 0.3, -1.0, 0.0, 0.7]
quantized, scale = quantize_symmetric(tensor)
# scale = 1.0 / 127 ~= 0.00787
# quantized ≈ [127, -64, 38, -127, 0, 89]

dequantized = dequantize_symmetric(quantized, scale)
# dequantized ≈ [1.0, -0.504, 0.299, -1.0, 0.0, 0.701]

# --- Asymmetric Quantization (e.g., post-ReLU activations) ---
activations = [0.0, 0.5, 1.0, 1.5, 2.0, 0.1]
quantized_a, scale_a, zp = quantize_asymmetric(activations)
# scale_a = 2.0 / 255 ~= 0.00784
# zero_point = 0 (since min is 0.0)
# quantized_a ≈ [0, 64, 128, 191, 255, 13]

# --- Quantized MatMul ---
# A (2x3 float), B (3x2 float) -> C (2x2 float)
A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
B = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
# Expected: [[58.0, 64.0], [139.0, 154.0]]

A_q, scale_a = quantize_symmetric([v for row in A for v in row])
B_q, scale_b = quantize_symmetric([v for row in B for v in row])
# Reshape back to matrices for matmul
A_q_mat = [A_q[i*3:(i+1)*3] for i in range(2)]
B_q_mat = [B_q[i*2:(i+1)*2] for i in range(3)]

result = quantized_matmul(A_q_mat, B_q_mat, scale_a, scale_b)
# result ≈ [[58.0, 64.0], [139.0, 154.0]]  (with small quantization error)

# --- Error Analysis ---
errors = quantization_error(tensor, quantized, scale, symmetric=True)
print(f"MSE: {errors['mse']:.6f}")
print(f"Max Error: {errors['max_error']:.6f}")
print(f"SNR: {errors['snr_db']:.1f} dB")
```

---

## Constraints

- Use only the Python standard library (no NumPy, PyTorch, etc.).
- All tensor/matrix values are Python floats; quantized values are Python ints.
- `bits` parameter will be between 2 and 32.
- Matrices for matmul will have compatible dimensions.
- Handle edge cases: all-zero tensors, single-element tensors, very large value ranges.
