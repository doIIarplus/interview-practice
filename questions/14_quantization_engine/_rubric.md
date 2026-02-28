# Rubric: Quantization Engine

**Total: 100 points**

---

## 1. Correct Symmetric Quantization/Dequantization (15 points)

### Full marks (15):
- Correct scale computation: `scale = max(abs(tensor)) / (2^(bits-1) - 1)`
- Correct quantization with proper rounding and clamping to `[-(2^(bits-1)), 2^(bits-1) - 1]`
- Uses Python's built-in `round()` (banker's rounding) or explicit `int(x + 0.5)` -- either is acceptable
- Dequantization: `value = quantized * scale`
- Edge case: all-zero tensor uses scale = 1.0 (or any positive value) to avoid division by zero
- Works correctly with different `bits` values (2, 4, 8, 16)

### Partial credit:
- (10) Correct formula but missing edge case handling
- (7) Mostly correct but off-by-one in range (e.g., using 128 instead of 127)
- (5) Understands concept but formula has bugs

### Common mistakes:
- Using `2^bits` instead of `2^(bits-1)` for signed range
- Not clamping quantized values to the valid range
- Division by zero when tensor is all zeros

---

## 2. Correct Asymmetric Quantization with Zero Point (15 points)

### Full marks (15):
- Correct scale: `(max_val - min_val) / (2^bits - 1)`
- Correct zero_point: `round(-min_val / scale)`, clamped to `[0, 2^bits - 1]`
- Zero point is an integer (rounded and clamped)
- Quantization: `round(clamp(value / scale + zero_point, 0, 2^bits - 1))`
- Dequantization: `(quantized - zero_point) * scale`
- Edge case: all identical values handled gracefully

### Partial credit:
- (10) Correct formulas but zero_point handling has issues
- (7) Confuses signed vs unsigned range for asymmetric
- (5) Partial understanding

### Key insight to assess:
- The candidate should understand WHY asymmetric is needed: symmetric wastes half the range for non-negative data (like post-ReLU activations). If activations are in [0, 2], symmetric maps [-2, 2] to [-128, 127], wasting the negative half. Asymmetric maps [0, 2] to [0, 255], using the full range.

---

## 3. Per-Channel Quantization (15 points)

### Full marks (15):
- Correctly iterates over rows (axis=0) or columns (axis=1)
- Computes independent scale factor for each channel
- Applies `quantize_symmetric` (or equivalent logic) per channel
- Returns properly shaped matrix and scale list
- Handles axis=1 correctly (requires column extraction/reconstruction)

### Partial credit:
- (10) Correct for axis=0 only
- (7) Logic is right but matrix reshaping is wrong for axis=1
- (5) Understands concept but implementation doesn't work

### Key insight to assess:
- The candidate should explain WHY per-channel is better: different output channels of a weight matrix can have very different magnitude ranges. A single scale factor is dominated by the channel with the largest values, causing precision loss for channels with smaller values. Per-channel gives each channel its own optimal scale.

---

## 4. Integer-Only Matrix Multiply with Correct Rescaling (20 points)

### Full marks (20):
- Inner loop uses only integer arithmetic (no float operations in the accumulation)
- Accumulates `A_int[i][k] * B_int[k][j]` as integers
- Applies `scale_a * scale_b` only at the end (outside inner loop)
- Correct matrix dimensions: (M x K) @ (K x N) -> (M x N)
- Results match float reference within reasonable quantization error

### Partial credit:
- (15) Correct result but mixes float and int in inner loop (defeats the purpose)
- (10) Correct matmul logic but wrong scaling
- (5) Basic matmul but doesn't understand the quantization scaling

### Key insight to assess:
- The mathematical identity: `(A * s_a) @ (B * s_b) = s_a * s_b * (A @ B)` where A, B are integer matrices
- This is exactly how INT8 tensor cores work: the matmul is in INT8, producing INT32 accumulators, then rescaled to FP32
- The candidate should understand that keeping the inner loop in integers is the whole point -- it maps to hardware INT8 multiply-accumulate

---

## 5. Error Analysis with Correct SNR Computation (10 points)

### Full marks (10):
- Correct MSE: `mean((original[i] - dequantized[i])^2)`
- Correct max_error: `max(|original[i] - dequantized[i]|)`
- Correct SNR: `10 * log10(signal_power / noise_power)` where signal_power = mean(x^2), noise_power = MSE
- Handles edge cases: zero signal power (SNR = 0), zero noise power (SNR = inf)
- Correctly dispatches to symmetric vs asymmetric dequantization

### Partial credit:
- (7) Correct MSE and max_error but wrong SNR formula
- (5) Most metrics correct but edge cases not handled
- (3) Partial implementation

---

## 6. Edge Cases (10 points)

### Full marks (10):
- All-zero tensor: no division by zero, quantizes to all zeros
- Single-value tensor: works correctly
- Very large range: clipping/clamping works
- Values near zero: zero point is correct in asymmetric
- Different bit widths (2, 4, 8, 16): range adjusts correctly
- Empty tensor: either handled or explicit error
- Matrix with mismatched channel ranges: per-channel handles correctly

### Partial credit:
- (7) Most edge cases handled
- (4) Some edge cases cause crashes
- (2) Only happy path works

---

## 7. Understanding of Symmetric vs Asymmetric Trade-offs (15 points)

*Assessed through code structure, comments, or verbal explanation.*

### Full marks (15):
- Can explain when to use symmetric (weights centered at 0) vs asymmetric (activations after ReLU)
- Understands that asymmetric uses the full integer range for non-negative data
- Understands that per-channel preserves accuracy for weight matrices with varying channel ranges
- Can discuss the trade-off: per-channel requires storing more scale factors but gives better accuracy
- Understands that INT8 quantization for typical weight distributions gives ~40-50 dB SNR

### Partial credit:
- (10) Understands symmetric vs asymmetric but not per-channel trade-offs
- (5) Basic understanding without nuance
- (0) Cannot explain trade-offs

---

## Bonus Observations (not scored, but positive signals):

- Mentions that accumulator width matters (INT8 * INT8 needs INT16 or INT32 accumulator)
- Discusses calibration: how to choose the quantization range from training data
- Mentions outlier-aware quantization (SmoothQuant, AWQ)
- Discusses the interaction between quantization and batch normalization / layer normalization
- Notes that per-channel quantization for activations is harder in practice (requires knowing channel ranges at inference time)
- Mentions mixed-precision approaches (some layers at INT8, attention at FP16)
- Discusses the hardware implications: INT8 tensor cores have 2x throughput vs FP16

---

## Red Flags:

- Cannot explain the difference between symmetric and asymmetric quantization
- Does not understand what scale and zero_point represent
- Uses floating-point operations in the "integer" matmul inner loop without realizing the issue
- Cannot handle different bit widths
- Does not understand why quantization introduces error
- Confuses quantization with normalization or standardization
- Uses NumPy (question specifies stdlib only)
