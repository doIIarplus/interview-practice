# Follow-Up Questions: Quantization Engine

---

## 1. When would you use symmetric vs asymmetric quantization?

**Expected answer:**

**Symmetric** is preferred for:
- **Model weights**: Neural network weights are typically centered around zero with a roughly symmetric distribution. Symmetric quantization maps zero to zero exactly, preserving this structure.
- **Simplicity**: No zero_point parameter to store or compute with. Dequantization is just `q * scale`, and the matmul simplification `scale_a * scale_b * (A_int @ B_int)` is cleaner.
- **Hardware efficiency**: Some accelerators optimize for symmetric quantization specifically.

**Asymmetric** is preferred for:
- **Activations after ReLU/GeLU**: These are non-negative, so the distribution is [0, max]. Symmetric would waste the entire negative range [-max, 0], effectively halving precision.
- **Biased distributions**: Any data with a non-zero mean benefits from asymmetric, as it shifts the zero_point to utilize the full integer range.

**In practice**, a common configuration is: symmetric quantization for weights, asymmetric for activations. PyTorch's default quantization uses this scheme.

---

## 2. What is dynamic quantization vs static quantization? How do you calibrate?

**Expected answer:**

**Dynamic quantization**:
- Scale and zero_point for activations are computed at runtime from the actual input data for each forward pass.
- Pros: Adapts to the actual data distribution; no calibration step needed.
- Cons: Overhead of computing min/max at runtime; cannot fuse quantization into preceding operations.

**Static quantization**:
- Scale and zero_point are pre-computed using a calibration dataset before deployment.
- A representative batch of inputs is passed through the model, and the activation ranges are recorded (using observers/hooks).
- Pros: No runtime overhead for range computation; allows operator fusion.
- Cons: Calibration dataset must be representative; distribution shift can cause accuracy loss.

**Calibration methods**:
- **MinMax**: Use observed min and max. Simple but sensitive to outliers.
- **Percentile**: Use the 99.99th percentile instead of absolute min/max to reduce outlier sensitivity.
- **Entropy (KL divergence)**: Choose the range that minimizes KL divergence between the original and quantized distributions. Used by TensorRT.
- **MSE minimization**: Choose the range that minimizes reconstruction MSE.

---

## 3. Explain FP8 (E4M3 and E5M2 formats). How do they differ from INT8?

**Expected answer:**

FP8 uses 8 bits with a floating-point representation (sign + exponent + mantissa), unlike INT8 which is a fixed-point integer.

**E4M3** (1 sign + 4 exponent + 3 mantissa bits):
- Range: up to ~448, precision ~0.0625 near 1.0
- Larger dynamic range than E5M2 near zero
- Better for **forward pass** (weights and activations) because it has more mantissa bits for precision

**E5M2** (1 sign + 5 exponent + 2 mantissa bits):
- Range: up to ~57344, less precision
- Larger overall dynamic range
- Better for **gradients** in training because gradients can have large dynamic range

**Key differences from INT8**:
- FP8 has a logarithmic distribution of representable values (denser near zero, sparser for large values). INT8 has uniform spacing.
- FP8 doesn't require explicit scale/zero_point computation -- the floating-point format handles dynamic range naturally (though per-tensor scaling is still used in practice).
- FP8 has native hardware support on NVIDIA Hopper (H100) and Ada Lovelace GPUs.
- FP8 training is practical; INT8 training is much harder due to gradient quantization challenges.

---

## 4. What is GPTQ / AWQ / SmoothQuant? How do they improve quantization quality?

**Expected answer:**

**GPTQ** (Generalized Post-Training Quantization):
- Post-training quantization for LLMs using second-order information (Hessian).
- Quantizes weights one column at a time, adjusting remaining columns to compensate for the quantization error of already-quantized columns.
- Based on Optimal Brain Quantization (OBQ) framework.
- Achieves good quality at 4-bit and even 3-bit for large models.

**AWQ** (Activation-Aware Weight Quantization):
- Observes that not all weights are equally important -- some weights correspond to activation channels with large magnitudes and are more sensitive to quantization.
- Instead of quantizing all weights equally, it finds per-channel scaling factors that protect the important (salient) weight channels.
- Does NOT quantize activations -- only weights. Simple and effective.

**SmoothQuant** (Xiao et al., 2023):
- Addresses the problem that activation outliers make INT8 quantization difficult (a few channels have very large values).
- Key insight: migrate the quantization difficulty from activations to weights using a mathematically equivalent per-channel scaling: `Y = (X * diag(s)^-1) @ (diag(s) * W)`. This smooths activations (divides by s) and amplifies weights (multiplies by s).
- After smoothing, both activations and weights are easier to quantize.

**Common theme**: All these methods recognize that naive uniform quantization is suboptimal and use data-dependent strategies to place the quantization budget where it matters most.

---

## 5. How does quantization interact with the KV-cache in transformer inference?

**Expected answer:**

The KV-cache stores past key and value tensors for autoregressive generation. For long sequences, it becomes the memory bottleneck:
- A 70B model with batch=1, seq_len=4096, FP16: KV-cache ~ 40 GB

**KV-cache quantization**:
- Quantize K and V to INT8 or INT4, reducing cache size by 2-4x.
- This is particularly impactful because KV-cache grows linearly with sequence length.
- Challenge: K and V distributions can shift over time (different positions have different ranges), so per-token or per-head calibration is needed.
- Methods like **KIVI** (KV-cache quantization) use per-channel quantization for keys and per-token quantization for values, reflecting their different access patterns.
- INT4 KV-cache is practical with careful per-group quantization (e.g., group size 128).

**Impact on attention computation**:
- Quantized KV requires dequantization before the attention dot product (or specialized quantized attention kernels).
- The memory savings directly translate to longer context lengths or larger batch sizes.
- Some approaches quantize to different precisions at different layers or for different heads.

---

## 6. What are the challenges of quantizing attention layers specifically?

**Expected answer:**

Attention layers are harder to quantize than feedforward layers for several reasons:

1. **Softmax sensitivity**: The softmax function is exponential, so small changes in the logits (Q @ K^T) can cause large changes in the attention weights. Quantization error in Q or K is amplified.

2. **Dynamic range of attention logits**: The dot product Q @ K^T can have a wide range, and the range depends on the input. Static calibration is less reliable.

3. **Outlier features**: Transformer activations often have a few channels with very large magnitudes ("outlier features," observed in LLMs at ~6B+ parameters). These outliers make per-tensor quantization very poor -- the scale is dominated by outliers, crushing precision for normal values.

4. **Softmax output**: After softmax, values are in [0, 1] with most near 0 and a few near 1 (sparse attention pattern). This is a challenging distribution for uniform quantization.

5. **Accumulation precision**: The matmul `softmax(QK^T/sqrt(d)) @ V` requires high precision in the accumulator to avoid compounding errors.

**Mitigations**:
- Per-head quantization (each attention head has its own scale)
- SmoothQuant to handle outlier channels
- FP8 for attention, INT8 for feedforward
- Flash Attention computes in higher precision, avoiding materialization of the attention matrix

---

## 7. How would you implement mixed-precision quantization (different layers at different precision)?

**Expected answer:**

Mixed-precision quantization assigns different bit-widths to different layers based on their sensitivity to quantization error.

**Sensitivity analysis approach**:
1. For each layer, quantize it to various bit-widths while keeping other layers at full precision.
2. Measure the impact on model output (perplexity or task accuracy).
3. Assign higher precision (more bits) to sensitive layers and lower precision to robust layers.
4. This can be framed as an optimization problem: minimize total model size subject to an accuracy constraint.

**Heuristic approaches**:
- First and last layers are often kept at higher precision (they handle raw inputs/outputs).
- Attention layers often need higher precision than feedforward layers.
- Embedding layers are kept at full precision (they are lookup tables, not compute-bound).
- Within feedforward: the up-projection may be more sensitive than the down-projection.

**Implementation considerations**:
- Need a dispatcher that selects the correct kernel for each layer's precision.
- Mixed precision increases code complexity (different quantization parameters per layer).
- Hardware must support the mix of precisions efficiently (e.g., INT4 for some ops, INT8 for others, FP16 for attention).
- Frameworks like TensorRT and ONNX Runtime support per-layer precision configuration.

---

## 8. What is the relationship between quantization and tensor cores on NVIDIA GPUs?

**Expected answer:**

Tensor cores are specialized hardware units that perform matrix multiply-accumulate (MMA) operations at very high throughput.

**Supported precision modes** (as of Hopper/H100):
- FP16 x FP16 -> FP16/FP32 accumulate
- BF16 x BF16 -> FP32 accumulate
- TF32 (19-bit) x TF32 -> FP32 accumulate
- INT8 x INT8 -> INT32 accumulate (2x throughput vs FP16)
- FP8 (E4M3/E5M2) x FP8 -> FP16/FP32 accumulate (2x throughput vs FP16)
- INT4 x INT4 -> INT32 accumulate (4x throughput vs FP16, limited availability)

**Key points**:
- Tensor cores require specific matrix tile sizes (e.g., 16x16 for INT8 on Ampere). Quantization must produce matrices aligned to these sizes.
- INT8 tensor cores perform the multiply-accumulate in INT32 to avoid overflow (127 * 127 * K could overflow INT16 for large K).
- The `scale_a * scale_b * (A_int @ B_int)` pattern maps directly to: tensor core does `A_int @ B_int` in INT8->INT32, then a separate FP32 scale multiplication.
- cuBLAS and CUTLASS provide optimized INT8 GEMM kernels that utilize tensor cores.
- **Throughput**: H100 achieves ~3958 TOPS for INT8 vs ~1979 TFLOPS for FP16 -- nearly 2x gain from quantization.
- The quantization scheme must match what the hardware supports. Custom quantization methods that don't map to tensor core operations lose the throughput advantage.
