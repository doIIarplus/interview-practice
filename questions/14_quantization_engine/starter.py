"""
Question 14: Quantization Engine

Implement quantization routines for deploying large language models at lower
precision. Covers symmetric/asymmetric quantization, per-channel quantization,
integer-only matrix multiplication, and error analysis.

Run this file directly to execute the test cases:
    python starter.py
"""

from __future__ import annotations

import math
import random


# --------------------------------------------------------------------------- #
#  Part 1: Symmetric Quantization                                             #
# --------------------------------------------------------------------------- #

def quantize_symmetric(
    tensor: list[float], bits: int = 8
) -> tuple[list[int], float]:
    """Quantize a float tensor to signed integers using symmetric quantization.

    Symmetric quantization maps [-max_abs, +max_abs] -> [-(2^(bits-1)), 2^(bits-1) - 1].

    Formula:
        scale = max(|x| for x in tensor) / (2^(bits-1) - 1)
        q = round(clamp(value / scale, -(2^(bits-1)), 2^(bits-1) - 1))

    Args:
        tensor: List of floating-point values to quantize.
        bits: Number of bits for the quantized representation (default 8).

    Returns:
        Tuple of (quantized_values: list[int], scale: float).

    Edge cases:
        - If all values are zero, use scale = 1.0 to avoid division by zero.
    """
    # TODO: Implement this function
    pass


def dequantize_symmetric(quantized: list[int], scale: float) -> list[float]:
    """Dequantize signed integers back to floating-point values.

    Formula: value = quantized * scale

    Args:
        quantized: List of quantized integer values.
        scale: The scale factor used during quantization.

    Returns:
        List of dequantized float values.
    """
    # TODO: Implement this function
    pass


# --------------------------------------------------------------------------- #
#  Part 2: Asymmetric Quantization                                            #
# --------------------------------------------------------------------------- #

def quantize_asymmetric(
    tensor: list[float], bits: int = 8
) -> tuple[list[int], float, int]:
    """Quantize a float tensor to unsigned integers using asymmetric quantization.

    Asymmetric quantization maps [min_val, max_val] -> [0, 2^bits - 1].

    Formulas:
        scale = (max_val - min_val) / (2^bits - 1)
        zero_point = round(-min_val / scale), clamped to [0, 2^bits - 1]
        q = round(clamp(value / scale + zero_point, 0, 2^bits - 1))

    Args:
        tensor: List of floating-point values to quantize.
        bits: Number of bits for the quantized representation (default 8).

    Returns:
        Tuple of (quantized_values: list[int], scale: float, zero_point: int).

    Edge cases:
        - If all values are identical, use scale = 1.0 and zero_point = 0.
    """
    # TODO: Implement this function
    pass


def dequantize_asymmetric(
    quantized: list[int], scale: float, zero_point: int
) -> list[float]:
    """Dequantize unsigned integers back to floating-point values.

    Formula: value = (quantized - zero_point) * scale

    Args:
        quantized: List of quantized unsigned integer values.
        scale: The scale factor used during quantization.
        zero_point: The zero-point offset used during quantization.

    Returns:
        List of dequantized float values.
    """
    # TODO: Implement this function
    pass


# --------------------------------------------------------------------------- #
#  Part 3: Per-Channel Quantization                                           #
# --------------------------------------------------------------------------- #

def quantize_per_channel(
    weight_matrix: list[list[float]], axis: int = 0, bits: int = 8
) -> tuple[list[list[int]], list[float]]:
    """Quantize a weight matrix per-channel using symmetric quantization.

    Each channel (row for axis=0, column for axis=1) gets its own scale factor,
    preserving accuracy better than a single per-tensor scale.

    Args:
        weight_matrix: 2D list of floats (M x N).
        axis: 0 for per-row quantization, 1 for per-column quantization.
        bits: Number of bits for quantized representation.

    Returns:
        Tuple of (quantized_matrix: list[list[int]], scales: list[float]).
        scales[i] is the scale factor for channel i.
    """
    # TODO: Implement this function
    pass


# --------------------------------------------------------------------------- #
#  Part 4: Quantized Matrix Multiply                                          #
# --------------------------------------------------------------------------- #

def quantized_matmul(
    A_quant: list[list[int]],
    B_quant: list[list[int]],
    scale_a: float,
    scale_b: float,
) -> list[list[float]]:
    """Perform matrix multiplication in integer arithmetic, then rescale.

    Key insight:
        C_float = A_float @ B_float
                = (A_int * scale_a) @ (B_int * scale_b)
                = scale_a * scale_b * (A_int @ B_int)

    The integer matmul should use ONLY integer arithmetic in the inner loop.
    The scale factors are applied only at the end.

    Args:
        A_quant: Quantized matrix A (M x K), integer values.
        B_quant: Quantized matrix B (K x N), integer values.
        scale_a: Scale factor for matrix A.
        scale_b: Scale factor for matrix B.

    Returns:
        Result matrix C (M x N) as floats.
    """
    # TODO: Implement this function
    pass


# --------------------------------------------------------------------------- #
#  Part 5: Quantization Error Analysis                                        #
# --------------------------------------------------------------------------- #

def quantization_error(
    original: list[float],
    quantized: list[int],
    scale: float,
    zero_point: int = 0,
    symmetric: bool = True,
) -> dict:
    """Analyze quantization error between original and quantized values.

    Dequantizes the quantized values and compares against the originals.

    Metrics:
        - mse: Mean Squared Error
        - max_error: Maximum absolute error across all elements
        - snr_db: Signal-to-Noise Ratio in decibels
            signal_power = mean(x^2 for x in original)
            noise_power = MSE
            snr_db = 10 * log10(signal_power / noise_power)

    Args:
        original: Original floating-point values.
        quantized: Quantized integer values.
        scale: Scale factor used in quantization.
        zero_point: Zero-point offset (0 for symmetric quantization).
        symmetric: If True, use symmetric dequantization; else asymmetric.

    Returns:
        Dict with keys "mse", "max_error", "snr_db".

    Edge cases:
        - If signal_power is 0, snr_db = 0.0
        - If noise_power is 0 (perfect quantization), snr_db = float('inf')
    """
    # TODO: Implement this function
    pass


# --------------------------------------------------------------------------- #
#  Helper Utilities                                                           #
# --------------------------------------------------------------------------- #

def generate_normal_weights(
    n: int, mean: float = 0.0, std: float = 1.0, seed: int = 42
) -> list[float]:
    """Generate normally distributed weight-like values.

    Uses Box-Muller transform for normal distribution.

    Args:
        n: Number of values to generate.
        mean: Mean of the distribution.
        std: Standard deviation.
        seed: Random seed.

    Returns:
        List of float values.
    """
    rng = random.Random(seed)
    values = []
    for _ in range(n):
        u1 = rng.random()
        u2 = rng.random()
        # Box-Muller transform
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        values.append(mean + std * z)
    return values


def generate_relu_activations(
    n: int, mean: float = 1.0, std: float = 0.5, seed: int = 42
) -> list[float]:
    """Generate ReLU-like activations (non-negative, skewed distribution).

    Args:
        n: Number of values.
        mean: Mean of the underlying normal distribution.
        std: Standard deviation.
        seed: Random seed.

    Returns:
        List of non-negative float values.
    """
    raw = generate_normal_weights(n, mean, std, seed)
    return [max(0.0, x) for x in raw]


def generate_weight_matrix(
    rows: int, cols: int, std: float = 0.02, seed: int = 42
) -> list[list[float]]:
    """Generate a weight matrix with typical LLM initialization scale.

    Args:
        rows: Number of rows.
        cols: Number of columns.
        std: Standard deviation (LLMs typically use small init values).
        seed: Random seed.

    Returns:
        2D list of floats.
    """
    flat = generate_normal_weights(rows * cols, mean=0.0, std=std, seed=seed)
    return [flat[i * cols : (i + 1) * cols] for i in range(rows)]


def float_matmul(
    A: list[list[float]], B: list[list[float]]
) -> list[list[float]]:
    """Reference floating-point matrix multiplication.

    Args:
        A: Matrix of shape (M, K).
        B: Matrix of shape (K, N).

    Returns:
        Result matrix of shape (M, N).
    """
    M = len(A)
    K = len(A[0])
    N = len(B[0])
    C = [[0.0] * N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def flatten_matrix(matrix: list[list[float]]) -> list[float]:
    """Flatten a 2D matrix to a 1D list."""
    return [v for row in matrix for v in row]


def reshape_to_matrix(flat: list, rows: int, cols: int) -> list[list]:
    """Reshape a 1D list into a 2D matrix."""
    return [flat[i * cols : (i + 1) * cols] for i in range(rows)]


# --------------------------------------------------------------------------- #
#  Test Cases                                                                 #
# --------------------------------------------------------------------------- #

def test_symmetric_quantization() -> None:
    """Test symmetric quantization and dequantization."""
    # Basic test
    tensor = [1.0, -0.5, 0.3, -1.0, 0.0, 0.7]
    quantized, scale = quantize_symmetric(tensor)

    assert len(quantized) == len(tensor), "Output length mismatch"
    assert isinstance(scale, float), "Scale should be float"
    assert scale > 0, "Scale should be positive"

    # Check that max value maps to 127
    assert quantized[0] == 127 or quantized[3] == -127, (
        "Extreme values should map to +/-127"
    )

    # Dequantize and check round-trip accuracy
    dequantized = dequantize_symmetric(quantized, scale)
    for orig, deq in zip(tensor, dequantized):
        assert abs(orig - deq) < 0.02, (
            f"Round-trip error too large: {orig} -> {deq}"
        )

    # Edge case: all zeros
    zeros = [0.0, 0.0, 0.0]
    q_zeros, s_zeros = quantize_symmetric(zeros)
    assert all(v == 0 for v in q_zeros), "All-zero tensor should quantize to all zeros"
    assert s_zeros > 0, "Scale should still be positive for all-zero tensor"

    # Edge case: single value
    single = [3.14]
    q_single, s_single = quantize_symmetric(single)
    assert len(q_single) == 1
    assert q_single[0] == 127 or q_single[0] == -127

    print("  [PASS] test_symmetric_quantization")


def test_asymmetric_quantization() -> None:
    """Test asymmetric quantization for ReLU-like activations."""
    activations = [0.0, 0.5, 1.0, 1.5, 2.0, 0.1]
    quantized, scale, zp = quantize_asymmetric(activations)

    assert len(quantized) == len(activations), "Output length mismatch"
    assert isinstance(zp, int), "Zero point should be integer"
    assert 0 <= zp <= 255, "Zero point out of range for 8-bit"

    # Min value should map near 0, max near 255
    assert quantized[0] <= 5, f"Min value should map near 0, got {quantized[0]}"
    assert quantized[4] >= 250, f"Max value should map near 255, got {quantized[4]}"

    # Dequantize round-trip
    dequantized = dequantize_asymmetric(quantized, scale, zp)
    for orig, deq in zip(activations, dequantized):
        assert abs(orig - deq) < 0.02, (
            f"Round-trip error too large: {orig} -> {deq}"
        )

    # Edge case: all same value
    same = [5.0, 5.0, 5.0]
    q_same, s_same, zp_same = quantize_asymmetric(same)
    d_same = dequantize_asymmetric(q_same, s_same, zp_same)
    for orig, deq in zip(same, d_same):
        assert abs(orig - deq) < 0.1, f"Same-value round-trip failed: {orig} -> {deq}"

    print("  [PASS] test_asymmetric_quantization")


def test_per_channel_quantization() -> None:
    """Test per-channel quantization preserves per-row accuracy."""
    # Matrix where rows have very different scales
    matrix = [
        [100.0, 50.0, -100.0, -50.0],   # large values
        [0.01, 0.005, -0.01, -0.005],    # tiny values
        [1.0, 0.5, -1.0, -0.5],          # moderate values
    ]

    quantized, scales = quantize_per_channel(matrix, axis=0)

    assert len(scales) == 3, f"Expected 3 scale factors, got {len(scales)}"
    assert len(quantized) == 3
    assert len(quantized[0]) == 4

    # Per-channel scales should vary significantly
    assert scales[0] > scales[2] > scales[1], (
        f"Scales should reflect row magnitudes: {scales}"
    )

    # Round-trip each channel
    for i, (row, s) in enumerate(zip(quantized, scales)):
        deq_row = dequantize_symmetric(row, s)
        for j, (orig, deq) in enumerate(zip(matrix[i], deq_row)):
            rel_error = abs(orig - deq) / max(abs(orig), 1e-10)
            assert rel_error < 0.02, (
                f"Per-channel round-trip error too large at [{i}][{j}]: "
                f"{orig} -> {deq} (rel error: {rel_error:.4f})"
            )

    print("  [PASS] test_per_channel_quantization")


def test_quantized_matmul() -> None:
    """Test quantized matrix multiplication against float reference."""
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    B = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]

    expected = float_matmul(A, B)
    # expected = [[58.0, 64.0], [139.0, 154.0]]

    # Quantize A and B as flat tensors, then reshape
    A_flat = flatten_matrix(A)
    B_flat = flatten_matrix(B)

    A_q, scale_a = quantize_symmetric(A_flat)
    B_q, scale_b = quantize_symmetric(B_flat)

    A_q_mat = reshape_to_matrix(A_q, 2, 3)
    B_q_mat = reshape_to_matrix(B_q, 3, 2)

    result = quantized_matmul(A_q_mat, B_q_mat, scale_a, scale_b)

    # Check shape
    assert len(result) == 2 and len(result[0]) == 2, f"Wrong shape: {len(result)}x{len(result[0])}"

    # Check values with tolerance (quantization introduces error)
    for i in range(2):
        for j in range(2):
            rel_error = abs(result[i][j] - expected[i][j]) / abs(expected[i][j])
            assert rel_error < 0.05, (
                f"MatMul error at [{i}][{j}]: expected {expected[i][j]:.2f}, "
                f"got {result[i][j]:.2f} (rel error: {rel_error:.4f})"
            )

    print("  [PASS] test_quantized_matmul")


def test_quantized_matmul_realistic() -> None:
    """Test quantized matmul with realistic LLM-like weight distributions."""
    M, K, N = 16, 32, 16
    A = generate_weight_matrix(M, K, std=0.5, seed=100)
    B = generate_weight_matrix(K, N, std=0.5, seed=200)

    expected = float_matmul(A, B)

    # Quantize
    A_flat = flatten_matrix(A)
    B_flat = flatten_matrix(B)
    A_q, sa = quantize_symmetric(A_flat)
    B_q, sb = quantize_symmetric(B_flat)

    A_q_mat = reshape_to_matrix(A_q, M, K)
    B_q_mat = reshape_to_matrix(B_q, K, N)

    result = quantized_matmul(A_q_mat, B_q_mat, sa, sb)

    # Compute overall relative error
    total_error = 0.0
    total_magnitude = 0.0
    for i in range(M):
        for j in range(N):
            total_error += (result[i][j] - expected[i][j]) ** 2
            total_magnitude += expected[i][j] ** 2

    if total_magnitude > 0:
        relative_rmse = math.sqrt(total_error / total_magnitude)
        assert relative_rmse < 0.1, (
            f"Quantized matmul relative RMSE too high: {relative_rmse:.4f}"
        )
        print(f"    Relative RMSE: {relative_rmse:.6f}")

    print("  [PASS] test_quantized_matmul_realistic")


def test_error_analysis() -> None:
    """Test quantization error metrics."""
    tensor = generate_normal_weights(100, mean=0.0, std=1.0, seed=42)
    quantized, scale = quantize_symmetric(tensor)

    errors = quantization_error(tensor, quantized, scale, symmetric=True)

    assert "mse" in errors, "Missing 'mse' key"
    assert "max_error" in errors, "Missing 'max_error' key"
    assert "snr_db" in errors, "Missing 'snr_db' key"

    assert errors["mse"] >= 0, "MSE should be non-negative"
    assert errors["max_error"] >= 0, "Max error should be non-negative"
    assert errors["mse"] < 0.001, (
        f"MSE seems too high for INT8: {errors['mse']:.6f}"
    )
    assert errors["snr_db"] > 30, (
        f"SNR should be >30 dB for INT8 quantization of normal weights, "
        f"got {errors['snr_db']:.1f}"
    )

    print(f"    MSE: {errors['mse']:.8f}")
    print(f"    Max error: {errors['max_error']:.6f}")
    print(f"    SNR: {errors['snr_db']:.1f} dB")

    # Edge case: perfect quantization (e.g., all zeros)
    zeros = [0.0] * 10
    q_zeros, s_zeros = quantize_symmetric(zeros)
    err_zeros = quantization_error(zeros, q_zeros, s_zeros, symmetric=True)
    assert err_zeros["mse"] == 0.0, "All-zero MSE should be 0"

    print("  [PASS] test_error_analysis")


def test_symmetric_vs_asymmetric_for_relu() -> None:
    """Demonstrate that asymmetric is better for ReLU-like activations."""
    activations = generate_relu_activations(200, mean=1.0, std=0.5, seed=77)

    # Symmetric quantization
    q_sym, s_sym = quantize_symmetric(activations)
    err_sym = quantization_error(activations, q_sym, s_sym, symmetric=True)

    # Asymmetric quantization
    q_asym, s_asym, zp_asym = quantize_asymmetric(activations)
    err_asym = quantization_error(
        activations, q_asym, s_asym, zero_point=zp_asym, symmetric=False
    )

    print(f"    Symmetric SNR:  {err_sym['snr_db']:.1f} dB")
    print(f"    Asymmetric SNR: {err_asym['snr_db']:.1f} dB")

    # Asymmetric should be better for non-negative activations
    assert err_asym["snr_db"] > err_sym["snr_db"], (
        "Asymmetric should have higher SNR for ReLU activations"
    )

    print("  [PASS] test_symmetric_vs_asymmetric_for_relu")


def test_bits_parameter() -> None:
    """Test quantization at different bit widths."""
    tensor = generate_normal_weights(50, std=1.0, seed=55)

    snrs = {}
    for bits in [2, 4, 8, 16]:
        q, s = quantize_symmetric(tensor, bits=bits)
        err = quantization_error(tensor, q, s, symmetric=True)
        snrs[bits] = err["snr_db"]

        # Check quantized values are in range
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        assert all(qmin <= v <= qmax for v in q), (
            f"Quantized values out of range for {bits}-bit"
        )

    print(f"    SNR by bits: { {b: f'{s:.1f} dB' for b, s in snrs.items()} }")

    # More bits should give higher SNR
    assert snrs[4] > snrs[2], "4-bit should be better than 2-bit"
    assert snrs[8] > snrs[4], "8-bit should be better than 4-bit"
    assert snrs[16] > snrs[8], "16-bit should be better than 8-bit"

    print("  [PASS] test_bits_parameter")


def run_all_tests() -> None:
    """Run all test cases."""
    print("Running Quantization Engine tests...\n")

    test_symmetric_quantization()
    test_asymmetric_quantization()
    test_per_channel_quantization()
    test_quantized_matmul()
    test_quantized_matmul_realistic()
    test_error_analysis()
    test_symmetric_vs_asymmetric_for_relu()
    test_bits_parameter()

    print("\nAll tests passed!")


if __name__ == "__main__":
    run_all_tests()
