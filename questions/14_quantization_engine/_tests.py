"""Hidden tests for Question 14: Quantization Engine
Run: python questions/14_quantization_engine/_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import math
from starter import (
    quantize_symmetric, dequantize_symmetric,
    quantize_asymmetric, dequantize_asymmetric,
    quantize_per_channel,
    quantized_matmul,
    quantization_error,
    generate_normal_weights, generate_relu_activations,
    generate_weight_matrix, float_matmul,
    flatten_matrix, reshape_to_matrix,
)


def test_symmetric_quantization():
    """Test symmetric quantization and dequantization."""
    tensor = [1.0, -0.5, 0.3, -1.0, 0.0, 0.7]
    quantized, scale = quantize_symmetric(tensor)

    assert len(quantized) == len(tensor), "Output length mismatch"
    assert isinstance(scale, float), "Scale should be float"
    assert scale > 0, "Scale should be positive"
    assert quantized[0] == 127 or quantized[3] == -127, "Extreme values should map to +/-127"

    dequantized = dequantize_symmetric(quantized, scale)
    for orig, deq in zip(tensor, dequantized):
        assert abs(orig - deq) < 0.02, f"Round-trip error too large: {orig} -> {deq}"

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


def test_asymmetric_quantization():
    """Test asymmetric quantization for ReLU-like activations."""
    activations = [0.0, 0.5, 1.0, 1.5, 2.0, 0.1]
    quantized, scale, zp = quantize_asymmetric(activations)

    assert len(quantized) == len(activations), "Output length mismatch"
    assert isinstance(zp, int), "Zero point should be integer"
    assert 0 <= zp <= 255, "Zero point out of range for 8-bit"
    assert quantized[0] <= 5, f"Min value should map near 0, got {quantized[0]}"
    assert quantized[4] >= 250, f"Max value should map near 255, got {quantized[4]}"

    dequantized = dequantize_asymmetric(quantized, scale, zp)
    for orig, deq in zip(activations, dequantized):
        assert abs(orig - deq) < 0.02, f"Round-trip error too large: {orig} -> {deq}"

    # Edge case: all same value
    same = [5.0, 5.0, 5.0]
    q_same, s_same, zp_same = quantize_asymmetric(same)
    d_same = dequantize_asymmetric(q_same, s_same, zp_same)
    for orig, deq in zip(same, d_same):
        assert abs(orig - deq) < 0.1, f"Same-value round-trip failed: {orig} -> {deq}"

    print("  [PASS] test_asymmetric_quantization")


def test_per_channel_quantization():
    """Test per-channel quantization preserves per-row accuracy."""
    matrix = [
        [100.0, 50.0, -100.0, -50.0],
        [0.01, 0.005, -0.01, -0.005],
        [1.0, 0.5, -1.0, -0.5],
    ]

    quantized, scales = quantize_per_channel(matrix, axis=0)

    assert len(scales) == 3, f"Expected 3 scale factors, got {len(scales)}"
    assert len(quantized) == 3
    assert len(quantized[0]) == 4
    assert scales[0] > scales[2] > scales[1], f"Scales should reflect row magnitudes: {scales}"

    for i, (row, s) in enumerate(zip(quantized, scales)):
        deq_row = dequantize_symmetric(row, s)
        for j, (orig, deq) in enumerate(zip(matrix[i], deq_row)):
            rel_error = abs(orig - deq) / max(abs(orig), 1e-10)
            assert rel_error < 0.02, (
                f"Per-channel round-trip error too large at [{i}][{j}]: "
                f"{orig} -> {deq} (rel error: {rel_error:.4f})"
            )

    print("  [PASS] test_per_channel_quantization")


def test_quantized_matmul():
    """Test quantized matrix multiplication against float reference."""
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    B = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]

    expected = float_matmul(A, B)

    A_flat = flatten_matrix(A)
    B_flat = flatten_matrix(B)
    A_q, scale_a = quantize_symmetric(A_flat)
    B_q, scale_b = quantize_symmetric(B_flat)
    A_q_mat = reshape_to_matrix(A_q, 2, 3)
    B_q_mat = reshape_to_matrix(B_q, 3, 2)

    result = quantized_matmul(A_q_mat, B_q_mat, scale_a, scale_b)

    assert len(result) == 2 and len(result[0]) == 2, f"Wrong shape: {len(result)}x{len(result[0])}"

    for i in range(2):
        for j in range(2):
            rel_error = abs(result[i][j] - expected[i][j]) / abs(expected[i][j])
            assert rel_error < 0.05, (
                f"MatMul error at [{i}][{j}]: expected {expected[i][j]:.2f}, "
                f"got {result[i][j]:.2f} (rel error: {rel_error:.4f})"
            )

    print("  [PASS] test_quantized_matmul")


def test_quantized_matmul_realistic():
    """Test quantized matmul with realistic LLM-like weight distributions."""
    M, K, N = 16, 32, 16
    A = generate_weight_matrix(M, K, std=0.5, seed=100)
    B = generate_weight_matrix(K, N, std=0.5, seed=200)
    expected = float_matmul(A, B)

    A_flat = flatten_matrix(A)
    B_flat = flatten_matrix(B)
    A_q, sa = quantize_symmetric(A_flat)
    B_q, sb = quantize_symmetric(B_flat)
    A_q_mat = reshape_to_matrix(A_q, M, K)
    B_q_mat = reshape_to_matrix(B_q, K, N)

    result = quantized_matmul(A_q_mat, B_q_mat, sa, sb)

    total_error = 0.0
    total_magnitude = 0.0
    for i in range(M):
        for j in range(N):
            total_error += (result[i][j] - expected[i][j]) ** 2
            total_magnitude += expected[i][j] ** 2

    if total_magnitude > 0:
        relative_rmse = math.sqrt(total_error / total_magnitude)
        assert relative_rmse < 0.1, f"Quantized matmul relative RMSE too high: {relative_rmse:.4f}"
        print(f"    Relative RMSE: {relative_rmse:.6f}")

    print("  [PASS] test_quantized_matmul_realistic")


def test_error_analysis():
    """Test quantization error metrics."""
    tensor = generate_normal_weights(100, mean=0.0, std=1.0, seed=42)
    quantized, scale = quantize_symmetric(tensor)

    errors = quantization_error(tensor, quantized, scale, symmetric=True)

    assert "mse" in errors, "Missing 'mse' key"
    assert "max_error" in errors, "Missing 'max_error' key"
    assert "snr_db" in errors, "Missing 'snr_db' key"
    assert errors["mse"] >= 0, "MSE should be non-negative"
    assert errors["max_error"] >= 0, "Max error should be non-negative"
    assert errors["mse"] < 0.001, f"MSE seems too high for INT8: {errors['mse']:.6f}"
    assert errors["snr_db"] > 30, (
        f"SNR should be >30 dB for INT8 quantization of normal weights, got {errors['snr_db']:.1f}"
    )

    print(f"    MSE: {errors['mse']:.8f}")
    print(f"    Max error: {errors['max_error']:.6f}")
    print(f"    SNR: {errors['snr_db']:.1f} dB")

    # Edge case: perfect quantization
    zeros = [0.0] * 10
    q_zeros, s_zeros = quantize_symmetric(zeros)
    err_zeros = quantization_error(zeros, q_zeros, s_zeros, symmetric=True)
    assert err_zeros["mse"] == 0.0, "All-zero MSE should be 0"

    print("  [PASS] test_error_analysis")


def test_symmetric_vs_asymmetric_for_relu():
    """Demonstrate that asymmetric is better for ReLU-like activations."""
    activations = generate_relu_activations(200, mean=1.0, std=0.5, seed=77)

    q_sym, s_sym = quantize_symmetric(activations)
    err_sym = quantization_error(activations, q_sym, s_sym, symmetric=True)

    q_asym, s_asym, zp_asym = quantize_asymmetric(activations)
    err_asym = quantization_error(
        activations, q_asym, s_asym, zero_point=zp_asym, symmetric=False
    )

    print(f"    Symmetric SNR:  {err_sym['snr_db']:.1f} dB")
    print(f"    Asymmetric SNR: {err_asym['snr_db']:.1f} dB")

    assert err_asym["snr_db"] > err_sym["snr_db"], (
        "Asymmetric should have higher SNR for ReLU activations"
    )

    print("  [PASS] test_symmetric_vs_asymmetric_for_relu")


def test_bits_parameter():
    """Test quantization at different bit widths."""
    tensor = generate_normal_weights(50, std=1.0, seed=55)

    snrs = {}
    for bits in [2, 4, 8, 16]:
        q, s = quantize_symmetric(tensor, bits=bits)
        err = quantization_error(tensor, q, s, symmetric=True)
        snrs[bits] = err["snr_db"]

        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        assert all(qmin <= v <= qmax for v in q), f"Quantized values out of range for {bits}-bit"

    print(f"    SNR by bits: { {b: f'{s:.1f} dB' for b, s in snrs.items()} }")

    assert snrs[4] > snrs[2], "4-bit should be better than 2-bit"
    assert snrs[8] > snrs[4], "8-bit should be better than 4-bit"
    assert snrs[16] > snrs[8], "16-bit should be better than 8-bit"

    print("  [PASS] test_bits_parameter")


def run_tests():
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
    run_tests()
