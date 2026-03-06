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


# =============================================================================
# Usage Example
# =============================================================================
if __name__ == "__main__":
    tensor = [1.0, -0.5, 0.3, -1.0, 0.0]
    quantized, scale = quantize_symmetric(tensor)
    print(f"Quantized: {quantized}")
    print(f"Scale: {scale}")
    dequantized = dequantize_symmetric(quantized, scale)
    print(f"Dequantized: {dequantized}")
