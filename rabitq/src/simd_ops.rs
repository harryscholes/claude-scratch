//! SIMD-accelerated operations for RaBitQ.
//!
//! Uses Rust's portable SIMD for cross-platform vectorized operations.

#![allow(dead_code)]

use std::simd::prelude::*;
use std::simd::{f32x8, u64x4, Simd};

/// Compute the dot product of two f32 slices using SIMD.
#[inline]
pub fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let lanes = 8;
    let chunks = n / lanes;

    let mut sum = f32x8::splat(0.0);

    for i in 0..chunks {
        let offset = i * lanes;
        let va = f32x8::from_slice(&a[offset..offset + lanes]);
        let vb = f32x8::from_slice(&b[offset..offset + lanes]);
        sum += va * vb;
    }

    let mut result = sum.reduce_sum();

    // Handle remainder
    for i in (chunks * lanes)..n {
        result += a[i] * b[i];
    }

    result
}

/// Compute the L2 norm squared of a vector using SIMD.
#[inline]
pub fn simd_norm_squared(v: &[f32]) -> f32 {
    let n = v.len();
    let lanes = 8;
    let chunks = n / lanes;

    let mut sum = f32x8::splat(0.0);

    for i in 0..chunks {
        let offset = i * lanes;
        let va = f32x8::from_slice(&v[offset..offset + lanes]);
        sum += va * va;
    }

    let mut result = sum.reduce_sum();

    // Handle remainder
    for i in (chunks * lanes)..n {
        result += v[i] * v[i];
    }

    result
}

/// Compute the Hamming distance between two bit vectors using SIMD.
#[inline]
pub fn simd_hamming_distance(a: &[u64], b: &[u64]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let lanes = 4;
    let chunks = n / lanes;

    let mut total: u64 = 0;

    for i in 0..chunks {
        let offset = i * lanes;
        let va = u64x4::from_slice(&a[offset..offset + lanes]);
        let vb = u64x4::from_slice(&b[offset..offset + lanes]);
        let xor = va ^ vb;

        // Count bits in each lane
        for j in 0..lanes {
            total += xor[j].count_ones() as u64;
        }
    }

    // Handle remainder
    for i in (chunks * lanes)..n {
        total += (a[i] ^ b[i]).count_ones() as u64;
    }

    total as u32
}

/// Compute sum of absolute values using SIMD.
#[inline]
pub fn simd_sum_abs(v: &[f32]) -> f32 {
    let n = v.len();
    let lanes = 8;
    let chunks = n / lanes;

    let mut sum = f32x8::splat(0.0);

    for i in 0..chunks {
        let offset = i * lanes;
        let va = f32x8::from_slice(&v[offset..offset + lanes]);
        sum += va.abs();
    }

    let mut result = sum.reduce_sum();

    // Handle remainder
    for i in (chunks * lanes)..n {
        result += v[i].abs();
    }

    result
}

/// SIMD-accelerated asymmetric inner product for RaBitQ.
///
/// Computes the estimated inner product between a full-precision query
/// and a quantized database vector.
#[inline]
pub fn simd_asymmetric_ip(
    query_rotated: &[f32],
    db_bits: &[u64],
    scale: f32,
    dim: usize,
) -> f32 {
    let mut ip = 0.0f32;
    let lanes = 8;
    let chunks = dim / lanes;

    // Process 8 dimensions at a time
    for chunk_idx in 0..chunks {
        let base = chunk_idx * lanes;

        // Load 8 query values
        let q_vec = f32x8::from_slice(&query_rotated[base..base + lanes]);

        // Extract 8 sign bits
        let mut signs = Simd::<f32, 8>::splat(0.0);
        for i in 0..8 {
            let actual_idx = base + i;
            let w_idx = actual_idx / 64;
            let b_idx = actual_idx % 64;
            let bit = (db_bits[w_idx] >> b_idx) & 1;
            signs[i] = if bit == 1 { 1.0 } else { -1.0 };
        }

        let contrib = q_vec * signs;
        ip += contrib.reduce_sum();
    }

    // Handle remainder
    for i in (chunks * lanes)..dim {
        let word_idx = i / 64;
        let bit_idx = i % 64;
        let sign_bit = (db_bits[word_idx] >> bit_idx) & 1;
        let sign = if sign_bit == 1 { 1.0 } else { -1.0 };
        ip += query_rotated[i] * sign;
    }

    ip * scale
}

/// Matrix-vector multiplication using SIMD.
///
/// Computes y = M * x where M is stored in row-major order.
#[inline]
pub fn simd_matvec(matrix: &[f32], x: &[f32], y: &mut [f32], dim: usize) {
    for i in 0..dim {
        let row_start = i * dim;
        y[i] = simd_dot_product(&matrix[row_start..row_start + dim], x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_dot_product() {
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..100).map(|i| (i * 2) as f32).collect();

        let simd_result = simd_dot_product(&a, &b);
        let scalar_result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!(
            (simd_result - scalar_result).abs() < 1e-3,
            "SIMD: {}, Scalar: {}",
            simd_result,
            scalar_result
        );
    }

    #[test]
    fn test_simd_norm_squared() {
        let v: Vec<f32> = (0..100).map(|i| i as f32 / 10.0).collect();

        let simd_result = simd_norm_squared(&v);
        let scalar_result: f32 = v.iter().map(|x| x * x).sum();

        assert!(
            (simd_result - scalar_result).abs() < 1e-3,
            "SIMD: {}, Scalar: {}",
            simd_result,
            scalar_result
        );
    }

    #[test]
    fn test_simd_hamming_distance() {
        let a = vec![0xFFFFFFFF_00000000u64, 0x0F0F0F0F_0F0F0F0Fu64];
        let b = vec![0x00000000_FFFFFFFFu64, 0xF0F0F0F0_F0F0F0F0u64];

        let simd_result = simd_hamming_distance(&a, &b);

        // First word: all 64 bits differ
        // Second word: each nibble differs by 4 bits, 16 nibbles = 64 bits
        assert_eq!(simd_result, 128);
    }

    #[test]
    fn test_simd_sum_abs() {
        let v: Vec<f32> = (-50..50).map(|i| i as f32).collect();

        let simd_result = simd_sum_abs(&v);
        let scalar_result: f32 = v.iter().map(|x| x.abs()).sum();

        assert!(
            (simd_result - scalar_result).abs() < 1e-3,
            "SIMD: {}, Scalar: {}",
            simd_result,
            scalar_result
        );
    }
}
