//! Vector quantization for RaBitQ.
//!
//! This module implements the 1-bit quantization scheme used in RaBitQ.
//! Each dimension is quantized to a single bit representing the sign
//! of the rotated vector component.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// A quantized vector using 1-bit per dimension.
///
/// The bits are packed into u64 words for efficient storage and SIMD operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedVector {
    /// Packed bits: bit i is 1 if the i-th component is positive after rotation.
    pub bits: Vec<u64>,
    /// Original L2 norm of the vector (before normalization).
    pub norm: f32,
    /// Sum of absolute values of rotated components (for error estimation).
    pub sum_abs: f32,
}

impl QuantizedVector {
    /// Quantize a rotated vector to 1-bit representation.
    ///
    /// # Arguments
    /// * `rotated` - The vector after orthogonal rotation
    /// * `original_norm` - The L2 norm of the original (non-rotated) vector
    pub fn from_rotated(rotated: &[f32], original_norm: f32) -> Self {
        let dim = rotated.len();
        let num_words = (dim + 63) / 64;
        let mut bits = vec![0u64; num_words];
        let mut sum_abs = 0.0f32;

        for (i, &val) in rotated.iter().enumerate() {
            sum_abs += val.abs();
            if val >= 0.0 {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                bits[word_idx] |= 1u64 << bit_idx;
            }
        }

        QuantizedVector {
            bits,
            norm: original_norm,
            sum_abs,
        }
    }

    /// Get the sign bit at a given index.
    #[inline]
    pub fn get_bit(&self, idx: usize) -> bool {
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        (self.bits[word_idx] >> bit_idx) & 1 == 1
    }

    /// Count the number of bits set to 1.
    #[inline]
    pub fn popcount(&self) -> u32 {
        self.bits.iter().map(|w| w.count_ones()).sum()
    }

    /// Compute the Hamming distance to another quantized vector.
    #[inline]
    pub fn hamming_distance(&self, other: &QuantizedVector) -> u32 {
        self.bits
            .iter()
            .zip(other.bits.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
}

/// Metadata for a subvector partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubvectorQuantized {
    /// Quantized bits for this subvector.
    pub bits: Vec<u64>,
    /// Dimension of this subvector.
    pub subdim: usize,
}

impl SubvectorQuantized {
    /// Create a quantized subvector from a slice of rotated values.
    pub fn from_rotated_slice(rotated: &[f32]) -> Self {
        let subdim = rotated.len();
        let num_words = (subdim + 63) / 64;
        let mut bits = vec![0u64; num_words];

        for (i, &val) in rotated.iter().enumerate() {
            if val >= 0.0 {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                bits[word_idx] |= 1u64 << bit_idx;
            }
        }

        SubvectorQuantized { bits, subdim }
    }

    /// Compute Hamming distance to another subvector.
    #[inline]
    pub fn hamming_distance(&self, other: &SubvectorQuantized) -> u32 {
        self.bits
            .iter()
            .zip(other.bits.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
}

/// Encode a floating point vector as signs (for asymmetric distance computation).
///
/// Returns a packed bit representation where bit i = 1 if v[i] >= 0.
#[inline]
pub fn encode_signs(v: &[f32]) -> Vec<u64> {
    let dim = v.len();
    let num_words = (dim + 63) / 64;
    let mut bits = vec![0u64; num_words];

    for (i, &val) in v.iter().enumerate() {
        if val >= 0.0 {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            bits[word_idx] |= 1u64 << bit_idx;
        }
    }

    bits
}

/// Compute the inner product estimator between a query vector and quantized database vector.
///
/// This implements the asymmetric distance estimation from RaBitQ:
/// The query is kept in full precision, while the database vector is quantized.
///
/// # Arguments
/// * `query_rotated` - Query vector after rotation (full precision)
/// * `db_quantized` - Database vector (quantized)
/// * `dim` - Vector dimension
///
/// # Returns
/// Estimated inner product <q, x> where x is the original database vector.
#[inline]
pub fn asymmetric_inner_product(
    query_rotated: &[f32],
    db_bits: &[u64],
    db_norm: f32,
    db_sum_abs: f32,
    dim: usize,
) -> f32 {
    // The RaBitQ estimator uses the fact that after rotation,
    // each component is approximately ±(norm / sqrt(d)) distributed.
    //
    // For the quantized vector b with signs s_i:
    //   x_i ≈ s_i * |x_rotated_i|
    //
    // The inner product estimate is:
    //   <q, x> ≈ sum_i q_i * s_i * (db_sum_abs / d)

    let scale = db_sum_abs / (dim as f32);
    let mut ip = 0.0f32;

    for (i, &q_val) in query_rotated.iter().enumerate() {
        let word_idx = i / 64;
        let bit_idx = i % 64;
        let sign_bit = (db_bits[word_idx] >> bit_idx) & 1;
        let sign = if sign_bit == 1 { 1.0 } else { -1.0 };
        ip += q_val * sign * scale;
    }

    // Scale by the ratio of norms for better estimation
    ip * (db_norm / db_sum_abs.max(1e-10))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_signs() {
        let v = vec![1.0, -2.0, 3.0, -4.0, 0.0, -0.1, 0.1, -0.5];
        let qv = QuantizedVector::from_rotated(&v, 1.0);

        assert!(qv.get_bit(0)); // 1.0 >= 0
        assert!(!qv.get_bit(1)); // -2.0 < 0
        assert!(qv.get_bit(2)); // 3.0 >= 0
        assert!(!qv.get_bit(3)); // -4.0 < 0
        assert!(qv.get_bit(4)); // 0.0 >= 0
        assert!(!qv.get_bit(5)); // -0.1 < 0
        assert!(qv.get_bit(6)); // 0.1 >= 0
        assert!(!qv.get_bit(7)); // -0.5 < 0
    }

    #[test]
    fn test_hamming_distance() {
        let v1 = vec![1.0, 1.0, 1.0, 1.0];
        let v2 = vec![1.0, -1.0, 1.0, -1.0];

        let q1 = QuantizedVector::from_rotated(&v1, 1.0);
        let q2 = QuantizedVector::from_rotated(&v2, 1.0);

        assert_eq!(q1.hamming_distance(&q2), 2);
    }

    #[test]
    fn test_popcount() {
        let v = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let qv = QuantizedVector::from_rotated(&v, 1.0);
        assert_eq!(qv.popcount(), 4);
    }
}
