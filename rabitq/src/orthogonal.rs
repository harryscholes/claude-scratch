//! Random orthogonal matrix generation using QR decomposition.
//!
//! RaBitQ requires random orthogonal transformations to ensure that the
//! quantization error is uniformly distributed across all dimensions.

#![allow(dead_code)]

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

/// A random orthogonal matrix for vector transformation.
///
/// The matrix is generated using QR decomposition of a random Gaussian matrix.
/// This ensures uniform distribution over the orthogonal group O(n).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrthogonalMatrix {
    /// The orthogonal matrix stored in row-major order.
    /// Shape: (dim, dim)
    pub data: Vec<f32>,
    /// Dimension of the square matrix.
    pub dim: usize,
}

impl OrthogonalMatrix {
    /// Generate a random orthogonal matrix of given dimension.
    ///
    /// Uses the QR decomposition method: generate a random Gaussian matrix,
    /// then compute its QR decomposition. Q is uniformly distributed over O(n).
    pub fn random<R: Rng>(dim: usize, rng: &mut R) -> Self {
        let normal = StandardNormal;

        // Generate random Gaussian matrix
        let mut a: Vec<f32> = (0..dim * dim).map(|_| normal.sample(rng)).collect();

        // Perform QR decomposition using modified Gram-Schmidt
        // This gives us an orthogonal matrix Q
        let q = gram_schmidt_qr(&mut a, dim);

        OrthogonalMatrix { data: q, dim }
    }

    /// Transform a vector by multiplying with the orthogonal matrix.
    ///
    /// Computes: y = Q * x
    #[inline]
    pub fn transform(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.dim);
        let mut result = vec![0.0; self.dim];
        self.transform_into(x, &mut result);
        result
    }

    /// Transform a vector into a pre-allocated buffer.
    #[inline]
    pub fn transform_into(&self, x: &[f32], result: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(result.len(), self.dim);

        for i in 0..self.dim {
            let row_start = i * self.dim;
            let mut sum = 0.0;
            for j in 0..self.dim {
                sum += self.data[row_start + j] * x[j];
            }
            result[i] = sum;
        }
    }

    /// Transform a vector by multiplying with the transpose (inverse) of the matrix.
    ///
    /// Computes: y = Q^T * x = Q^{-1} * x
    #[inline]
    pub fn inverse_transform(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.dim);
        let mut result = vec![0.0; self.dim];

        for i in 0..self.dim {
            let mut sum = 0.0;
            for j in 0..self.dim {
                // Transpose: access column i of row j
                sum += self.data[j * self.dim + i] * x[j];
            }
            result[i] = sum;
        }
        result
    }
}

/// Perform QR decomposition using modified Gram-Schmidt and return Q.
///
/// The input matrix `a` is in row-major order (dim x dim).
/// Returns Q, also in row-major order.
fn gram_schmidt_qr(a: &mut [f32], dim: usize) -> Vec<f32> {
    let mut q = vec![0.0; dim * dim];

    // Work column by column
    for j in 0..dim {
        // Copy column j of A to column j of Q
        for i in 0..dim {
            q[i * dim + j] = a[i * dim + j];
        }

        // Orthogonalize against previous columns
        for k in 0..j {
            // Compute dot product of column k and column j
            let mut dot = 0.0;
            for i in 0..dim {
                dot += q[i * dim + k] * q[i * dim + j];
            }

            // Subtract projection
            for i in 0..dim {
                q[i * dim + j] -= dot * q[i * dim + k];
            }
        }

        // Normalize column j
        let mut norm = 0.0;
        for i in 0..dim {
            norm += q[i * dim + j] * q[i * dim + j];
        }
        norm = norm.sqrt();

        if norm > 1e-10 {
            for i in 0..dim {
                q[i * dim + j] /= norm;
            }
        }
    }

    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_orthogonality() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dim = 64;
        let q = OrthogonalMatrix::random(dim, &mut rng);

        // Check Q * Q^T = I (orthogonality)
        for i in 0..dim {
            for j in 0..dim {
                let mut dot = 0.0;
                for k in 0..dim {
                    dot += q.data[i * dim + k] * q.data[j * dim + k];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-5,
                    "Q*Q^T[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_preserves_norm() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dim = 32;
        let q = OrthogonalMatrix::random(dim, &mut rng);

        let x: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let y = q.transform(&x);

        let norm_x: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_y: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();

        assert!(
            (norm_x - norm_y).abs() < 1e-4,
            "Norm not preserved: {} vs {}",
            norm_x,
            norm_y
        );
    }

    #[test]
    fn test_inverse_transform() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dim = 16;
        let q = OrthogonalMatrix::random(dim, &mut rng);

        let x: Vec<f32> = (1..=dim).map(|i| (i as f32) / 10.0).collect();
        let y = q.transform(&x);
        let x_recovered = q.inverse_transform(&y);

        for i in 0..dim {
            assert!(
                (x[i] - x_recovered[i]).abs() < 1e-4,
                "Inverse failed at {}: {} vs {}",
                i,
                x[i],
                x_recovered[i]
            );
        }
    }
}
