//! Distance metrics for similarity search.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Distance metric for nearest neighbor search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance: sqrt(sum((a_i - b_i)^2))
    Euclidean,
    /// Cosine distance: 1 - (a · b) / (||a|| * ||b||)
    Cosine,
    /// Inner product (negative for max similarity): -a · b
    InnerProduct,
}

impl DistanceMetric {
    /// Compute the exact distance between two vectors.
    #[inline]
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        match self {
            DistanceMetric::Euclidean => euclidean_distance(a, b),
            DistanceMetric::Cosine => cosine_distance(a, b),
            DistanceMetric::InnerProduct => negative_inner_product(a, b),
        }
    }

    /// Whether lower values indicate more similar vectors.
    #[inline]
    pub fn lower_is_better(&self) -> bool {
        true // All our metrics are defined so lower = more similar
    }
}

/// Compute Euclidean (L2) distance between two vectors.
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let sum_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    sum_sq.sqrt()
}

/// Compute squared Euclidean distance (avoids sqrt for comparisons).
#[inline]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Compute cosine distance: 1 - cosine_similarity.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Maximum distance for zero vectors
    }

    1.0 - (dot / (norm_a * norm_b))
}

/// Compute negative inner product (so lower = more similar).
#[inline]
pub fn negative_inner_product(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    -dot
}

/// Compute the L2 norm of a vector.
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Normalize a vector in place.
#[inline]
pub fn normalize_inplace(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Compute dot product between two vectors.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - std::f32::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let dist = cosine_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6); // Orthogonal vectors

        let c = vec![1.0, 1.0];
        let d = vec![1.0, 1.0];
        let dist2 = cosine_distance(&c, &d);
        assert!(dist2.abs() < 1e-6); // Same direction
    }

    #[test]
    fn test_inner_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let ip = negative_inner_product(&a, &b);
        assert!((ip - (-32.0)).abs() < 1e-6);
    }
}
