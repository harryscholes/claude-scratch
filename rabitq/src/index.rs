//! RaBitQ index for approximate nearest neighbor search.
//!
//! The index stores quantized representations of vectors along with
//! auxiliary data needed for asymmetric distance estimation.

use crate::distance::{self, DistanceMetric};
use crate::error::{RaBitQError, Result};
use crate::orthogonal::OrthogonalMatrix;
use crate::simd_ops;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// A quantized database vector with all necessary metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedEntry {
    /// Packed sign bits for each subvector.
    /// Outer vec: subvectors, Inner vec: u64 words for bits.
    pub subvector_bits: Vec<Vec<u64>>,
    /// L2 norm of the original vector.
    pub norm: f32,
    /// Sum of absolute values after rotation (per subvector).
    pub subvector_sum_abs: Vec<f32>,
    /// Centroid-adjusted norm component for distance estimation.
    pub norm_sq: f32,
}

/// RaBitQ index for approximate nearest neighbor search.
///
/// This index implements the RaBitQ algorithm with subvector splitting
/// for improved accuracy. Vectors are quantized to 1 bit per dimension
/// after random orthogonal rotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaBitQIndex {
    /// Random orthogonal matrices for each subvector.
    orthogonal_matrices: Vec<OrthogonalMatrix>,
    /// Quantized database entries.
    entries: Vec<QuantizedEntry>,
    /// Centroid of the database vectors.
    centroid: Vec<f32>,
    /// Original dimension of vectors.
    dim: usize,
    /// Number of subvectors.
    num_subvectors: usize,
    /// Dimension of each subvector.
    subdim: usize,
    /// Distance metric used for search.
    metric: DistanceMetric,
    /// Random seed used for reproducibility.
    seed: u64,
}

impl RaBitQIndex {
    /// Build a new RaBitQ index from a collection of vectors.
    ///
    /// # Arguments
    /// * `vectors` - Slice of vectors to index. All vectors must have the same dimension.
    /// * `metric` - Distance metric to use for search.
    /// * `num_subvectors` - Number of subvectors to split each vector into.
    ///   Must evenly divide the dimension. More subvectors = higher accuracy but slower.
    /// * `seed` - Random seed for reproducibility.
    ///
    /// # Returns
    /// A new RaBitQ index ready for queries.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The vector set is empty
    /// - Vectors have inconsistent dimensions
    /// - The dimension is not divisible by num_subvectors
    pub fn build(
        vectors: &[Vec<f32>],
        metric: DistanceMetric,
        num_subvectors: usize,
        seed: u64,
    ) -> Result<Self> {
        if vectors.is_empty() {
            return Err(RaBitQError::EmptyVectorSet);
        }

        let dim = vectors[0].len();

        // Validate all vectors have same dimension
        for v in vectors.iter() {
            if v.len() != dim {
                return Err(RaBitQError::DimensionMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
        }

        // Check subvector division
        if dim % num_subvectors != 0 {
            return Err(RaBitQError::InvalidSubvectorCount { dim, num_subvectors });
        }

        let subdim = dim / num_subvectors;

        // Compute centroid
        let mut centroid = vec![0.0f32; dim];
        for v in vectors.iter() {
            for (i, &val) in v.iter().enumerate() {
                centroid[i] += val;
            }
        }
        let n = vectors.len() as f32;
        for c in centroid.iter_mut() {
            *c /= n;
        }

        // Generate random orthogonal matrices for each subvector
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let orthogonal_matrices: Vec<OrthogonalMatrix> = (0..num_subvectors)
            .map(|_| OrthogonalMatrix::random(subdim, &mut rng))
            .collect();

        // Quantize all vectors
        let entries: Vec<QuantizedEntry> = vectors
            .iter()
            .map(|v| Self::quantize_vector(v, &centroid, &orthogonal_matrices, num_subvectors, subdim, metric))
            .collect();

        Ok(RaBitQIndex {
            orthogonal_matrices,
            entries,
            centroid,
            dim,
            num_subvectors,
            subdim,
            metric,
            seed,
        })
    }

    /// Quantize a single vector.
    fn quantize_vector(
        v: &[f32],
        centroid: &[f32],
        ortho_matrices: &[OrthogonalMatrix],
        num_subvectors: usize,
        subdim: usize,
        _metric: DistanceMetric,
    ) -> QuantizedEntry {
        // Center the vector (for Euclidean distance)
        let centered: Vec<f32> = v.iter().zip(centroid.iter()).map(|(a, b)| a - b).collect();

        // Compute norms
        let norm = distance::l2_norm(v);
        let norm_sq = simd_ops::simd_norm_squared(&centered);

        // Process each subvector
        let mut subvector_bits = Vec::with_capacity(num_subvectors);
        let mut subvector_sum_abs = Vec::with_capacity(num_subvectors);

        for s in 0..num_subvectors {
            let start = s * subdim;
            let end = start + subdim;
            let subvec = &centered[start..end];

            // Rotate the subvector
            let rotated = ortho_matrices[s].transform(subvec);

            // Compute sum of absolute values
            let sum_abs = simd_ops::simd_sum_abs(&rotated);
            subvector_sum_abs.push(sum_abs);

            // Quantize to sign bits
            let num_words = (subdim + 63) / 64;
            let mut bits = vec![0u64; num_words];
            for (i, &val) in rotated.iter().enumerate() {
                if val >= 0.0 {
                    let word_idx = i / 64;
                    let bit_idx = i % 64;
                    bits[word_idx] |= 1u64 << bit_idx;
                }
            }
            subvector_bits.push(bits);
        }

        QuantizedEntry {
            subvector_bits,
            norm,
            subvector_sum_abs,
            norm_sq,
        }
    }

    /// Query the index for the k nearest neighbors.
    ///
    /// # Arguments
    /// * `query` - Query vector. Must have the same dimension as indexed vectors.
    /// * `k` - Number of neighbors to return.
    ///
    /// # Returns
    /// Vector of (index, distance) pairs sorted by distance (ascending).
    pub fn query(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        if query.len() != self.dim {
            return Err(RaBitQError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        if k > self.entries.len() {
            return Err(RaBitQError::KTooLarge {
                k,
                num_vectors: self.entries.len(),
            });
        }

        // Prepare query based on metric
        let (query_centered, query_rotated_subvecs) = self.prepare_query(query);

        // Compute estimated distances to all entries
        let mut distances: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| {
                let dist = self.estimate_distance(
                    &query_centered,
                    &query_rotated_subvecs,
                    entry,
                    query,
                );
                (idx, dist)
            })
            .collect();

        // Sort by distance and return top k
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);

        Ok(distances)
    }

    /// Prepare a query vector for distance estimation.
    fn prepare_query(&self, query: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        // Center the query
        let centered: Vec<f32> = query
            .iter()
            .zip(self.centroid.iter())
            .map(|(a, b)| a - b)
            .collect();

        // Rotate each subvector
        let rotated_subvecs: Vec<Vec<f32>> = (0..self.num_subvectors)
            .map(|s| {
                let start = s * self.subdim;
                let end = start + self.subdim;
                self.orthogonal_matrices[s].transform(&centered[start..end])
            })
            .collect();

        (centered, rotated_subvecs)
    }

    /// Estimate the distance between a query and a database entry.
    fn estimate_distance(
        &self,
        query_centered: &[f32],
        query_rotated_subvecs: &[Vec<f32>],
        entry: &QuantizedEntry,
        original_query: &[f32],
    ) -> f32 {
        match self.metric {
            DistanceMetric::Euclidean => {
                self.estimate_euclidean(query_centered, query_rotated_subvecs, entry)
            }
            DistanceMetric::InnerProduct => {
                self.estimate_inner_product(query_rotated_subvecs, entry)
            }
            DistanceMetric::Cosine => {
                self.estimate_cosine(query_rotated_subvecs, entry, original_query)
            }
        }
    }

    /// Estimate Euclidean distance using RaBitQ.
    ///
    /// Uses the identity: ||q - x||^2 = ||q||^2 + ||x||^2 - 2<q, x>
    fn estimate_euclidean(
        &self,
        query_centered: &[f32],
        query_rotated_subvecs: &[Vec<f32>],
        entry: &QuantizedEntry,
    ) -> f32 {
        let query_norm_sq = simd_ops::simd_norm_squared(query_centered);

        // Estimate inner product
        let ip = self.estimate_ip_raw(query_rotated_subvecs, entry);

        // ||q - x||^2 = ||q||^2 + ||x||^2 - 2<q,x>
        let dist_sq = query_norm_sq + entry.norm_sq - 2.0 * ip;
        dist_sq.max(0.0).sqrt()
    }

    /// Estimate negative inner product (for IP metric).
    fn estimate_inner_product(
        &self,
        query_rotated_subvecs: &[Vec<f32>],
        entry: &QuantizedEntry,
    ) -> f32 {
        -self.estimate_ip_raw(query_rotated_subvecs, entry)
    }

    /// Estimate cosine distance.
    fn estimate_cosine(
        &self,
        query_rotated_subvecs: &[Vec<f32>],
        entry: &QuantizedEntry,
        original_query: &[f32],
    ) -> f32 {
        let query_norm = distance::l2_norm(original_query);
        if query_norm < 1e-10 || entry.norm < 1e-10 {
            return 1.0;
        }

        let ip = self.estimate_ip_raw(query_rotated_subvecs, entry);
        let cos_sim = ip / (query_norm * entry.norm);
        1.0 - cos_sim.clamp(-1.0, 1.0)
    }

    /// Raw inner product estimation using asymmetric quantization.
    fn estimate_ip_raw(
        &self,
        query_rotated_subvecs: &[Vec<f32>],
        entry: &QuantizedEntry,
    ) -> f32 {
        let mut total_ip = 0.0f32;

        for s in 0..self.num_subvectors {
            let query_sub = &query_rotated_subvecs[s];
            let db_bits = &entry.subvector_bits[s];
            let sum_abs = entry.subvector_sum_abs[s];

            // Scale factor based on the magnitude of the rotated subvector
            let scale = sum_abs / (self.subdim as f32);

            // Compute signed inner product
            let ip = simd_ops::simd_asymmetric_ip(query_sub, db_bits, scale, self.subdim);
            total_ip += ip;
        }

        total_ip
    }

    /// Get the number of indexed vectors.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the dimension of indexed vectors.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the distance metric used by this index.
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Get the number of subvectors.
    pub fn num_subvectors(&self) -> usize {
        self.num_subvectors
    }

    /// Save the index to a file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)
            .map_err(|e| RaBitQError::SerializationError(e.to_string()))?;
        Ok(())
    }

    /// Load an index from a file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let index: RaBitQIndex = bincode::deserialize_from(reader)
            .map_err(|e| RaBitQError::SerializationError(e.to_string()))?;
        Ok(index)
    }

    /// Compute exact distances (for recall testing).
    pub fn query_exact(&self, query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
        let mut distances: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(idx, v)| (idx, self.metric.compute(query, v)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);
        distances
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn generate_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect())
            .collect()
    }

    #[test]
    fn test_build_index() {
        let vectors = generate_random_vectors(100, 64, 42);
        let index = RaBitQIndex::build(&vectors, DistanceMetric::Euclidean, 4, 42).unwrap();

        assert_eq!(index.len(), 100);
        assert_eq!(index.dim(), 64);
        assert_eq!(index.num_subvectors(), 4);
    }

    #[test]
    fn test_query() {
        let vectors = generate_random_vectors(100, 64, 42);
        let index = RaBitQIndex::build(&vectors, DistanceMetric::Euclidean, 4, 42).unwrap();

        let query = &vectors[0];
        let results = index.query(query, 10).unwrap();

        assert_eq!(results.len(), 10);
        // The first result should ideally be the query itself (index 0)
        // with distance close to 0, but due to quantization this isn't guaranteed
    }

    #[test]
    fn test_dimension_mismatch() {
        let vectors = generate_random_vectors(100, 64, 42);
        let index = RaBitQIndex::build(&vectors, DistanceMetric::Euclidean, 4, 42).unwrap();

        let bad_query: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let result = index.query(&bad_query, 10);

        assert!(matches!(result, Err(RaBitQError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_save_load() {
        let vectors = generate_random_vectors(50, 32, 42);
        let index = RaBitQIndex::build(&vectors, DistanceMetric::Euclidean, 4, 42).unwrap();

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test_index.bin");

        index.save(&path).unwrap();
        let loaded = RaBitQIndex::load(&path).unwrap();

        assert_eq!(index.len(), loaded.len());
        assert_eq!(index.dim(), loaded.dim());
        assert_eq!(index.metric(), loaded.metric());

        // Query should give same results
        let query = &vectors[0];
        let results1 = index.query(query, 5).unwrap();
        let results2 = loaded.query(query, 5).unwrap();

        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert_eq!(r1.0, r2.0);
            assert!((r1.1 - r2.1).abs() < 1e-6);
        }
    }

    #[test]
    fn test_all_metrics() {
        let vectors = generate_random_vectors(100, 64, 42);

        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Cosine,
            DistanceMetric::InnerProduct,
        ] {
            let index = RaBitQIndex::build(&vectors, metric, 4, 42).unwrap();
            let query = &vectors[0];
            let results = index.query(query, 10).unwrap();
            assert_eq!(results.len(), 10);
        }
    }
}
