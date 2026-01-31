//! # RaBitQ
//!
//! Implementation of the RaBitQ algorithm for approximate nearest neighbor search.
//!
//! RaBitQ (Random Bit Quantization) quantizes high-dimensional vectors into compact
//! binary codes using random orthogonal transformations, enabling fast similarity search
//! with theoretical error bounds.
//!
//! ## Features
//!
//! - 1-bit per dimension quantization with random orthogonal rotation
//! - Subvector splitting for improved accuracy
//! - Support for Euclidean, Cosine, and Inner Product distance metrics
//! - Serialization support for index persistence
//!
//! ## Example
//!
//! ```
//! use rabitq::{RaBitQIndex, DistanceMetric};
//!
//! // Create some sample vectors
//! let vectors: Vec<Vec<f32>> = (0..1000)
//!     .map(|i| (0..128).map(|j| (i * j) as f32 / 1000.0).collect())
//!     .collect();
//!
//! // Build the index
//! let index = RaBitQIndex::build(&vectors, DistanceMetric::Euclidean, 4, 42).unwrap();
//!
//! // Query for nearest neighbors
//! let query: Vec<f32> = (0..128).map(|j| j as f32 / 100.0).collect();
//! let results = index.query(&query, 10).unwrap();
//!
//! println!("Top 10 nearest neighbors: {:?}", results);
//! ```

#![feature(portable_simd)]

mod distance;
mod error;
mod index;
mod orthogonal;
mod quantize;
mod simd_ops;

pub use distance::DistanceMetric;
pub use error::{RaBitQError, Result};
pub use index::RaBitQIndex;

#[cfg(test)]
mod tests;
