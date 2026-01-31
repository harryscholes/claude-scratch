//! Error types for RaBitQ.

use thiserror::Error;

/// Result type for RaBitQ operations.
pub type Result<T> = std::result::Result<T, RaBitQError>;

/// Errors that can occur during RaBitQ operations.
#[derive(Error, Debug)]
pub enum RaBitQError {
    /// The input vectors have inconsistent dimensions.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// No vectors were provided for indexing.
    #[error("cannot build index with empty vector set")]
    EmptyVectorSet,

    /// The number of subvectors doesn't evenly divide the dimension.
    #[error("dimension {dim} is not divisible by number of subvectors {num_subvectors}")]
    InvalidSubvectorCount { dim: usize, num_subvectors: usize },

    /// The requested k exceeds the number of indexed vectors.
    #[error("k={k} exceeds number of vectors {num_vectors}")]
    KTooLarge { k: usize, num_vectors: usize },

    /// Serialization or deserialization error.
    #[error("serialization error: {0}")]
    SerializationError(String),

    /// IO error during persistence operations.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Invalid vector (e.g., zero norm for cosine similarity).
    #[error("invalid vector: {0}")]
    InvalidVector(String),
}
