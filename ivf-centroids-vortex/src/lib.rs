//! # IVF Centroids Vortex
//!
//! This crate demonstrates how to store IVF (Inverted File) index data inside Vortex files
//! with centroids embedded in the file structure. The approach uses a composite file format
//! that wraps a standard Vortex file with IVF-specific metadata.
//!
//! ## Overview
//!
//! IVF (Inverted File Index) is a popular technique for approximate nearest neighbor search.
//! It works by:
//! 1. Clustering vectors into K partitions using K-means
//! 2. Storing each vector with its assigned partition ID
//! 3. At query time, only searching the closest partitions to the query vector
//!
//! This crate shows how to embed the IVF centroids directly into a file alongside Vortex data,
//! allowing the index and data to be stored together in a single file.
//!
//! ## File Format
//!
//! The IVF Vortex file format consists of:
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │  IVF Header (20 bytes)                      │
//! │  ├── magic: "IVFV" (4 bytes)                │
//! │  ├── version: u32 (4 bytes)                 │
//! │  ├── vortex_offset: u64 (8 bytes)           │
//! │  └── vortex_length: u64 (8 bytes)           │ (removed, use centroids_offset - vortex_offset)
//! ├─────────────────────────────────────────────┤
//! │  Vortex Data (variable length)              │
//! │  └── StructArray with:                      │
//! │      ├── row_id: u64                        │
//! │      ├── vector: List<f32>                  │
//! │      └── ivf_partition_id: u32              │
//! ├─────────────────────────────────────────────┤
//! │  IVF Metadata                               │
//! │  ├── num_partitions: u32 (4 bytes)          │
//! │  ├── dimensions: u32 (4 bytes)              │
//! │  └── centroids: [f32; K * D]                │
//! ├─────────────────────────────────────────────┤
//! │  IVF Footer (16 bytes)                      │
//! │  ├── centroids_offset: u64 (8 bytes)        │
//! │  ├── footer_magic: "FVFI" (4 bytes)         │
//! │  └── checksum: u32 (4 bytes)                │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ivf_centroids_vortex::{IvfConfig, IvfVortexWriter, IvfVortexReader};
//!
//! // Create IVF configuration with centroids
//! let config = IvfConfig::new(num_partitions, dimensions, centroids)?;
//!
//! // Write IVF data to a file
//! let mut writer = IvfVortexWriter::new(config);
//! writer.add_vector(row_id, vector, partition_id)?;
//! writer.write_to_file("index.ivfvortex").await?;
//!
//! // Read IVF data from a file
//! let reader = IvfVortexReader::open("index.ivfvortex").await?;
//! let config = reader.config();
//! let nearest = config.find_nearest_partitions(&query, nprobe)?;
//! ```

mod file;
mod metadata;

pub use file::*;
pub use metadata::*;

use std::sync::Arc;

use vortex_array::arrays::{ListArray, PrimitiveArray, StructArray};
use vortex_array::validity::Validity;
use vortex_array::{ArrayRef, IntoArray};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, FieldName, FieldNames, Nullability, PType, StructFields};
use vortex_error::{VortexResult, vortex_bail};

/// Column names for the IVF data schema
pub const ROW_ID_COLUMN: &str = "row_id";
pub const VECTOR_COLUMN: &str = "vector";
pub const PARTITION_ID_COLUMN: &str = "ivf_partition_id";

/// Returns the DType for the IVF data schema with the given vector dimensions.
///
/// The schema is:
/// - `row_id`: u64 (non-nullable)
/// - `vector`: list of f32 (non-nullable)
/// - `ivf_partition_id`: u32 (non-nullable)
pub fn ivf_data_dtype() -> DType {
    let fields = StructFields::from_iter([
        (
            FieldName::from(ROW_ID_COLUMN),
            DType::Primitive(PType::U64, Nullability::NonNullable),
        ),
        (
            FieldName::from(VECTOR_COLUMN),
            DType::List(
                Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
                Nullability::NonNullable,
            ),
        ),
        (
            FieldName::from(PARTITION_ID_COLUMN),
            DType::Primitive(PType::U32, Nullability::NonNullable),
        ),
    ]);
    DType::Struct(fields, Nullability::NonNullable)
}

/// Creates a StructArray containing IVF vector data.
///
/// # Arguments
/// * `row_ids` - Row identifiers
/// * `vectors` - Vectors as a flat f32 buffer with shape [num_vectors, dimensions]
/// * `partition_ids` - IVF partition assignments for each vector
/// * `dimensions` - Number of dimensions per vector
pub fn create_ivf_data_array(
    row_ids: &[u64],
    vectors: &[f32],
    partition_ids: &[u32],
    dimensions: usize,
) -> VortexResult<ArrayRef> {
    let num_vectors = row_ids.len();

    if vectors.len() != num_vectors * dimensions {
        vortex_bail!(
            "vectors length {} does not match expected {} (num_vectors={} * dimensions={})",
            vectors.len(),
            num_vectors * dimensions,
            num_vectors,
            dimensions
        );
    }

    if partition_ids.len() != num_vectors {
        vortex_bail!(
            "partition_ids length {} does not match num_vectors {}",
            partition_ids.len(),
            num_vectors
        );
    }

    // Create row_id array
    let row_id_array = PrimitiveArray::new(
        Buffer::copy_from(row_ids),
        Validity::NonNullable,
    );

    // Create vector array as a list of f32 values
    let flat_values = PrimitiveArray::new(
        Buffer::copy_from(vectors),
        Validity::NonNullable,
    );

    // Create offsets for the list array (0, D, 2D, 3D, ...)
    let offsets: Vec<i64> = (0..=num_vectors)
        .map(|i| (i * dimensions) as i64)
        .collect();
    let offsets_array = PrimitiveArray::new(
        Buffer::copy_from(&offsets),
        Validity::NonNullable,
    );

    let vector_array = ListArray::try_new(
        flat_values.into_array(),
        offsets_array.into_array(),
        Validity::NonNullable,
    )?;

    // Create partition_id array
    let partition_id_array = PrimitiveArray::new(
        Buffer::copy_from(partition_ids),
        Validity::NonNullable,
    );

    // Create the struct array
    let struct_array = StructArray::try_new(
        FieldNames::from_iter([
            FieldName::from(ROW_ID_COLUMN),
            FieldName::from(VECTOR_COLUMN),
            FieldName::from(PARTITION_ID_COLUMN),
        ]),
        vec![
            row_id_array.into_array(),
            vector_array.into_array(),
            partition_id_array.into_array(),
        ],
        num_vectors,
        Validity::NonNullable,
    )?;

    Ok(struct_array.into_array())
}

/// Configuration for an IVF index.
#[derive(Debug, Clone, PartialEq)]
pub struct IvfConfig {
    /// Number of partitions (K in K-means)
    pub num_partitions: u32,
    /// Vector dimensions
    pub dimensions: u32,
    /// Centroids as a flat f32 buffer with shape [num_partitions, dimensions]
    pub centroids: Vec<f32>,
}

impl IvfConfig {
    /// Creates a new IVF configuration.
    pub fn new(num_partitions: u32, dimensions: u32, centroids: Vec<f32>) -> VortexResult<Self> {
        let expected_len = (num_partitions as usize) * (dimensions as usize);
        if centroids.len() != expected_len {
            vortex_bail!(
                "centroids length {} does not match expected {} (num_partitions={} * dimensions={})",
                centroids.len(),
                expected_len,
                num_partitions,
                dimensions
            );
        }

        Ok(Self {
            num_partitions,
            dimensions,
            centroids,
        })
    }

    /// Returns the centroid for a given partition.
    pub fn centroid(&self, partition: u32) -> Option<&[f32]> {
        if partition >= self.num_partitions {
            return None;
        }
        let start = (partition as usize) * (self.dimensions as usize);
        let end = start + (self.dimensions as usize);
        Some(&self.centroids[start..end])
    }

    /// Finds the nearest partition for a query vector using L2 distance.
    pub fn find_nearest_partition(&self, query: &[f32]) -> VortexResult<u32> {
        if query.len() != self.dimensions as usize {
            vortex_bail!(
                "query dimensions {} does not match expected {}",
                query.len(),
                self.dimensions
            );
        }

        let mut best_partition = 0;
        let mut best_distance = f32::MAX;

        for p in 0..self.num_partitions {
            let centroid = self.centroid(p).unwrap();
            let distance: f32 = query
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();

            if distance < best_distance {
                best_distance = distance;
                best_partition = p;
            }
        }

        Ok(best_partition)
    }

    /// Finds the N nearest partitions for a query vector using L2 distance.
    pub fn find_nearest_partitions(&self, query: &[f32], n: usize) -> VortexResult<Vec<u32>> {
        if query.len() != self.dimensions as usize {
            vortex_bail!(
                "query dimensions {} does not match expected {}",
                query.len(),
                self.dimensions
            );
        }

        let mut distances: Vec<(u32, f32)> = (0..self.num_partitions)
            .map(|p| {
                let centroid = self.centroid(p).unwrap();
                let distance: f32 = query
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                (p, distance)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(distances.into_iter().take(n).map(|(p, _)| p).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ivf_config() {
        // 3 partitions, 4 dimensions
        let centroids = vec![
            0.0, 0.0, 0.0, 0.0, // Partition 0
            1.0, 1.0, 1.0, 1.0, // Partition 1
            2.0, 2.0, 2.0, 2.0, // Partition 2
        ];

        let config = IvfConfig::new(3, 4, centroids).unwrap();

        assert_eq!(config.centroid(0).unwrap(), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(config.centroid(1).unwrap(), &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(config.centroid(2).unwrap(), &[2.0, 2.0, 2.0, 2.0]);
        assert!(config.centroid(3).is_none());
    }

    #[test]
    fn test_find_nearest_partition() {
        let centroids = vec![
            0.0, 0.0, 0.0, 0.0, // Partition 0
            1.0, 1.0, 1.0, 1.0, // Partition 1
            2.0, 2.0, 2.0, 2.0, // Partition 2
        ];

        let config = IvfConfig::new(3, 4, centroids).unwrap();

        // Query close to partition 0
        assert_eq!(
            config.find_nearest_partition(&[0.1, 0.1, 0.1, 0.1]).unwrap(),
            0
        );

        // Query close to partition 1
        assert_eq!(
            config.find_nearest_partition(&[0.9, 1.1, 0.9, 1.1]).unwrap(),
            1
        );

        // Query close to partition 2
        assert_eq!(
            config.find_nearest_partition(&[2.1, 1.9, 2.0, 2.0]).unwrap(),
            2
        );
    }

    #[test]
    fn test_find_nearest_partitions() {
        let centroids = vec![
            0.0, 0.0, 0.0, 0.0, // Partition 0
            1.0, 1.0, 1.0, 1.0, // Partition 1
            2.0, 2.0, 2.0, 2.0, // Partition 2
        ];

        let config = IvfConfig::new(3, 4, centroids).unwrap();

        // Query between partitions 0 and 1
        let nearest = config
            .find_nearest_partitions(&[0.6, 0.6, 0.6, 0.6], 2)
            .unwrap();
        assert_eq!(nearest.len(), 2);
        // Should be partition 0 and 1 in some order (0 is closer)
        assert!(nearest.contains(&0));
        assert!(nearest.contains(&1));
    }

    #[test]
    fn test_create_ivf_data_array() {
        let row_ids = vec![0u64, 1, 2];
        let vectors = vec![
            0.0f32, 0.1, 0.2, 0.3, // Vector 0
            1.0, 1.1, 1.2, 1.3, // Vector 1
            2.0, 2.1, 2.2, 2.3, // Vector 2
        ];
        let partition_ids = vec![0u32, 1, 2];

        let array = create_ivf_data_array(&row_ids, &vectors, &partition_ids, 4).unwrap();

        // Check the array type
        assert!(array.dtype().is_struct());
    }
}
