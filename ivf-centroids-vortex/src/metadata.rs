//! IVF Metadata Serialization
//!
//! This module handles serialization and deserialization of IVF metadata,
//! including centroids and configuration parameters.

use vortex_error::{VortexResult, vortex_bail};

use crate::IvfConfig;

/// Magic bytes for IVF metadata: "IVFM"
pub const IVF_METADATA_MAGIC: [u8; 4] = *b"IVFM";

/// Version of the IVF metadata format
pub const IVF_METADATA_VERSION: u32 = 1;

/// IVF metadata that can be serialized and embedded in a file.
///
/// Binary format:
/// ```text
/// ┌──────────────────────────────────────────┐
/// │ magic: [u8; 4] = "IVFM"                  │
/// │ version: u32 (little-endian)             │
/// │ num_partitions: u32 (little-endian)      │
/// │ dimensions: u32 (little-endian)          │
/// │ centroids: [f32; num_partitions * dims]  │
/// └──────────────────────────────────────────┘
/// ```
#[derive(Debug, Clone)]
pub struct IvfMetadata {
    /// IVF index configuration
    pub config: IvfConfig,
}

impl IvfMetadata {
    /// Creates a new IVF metadata from the configuration.
    pub fn new(config: IvfConfig) -> Self {
        Self { config }
    }

    /// Returns the size of the serialized metadata in bytes.
    pub fn serialized_size(&self) -> usize {
        // magic + version + num_partitions + dimensions + centroids
        4 + 4 + 4 + 4 + (self.config.centroids.len() * 4)
    }

    /// Serializes the metadata to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.serialized_size());

        // Magic bytes
        bytes.extend_from_slice(&IVF_METADATA_MAGIC);

        // Version
        bytes.extend_from_slice(&IVF_METADATA_VERSION.to_le_bytes());

        // Number of partitions
        bytes.extend_from_slice(&self.config.num_partitions.to_le_bytes());

        // Dimensions
        bytes.extend_from_slice(&self.config.dimensions.to_le_bytes());

        // Centroids (as raw f32 bytes)
        for &c in &self.config.centroids {
            bytes.extend_from_slice(&c.to_le_bytes());
        }

        bytes
    }

    /// Deserializes metadata from bytes.
    pub fn deserialize(bytes: &[u8]) -> VortexResult<Self> {
        if bytes.len() < 16 {
            vortex_bail!(
                "IVF metadata too short: expected at least 16 bytes, got {}",
                bytes.len()
            );
        }

        // Check magic bytes
        let magic = &bytes[0..4];
        if magic != IVF_METADATA_MAGIC {
            vortex_bail!(
                "Invalid IVF metadata magic: expected {:?}, got {:?}",
                IVF_METADATA_MAGIC,
                magic
            );
        }

        // Check version
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version != IVF_METADATA_VERSION {
            vortex_bail!(
                "Unsupported IVF metadata version: expected {}, got {}",
                IVF_METADATA_VERSION,
                version
            );
        }

        // Read num_partitions
        let num_partitions = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);

        // Read dimensions
        let dimensions = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);

        // Calculate expected centroids length
        let centroids_count = (num_partitions as usize) * (dimensions as usize);
        let expected_len = 16 + centroids_count * 4;

        if bytes.len() < expected_len {
            vortex_bail!(
                "IVF metadata too short: expected {} bytes, got {}",
                expected_len,
                bytes.len()
            );
        }

        // Read centroids
        let centroids: Vec<f32> = bytes[16..]
            .chunks_exact(4)
            .take(centroids_count)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let config = IvfConfig::new(num_partitions, dimensions, centroids)?;
        Ok(Self { config })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_roundtrip() {
        let centroids = vec![
            0.0, 0.1, 0.2, 0.3, // Partition 0
            1.0, 1.1, 1.2, 1.3, // Partition 1
            2.0, 2.1, 2.2, 2.3, // Partition 2
        ];

        let config = IvfConfig::new(3, 4, centroids.clone()).unwrap();
        let metadata = IvfMetadata::new(config);

        let bytes = metadata.serialize();
        assert_eq!(bytes.len(), metadata.serialized_size());

        let deserialized = IvfMetadata::deserialize(&bytes).unwrap();

        assert_eq!(deserialized.config.num_partitions, 3);
        assert_eq!(deserialized.config.dimensions, 4);
        assert_eq!(deserialized.config.centroids, centroids);
    }

    #[test]
    fn test_metadata_invalid_magic() {
        let bytes = b"XXXX\x01\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00";
        let result = IvfMetadata::deserialize(bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_metadata_invalid_version() {
        let bytes = b"IVFM\xFF\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00";
        let result = IvfMetadata::deserialize(bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_metadata_too_short() {
        let bytes = b"IVFM\x01\x00\x00\x00";
        let result = IvfMetadata::deserialize(bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialized_size() {
        let centroids = vec![0.0; 12]; // 3 partitions * 4 dimensions
        let config = IvfConfig::new(3, 4, centroids).unwrap();
        let metadata = IvfMetadata::new(config);

        // 4 (magic) + 4 (version) + 4 (num_partitions) + 4 (dimensions) + 48 (12 * 4 bytes)
        assert_eq!(metadata.serialized_size(), 64);
    }
}
