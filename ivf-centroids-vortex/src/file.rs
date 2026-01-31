//! IVF Vortex File Format
//!
//! This module implements reading and writing IVF Vortex files, which combine
//! standard Vortex data with IVF centroid metadata.
//!
//! ## File Structure
//!
//! The file format wraps a Vortex file with IVF-specific header and footer:
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │  IVF Header (16 bytes)                      │
//! │  ├── magic: "IVFV" (4 bytes)                │
//! │  ├── version: u32 (4 bytes)                 │
//! │  └── vortex_offset: u64 (8 bytes)           │
//! ├─────────────────────────────────────────────┤
//! │  Vortex Data (variable length)              │
//! ├─────────────────────────────────────────────┤
//! │  IVF Metadata (variable length)             │
//! ├─────────────────────────────────────────────┤
//! │  IVF Footer (16 bytes)                      │
//! │  ├── centroids_offset: u64 (8 bytes)        │
//! │  ├── footer_magic: "FVFI" (4 bytes)         │
//! │  └── checksum: u32 (4 bytes)                │
//! └─────────────────────────────────────────────┘
//! ```

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use vortex_array::ArrayRef;
use vortex_error::{VortexResult, vortex_bail};

use crate::metadata::IvfMetadata;
use crate::{IvfConfig, create_ivf_data_array};

/// Magic bytes for IVF Vortex file header: "IVFV"
pub const IVF_FILE_MAGIC: [u8; 4] = *b"IVFV";

/// Magic bytes for IVF Vortex file footer: "FVFI"
pub const IVF_FOOTER_MAGIC: [u8; 4] = *b"FVFI";

/// Version of the IVF Vortex file format
pub const IVF_FILE_VERSION: u32 = 1;

/// Size of the IVF file header in bytes
pub const IVF_HEADER_SIZE: usize = 16;

/// Size of the IVF file footer in bytes
pub const IVF_FOOTER_SIZE: usize = 16;

/// A single vector record for IVF indexing.
#[derive(Debug, Clone)]
pub struct IvfVectorRecord {
    /// Unique row identifier
    pub row_id: u64,
    /// The vector data
    pub vector: Vec<f32>,
    /// Assigned IVF partition ID
    pub partition_id: u32,
}

/// Writer for IVF Vortex files.
///
/// This writer accumulates vector data and writes it along with IVF centroids
/// to a composite file format.
#[derive(Debug)]
pub struct IvfVortexWriter {
    /// IVF configuration including centroids
    config: IvfConfig,
    /// Accumulated vector records
    records: Vec<IvfVectorRecord>,
}

impl IvfVortexWriter {
    /// Creates a new IVF Vortex writer with the given configuration.
    pub fn new(config: IvfConfig) -> Self {
        Self {
            config,
            records: Vec::new(),
        }
    }

    /// Returns the IVF configuration.
    pub fn config(&self) -> &IvfConfig {
        &self.config
    }

    /// Returns the number of vectors added.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns true if no vectors have been added.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Adds a vector with an explicit partition assignment.
    pub fn add_vector(
        &mut self,
        row_id: u64,
        vector: Vec<f32>,
        partition_id: u32,
    ) -> VortexResult<()> {
        if vector.len() != self.config.dimensions as usize {
            vortex_bail!(
                "Vector dimension {} does not match expected {}",
                vector.len(),
                self.config.dimensions
            );
        }

        if partition_id >= self.config.num_partitions {
            vortex_bail!(
                "Partition ID {} exceeds num_partitions {}",
                partition_id,
                self.config.num_partitions
            );
        }

        self.records.push(IvfVectorRecord {
            row_id,
            vector,
            partition_id,
        });

        Ok(())
    }

    /// Adds a vector and automatically assigns it to the nearest partition.
    pub fn add_vector_auto_partition(&mut self, row_id: u64, vector: Vec<f32>) -> VortexResult<()> {
        let partition_id = self.config.find_nearest_partition(&vector)?;
        self.add_vector(row_id, vector, partition_id)
    }

    /// Builds the Vortex array from accumulated records.
    pub fn build_array(&self) -> VortexResult<ArrayRef> {
        let row_ids: Vec<u64> = self.records.iter().map(|r| r.row_id).collect();
        let vectors: Vec<f32> = self
            .records
            .iter()
            .flat_map(|r| r.vector.iter().copied())
            .collect();
        let partition_ids: Vec<u32> = self.records.iter().map(|r| r.partition_id).collect();

        create_ivf_data_array(
            &row_ids,
            &vectors,
            &partition_ids,
            self.config.dimensions as usize,
        )
    }

    /// Writes the IVF Vortex file to a writer.
    ///
    /// This writes:
    /// 1. IVF header
    /// 2. Vortex data (as IPC format for simplicity in this demo)
    /// 3. IVF metadata (centroids)
    /// 4. IVF footer
    pub fn write<W: Write + Seek>(&self, writer: &mut W) -> VortexResult<IvfWriteSummary> {
        // Write header
        let header = IvfHeader {
            magic: IVF_FILE_MAGIC,
            version: IVF_FILE_VERSION,
            vortex_offset: IVF_HEADER_SIZE as u64,
        };
        header.write(writer)?;

        // Build the Vortex array
        let array = self.build_array()?;

        // Write Vortex data as serialized bytes
        // For this demonstration, we serialize the array metadata and data
        let vortex_data = self.serialize_vortex_data(&array)?;
        writer.write_all(&vortex_data)?;

        let centroids_offset = writer.stream_position()?;

        // Write IVF metadata
        let metadata = IvfMetadata::new(self.config.clone());
        let metadata_bytes = metadata.serialize();
        writer.write_all(&metadata_bytes)?;

        // Calculate checksum (simple sum of all metadata bytes)
        let checksum: u32 = metadata_bytes.iter().map(|&b| b as u32).sum();

        // Write footer
        let footer = IvfFooter {
            centroids_offset,
            magic: IVF_FOOTER_MAGIC,
            checksum,
        };
        footer.write(writer)?;

        let total_size = writer.stream_position()?;

        Ok(IvfWriteSummary {
            total_size,
            vortex_data_size: vortex_data.len() as u64,
            metadata_size: metadata_bytes.len() as u64,
            num_vectors: self.records.len(),
            num_partitions: self.config.num_partitions as usize,
        })
    }

    /// Writes the IVF Vortex file to a path.
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> VortexResult<IvfWriteSummary> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        self.write(&mut writer)
    }

    /// Serializes Vortex array data.
    ///
    /// In a full implementation, this would use Vortex's file writer.
    /// For this demonstration, we serialize a simple representation.
    fn serialize_vortex_data(&self, _array: &ArrayRef) -> VortexResult<Vec<u8>> {
        // Serialize the raw data in a simple format for demonstration
        // Format: num_vectors (u64) + row_ids + vectors + partition_ids
        let num_vectors = self.records.len() as u64;
        let dimensions = self.config.dimensions as usize;

        let mut data = Vec::new();

        // Number of vectors
        data.extend_from_slice(&num_vectors.to_le_bytes());

        // Dimensions
        data.extend_from_slice(&(dimensions as u64).to_le_bytes());

        // Row IDs
        for record in &self.records {
            data.extend_from_slice(&record.row_id.to_le_bytes());
        }

        // Vectors (flattened)
        for record in &self.records {
            for &v in &record.vector {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }

        // Partition IDs
        for record in &self.records {
            data.extend_from_slice(&record.partition_id.to_le_bytes());
        }

        Ok(data)
    }
}

/// Reader for IVF Vortex files.
#[derive(Debug)]
pub struct IvfVortexReader {
    /// IVF configuration including centroids
    config: IvfConfig,
    /// Vector records
    records: Vec<IvfVectorRecord>,
}

impl IvfVortexReader {
    /// Opens an IVF Vortex file from a reader.
    pub fn read<R: Read + Seek>(reader: &mut R) -> VortexResult<Self> {
        // Read and validate header
        let header = IvfHeader::read(reader)?;

        // Read footer (at end of file)
        reader.seek(SeekFrom::End(-(IVF_FOOTER_SIZE as i64)))?;
        let footer = IvfFooter::read(reader)?;

        // Read IVF metadata
        reader.seek(SeekFrom::Start(footer.centroids_offset))?;
        let _metadata_size = reader.stream_position()? as usize;
        let file_end = reader.seek(SeekFrom::End(-(IVF_FOOTER_SIZE as i64)))? as usize;
        let metadata_len = file_end - footer.centroids_offset as usize;

        reader.seek(SeekFrom::Start(footer.centroids_offset))?;
        let mut metadata_bytes = vec![0u8; metadata_len];
        reader.read_exact(&mut metadata_bytes)?;

        let metadata = IvfMetadata::deserialize(&metadata_bytes)?;

        // Verify checksum
        let expected_checksum: u32 = metadata_bytes.iter().map(|&b| b as u32).sum();
        if expected_checksum != footer.checksum {
            vortex_bail!(
                "IVF metadata checksum mismatch: expected {}, got {}",
                expected_checksum,
                footer.checksum
            );
        }

        // Read Vortex data
        reader.seek(SeekFrom::Start(header.vortex_offset))?;
        let vortex_data_len = footer.centroids_offset - header.vortex_offset;
        let mut vortex_data = vec![0u8; vortex_data_len as usize];
        reader.read_exact(&mut vortex_data)?;

        let records = Self::deserialize_vortex_data(&vortex_data)?;

        Ok(Self {
            config: metadata.config,
            records,
        })
    }

    /// Opens an IVF Vortex file from a path.
    pub fn open<P: AsRef<Path>>(path: P) -> VortexResult<Self> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        Self::read(&mut reader)
    }

    /// Returns the IVF configuration including centroids.
    pub fn config(&self) -> &IvfConfig {
        &self.config
    }

    /// Returns the vector records.
    pub fn records(&self) -> &[IvfVectorRecord] {
        &self.records
    }

    /// Returns the number of vectors.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns true if there are no vectors.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Finds vectors in the nearest partitions to the query.
    ///
    /// Returns records whose partition_id is in the set of nearest partitions.
    pub fn find_in_nearest_partitions(
        &self,
        query: &[f32],
        nprobe: usize,
    ) -> VortexResult<Vec<&IvfVectorRecord>> {
        let nearest_partitions = self.config.find_nearest_partitions(query, nprobe)?;

        let results: Vec<_> = self
            .records
            .iter()
            .filter(|r| nearest_partitions.contains(&r.partition_id))
            .collect();

        Ok(results)
    }

    /// Builds a Vortex array from the records.
    pub fn build_array(&self) -> VortexResult<ArrayRef> {
        let row_ids: Vec<u64> = self.records.iter().map(|r| r.row_id).collect();
        let vectors: Vec<f32> = self
            .records
            .iter()
            .flat_map(|r| r.vector.iter().copied())
            .collect();
        let partition_ids: Vec<u32> = self.records.iter().map(|r| r.partition_id).collect();

        create_ivf_data_array(
            &row_ids,
            &vectors,
            &partition_ids,
            self.config.dimensions as usize,
        )
    }

    /// Deserializes Vortex data from bytes.
    fn deserialize_vortex_data(data: &[u8]) -> VortexResult<Vec<IvfVectorRecord>> {
        if data.len() < 16 {
            vortex_bail!("Vortex data too short");
        }

        let mut offset = 0;

        // Read number of vectors
        let num_vectors =
            u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        // Read dimensions
        let dimensions = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        // Read row IDs
        let mut row_ids = Vec::with_capacity(num_vectors);
        for _ in 0..num_vectors {
            let row_id = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            row_ids.push(row_id);
            offset += 8;
        }

        // Read vectors
        let mut vectors = Vec::with_capacity(num_vectors);
        for _ in 0..num_vectors {
            let mut vector = Vec::with_capacity(dimensions);
            for _ in 0..dimensions {
                let v = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                vector.push(v);
                offset += 4;
            }
            vectors.push(vector);
        }

        // Read partition IDs
        let mut partition_ids = Vec::with_capacity(num_vectors);
        for _ in 0..num_vectors {
            let partition_id = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            partition_ids.push(partition_id);
            offset += 4;
        }

        // Build records
        let records: Vec<_> = row_ids
            .into_iter()
            .zip(vectors)
            .zip(partition_ids)
            .map(|((row_id, vector), partition_id)| IvfVectorRecord {
                row_id,
                vector,
                partition_id,
            })
            .collect();

        Ok(records)
    }
}

/// Summary of a write operation.
#[derive(Debug, Clone)]
pub struct IvfWriteSummary {
    /// Total file size in bytes
    pub total_size: u64,
    /// Size of the Vortex data section
    pub vortex_data_size: u64,
    /// Size of the IVF metadata section
    pub metadata_size: u64,
    /// Number of vectors written
    pub num_vectors: usize,
    /// Number of partitions
    pub num_partitions: usize,
}

/// IVF file header.
#[derive(Debug, Clone)]
struct IvfHeader {
    magic: [u8; 4],
    version: u32,
    vortex_offset: u64,
}

impl IvfHeader {
    fn write<W: Write>(&self, writer: &mut W) -> VortexResult<()> {
        writer.write_all(&self.magic)?;
        writer.write_all(&self.version.to_le_bytes())?;
        writer.write_all(&self.vortex_offset.to_le_bytes())?;
        Ok(())
    }

    fn read<R: Read>(reader: &mut R) -> VortexResult<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;

        if magic != IVF_FILE_MAGIC {
            vortex_bail!(
                "Invalid IVF file magic: expected {:?}, got {:?}",
                IVF_FILE_MAGIC,
                magic
            );
        }

        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);

        if version != IVF_FILE_VERSION {
            vortex_bail!(
                "Unsupported IVF file version: expected {}, got {}",
                IVF_FILE_VERSION,
                version
            );
        }

        let mut offset_bytes = [0u8; 8];
        reader.read_exact(&mut offset_bytes)?;
        let vortex_offset = u64::from_le_bytes(offset_bytes);

        Ok(Self {
            magic,
            version,
            vortex_offset,
        })
    }
}

/// IVF file footer.
#[derive(Debug, Clone)]
struct IvfFooter {
    centroids_offset: u64,
    magic: [u8; 4],
    checksum: u32,
}

impl IvfFooter {
    fn write<W: Write>(&self, writer: &mut W) -> VortexResult<()> {
        writer.write_all(&self.centroids_offset.to_le_bytes())?;
        writer.write_all(&self.magic)?;
        writer.write_all(&self.checksum.to_le_bytes())?;
        Ok(())
    }

    fn read<R: Read>(reader: &mut R) -> VortexResult<Self> {
        let mut offset_bytes = [0u8; 8];
        reader.read_exact(&mut offset_bytes)?;
        let centroids_offset = u64::from_le_bytes(offset_bytes);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;

        if magic != IVF_FOOTER_MAGIC {
            vortex_bail!(
                "Invalid IVF footer magic: expected {:?}, got {:?}",
                IVF_FOOTER_MAGIC,
                magic
            );
        }

        let mut checksum_bytes = [0u8; 4];
        reader.read_exact(&mut checksum_bytes)?;
        let checksum = u32::from_le_bytes(checksum_bytes);

        Ok(Self {
            centroids_offset,
            magic,
            checksum,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn create_test_config() -> IvfConfig {
        let centroids = vec![
            0.0, 0.0, 0.0, 0.0, // Partition 0
            1.0, 1.0, 1.0, 1.0, // Partition 1
            2.0, 2.0, 2.0, 2.0, // Partition 2
        ];
        IvfConfig::new(3, 4, centroids).unwrap()
    }

    #[test]
    fn test_write_read_roundtrip() {
        let config = create_test_config();
        let mut writer = IvfVortexWriter::new(config.clone());

        // Add some vectors
        writer
            .add_vector(0, vec![0.1, 0.1, 0.1, 0.1], 0)
            .unwrap();
        writer
            .add_vector(1, vec![1.1, 1.1, 1.1, 1.1], 1)
            .unwrap();
        writer
            .add_vector(2, vec![2.1, 2.1, 2.1, 2.1], 2)
            .unwrap();

        // Write to buffer
        let mut buffer = Cursor::new(Vec::new());
        let summary = writer.write(&mut buffer).unwrap();

        assert_eq!(summary.num_vectors, 3);
        assert_eq!(summary.num_partitions, 3);

        // Read back
        buffer.set_position(0);
        let reader = IvfVortexReader::read(&mut buffer).unwrap();

        assert_eq!(reader.len(), 3);
        assert_eq!(reader.config(), &config);

        // Check records
        let records = reader.records();
        assert_eq!(records[0].row_id, 0);
        assert_eq!(records[0].partition_id, 0);
        assert_eq!(records[1].row_id, 1);
        assert_eq!(records[1].partition_id, 1);
        assert_eq!(records[2].row_id, 2);
        assert_eq!(records[2].partition_id, 2);
    }

    #[test]
    fn test_auto_partition() {
        let config = create_test_config();
        let mut writer = IvfVortexWriter::new(config);

        // Add vector close to partition 1
        writer
            .add_vector_auto_partition(0, vec![0.9, 1.1, 0.9, 1.1])
            .unwrap();

        assert_eq!(writer.records[0].partition_id, 1);
    }

    #[test]
    fn test_find_in_nearest_partitions() {
        let config = create_test_config();
        let mut writer = IvfVortexWriter::new(config);

        // Add vectors to different partitions
        for i in 0..10 {
            let partition = (i % 3) as u32;
            let base = partition as f32;
            writer
                .add_vector(
                    i,
                    vec![base + 0.1, base + 0.1, base + 0.1, base + 0.1],
                    partition,
                )
                .unwrap();
        }

        // Write and read back
        let mut buffer = Cursor::new(Vec::new());
        writer.write(&mut buffer).unwrap();
        buffer.set_position(0);
        let reader = IvfVortexReader::read(&mut buffer).unwrap();

        // Query close to partition 0
        let results = reader
            .find_in_nearest_partitions(&[0.0, 0.0, 0.0, 0.0], 1)
            .unwrap();

        // Should only return vectors in partition 0
        for r in &results {
            assert_eq!(r.partition_id, 0);
        }

        // Query with nprobe=2
        let results = reader
            .find_in_nearest_partitions(&[0.5, 0.5, 0.5, 0.5], 2)
            .unwrap();

        // Should return vectors from partitions 0 and 1
        for r in &results {
            assert!(r.partition_id == 0 || r.partition_id == 1);
        }
    }

    #[test]
    fn test_invalid_dimension() {
        let config = create_test_config();
        let mut writer = IvfVortexWriter::new(config);

        // Wrong dimension
        let result = writer.add_vector(0, vec![0.1, 0.1, 0.1], 0); // 3 instead of 4
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_partition() {
        let config = create_test_config();
        let mut writer = IvfVortexWriter::new(config);

        // Partition out of range
        let result = writer.add_vector(0, vec![0.1, 0.1, 0.1, 0.1], 5); // Only 3 partitions
        assert!(result.is_err());
    }
}
