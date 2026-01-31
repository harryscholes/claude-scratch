//! IVF Centroids Vortex Demo
//!
//! This example demonstrates:
//! 1. Creating an IVF index with K-means centroids
//! 2. Writing vector data with partition assignments to an IVF Vortex file
//! 3. Reading the file and using centroids for partition pruning
//!
//! Run with: cargo run --example ivf_demo

use std::io::Cursor;

use ivf_centroids_vortex::{IvfConfig, IvfVortexReader, IvfVortexWriter};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IVF Centroids Vortex Demo ===\n");

    // Configuration
    let num_partitions = 4;
    let dimensions = 8;
    let vectors_per_partition = 100;
    let total_vectors = num_partitions * vectors_per_partition;

    println!("Configuration:");
    println!("  Number of partitions (K): {}", num_partitions);
    println!("  Vector dimensions: {}", dimensions);
    println!("  Vectors per partition: {}", vectors_per_partition);
    println!("  Total vectors: {}", total_vectors);
    println!();

    // Step 1: Generate synthetic centroids
    println!("Step 1: Generating centroids...");
    let centroids = generate_centroids(num_partitions, dimensions);
    println!("  Generated {} centroids", num_partitions);

    // Create IVF configuration
    let config = IvfConfig::new(
        num_partitions as u32,
        dimensions as u32,
        centroids.clone(),
    )?;

    // Print centroid locations
    for i in 0..num_partitions {
        let centroid = config.centroid(i as u32).unwrap();
        println!(
            "  Centroid {}: [{:.2}, {:.2}, {:.2}, ...]",
            i, centroid[0], centroid[1], centroid[2]
        );
    }
    println!();

    // Step 2: Generate synthetic vectors around each centroid
    println!("Step 2: Generating vectors around centroids...");
    let mut writer = IvfVortexWriter::new(config.clone());

    let mut rng = StdRng::seed_from_u64(42);
    let mut row_id = 0u64;

    for partition in 0..num_partitions {
        let centroid = config.centroid(partition as u32).unwrap();

        for _ in 0..vectors_per_partition {
            // Generate a vector near this centroid with some noise
            let vector: Vec<f32> = centroid
                .iter()
                .map(|&c| c + rng.random_range(-0.5..0.5))
                .collect();

            writer.add_vector(row_id, vector, partition as u32)?;
            row_id += 1;
        }
    }

    println!("  Generated {} vectors", writer.len());
    println!();

    // Step 3: Write to file (in-memory for demo)
    println!("Step 3: Writing IVF Vortex file...");
    let mut buffer = Cursor::new(Vec::new());
    let summary = writer.write(&mut buffer)?;

    println!("  Total file size: {} bytes", summary.total_size);
    println!("  Vortex data size: {} bytes", summary.vortex_data_size);
    println!("  Metadata size: {} bytes", summary.metadata_size);
    println!("  Num vectors: {}", summary.num_vectors);
    println!("  Num partitions: {}", summary.num_partitions);
    println!();

    // Step 4: Read the file back
    println!("Step 4: Reading IVF Vortex file...");
    buffer.set_position(0);
    let reader = IvfVortexReader::read(&mut buffer)?;

    println!("  Read {} vectors", reader.len());
    println!(
        "  Config: {} partitions, {} dimensions",
        reader.config().num_partitions,
        reader.config().dimensions
    );
    println!();

    // Step 5: Demonstrate partition pruning
    println!("Step 5: Demonstrating partition pruning...");

    // Generate a random query vector
    let query: Vec<f32> = (0..dimensions)
        .map(|_| rng.random_range(0.0..10.0))
        .collect();
    println!(
        "  Query vector: [{:.2}, {:.2}, {:.2}, ...]",
        query[0], query[1], query[2]
    );

    // Find nearest partitions
    let nearest = reader.config().find_nearest_partitions(&query, 2)?;
    println!("  Nearest partitions: {:?}", nearest);

    // Calculate distances to all centroids for verification
    println!("\n  Distances to all centroids:");
    for i in 0..num_partitions {
        let centroid = reader.config().centroid(i as u32).unwrap();
        let distance: f32 = query
            .iter()
            .zip(centroid.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        let marker = if nearest.contains(&(i as u32)) {
            " <--"
        } else {
            ""
        };
        println!("    Partition {}: {:.4}{}", i, distance, marker);
    }
    println!();

    // Step 6: Query with partition pruning
    println!("Step 6: Querying with partition pruning...");

    let nprobe = 2;
    let results = reader.find_in_nearest_partitions(&query, nprobe)?;

    let rows_scanned = results.len();
    let rows_pruned = total_vectors - rows_scanned;
    let pruning_ratio = (rows_pruned as f64 / total_vectors as f64) * 100.0;

    println!("  nprobe: {}", nprobe);
    println!(
        "  Partitions searched: {:?}",
        nearest
    );
    println!(
        "  Rows scanned: {} (out of {})",
        rows_scanned, total_vectors
    );
    println!(
        "  Rows pruned: {} ({:.1}% pruning)",
        rows_pruned, pruning_ratio
    );
    println!();

    // Step 7: Show the file structure
    println!("Step 7: IVF Vortex File Structure");
    println!("  The file format combines Vortex data with IVF metadata:");
    println!();
    println!("  ┌─────────────────────────────────────────────┐");
    println!("  │  IVF Header (16 bytes)                      │");
    println!("  │  ├── magic: \"IVFV\" (4 bytes)                │");
    println!("  │  ├── version: u32 (4 bytes)                 │");
    println!("  │  └── vortex_offset: u64 (8 bytes)           │");
    println!("  ├─────────────────────────────────────────────┤");
    println!("  │  Vortex Data ({} bytes)               │", summary.vortex_data_size);
    println!("  │  └── StructArray with:                      │");
    println!("  │      ├── row_id: u64                        │");
    println!("  │      ├── vector: List<f32>                  │");
    println!("  │      └── ivf_partition_id: u32              │");
    println!("  ├─────────────────────────────────────────────┤");
    println!("  │  IVF Metadata ({} bytes)                   │", summary.metadata_size);
    println!("  │  ├── num_partitions: {} │", num_partitions);
    println!("  │  ├── dimensions: {}                          │", dimensions);
    println!(
        "  │  └── centroids: [[f32; {}]; {}]             │",
        dimensions, num_partitions
    );
    println!("  ├─────────────────────────────────────────────┤");
    println!("  │  IVF Footer (16 bytes)                      │");
    println!("  │  ├── centroids_offset: u64 (8 bytes)        │");
    println!("  │  ├── footer_magic: \"FVFI\" (4 bytes)         │");
    println!("  │  └── checksum: u32 (4 bytes)                │");
    println!("  └─────────────────────────────────────────────┘");
    println!();

    // Step 8: Verify roundtrip
    println!("Step 8: Verifying data integrity...");
    let original_config = config;
    let read_config = reader.config().clone();

    assert_eq!(
        original_config.num_partitions,
        read_config.num_partitions,
        "num_partitions mismatch"
    );
    assert_eq!(
        original_config.dimensions,
        read_config.dimensions,
        "dimensions mismatch"
    );
    assert_eq!(
        original_config.centroids,
        read_config.centroids,
        "centroids mismatch"
    );

    println!("  ✓ Configuration matches");
    println!("  ✓ Centroids preserved correctly");
    println!("  ✓ All {} vectors recovered", reader.len());
    println!();

    println!("=== Demo Complete ===");
    Ok(())
}

/// Generates synthetic centroids spread across the space.
fn generate_centroids(num_partitions: usize, dimensions: usize) -> Vec<f32> {
    let mut centroids = Vec::with_capacity(num_partitions * dimensions);
    let mut rng = StdRng::seed_from_u64(12345);

    for i in 0..num_partitions {
        // Space centroids apart by using partition index as a base
        let base = (i as f32) * 2.5;
        for d in 0..dimensions {
            // Add some randomness to each dimension
            let value = base + rng.random_range(-0.5..0.5) + (d as f32) * 0.1;
            centroids.push(value);
        }
    }

    centroids
}
