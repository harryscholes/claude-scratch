# RaBitQ

A Rust implementation of the RaBitQ algorithm for approximate nearest neighbor search.

RaBitQ (Random Bit Quantization) quantizes high-dimensional vectors into compact binary codes using random orthogonal transformations, enabling fast similarity search with theoretical error bounds.

Based on the paper: "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search"

## Features

- **1-bit per dimension quantization** with random orthogonal rotation
- **Subvector splitting** for improved accuracy
- **Multiple distance metrics**: Euclidean, Cosine, and Inner Product
- **SIMD acceleration** using Rust's portable SIMD (requires nightly)
- **Serialization support** for index persistence via serde/bincode

## Requirements

- Rust nightly (for `portable_simd` feature)

## Usage

```rust
use rabitq::{RaBitQIndex, DistanceMetric};

// Create some sample vectors
let vectors: Vec<Vec<f32>> = (0..1000)
    .map(|i| (0..128).map(|j| (i * j) as f32 / 1000.0).collect())
    .collect();

// Build the index with 8 subvectors
let index = RaBitQIndex::build(&vectors, DistanceMetric::Euclidean, 8, 42).unwrap();

// Query for 10 nearest neighbors
let query: Vec<f32> = (0..128).map(|j| j as f32 / 100.0).collect();
let results = index.query(&query, 10).unwrap();

// Results are (index, distance) pairs sorted by distance
for (idx, dist) in results {
    println!("Vector {}: distance = {}", idx, dist);
}
```

## Configuration

### Number of Subvectors

The `num_subvectors` parameter controls the accuracy/speed tradeoff:
- More subvectors = higher accuracy, slower queries
- Must evenly divide the vector dimension
- Recommended: start with dim/16 or dim/8

### Distance Metrics

- `DistanceMetric::Euclidean` - L2 distance
- `DistanceMetric::Cosine` - Cosine distance (1 - cosine similarity)
- `DistanceMetric::InnerProduct` - Negative inner product

## Persistence

```rust
// Save index to file
index.save("my_index.bin")?;

// Load index from file
let loaded = RaBitQIndex::load("my_index.bin")?;
```

## Benchmarks

Run benchmarks:

```bash
cargo bench --bench rabitq_bench
```

### SIFT Benchmark

To run benchmarks with the SIFT-128 dataset from ann-benchmarks:

1. Install the HDF5 library on your system:
   - Ubuntu/Debian: `sudo apt-get install libhdf5-dev`
   - macOS: `brew install hdf5`
   - Fedora: `sudo dnf install hdf5-devel`

2. Download the dataset:
   ```bash
   wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
   ```

3. Run the benchmark:
   ```bash
   SIFT_PATH=./sift-128-euclidean.hdf5 cargo bench --features bench-hdf5 --bench sift_bench
   ```

## Algorithm Overview

RaBitQ works as follows:

1. **Preprocessing**: Compute the centroid of all database vectors
2. **Rotation**: Apply a random orthogonal transformation to each centered vector
3. **Quantization**: Store the sign of each component (1 bit per dimension)
4. **Auxiliary data**: Store norms and sum of absolute values for distance estimation

During queries:
1. Center and rotate the query vector
2. Use asymmetric distance estimation (full-precision query vs. quantized database)
3. Return approximate k nearest neighbors

Subvector splitting applies this process independently to segments of the vector for improved accuracy.

## License

MIT
