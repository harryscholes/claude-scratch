//! SIFT dataset benchmark for RaBitQ.
//!
//! This benchmark uses the SIFT dataset from ann-benchmarks (HDF5 format).
//! Download the dataset from: http://ann-benchmarks.com/sift-128-euclidean.hdf5
//!
//! Run with:
//!   cargo bench --features bench-hdf5 --bench sift_bench
//!
//! Set SIFT_PATH environment variable to specify dataset location:
//!   SIFT_PATH=/path/to/sift-128-euclidean.hdf5 cargo bench --features bench-hdf5 --bench sift_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hdf5::File as H5File;
use rabitq::{DistanceMetric, RaBitQIndex};
use std::collections::HashSet;
use std::env;
use std::path::PathBuf;

/// SIFT dataset loaded from HDF5.
struct SiftDataset {
    train: Vec<Vec<f32>>,
    test: Vec<Vec<f32>>,
    neighbors: Vec<Vec<usize>>,
    distances: Vec<Vec<f32>>,
}

impl SiftDataset {
    fn load(path: &PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let file = H5File::open(path)?;

        // Load training vectors
        let train_ds = file.dataset("train")?;
        let train_data: ndarray::Array2<f32> = train_ds.read()?;
        let train: Vec<Vec<f32>> = train_data.rows().into_iter().map(|r| r.to_vec()).collect();

        // Load test queries
        let test_ds = file.dataset("test")?;
        let test_data: ndarray::Array2<f32> = test_ds.read()?;
        let test: Vec<Vec<f32>> = test_data.rows().into_iter().map(|r| r.to_vec()).collect();

        // Load ground truth neighbors
        let neighbors_ds = file.dataset("neighbors")?;
        let neighbors_data: ndarray::Array2<i32> = neighbors_ds.read()?;
        let neighbors: Vec<Vec<usize>> = neighbors_data
            .rows()
            .into_iter()
            .map(|r| r.iter().map(|&x| x as usize).collect())
            .collect();

        // Load ground truth distances
        let distances_ds = file.dataset("distances")?;
        let distances_data: ndarray::Array2<f32> = distances_ds.read()?;
        let distances: Vec<Vec<f32>> = distances_data
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect();

        Ok(SiftDataset {
            train,
            test,
            neighbors,
            distances,
        })
    }
}

fn get_sift_path() -> Option<PathBuf> {
    // Try environment variable first
    if let Ok(path) = env::var("SIFT_PATH") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    // Try common locations
    let candidates = [
        "sift-128-euclidean.hdf5",
        "data/sift-128-euclidean.hdf5",
        "../data/sift-128-euclidean.hdf5",
        "~/data/sift-128-euclidean.hdf5",
    ];

    for candidate in candidates {
        let p = PathBuf::from(shellexpand::tilde(candidate).to_string());
        if p.exists() {
            return Some(p);
        }
    }

    None
}

fn compute_recall(approx: &[(usize, f32)], ground_truth: &[usize], k: usize) -> f32 {
    let gt_set: HashSet<usize> = ground_truth.iter().take(k).copied().collect();
    let found = approx.iter().take(k).filter(|(idx, _)| gt_set.contains(idx)).count();
    found as f32 / k as f32
}

fn bench_sift_indexing(c: &mut Criterion) {
    let path = match get_sift_path() {
        Some(p) => p,
        None => {
            eprintln!("SIFT dataset not found. Set SIFT_PATH environment variable.");
            eprintln!("Download from: http://ann-benchmarks.com/sift-128-euclidean.hdf5");
            return;
        }
    };

    let dataset = match SiftDataset::load(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to load SIFT dataset: {}", e);
            return;
        }
    };

    println!("\nSIFT Dataset loaded:");
    println!("  Train vectors: {}", dataset.train.len());
    println!("  Test queries: {}", dataset.test.len());
    println!("  Dimension: {}", dataset.train[0].len());

    let mut group = c.benchmark_group("sift_indexing");
    group.sample_size(10);

    // Benchmark full dataset indexing
    group.throughput(Throughput::Elements(dataset.train.len() as u64));
    group.bench_function("build_full", |b| {
        b.iter(|| {
            black_box(
                RaBitQIndex::build(&dataset.train, DistanceMetric::Euclidean, 8, 42).unwrap(),
            )
        })
    });

    // Benchmark with different subvector counts
    for num_subvectors in [4, 8, 16, 32] {
        group.bench_with_input(
            BenchmarkId::new("build_subvec", num_subvectors),
            &num_subvectors,
            |b, &num_subvectors| {
                b.iter(|| {
                    black_box(
                        RaBitQIndex::build(&dataset.train, DistanceMetric::Euclidean, num_subvectors, 42)
                            .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_sift_query(c: &mut Criterion) {
    let path = match get_sift_path() {
        Some(p) => p,
        None => {
            eprintln!("SIFT dataset not found. Set SIFT_PATH environment variable.");
            return;
        }
    };

    let dataset = match SiftDataset::load(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to load SIFT dataset: {}", e);
            return;
        }
    };

    let mut group = c.benchmark_group("sift_query");

    for num_subvectors in [4, 8, 16] {
        let index = RaBitQIndex::build(&dataset.train, DistanceMetric::Euclidean, num_subvectors, 42)
            .unwrap();

        for k in [1, 10, 100] {
            group.throughput(Throughput::Elements(1));
            group.bench_with_input(
                BenchmarkId::new(format!("subvec={}", num_subvectors), format!("k={}", k)),
                &(&index, &dataset.test),
                |b, (index, queries)| {
                    let mut query_idx = 0;
                    b.iter(|| {
                        let result = index.query(&queries[query_idx], k).unwrap();
                        query_idx = (query_idx + 1) % queries.len();
                        black_box(result)
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_sift_recall(c: &mut Criterion) {
    let path = match get_sift_path() {
        Some(p) => p,
        None => {
            eprintln!("SIFT dataset not found. Set SIFT_PATH environment variable.");
            return;
        }
    };

    let dataset = match SiftDataset::load(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to load SIFT dataset: {}", e);
            return;
        }
    };

    println!("\n=== SIFT Recall Evaluation ===\n");

    for num_subvectors in [4, 8, 16, 32] {
        let index = RaBitQIndex::build(&dataset.train, DistanceMetric::Euclidean, num_subvectors, 42)
            .unwrap();

        for k in [1, 10, 100] {
            let mut total_recall = 0.0f32;
            let num_queries = dataset.test.len().min(1000); // Limit for speed

            for i in 0..num_queries {
                let query = &dataset.test[i];
                let approx_results = index.query(query, k).unwrap();
                let recall = compute_recall(&approx_results, &dataset.neighbors[i], k);
                total_recall += recall;
            }

            let avg_recall = total_recall / num_queries as f32;
            println!(
                "num_subvectors={:2}, k={:3}: Recall@{} = {:.4}",
                num_subvectors, k, k, avg_recall
            );
        }
        println!();
    }

    // Benchmark with recall measurement
    let mut group = c.benchmark_group("sift_recall_benchmark");
    group.sample_size(10);

    let index = RaBitQIndex::build(&dataset.train, DistanceMetric::Euclidean, 8, 42).unwrap();
    let k = 10;

    group.bench_function("query_with_recall", |b| {
        let mut query_idx = 0;
        b.iter(|| {
            let query = &dataset.test[query_idx];
            let approx_results = index.query(query, k).unwrap();
            let recall = compute_recall(&approx_results, &dataset.neighbors[query_idx], k);
            query_idx = (query_idx + 1) % dataset.test.len();
            black_box((approx_results, recall))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sift_indexing,
    bench_sift_query,
    bench_sift_recall,
);

criterion_main!(benches);
