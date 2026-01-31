//! Benchmarks for RaBitQ indexing and querying.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rabitq::{DistanceMetric, RaBitQIndex};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn generate_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect())
        .collect()
}

fn bench_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("indexing");

    for n in [1000, 10000] {
        for dim in [64, 128, 256] {
            let vectors = generate_random_vectors(n, dim, 42);

            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(
                BenchmarkId::new("build", format!("n={}_dim={}", n, dim)),
                &vectors,
                |b, vectors| {
                    b.iter(|| {
                        black_box(
                            RaBitQIndex::build(vectors, DistanceMetric::Euclidean, 8, 42).unwrap(),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("query");

    let n = 10000;
    let dim = 128;
    let vectors = generate_random_vectors(n, dim, 42);
    let queries = generate_random_vectors(100, dim, 123);

    for num_subvectors in [4, 8, 16] {
        let index = RaBitQIndex::build(&vectors, DistanceMetric::Euclidean, num_subvectors, 42)
            .unwrap();

        for k in [10, 50, 100] {
            group.throughput(Throughput::Elements(1));
            group.bench_with_input(
                BenchmarkId::new(
                    "query",
                    format!("n={}_subvec={}_k={}", n, num_subvectors, k),
                ),
                &(&index, &queries),
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

fn bench_query_by_metric(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_by_metric");

    let n = 10000;
    let dim = 128;
    let k = 10;
    let vectors = generate_random_vectors(n, dim, 42);
    let queries = generate_random_vectors(100, dim, 123);

    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::Cosine,
        DistanceMetric::InnerProduct,
    ] {
        let index = RaBitQIndex::build(&vectors, metric, 8, 42).unwrap();

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("query", format!("{:?}", metric)),
            &(&index, &queries),
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

    group.finish();
}

fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    let n = 10000;
    let dim = 128;
    let vectors = generate_random_vectors(n, dim, 42);
    let index = RaBitQIndex::build(&vectors, DistanceMetric::Euclidean, 8, 42).unwrap();

    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("bench_index.bin");

    group.bench_function("save", |b| {
        b.iter(|| {
            index.save(&path).unwrap();
        })
    });

    index.save(&path).unwrap();

    group.bench_function("load", |b| {
        b.iter(|| black_box(RaBitQIndex::load(&path).unwrap()))
    });

    group.finish();
}

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");
    group.sample_size(20);

    let dim = 128;
    let k = 10;
    let queries = generate_random_vectors(50, dim, 123);

    for n in [1000, 5000, 10000, 50000] {
        let vectors = generate_random_vectors(n, dim, 42);
        let index = RaBitQIndex::build(&vectors, DistanceMetric::Euclidean, 8, 42).unwrap();

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("query_vs_n", format!("n={}", n)),
            &(&index, &queries),
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

    group.finish();
}

criterion_group!(
    benches,
    bench_indexing,
    bench_query,
    bench_query_by_metric,
    bench_serialization,
    bench_scaling,
);

criterion_main!(benches);
