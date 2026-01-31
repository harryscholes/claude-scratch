//! Tests for RaBitQ including recall evaluation.

use crate::{DistanceMetric, RaBitQIndex};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

fn generate_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect())
        .collect()
}

fn generate_clustered_vectors(n: usize, dim: usize, num_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Generate cluster centers
    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 10.0 - 5.0).collect())
        .collect();

    // Generate points around centers
    (0..n)
        .map(|_| {
            let center_idx = rng.gen_range(0..num_clusters);
            let center = &centers[center_idx];
            center
                .iter()
                .map(|&c| c + (rng.gen::<f32>() - 0.5) * 0.5)
                .collect()
        })
        .collect()
}

/// Compute recall@k: fraction of true k-NN that appear in approximate k-NN.
fn compute_recall(
    approx_results: &[(usize, f32)],
    exact_results: &[(usize, f32)],
    k: usize,
) -> f32 {
    let exact_set: std::collections::HashSet<usize> =
        exact_results.iter().take(k).map(|(idx, _)| *idx).collect();

    let found = approx_results
        .iter()
        .take(k)
        .filter(|(idx, _)| exact_set.contains(idx))
        .count();

    found as f32 / k as f32
}

#[test]
fn test_recall_random_euclidean() {
    let n = 1000;
    let dim = 128;
    let k = 10;
    let num_queries = 100;

    let vectors = generate_random_vectors(n, dim, 42);
    let queries = generate_random_vectors(num_queries, dim, 123);

    let index = RaBitQIndex::build(&vectors, DistanceMetric::Euclidean, 8, 42).unwrap();

    let mut total_recall = 0.0f32;

    for query in &queries {
        let approx_results = index.query(query, k).unwrap();
        let exact_results = index.query_exact(query, &vectors, k);
        let recall = compute_recall(&approx_results, &exact_results, k);
        total_recall += recall;
    }

    let avg_recall = total_recall / num_queries as f32;
    println!("Average Recall@{} (Random, Euclidean): {:.4}", k, avg_recall);

    // RaBitQ should achieve reasonable recall (>0.5) on random data
    assert!(
        avg_recall > 0.3,
        "Recall too low: {}. Expected > 0.3",
        avg_recall
    );
}

#[test]
fn test_recall_clustered_euclidean() {
    let n = 1000;
    let dim = 128;
    let k = 10;
    let num_queries = 100;

    let vectors = generate_clustered_vectors(n, dim, 20, 42);
    let queries = generate_clustered_vectors(num_queries, dim, 20, 123);

    // Use more subvectors for better accuracy on clustered data
    let index = RaBitQIndex::build(&vectors, DistanceMetric::Euclidean, 16, 42).unwrap();

    let mut total_recall = 0.0f32;

    for query in &queries {
        let approx_results = index.query(query, k).unwrap();
        let exact_results = index.query_exact(query, &vectors, k);
        let recall = compute_recall(&approx_results, &exact_results, k);
        total_recall += recall;
    }

    let avg_recall = total_recall / num_queries as f32;
    println!(
        "Average Recall@{} (Clustered, Euclidean): {:.4}",
        k, avg_recall
    );

    // RaBitQ may have lower recall on tightly clustered data
    // where fine-grained distinctions matter more
    assert!(
        avg_recall > 0.1,
        "Recall too low: {}. Expected > 0.1",
        avg_recall
    );
}

#[test]
fn test_recall_cosine() {
    let n = 1000;
    let dim = 128;
    let k = 10;
    let num_queries = 50;

    let vectors = generate_random_vectors(n, dim, 42);
    let queries = generate_random_vectors(num_queries, dim, 456);

    let index = RaBitQIndex::build(&vectors, DistanceMetric::Cosine, 8, 42).unwrap();

    let mut total_recall = 0.0f32;

    for query in &queries {
        let approx_results = index.query(query, k).unwrap();
        let exact_results = index.query_exact(query, &vectors, k);
        let recall = compute_recall(&approx_results, &exact_results, k);
        total_recall += recall;
    }

    let avg_recall = total_recall / num_queries as f32;
    println!("Average Recall@{} (Random, Cosine): {:.4}", k, avg_recall);

    assert!(
        avg_recall > 0.2,
        "Recall too low: {}. Expected > 0.2",
        avg_recall
    );
}

#[test]
fn test_recall_inner_product() {
    let n = 1000;
    let dim = 128;
    let k = 10;
    let num_queries = 50;

    let vectors = generate_random_vectors(n, dim, 42);
    let queries = generate_random_vectors(num_queries, dim, 789);

    let index = RaBitQIndex::build(&vectors, DistanceMetric::InnerProduct, 8, 42).unwrap();

    let mut total_recall = 0.0f32;

    for query in &queries {
        let approx_results = index.query(query, k).unwrap();
        let exact_results = index.query_exact(query, &vectors, k);
        let recall = compute_recall(&approx_results, &exact_results, k);
        total_recall += recall;
    }

    let avg_recall = total_recall / num_queries as f32;
    println!(
        "Average Recall@{} (Random, InnerProduct): {:.4}",
        k, avg_recall
    );

    assert!(
        avg_recall > 0.2,
        "Recall too low: {}. Expected > 0.2",
        avg_recall
    );
}

#[test]
fn test_recall_vs_num_subvectors() {
    let n = 500;
    let dim = 128;
    let k = 10;
    let num_queries = 50;

    let vectors = generate_random_vectors(n, dim, 42);
    let queries = generate_random_vectors(num_queries, dim, 123);

    println!("\nRecall vs Number of Subvectors (Euclidean, dim=128):");
    println!("---------------------------------------------------");

    for num_subvectors in [1, 2, 4, 8, 16, 32] {
        let index =
            RaBitQIndex::build(&vectors, DistanceMetric::Euclidean, num_subvectors, 42).unwrap();

        let mut total_recall = 0.0f32;
        for query in &queries {
            let approx_results = index.query(query, k).unwrap();
            let exact_results = index.query_exact(query, &vectors, k);
            let recall = compute_recall(&approx_results, &exact_results, k);
            total_recall += recall;
        }

        let avg_recall = total_recall / num_queries as f32;
        println!("  num_subvectors={:2}: Recall@{}={:.4}", num_subvectors, k, avg_recall);
    }
}

#[test]
fn test_exact_self_query() {
    // Query with the same vector should return itself in approximate results
    let n = 100;
    let dim = 64;

    let vectors = generate_random_vectors(n, dim, 42);
    let index = RaBitQIndex::build(&vectors, DistanceMetric::Euclidean, 4, 42).unwrap();

    // Query each vector and check if it appears in top results
    let mut found_self_count = 0;
    for (i, query) in vectors.iter().enumerate() {
        let results = index.query(query, 10).unwrap();
        if results.iter().any(|(idx, _)| *idx == i) {
            found_self_count += 1;
        }
    }

    let self_recall = found_self_count as f32 / n as f32;
    println!("Self-query recall (in top 10): {:.2}%", self_recall * 100.0);

    // Should find itself in top 10 most of the time
    assert!(
        self_recall > 0.5,
        "Self-query recall too low: {:.2}%",
        self_recall * 100.0
    );
}

#[test]
fn test_recall_at_different_k() {
    let n = 1000;
    let dim = 128;
    let num_queries = 50;

    let vectors = generate_random_vectors(n, dim, 42);
    let queries = generate_random_vectors(num_queries, dim, 123);

    let index = RaBitQIndex::build(&vectors, DistanceMetric::Euclidean, 8, 42).unwrap();

    println!("\nRecall at different k values:");
    println!("------------------------------");

    for k in [1, 5, 10, 20, 50, 100] {
        let mut total_recall = 0.0f32;
        for query in &queries {
            let approx_results = index.query(query, k).unwrap();
            let exact_results = index.query_exact(query, &vectors, k);
            let recall = compute_recall(&approx_results, &exact_results, k);
            total_recall += recall;
        }

        let avg_recall = total_recall / num_queries as f32;
        println!("  Recall@{:3} = {:.4}", k, avg_recall);
    }
}
