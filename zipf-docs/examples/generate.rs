//! Example: Generate sample documents with different configurations.

use zipf_docs::{DocumentGenerator, GeneratorConfig};

fn main() {
    println!("=== zipf-docs Document Generator ===\n");

    // Generate with default configuration
    println!("--- Default Configuration (medium-length documents) ---");
    let mut generator = DocumentGenerator::new(42);
    for i in 1..=3 {
        let doc = generator.generate();
        println!(
            "Document {}: ({} words)\n{}\n",
            i,
            doc.split_whitespace().count(),
            doc
        );
    }

    // Generate short documents (tweets, titles)
    println!("\n--- Short Documents Configuration ---");
    let config = GeneratorConfig::short_documents();
    let mut generator = DocumentGenerator::with_config(123, config);
    for i in 1..=5 {
        let doc = generator.generate();
        println!("Short {}: {}", i, doc);
    }

    // Generate long documents (articles)
    println!("\n--- Long Documents Configuration ---");
    let config = GeneratorConfig::long_documents();
    let mut generator = DocumentGenerator::with_config(456, config);
    let doc = generator.generate();
    println!(
        "Long document ({} words):\n{}...\n",
        doc.split_whitespace().count(),
        &doc[..500.min(doc.len())]
    );

    // Generate documents with specific length
    println!("\n--- Fixed Length Documents (exactly 20 words) ---");
    let mut generator = DocumentGenerator::new(789);
    for i in 1..=3 {
        let doc = generator.generate_with_length(20);
        println!("Fixed {}: {}", i, doc);
    }

    // Demonstrate batch generation
    println!("\n--- Batch Generation (1000 documents) ---");
    let mut generator = DocumentGenerator::new(42);
    let docs: Vec<_> = generator.generate_batch(1000).collect();

    let total_words: usize = docs.iter().map(|d| d.split_whitespace().count()).sum();
    let avg_words = total_words as f64 / docs.len() as f64;
    let min_words = docs
        .iter()
        .map(|d| d.split_whitespace().count())
        .min()
        .unwrap();
    let max_words = docs
        .iter()
        .map(|d| d.split_whitespace().count())
        .max()
        .unwrap();

    println!("Generated {} documents", docs.len());
    println!("Average length: {:.1} words", avg_words);
    println!("Min length: {} words", min_words);
    println!("Max length: {} words", max_words);
}
