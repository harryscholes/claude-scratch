//! # zipf-docs
//!
//! Generate random documents with realistic word frequency distributions for benchmarking
//! text search engines like Tantivy.
//!
//! Documents are generated with:
//! - **Zipf-distributed words**: Word selection follows Zipf's law, mimicking natural language
//!   where a few words are very common and most words are rare.
//! - **Log-normally distributed lengths**: Document lengths follow a log-normal distribution,
//!   which models real-world document collections where most documents are moderate length
//!   with a long tail of longer documents.
//!
//! ## Example
//!
//! ```rust
//! use zipf_docs::DocumentGenerator;
//!
//! let mut generator = DocumentGenerator::new(42); // seed for reproducibility
//!
//! // Generate a single document
//! let doc = generator.generate();
//! println!("Document: {}", doc);
//!
//! // Generate multiple documents
//! let docs: Vec<String> = generator.generate_batch(1000).collect();
//! ```

mod vocabulary;

use rand::prelude::*;
use rand_distr::{LogNormal, Zipf};

pub use vocabulary::VOCABULARY;

/// Configuration for document generation.
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Zipf exponent (typically 1.0-1.5 for natural language). Higher = more skewed.
    pub zipf_exponent: f64,
    /// Mean of the log-normal distribution for document length (in ln-space).
    pub length_mean: f64,
    /// Standard deviation of the log-normal distribution for document length (in ln-space).
    pub length_std: f64,
    /// Minimum document length in words.
    pub min_length: usize,
    /// Maximum document length in words.
    pub max_length: usize,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            // Zipf exponent ~1.07 is typical for English text
            zipf_exponent: 1.07,
            // These parameters give a median around 150 words with reasonable spread
            // exp(5.0) ≈ 148 words median
            length_mean: 5.0,
            length_std: 0.8,
            min_length: 10,
            max_length: 10000,
        }
    }
}

impl GeneratorConfig {
    /// Create a new configuration with custom parameters.
    pub fn new(
        zipf_exponent: f64,
        length_mean: f64,
        length_std: f64,
        min_length: usize,
        max_length: usize,
    ) -> Self {
        Self {
            zipf_exponent,
            length_mean,
            length_std,
            min_length,
            max_length,
        }
    }

    /// Configuration for short documents (e.g., tweets, titles).
    pub fn short_documents() -> Self {
        Self {
            zipf_exponent: 1.07,
            length_mean: 2.5,  // exp(2.5) ≈ 12 words median
            length_std: 0.5,
            min_length: 3,
            max_length: 50,
        }
    }

    /// Configuration for medium documents (e.g., paragraphs, abstracts).
    pub fn medium_documents() -> Self {
        Self {
            zipf_exponent: 1.07,
            length_mean: 4.5,  // exp(4.5) ≈ 90 words median
            length_std: 0.6,
            min_length: 20,
            max_length: 500,
        }
    }

    /// Configuration for long documents (e.g., articles, papers).
    pub fn long_documents() -> Self {
        Self {
            zipf_exponent: 1.07,
            length_mean: 6.5,  // exp(6.5) ≈ 665 words median
            length_std: 0.7,
            min_length: 100,
            max_length: 20000,
        }
    }
}

/// A generator for creating random documents with realistic word distributions.
pub struct DocumentGenerator<R: Rng = StdRng> {
    rng: R,
    config: GeneratorConfig,
    zipf: Zipf<f64>,
    length_dist: LogNormal<f64>,
    vocab_size: usize,
}

impl DocumentGenerator<StdRng> {
    /// Create a new document generator with a seed for reproducibility.
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, GeneratorConfig::default())
    }

    /// Create a new document generator with custom configuration.
    pub fn with_config(seed: u64, config: GeneratorConfig) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Self::with_rng_and_config(rng, config)
    }
}

impl<R: Rng> DocumentGenerator<R> {
    /// Create a document generator with a custom RNG.
    pub fn with_rng(rng: R) -> Self {
        Self::with_rng_and_config(rng, GeneratorConfig::default())
    }

    /// Create a document generator with a custom RNG and configuration.
    pub fn with_rng_and_config(rng: R, config: GeneratorConfig) -> Self {
        let vocab_size = VOCABULARY.len();
        let zipf = Zipf::new(vocab_size as f64, config.zipf_exponent)
            .expect("Invalid Zipf parameters");
        let length_dist = LogNormal::new(config.length_mean, config.length_std)
            .expect("Invalid LogNormal parameters");

        Self {
            rng,
            config,
            zipf,
            length_dist,
            vocab_size,
        }
    }

    /// Generate a single document.
    pub fn generate(&mut self) -> String {
        let length = self.sample_length();
        self.generate_with_length(length)
    }

    /// Generate a document with a specific word count.
    pub fn generate_with_length(&mut self, word_count: usize) -> String {
        let mut words = Vec::with_capacity(word_count);

        for _ in 0..word_count {
            let word = self.sample_word();
            words.push(word);
        }

        // Capitalize first word of each sentence and add punctuation
        self.format_document(words)
    }

    /// Generate a batch of documents.
    pub fn generate_batch(&mut self, count: usize) -> impl Iterator<Item = String> + '_ {
        (0..count).map(move |_| self.generate())
    }

    /// Generate documents indefinitely.
    pub fn generate_iter(&mut self) -> impl Iterator<Item = String> + '_ {
        std::iter::from_fn(move || Some(self.generate()))
    }

    /// Sample a document length using the log-normal distribution.
    fn sample_length(&mut self) -> usize {
        let raw_length: f64 = self.length_dist.sample(&mut self.rng);
        let length = raw_length.round() as usize;
        length.clamp(self.config.min_length, self.config.max_length)
    }

    /// Sample a word using the Zipf distribution.
    fn sample_word(&mut self) -> &'static str {
        // Zipf returns 1-indexed values
        let idx: f64 = self.zipf.sample(&mut self.rng);
        let idx = (idx as usize).saturating_sub(1).min(self.vocab_size - 1);
        VOCABULARY[idx]
    }

    /// Format words into a document with sentences.
    fn format_document(&mut self, words: Vec<&'static str>) -> String {
        if words.is_empty() {
            return String::new();
        }

        let mut result = String::with_capacity(words.len() * 6);
        let mut sentence_length = 0;
        let avg_sentence_length = 15; // Average words per sentence

        for (i, word) in words.iter().enumerate() {
            // Capitalize first word of document or sentence
            if i == 0 || sentence_length == 0 {
                let mut chars = word.chars();
                if let Some(first) = chars.next() {
                    result.push(first.to_ascii_uppercase());
                    result.extend(chars);
                }
            } else {
                result.push(' ');
                result.push_str(word);
            }

            sentence_length += 1;

            // End sentence with some randomness around average length
            let end_sentence = if i == words.len() - 1 {
                true
            } else {
                sentence_length >= avg_sentence_length / 2
                    && self.rng.random_bool(1.0 / (avg_sentence_length as f64 / 2.0))
            };

            if end_sentence {
                // Vary punctuation
                let punct = match self.rng.random_range(0..10) {
                    0 => '?',
                    1 => '!',
                    _ => '.',
                };
                result.push(punct);
                // Add space after punctuation unless this is the last word
                if i < words.len() - 1 {
                    result.push(' ');
                }
                sentence_length = 0;
            }
        }

        result
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &GeneratorConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_document() {
        let mut generator = DocumentGenerator::new(42);
        let doc = generator.generate();
        assert!(!doc.is_empty());
        assert!(doc.ends_with('.') || doc.ends_with('!') || doc.ends_with('?'));
    }

    #[test]
    fn test_reproducibility() {
        let mut generator1 = DocumentGenerator::new(42);
        let mut generator2 = DocumentGenerator::new(42);

        let doc1 = generator1.generate();
        let doc2 = generator2.generate();

        assert_eq!(doc1, doc2);
    }

    #[test]
    fn test_generate_batch() {
        let mut generator = DocumentGenerator::new(42);
        let docs: Vec<_> = generator.generate_batch(10).collect();
        assert_eq!(docs.len(), 10);
        for doc in docs {
            assert!(!doc.is_empty());
        }
    }

    #[test]
    fn test_generate_with_length() {
        let mut generator = DocumentGenerator::new(42);
        let doc = generator.generate_with_length(50);
        // Count words (split by whitespace)
        let word_count = doc.split_whitespace().count();
        assert_eq!(word_count, 50);
    }

    #[test]
    fn test_short_documents_config() {
        let config = GeneratorConfig::short_documents();
        let mut generator = DocumentGenerator::with_config(42, config);

        let docs: Vec<_> = generator.generate_batch(100).collect();
        let avg_words: f64 = docs
            .iter()
            .map(|d| d.split_whitespace().count() as f64)
            .sum::<f64>()
            / 100.0;

        // Short docs should average under 30 words
        assert!(avg_words < 30.0, "Average was {}", avg_words);
    }

    #[test]
    fn test_long_documents_config() {
        let config = GeneratorConfig::long_documents();
        let mut generator = DocumentGenerator::with_config(42, config);

        let docs: Vec<_> = generator.generate_batch(50).collect();
        let avg_words: f64 = docs
            .iter()
            .map(|d| d.split_whitespace().count() as f64)
            .sum::<f64>()
            / 50.0;

        // Long docs should average over 200 words
        assert!(avg_words > 200.0, "Average was {}", avg_words);
    }

    #[test]
    fn test_zipf_distribution() {
        let mut generator = DocumentGenerator::new(42);
        let mut word_counts = std::collections::HashMap::new();

        // Generate many words and count frequencies
        for _ in 0..10000 {
            let word = generator.sample_word();
            *word_counts.entry(word).or_insert(0) += 1;
        }

        // "the" should be very common (in top few)
        let the_count = word_counts.get("the").unwrap_or(&0);
        // Less common words should appear less
        let less_common = word_counts.get("revolution").unwrap_or(&0);

        assert!(
            the_count > less_common,
            "'the' ({}) should appear more than 'revolution' ({})",
            the_count,
            less_common
        );
    }

    #[test]
    fn test_vocabulary_not_empty() {
        assert!(!VOCABULARY.is_empty());
        assert!(VOCABULARY.len() >= 500);
    }
}
