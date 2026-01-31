use blake3::Hasher;
use clap::Parser;
use colored::Colorize;
use crossbeam_channel::bounded;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::SystemTime;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexWriter, ReloadPolicy};
use tempfile::TempDir;
use walkdir::WalkDir;

/// rt - Ripgrep-like search tool powered by Tantivy
///
/// Recursively indexes files and searches using full-text search capabilities.
#[derive(Parser, Debug)]
#[command(name = "rt", version, about, long_about = None)]
struct Args {
    /// The search query (supports Tantivy query syntax)
    query: String,

    /// Path to search in (defaults to current directory)
    #[arg(default_value = ".")]
    path: PathBuf,

    /// Number of context lines to show before and after matches
    #[arg(short = 'C', long, default_value = "2")]
    context: usize,

    /// Maximum number of results to return (0 for unlimited)
    #[arg(short = 'n', long, default_value = "100")]
    max_results: usize,

    /// Include hidden files and directories
    #[arg(short = 'H', long)]
    hidden: bool,

    /// File extensions to include (e.g., -e rs -e py)
    #[arg(short = 'e', long = "ext", value_name = "EXT")]
    extensions: Vec<String>,

    /// Follow symbolic links
    #[arg(short = 'L', long)]
    follow_links: bool,

    /// Maximum depth to recurse (0 for unlimited)
    #[arg(short = 'd', long, default_value = "0")]
    max_depth: usize,

    /// Use hierarchical indexing for faster streaming results
    #[arg(long)]
    hierarchical: bool,

    /// Memory budget for indexing in MB
    #[arg(long, default_value = "50")]
    memory_mb: usize,

    /// Number of parallel indexing threads (0 for auto)
    #[arg(short = 'j', long, default_value = "0")]
    threads: usize,

    /// Only show file paths, not content
    #[arg(short = 'l', long)]
    files_only: bool,

    /// Case-insensitive search
    #[arg(short = 'i', long)]
    ignore_case: bool,

    /// Disable index caching (always rebuild)
    #[arg(long)]
    no_cache: bool,

    /// Clear the cache for this directory before searching
    #[arg(long)]
    clear_cache: bool,

    /// Custom cache directory (defaults to ~/.cache/rt)
    #[arg(long, value_name = "DIR")]
    cache_dir: Option<PathBuf>,

    /// Show cache statistics
    #[arg(long)]
    cache_stats: bool,
}

/// Represents a search match with context
#[derive(Debug, Clone)]
struct SearchMatch {
    file_path: PathBuf,
    line_number: usize,
    line_content: String,
    context_before: Vec<(usize, String)>,
    context_after: Vec<(usize, String)>,
    score: f32,
}

/// Represents a file to be indexed
#[derive(Debug)]
struct FileEntry {
    path: PathBuf,
    content: String,
    lines: Vec<(usize, usize)>, // (start_offset, end_offset) for each line
}

/// Metadata for a single file used for cache invalidation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileMetadata {
    path: String,
    size: u64,
    modified: u64, // seconds since UNIX epoch
}

/// Cache manifest stored alongside the index
#[derive(Debug, Serialize, Deserialize)]
struct CacheManifest {
    version: u32,
    hash: String,
    file_count: usize,
    created: u64,
    search_path: String,
    extensions: Vec<String>,
    hidden: bool,
    max_depth: usize,
    follow_links: bool,
}

const CACHE_VERSION: u32 = 1;

/// Manages the disk cache for Tantivy indexes
struct CacheManager {
    cache_dir: PathBuf,
}

impl CacheManager {
    fn new(custom_dir: Option<PathBuf>) -> Option<Self> {
        let cache_dir = custom_dir.or_else(|| {
            dirs::cache_dir().map(|d| d.join("rt"))
        })?;

        // Create cache directory if it doesn't exist
        fs::create_dir_all(&cache_dir).ok()?;

        Some(CacheManager { cache_dir })
    }

    /// Get the cache directory for a specific hash
    fn get_index_dir(&self, hash: &str) -> PathBuf {
        self.cache_dir.join(hash)
    }

    /// Check if a valid cached index exists
    fn get_cached_index(&self, hash: &str) -> Option<Index> {
        let index_dir = self.get_index_dir(hash);
        let manifest_path = index_dir.join("manifest.json");

        // Check if manifest exists and is valid
        let manifest_data = fs::read_to_string(&manifest_path).ok()?;
        let manifest: CacheManifest = serde_json::from_str(&manifest_data).ok()?;

        // Verify version compatibility
        if manifest.version != CACHE_VERSION {
            return None;
        }

        // Verify hash matches
        if manifest.hash != hash {
            return None;
        }

        // Try to open the index
        Index::open_in_dir(&index_dir).ok()
    }

    /// Save an index to the cache
    fn save_index(
        &self,
        hash: &str,
        files: &[FileEntry],
        args: &Args,
        search_path: &Path,
        memory_mb: usize,
    ) -> Option<Index> {
        let index_dir = self.get_index_dir(hash);

        // Remove old cache if it exists
        if index_dir.exists() {
            fs::remove_dir_all(&index_dir).ok()?;
        }

        fs::create_dir_all(&index_dir).ok()?;

        // Create and populate the index
        let index = create_index(&index_dir).ok()?;
        let file_count = Arc::new(AtomicUsize::new(0));
        index_files(&index, files, memory_mb, file_count);

        // Write manifest
        let manifest = CacheManifest {
            version: CACHE_VERSION,
            hash: hash.to_string(),
            file_count: files.len(),
            created: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            search_path: search_path.to_string_lossy().to_string(),
            extensions: args.extensions.clone(),
            hidden: args.hidden,
            max_depth: args.max_depth,
            follow_links: args.follow_links,
        };

        let manifest_json = serde_json::to_string_pretty(&manifest).ok()?;
        fs::write(index_dir.join("manifest.json"), manifest_json).ok()?;

        Some(index)
    }

    /// Clear cache for a specific hash
    fn clear_cache(&self, hash: &str) -> bool {
        let index_dir = self.get_index_dir(hash);
        if index_dir.exists() {
            fs::remove_dir_all(&index_dir).is_ok()
        } else {
            true
        }
    }

    /// Get cache statistics
    fn get_stats(&self) -> CacheStats {
        let mut stats = CacheStats::default();

        if let Ok(entries) = fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    stats.index_count += 1;
                    if let Ok(size) = dir_size(&entry.path()) {
                        stats.total_size += size;
                    }

                    // Read manifest for details
                    let manifest_path = entry.path().join("manifest.json");
                    if let Ok(data) = fs::read_to_string(&manifest_path) {
                        if let Ok(manifest) = serde_json::from_str::<CacheManifest>(&data) {
                            stats.total_files += manifest.file_count;
                        }
                    }
                }
            }
        }

        stats
    }
}

#[derive(Default)]
struct CacheStats {
    index_count: usize,
    total_size: u64,
    total_files: usize,
}

/// Calculate directory size recursively
fn dir_size(path: &Path) -> std::io::Result<u64> {
    let mut size = 0;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            size += dir_size(&entry.path())?;
        } else {
            size += metadata.len();
        }
    }
    Ok(size)
}

/// Compute a hash of file metadata for cache invalidation
fn compute_files_hash(args: &Args, search_path: &Path) -> (String, Vec<FileMetadata>) {
    let mut walker = WalkDir::new(search_path).follow_links(args.follow_links);

    if args.max_depth > 0 {
        walker = walker.max_depth(args.max_depth);
    }

    let extensions: HashSet<&str> = args.extensions.iter().map(|s| s.as_str()).collect();

    let mut file_metas: Vec<FileMetadata> = walker
        .into_iter()
        .filter_entry(|e| {
            if !args.hidden && e.file_name().to_string_lossy().starts_with('.') {
                return false;
            }
            true
        })
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            if extensions.is_empty() {
                return true;
            }
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| extensions.contains(ext))
                .unwrap_or(false)
        })
        .filter(|e| !is_binary_file(e.path()))
        .filter_map(|e| {
            let metadata = e.metadata().ok()?;
            let modified = metadata
                .modified()
                .ok()?
                .duration_since(SystemTime::UNIX_EPOCH)
                .ok()?
                .as_secs();
            Some(FileMetadata {
                path: e.path().to_string_lossy().to_string(),
                size: metadata.len(),
                modified,
            })
        })
        .collect();

    // Sort for deterministic hashing
    file_metas.sort_by(|a, b| a.path.cmp(&b.path));

    // Compute hash
    let mut hasher = Hasher::new();

    // Include search parameters in hash
    hasher.update(search_path.to_string_lossy().as_bytes());
    hasher.update(&[args.hidden as u8]);
    hasher.update(&args.max_depth.to_le_bytes());
    hasher.update(&[args.follow_links as u8]);
    for ext in &args.extensions {
        hasher.update(ext.as_bytes());
    }

    // Include file metadata
    for meta in &file_metas {
        hasher.update(meta.path.as_bytes());
        hasher.update(&meta.size.to_le_bytes());
        hasher.update(&meta.modified.to_le_bytes());
    }

    let hash = hasher.finalize();
    (hash.to_hex().to_string(), file_metas)
}

fn main() {
    let args = Args::parse();

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .ok();
    }

    let search_path = args.path.canonicalize().unwrap_or_else(|_| args.path.clone());

    if !search_path.exists() {
        eprintln!("{}: Path does not exist: {}", "error".red().bold(), search_path.display());
        std::process::exit(1);
    }

    // Handle cache stats request
    if args.cache_stats {
        if let Some(cache_mgr) = CacheManager::new(args.cache_dir.clone()) {
            let stats = cache_mgr.get_stats();
            println!("{}", "Cache Statistics".bold().cyan());
            println!("  Cache directory: {}", cache_mgr.cache_dir.display());
            println!("  Cached indexes:  {}", stats.index_count);
            println!("  Total files:     {}", stats.total_files);
            println!("  Total size:      {}", format_size(stats.total_size));
        } else {
            eprintln!("{}: Could not access cache directory", "error".red().bold());
        }
        return;
    }

    if args.hierarchical {
        run_hierarchical_search(&args, &search_path);
    } else {
        run_flat_search(&args, &search_path);
    }
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Run a flat search that indexes all files first, then searches
fn run_flat_search(args: &Args, search_path: &Path) {
    let cache_mgr = if args.no_cache {
        None
    } else {
        CacheManager::new(args.cache_dir.clone())
    };

    // Compute hash for cache lookup
    let (hash, _file_metas) = compute_files_hash(args, search_path);

    // Handle clear cache request
    if args.clear_cache {
        if let Some(ref mgr) = cache_mgr {
            if mgr.clear_cache(&hash) {
                eprintln!("{}: Cache cleared", "info".blue().bold());
            }
        }
    }

    // Try to use cached index
    let (index, files, used_cache) = if let Some(ref mgr) = cache_mgr {
        if let Some(cached_index) = mgr.get_cached_index(&hash) {
            // Load files for display (we still need content for showing matches)
            let files = collect_files(args, search_path);
            (cached_index, files, true)
        } else {
            // Build new index and cache it
            let files = collect_files(args, search_path);
            if files.is_empty() {
                eprintln!("{}: No files found to search", "warning".yellow().bold());
                return;
            }

            eprint!("Indexing {} files... ", files.len());
            let index = mgr
                .save_index(&hash, &files, args, search_path, args.memory_mb)
                .expect("Failed to create cached index");
            eprintln!("{} {}", "done".green(), "(cached)".dimmed());

            (index, files, false)
        }
    } else {
        // No caching - use temp directory
        let files = collect_files(args, search_path);
        if files.is_empty() {
            eprintln!("{}: No files found to search", "warning".yellow().bold());
            return;
        }

        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let index = create_index(temp_dir.path()).expect("Failed to create index");

        let file_count = Arc::new(AtomicUsize::new(0));
        eprint!("Indexing {} files... ", files.len());
        index_files(&index, &files, args.memory_mb, Arc::clone(&file_count));
        eprintln!("{}", "done".green());

        // Keep temp_dir alive by leaking it (it will be cleaned up on process exit)
        std::mem::forget(temp_dir);

        (index, files, false)
    };

    if used_cache {
        eprintln!("{}: Using cached index ({} files)", "cache".blue().bold(), files.len());
    }

    let matches = search_index(&index, &args.query, args.max_results, args.ignore_case);
    display_matches(args, &files, &matches);
}

/// Run a hierarchical search that indexes and searches directory by directory
fn run_hierarchical_search(args: &Args, search_path: &Path) {
    let (tx, rx) = bounded::<SearchMatch>(1000);

    let args_clone = args.clone();
    let search_path_clone = search_path.to_path_buf();

    // Spawn display thread
    let display_handle = thread::spawn(move || {
        let mut seen_files: HashSet<PathBuf> = HashSet::new();
        let mut match_count = 0;

        for search_match in rx {
            if args_clone.files_only {
                if seen_files.insert(search_match.file_path.clone()) {
                    println!("{}", search_match.file_path.display());
                }
            } else {
                display_single_match(&args_clone, &search_match, &mut seen_files);
            }
            match_count += 1;
            if args_clone.max_results > 0 && match_count >= args_clone.max_results {
                break;
            }
        }
    });

    // Group files by directory depth for hierarchical processing
    let files = collect_files(args, &search_path_clone);
    let mut depth_groups: HashMap<usize, Vec<&FileEntry>> = HashMap::new();

    for file in &files {
        let depth = file
            .path
            .strip_prefix(&search_path_clone)
            .map(|p| p.components().count())
            .unwrap_or(0);
        depth_groups.entry(depth).or_default().push(file);
    }

    let mut depths: Vec<usize> = depth_groups.keys().copied().collect();
    depths.sort();

    for depth in depths {
        if let Some(files_at_depth) = depth_groups.get(&depth) {
            if files_at_depth.is_empty() {
                continue;
            }

            let temp_dir = TempDir::new().expect("Failed to create temp directory");
            let index = create_index(temp_dir.path()).expect("Failed to create index");

            let file_count = Arc::new(AtomicUsize::new(0));
            index_files_slice(&index, files_at_depth, args.memory_mb, file_count);

            let matches = search_index(&index, &args.query, 0, args.ignore_case);

            for (doc_id, score) in matches {
                if let Some(file) = files_at_depth.get(doc_id as usize) {
                    let file_matches = find_matches_in_file(file, &args.query, args.context, score, args.ignore_case);
                    for m in file_matches {
                        if tx.send(m).is_err() {
                            break;
                        }
                    }
                }
            }
        }
    }

    drop(tx);
    display_handle.join().ok();
}

/// Collect all files to be indexed
fn collect_files(args: &Args, search_path: &Path) -> Vec<FileEntry> {
    let mut walker = WalkDir::new(search_path).follow_links(args.follow_links);

    if args.max_depth > 0 {
        walker = walker.max_depth(args.max_depth);
    }

    let extensions: HashSet<&str> = args.extensions.iter().map(|s| s.as_str()).collect();

    let paths: Vec<PathBuf> = walker
        .into_iter()
        .filter_entry(|e| {
            if !args.hidden && e.file_name().to_string_lossy().starts_with('.') {
                return false;
            }
            true
        })
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            if extensions.is_empty() {
                return true;
            }
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| extensions.contains(ext))
                .unwrap_or(false)
        })
        .filter(|e| !is_binary_file(e.path()))
        .map(|e| e.path().to_path_buf())
        .collect();

    paths
        .par_iter()
        .filter_map(|path| {
            let content = read_file_content(path)?;
            let lines = compute_line_offsets(&content);
            Some(FileEntry {
                path: path.clone(),
                content,
                lines,
            })
        })
        .collect()
}

/// Check if a file appears to be binary
fn is_binary_file(path: &Path) -> bool {
    // Check extension first for common binary types
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        let binary_extensions = [
            "png", "jpg", "jpeg", "gif", "bmp", "ico", "webp", "svg",
            "mp3", "mp4", "avi", "mov", "mkv", "wav", "flac",
            "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
            "zip", "tar", "gz", "bz2", "xz", "7z", "rar",
            "exe", "dll", "so", "dylib", "a", "o", "obj",
            "class", "pyc", "pyo", "wasm",
            "ttf", "otf", "woff", "woff2", "eot",
            "db", "sqlite", "sqlite3",
        ];
        if binary_extensions.contains(&ext.to_lowercase().as_str()) {
            return true;
        }
    }

    // Read first 8KB and check for null bytes
    if let Ok(file) = fs::File::open(path) {
        let mut reader = BufReader::new(file);
        let mut buffer = [0u8; 8192];
        if let Ok(bytes_read) = std::io::Read::read(&mut reader, &mut buffer) {
            return buffer[..bytes_read].contains(&0);
        }
    }

    false
}

/// Read file content, handling various encodings
fn read_file_content(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;

    // Try UTF-8 first
    if let Ok(content) = String::from_utf8(bytes.clone()) {
        return Some(content);
    }

    // Try to detect and convert encoding
    let (cow, _, had_errors) = encoding_rs::UTF_8.decode(&bytes);
    if !had_errors {
        return Some(cow.into_owned());
    }

    // Try other common encodings
    let encodings = [
        encoding_rs::WINDOWS_1252,
        encoding_rs::ISO_8859_2,
        encoding_rs::UTF_16LE,
        encoding_rs::UTF_16BE,
    ];
    for encoding in encodings {
        let (cow, _, had_errors) = encoding.decode(&bytes);
        if !had_errors {
            return Some(cow.into_owned());
        }
    }

    // Last resort: lossy UTF-8
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

/// Compute line offsets for a file's content
fn compute_line_offsets(content: &str) -> Vec<(usize, usize)> {
    let mut offsets = Vec::new();
    let mut start = 0;

    for (i, c) in content.char_indices() {
        if c == '\n' {
            offsets.push((start, i));
            start = i + 1;
        }
    }

    // Handle last line without newline
    if start < content.len() {
        offsets.push((start, content.len()));
    }

    offsets
}

/// Create a Tantivy index with our schema
fn create_index(path: &Path) -> tantivy::Result<Index> {
    let mut schema_builder = Schema::builder();

    schema_builder.add_text_field("path", STRING | STORED);
    schema_builder.add_text_field("content", TEXT | STORED);
    schema_builder.add_u64_field("doc_id", INDEXED | STORED);

    let schema = schema_builder.build();
    Index::create_in_dir(path, schema)
}

/// Index files into the Tantivy index
fn index_files(index: &Index, files: &[FileEntry], memory_mb: usize, file_count: Arc<AtomicUsize>) {
    let mut writer: IndexWriter = index
        .writer(memory_mb * 1024 * 1024)
        .expect("Failed to create index writer");

    let schema = index.schema();
    let path_field = schema.get_field("path").unwrap();
    let content_field = schema.get_field("content").unwrap();
    let doc_id_field = schema.get_field("doc_id").unwrap();

    for (doc_id, file) in files.iter().enumerate() {
        let doc = doc!(
            path_field => file.path.to_string_lossy().to_string(),
            content_field => file.content.clone(),
            doc_id_field => doc_id as u64
        );
        writer.add_document(doc).ok();
        file_count.fetch_add(1, Ordering::Relaxed);
    }

    writer.commit().expect("Failed to commit index");
}

/// Index a slice of file references
fn index_files_slice(
    index: &Index,
    files: &[&FileEntry],
    memory_mb: usize,
    file_count: Arc<AtomicUsize>,
) {
    let mut writer: IndexWriter = index
        .writer(memory_mb * 1024 * 1024)
        .expect("Failed to create index writer");

    let schema = index.schema();
    let path_field = schema.get_field("path").unwrap();
    let content_field = schema.get_field("content").unwrap();
    let doc_id_field = schema.get_field("doc_id").unwrap();

    for (doc_id, file) in files.iter().enumerate() {
        let doc = doc!(
            path_field => file.path.to_string_lossy().to_string(),
            content_field => file.content.clone(),
            doc_id_field => doc_id as u64
        );
        writer.add_document(doc).ok();
        file_count.fetch_add(1, Ordering::Relaxed);
    }

    writer.commit().expect("Failed to commit index");
}

/// Search the index and return matching document IDs with scores
fn search_index(index: &Index, query_str: &str, max_results: usize, ignore_case: bool) -> Vec<(u64, f32)> {
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .expect("Failed to create reader");

    let searcher = reader.searcher();
    let schema = index.schema();
    let content_field = schema.get_field("content").unwrap();
    let doc_id_field = schema.get_field("doc_id").unwrap();

    let query_parser = QueryParser::for_index(index, vec![content_field]);

    let query_string = if ignore_case {
        query_str.to_lowercase()
    } else {
        query_str.to_string()
    };

    let query = match query_parser.parse_query(&query_string) {
        Ok(q) => q,
        Err(e) => {
            eprintln!("{}: Failed to parse query: {}", "error".red().bold(), e);
            return Vec::new();
        }
    };

    let limit = if max_results == 0 { 10000 } else { max_results };
    let top_docs = match searcher.search(&query, &TopDocs::with_limit(limit)) {
        Ok(docs) => docs,
        Err(e) => {
            eprintln!("{}: Search failed: {}", "error".red().bold(), e);
            return Vec::new();
        }
    };

    top_docs
        .into_iter()
        .filter_map(|(score, doc_address)| {
            let doc: tantivy::TantivyDocument = searcher.doc(doc_address).ok()?;
            let doc_id = doc.get_first(doc_id_field)?.as_u64()?;
            Some((doc_id, score))
        })
        .collect()
}

/// Find matches within a specific file
fn find_matches_in_file(
    file: &FileEntry,
    query: &str,
    context_lines: usize,
    score: f32,
    ignore_case: bool,
) -> Vec<SearchMatch> {
    let mut matches = Vec::new();

    // Extract search terms from query (simple tokenization)
    let search_terms: Vec<&str> = query
        .split_whitespace()
        .filter(|s| !["AND", "OR", "NOT", "(", ")"].contains(s))
        .collect();

    if search_terms.is_empty() {
        return matches;
    }

    let content_to_search = if ignore_case {
        file.content.to_lowercase()
    } else {
        file.content.clone()
    };

    let terms_to_search: Vec<String> = if ignore_case {
        search_terms.iter().map(|s| s.to_lowercase()).collect()
    } else {
        search_terms.iter().map(|s| s.to_string()).collect()
    };

    for (line_idx, (start, end)) in file.lines.iter().enumerate() {
        let line_content = &content_to_search[*start..*end];

        let has_match = terms_to_search.iter().any(|term| line_content.contains(term));

        if has_match {
            let original_line = &file.content[*start..*end];

            let context_before: Vec<(usize, String)> = (0..context_lines)
                .rev()
                .filter_map(|i| {
                    let ctx_idx = line_idx.checked_sub(i + 1)?;
                    let (s, e) = file.lines.get(ctx_idx)?;
                    Some((ctx_idx + 1, file.content[*s..*e].to_string()))
                })
                .collect();

            let context_after: Vec<(usize, String)> = (1..=context_lines)
                .filter_map(|i| {
                    let ctx_idx = line_idx + i;
                    let (s, e) = file.lines.get(ctx_idx)?;
                    Some((ctx_idx + 1, file.content[*s..*e].to_string()))
                })
                .collect();

            matches.push(SearchMatch {
                file_path: file.path.clone(),
                line_number: line_idx + 1,
                line_content: original_line.to_string(),
                context_before,
                context_after,
                score,
            });
        }
    }

    matches
}

/// Display all matches
fn display_matches(args: &Args, files: &[FileEntry], matches: &[(u64, f32)]) {
    if matches.is_empty() {
        eprintln!("{}: No matches found", "info".blue().bold());
        return;
    }

    let mut seen_files: HashSet<PathBuf> = HashSet::new();

    for (doc_id, score) in matches {
        if let Some(file) = files.get(*doc_id as usize) {
            if args.files_only {
                if seen_files.insert(file.path.clone()) {
                    println!("{}", file.path.display());
                }
            } else {
                let file_matches =
                    find_matches_in_file(file, &args.query, args.context, *score, args.ignore_case);
                for m in file_matches {
                    display_single_match(args, &m, &mut seen_files);
                }
            }
        }
    }
}

/// Display a single match with context
fn display_single_match(args: &Args, m: &SearchMatch, seen_files: &mut HashSet<PathBuf>) {
    let is_new_file = seen_files.insert(m.file_path.clone());

    if is_new_file && !args.files_only {
        println!();
        println!("{}", m.file_path.display().to_string().magenta().bold());
    }

    // Print context before
    for (line_num, content) in &m.context_before {
        println!(
            "{}{} {}",
            line_num.to_string().green(),
            "-".dimmed(),
            content.dimmed()
        );
    }

    // Print matching line with highlighting
    let highlighted_line = highlight_matches(&m.line_content, &args.query, args.ignore_case);
    println!(
        "{}{} {}",
        m.line_number.to_string().green().bold(),
        ":".bold(),
        highlighted_line
    );

    // Print context after
    for (line_num, content) in &m.context_after {
        println!(
            "{}{} {}",
            line_num.to_string().green(),
            "-".dimmed(),
            content.dimmed()
        );
    }

    if !m.context_before.is_empty() || !m.context_after.is_empty() {
        println!("{}", "--".dimmed());
    }
}

/// Highlight search terms in the line
fn highlight_matches(line: &str, query: &str, ignore_case: bool) -> String {
    let terms: Vec<&str> = query
        .split_whitespace()
        .filter(|s| !["AND", "OR", "NOT", "(", ")"].contains(s))
        .collect();

    let mut result = line.to_string();

    for term in terms {
        if ignore_case {
            let lower_result = result.to_lowercase();
            let lower_term = term.to_lowercase();
            let mut new_result = String::new();
            let mut last_end = 0;

            for (start, _) in lower_result.match_indices(&lower_term) {
                new_result.push_str(&result[last_end..start]);
                let matched_text = &result[start..start + term.len()];
                new_result.push_str(&matched_text.red().bold().to_string());
                last_end = start + term.len();
            }
            new_result.push_str(&result[last_end..]);
            result = new_result;
        } else {
            result = result.replace(term, &term.red().bold().to_string());
        }
    }

    result
}

impl Clone for Args {
    fn clone(&self) -> Self {
        Args {
            query: self.query.clone(),
            path: self.path.clone(),
            context: self.context,
            max_results: self.max_results,
            hidden: self.hidden,
            extensions: self.extensions.clone(),
            follow_links: self.follow_links,
            max_depth: self.max_depth,
            hierarchical: self.hierarchical,
            memory_mb: self.memory_mb,
            threads: self.threads,
            files_only: self.files_only,
            ignore_case: self.ignore_case,
            no_cache: self.no_cache,
            clear_cache: self.clear_cache,
            cache_dir: self.cache_dir.clone(),
            cache_stats: self.cache_stats,
        }
    }
}
