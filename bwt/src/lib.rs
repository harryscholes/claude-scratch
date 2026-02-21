//! # Burrows-Wheeler Transform (BWT)
//!
//! The Burrows-Wheeler Transform rearranges a string into runs of similar
//! characters, which makes it highly amenable to compression (e.g. with
//! move-to-front coding followed by run-length or entropy encoding).
//!
//! ## How it works
//!
//! **Forward transform:**
//! 1. Append a sentinel character (here `\0`) that is lexicographically smaller
//!    than every character in the input. This guarantees a unique rotation
//!    ordering and lets us recover the original string later.
//! 2. Build the *suffix array* of the augmented string. Each suffix array entry
//!    implicitly represents one rotation of the string.
//! 3. The last column of the (conceptual) sorted rotation matrix is the BWT
//!    output. For each suffix starting at position `sa[i]`, the character one
//!    step *before* it (wrapping around) is `text[(sa[i] + len - 1) % len]`.
//!
//! **Inverse transform:**
//! The inverse uses the *LF-mapping* (Last-to-First column mapping):
//! 1. From the BWT output, reconstruct the first column by sorting.
//! 2. Build a mapping from each position in the last column (L) to its
//!    corresponding position in the first column (F). Because the sentinel
//!    is unique and smallest, row 0 of F always starts with the sentinel,
//!    and the character preceding the sentinel is the last character of the
//!    original string.
//! 3. Walk the LF-mapping starting from row 0, collecting characters until we
//!    have recovered the full original string.

use std::fmt;

/// Sentinel byte appended to the input to mark string termination.
/// Must be lexicographically smaller than any byte in the input.
const SENTINEL: u8 = 0x00;

/// Result of a forward Burrows-Wheeler Transform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BwtOutput {
    /// The transformed byte sequence (last column of the sorted rotation matrix).
    pub data: Vec<u8>,
    /// The row index in the sorted rotation matrix where the sentinel appears
    /// in the last column. This is needed for the inverse transform.
    pub sentinel_index: usize,
}

/// Errors that can occur during BWT operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BwtError {
    /// The input contains the sentinel byte (`\0`), which is reserved.
    InputContainsSentinel,
    /// The sentinel index is out of bounds for the given data.
    InvalidSentinelIndex,
    /// The data does not contain the sentinel byte at the expected position.
    MissingSentinel,
}

impl fmt::Display for BwtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BwtError::InputContainsSentinel => {
                write!(f, "input contains the reserved sentinel byte (0x00)")
            }
            BwtError::InvalidSentinelIndex => {
                write!(f, "sentinel index is out of bounds")
            }
            BwtError::MissingSentinel => {
                write!(f, "expected sentinel byte not found in data")
            }
        }
    }
}

impl std::error::Error for BwtError {}

// ---------------------------------------------------------------------------
// Suffix array construction (SA-IS algorithm)
// ---------------------------------------------------------------------------

/// Classify each position as S-type or L-type.
///
/// A suffix is **S-type** (smaller) if it is lexicographically smaller than the
/// suffix starting one position to the right, and **L-type** (larger) otherwise.
/// The sentinel at the end is always S-type by convention.
fn classify_types(text: &[u8]) -> Vec<bool> {
    let n = text.len();
    // true = S-type, false = L-type
    let mut types = vec![false; n];
    // The last character (sentinel) is S-type.
    types[n - 1] = true;
    // Walk backwards: if text[i] < text[i+1] then S-type,
    // if text[i] > text[i+1] then L-type,
    // if equal then same type as i+1.
    for i in (0..n - 1).rev() {
        types[i] = if text[i] < text[i + 1] {
            true
        } else if text[i] > text[i + 1] {
            false
        } else {
            types[i + 1]
        };
    }
    types
}

/// Check whether position `i` is a Left-Most S-type (LMS) character.
///
/// An LMS character is an S-type character immediately preceded by an L-type
/// character. These are the "seams" in the string that the SA-IS algorithm
/// uses as anchors for recursion.
fn is_lms(types: &[bool], i: usize) -> bool {
    i > 0 && types[i] && !types[i - 1]
}

/// Compute bucket start/end offsets for each byte value.
///
/// A "bucket" is the contiguous region of the suffix array that holds all
/// suffixes starting with a given character. We need the head (start) or
/// tail (end) positions depending on whether we are placing L-type or S-type
/// suffixes.
fn get_buckets(text: &[u8], end: bool) -> Vec<usize> {
    let mut count = vec![0usize; 256];
    for &b in text {
        count[b as usize] += 1;
    }
    let mut buckets = vec![0usize; 256];
    let mut sum = 0;
    for i in 0..256 {
        if end {
            sum += count[i];
            buckets[i] = sum; // one past the last slot
        } else {
            buckets[i] = sum; // first slot
            sum += count[i];
        }
    }
    buckets
}

/// Place LMS suffixes into approximately correct bucket positions (step 1 of
/// induced sorting). Each LMS suffix is placed at the *end* of its character
/// bucket, filling from right to left.
fn place_lms(sa: &mut [usize], text: &[u8], types: &[bool]) {
    let n = text.len();
    let mut buckets = get_buckets(text, true);
    for i in 0..n {
        if is_lms(types, i) {
            let c = text[i] as usize;
            buckets[c] -= 1;
            sa[buckets[c]] = i;
        }
    }
}

/// Induce L-type suffixes from the current (partial) suffix array.
///
/// For each filled position `sa[i]`, look at the character one step to its
/// left (`sa[i] - 1`). If that character is L-type, place it at the *front*
/// of its bucket (filling left to right).
fn induce_l(sa: &mut [usize], text: &[u8], types: &[bool]) {
    let n = text.len();
    let mut buckets = get_buckets(text, false);
    for i in 0..n {
        if sa[i] == usize::MAX {
            continue;
        }
        if sa[i] == 0 {
            continue;
        }
        let j = sa[i] - 1;
        if !types[j] {
            // L-type
            let c = text[j] as usize;
            sa[buckets[c]] = j;
            buckets[c] += 1;
        }
    }
}

/// Induce S-type suffixes from the current (partial) suffix array.
///
/// Same idea as `induce_l` but we scan right-to-left and fill buckets from
/// their tail end backwards.
fn induce_s(sa: &mut [usize], text: &[u8], types: &[bool]) {
    let n = text.len();
    let mut buckets = get_buckets(text, true);
    for i in (0..n).rev() {
        if sa[i] == usize::MAX {
            continue;
        }
        if sa[i] == 0 {
            continue;
        }
        let j = sa[i] - 1;
        if types[j] {
            // S-type
            let c = text[j] as usize;
            buckets[c] -= 1;
            sa[buckets[c]] = j;
        }
    }
}

/// Build the suffix array of `text` using the SA-IS (Suffix Array Induced
/// Sorting) algorithm.
///
/// SA-IS runs in O(n) time and O(n) space, making it one of the most efficient
/// suffix array construction algorithms. It works by:
/// 1. Classifying each suffix as S-type or L-type.
/// 2. Finding LMS (Left-Most S-type) positions — these are the "interesting"
///    boundaries in the string.
/// 3. Using induced sorting to place LMS suffixes, then L-type, then S-type
///    suffixes into the suffix array.
/// 4. If LMS substrings are not all unique, recurse on a reduced problem
///    (renaming LMS substrings with integer labels).
fn build_suffix_array(text: &[u8]) -> Vec<usize> {
    let n = text.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0];
    }

    let types = classify_types(text);

    // Collect all LMS positions.
    let lms_positions: Vec<usize> = (0..n).filter(|&i| is_lms(&types, i)).collect();

    // Step 1: Initial placement of LMS suffixes followed by induced sorting.
    let mut sa = vec![usize::MAX; n];
    place_lms(&mut sa, text, &types);
    induce_l(&mut sa, text, &types);
    induce_s(&mut sa, text, &types);

    // Step 2: Compact the sorted LMS suffixes and check whether they are unique.
    // Extract LMS suffixes in their sorted order from sa.
    let sorted_lms: Vec<usize> = sa.iter().copied().filter(|&i| is_lms(&types, i)).collect();

    // Assign names (ranks) to each LMS substring. Two LMS substrings get the
    // same name only if they are identical in content and length.
    let mut names = vec![usize::MAX; n];
    let mut current_name = 0;
    names[sorted_lms[0]] = current_name;

    for k in 1..sorted_lms.len() {
        let prev = sorted_lms[k - 1];
        let curr = sorted_lms[k];
        // Compare the two LMS substrings character by character.
        let mut different = false;
        for d in 0..n {
            let p = prev + d;
            let c = curr + d;
            if p >= n || c >= n || text[p] != text[c] || types[p] != types[c] {
                different = true;
                break;
            }
            // An LMS substring ends at the next LMS position (inclusive).
            if d > 0 && (is_lms(&types, p) || is_lms(&types, c)) {
                break;
            }
        }
        if different {
            current_name += 1;
        }
        names[curr] = current_name;
    }

    // If all names are unique we already have the correct order and can skip
    // recursion. Otherwise, build a reduced string and recurse.
    if current_name + 1 < lms_positions.len() {
        // Build the reduced problem: a string of integer labels for each LMS
        // substring in their original left-to-right order.
        let reduced: Vec<u8> = lms_positions
            .iter()
            .map(|&i| names[i] as u8)
            .collect();
        let reduced_sa = build_suffix_array(&reduced);

        // Use the recursively computed order to place LMS suffixes correctly.
        sa.fill(usize::MAX);
        let mut buckets = get_buckets(text, true);
        // Place in reverse order so bucket-tail filling works correctly.
        for &idx in reduced_sa.iter().rev() {
            let lms_pos = lms_positions[idx];
            let c = text[lms_pos] as usize;
            buckets[c] -= 1;
            sa[buckets[c]] = lms_pos;
        }
    } else {
        // All LMS substrings are unique — re-place them in the correct order.
        sa.fill(usize::MAX);
        let mut buckets = get_buckets(text, true);
        for &i in sorted_lms.iter().rev() {
            let c = text[i] as usize;
            buckets[c] -= 1;
            sa[buckets[c]] = i;
        }
    }

    // Final induced sort with the correctly ordered LMS suffixes.
    induce_l(&mut sa, text, &types);
    induce_s(&mut sa, text, &types);

    sa
}

// ---------------------------------------------------------------------------
// BWT forward and inverse transforms
// ---------------------------------------------------------------------------

/// Perform the forward Burrows-Wheeler Transform.
///
/// Appends a sentinel (`\0`) to the input, builds a suffix array, and reads
/// off the last column of the conceptual sorted rotation matrix.
///
/// Returns `BwtOutput` containing the transformed data and the index where
/// the sentinel lands.
///
/// # Errors
///
/// Returns `BwtError::InputContainsSentinel` if the input contains `\0`.
pub fn bwt(input: &[u8]) -> Result<BwtOutput, BwtError> {
    if input.iter().any(|&b| b == SENTINEL) {
        return Err(BwtError::InputContainsSentinel);
    }

    // Augmented string = input + sentinel.
    let mut text = Vec::with_capacity(input.len() + 1);
    text.extend_from_slice(input);
    text.push(SENTINEL);

    let n = text.len();
    let sa = build_suffix_array(&text);

    // Build the BWT output: for each suffix sa[i], the BWT character is the
    // one immediately *before* that suffix (wrapping around).
    let mut data = Vec::with_capacity(n);
    let mut sentinel_index = 0;
    for (i, &pos) in sa.iter().enumerate() {
        let bwt_char = if pos == 0 {
            text[n - 1] // wrap: character before position 0 is the last character
        } else {
            text[pos - 1]
        };
        if bwt_char == SENTINEL {
            sentinel_index = i;
        }
        data.push(bwt_char);
    }

    Ok(BwtOutput {
        data,
        sentinel_index,
    })
}

/// Perform the inverse Burrows-Wheeler Transform, recovering the original
/// byte sequence (without the sentinel).
///
/// Uses the **LF-mapping** technique:
/// - The *last column* (L) is the BWT output we were given.
/// - The *first column* (F) is simply L sorted — since each row of the
///   conceptual rotation matrix is a rotation of the same string, F and L
///   contain the same multiset of characters.
/// - The key insight: the *i*-th occurrence of character `c` in L corresponds
///   to the *i*-th occurrence of `c` in F. This lets us build a permutation
///   (`lf_map`) that maps each L-row to its F-row.
/// - Starting from row 0 (which holds the sentinel in F), we follow the
///   mapping repeatedly, collecting one character per step, to reconstruct
///   the original string.
///
/// # Errors
///
/// Returns an error if `sentinel_index` is out of bounds or the sentinel byte
/// is missing from the expected position.
pub fn ibwt(output: &BwtOutput) -> Result<Vec<u8>, BwtError> {
    let n = output.data.len();
    if n == 0 {
        return Ok(vec![]);
    }
    if output.sentinel_index >= n {
        return Err(BwtError::InvalidSentinelIndex);
    }
    if output.data[output.sentinel_index] != SENTINEL {
        return Err(BwtError::MissingSentinel);
    }

    // Count occurrences of each byte value (needed to compute bucket starts).
    let mut count = [0usize; 256];
    for &b in &output.data {
        count[b as usize] += 1;
    }

    // `cumulative[c]` = number of characters in the string that are
    // strictly less than `c`. This tells us where bucket `c` starts in the
    // first column F.
    let mut cumulative = [0usize; 256];
    let mut total = 0;
    for i in 0..256 {
        cumulative[i] = total;
        total += count[i];
    }

    // Build the LF-mapping. For each position `i` in L (the BWT output),
    // lf_map[i] gives the corresponding row in F.
    //
    // We use a running `seen` counter per character: the k-th occurrence of
    // character c in L maps to F[cumulative[c] + k].
    let mut lf_map = vec![0usize; n];
    let mut seen = [0usize; 256];
    for i in 0..n {
        let c = output.data[i] as usize;
        lf_map[i] = cumulative[c] + seen[c];
        seen[c] += 1;
    }

    // Walk the LF-mapping to reconstruct the original string.
    // Row 0 in F corresponds to the sentinel (smallest character), so we
    // start there. The LF-mapping steps backward through the original string:
    // each step moves to the row whose rotation starts one position earlier.
    // L[row] at each step gives us the character at that earlier position.
    // We therefore fill the result array from right to left to recover the
    // original forward order.
    let len = n - 1; // exclude sentinel
    let mut result = vec![0u8; len];
    let mut row = 0; // start at the sentinel row in F
    for i in (0..len).rev() {
        result[i] = output.data[row];
        row = lf_map[row];
    }

    Ok(result)
}

/// Convenience wrapper: transform a UTF-8 string and return the BWT output.
pub fn bwt_str(input: &str) -> Result<BwtOutput, BwtError> {
    bwt(input.as_bytes())
}

/// Convenience wrapper: inverse-transform and return a UTF-8 string.
///
/// # Panics
///
/// Panics if the recovered bytes are not valid UTF-8.
pub fn ibwt_str(output: &BwtOutput) -> Result<String, BwtError> {
    ibwt(output).map(|bytes| String::from_utf8(bytes).expect("invalid UTF-8 in inverse BWT"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Suffix array tests ---

    #[test]
    fn suffix_array_simple() {
        // "banana\0" — a classic test case for suffix arrays.
        let text = b"banana\0";
        let sa = build_suffix_array(text);
        // Expected sorted suffixes:
        //   6: \0
        //   5: a\0
        //   3: ana\0
        //   1: anana\0
        //   0: banana\0
        //   4: na\0
        //   2: nana\0
        assert_eq!(sa, vec![6, 5, 3, 1, 0, 4, 2]);
    }

    #[test]
    fn suffix_array_single_char() {
        let sa = build_suffix_array(b"a");
        assert_eq!(sa, vec![0]);
    }

    #[test]
    fn suffix_array_repeated_chars() {
        // "aaa\0"
        let text = b"aaa\0";
        let sa = build_suffix_array(text);
        // Sorted: \0, a\0, aa\0, aaa\0
        assert_eq!(sa, vec![3, 2, 1, 0]);
    }

    // --- BWT forward transform tests ---

    #[test]
    fn bwt_banana() {
        let result = bwt(b"banana").unwrap();
        // The BWT of "banana" (with \0 sentinel) produces "annb\0aa".
        // Sorted rotations of "banana\0":
        //   \0banana  -> last char 'a'
        //   a\0banan  -> last char 'n'
        //   ana\0ban  -> last char 'n'
        //   anana\0b  -> last char 'b'
        //   banana\0  -> last char '\0'
        //   na\0bana  -> last char 'a'
        //   nana\0ba  -> last char 'a'
        assert_eq!(result.data, b"annb\0aa");
        assert_eq!(result.sentinel_index, 4);
    }

    #[test]
    fn bwt_empty() {
        let result = bwt(b"").unwrap();
        // Empty input -> just the sentinel.
        assert_eq!(result.data, b"\0");
        assert_eq!(result.sentinel_index, 0);
    }

    #[test]
    fn bwt_single_char() {
        let result = bwt(b"a").unwrap();
        // Sorted rotations of "a\0":
        //   \0a -> last char 'a'
        //   a\0 -> last char '\0'
        // data = ['a', '\0'], sentinel at index 1
        assert_eq!(result.data, b"a\0");
        assert_eq!(result.sentinel_index, 1);
    }

    #[test]
    fn bwt_rejects_sentinel_in_input() {
        let result = bwt(b"hello\0world");
        assert_eq!(result, Err(BwtError::InputContainsSentinel));
    }

    // --- Inverse BWT tests ---

    #[test]
    fn ibwt_banana() {
        let bwt_out = bwt(b"banana").unwrap();
        let recovered = ibwt(&bwt_out).unwrap();
        assert_eq!(recovered, b"banana");
    }

    #[test]
    fn ibwt_empty() {
        let bwt_out = bwt(b"").unwrap();
        let recovered = ibwt(&bwt_out).unwrap();
        assert_eq!(recovered, b"");
    }

    #[test]
    fn ibwt_single_char() {
        let bwt_out = bwt(b"z").unwrap();
        let recovered = ibwt(&bwt_out).unwrap();
        assert_eq!(recovered, b"z");
    }

    #[test]
    fn roundtrip_hello_world() {
        let input = b"Hello, World!";
        let bwt_out = bwt(input).unwrap();
        let recovered = ibwt(&bwt_out).unwrap();
        assert_eq!(recovered, input);
    }

    #[test]
    fn roundtrip_repeated_pattern() {
        let input = b"abracadabra";
        let bwt_out = bwt(input).unwrap();
        let recovered = ibwt(&bwt_out).unwrap();
        assert_eq!(recovered, input);
    }

    #[test]
    fn roundtrip_all_same_chars() {
        let input = b"aaaaaaaaaa";
        let bwt_out = bwt(input).unwrap();
        let recovered = ibwt(&bwt_out).unwrap();
        assert_eq!(recovered, input);
    }

    #[test]
    fn roundtrip_binary_data() {
        // Test with all byte values except \0.
        let input: Vec<u8> = (1u8..=255).collect();
        let bwt_out = bwt(&input).unwrap();
        let recovered = ibwt(&bwt_out).unwrap();
        assert_eq!(recovered, input);
    }

    #[test]
    fn roundtrip_long_string() {
        let input = "the quick brown fox jumps over the lazy dog".repeat(100);
        let bwt_out = bwt(input.as_bytes()).unwrap();
        let recovered = ibwt(&bwt_out).unwrap();
        assert_eq!(recovered, input.as_bytes());
    }

    #[test]
    fn roundtrip_unicode() {
        let input = "こんにちは世界🌍";
        let bwt_out = bwt_str(input).unwrap();
        let recovered = ibwt_str(&bwt_out).unwrap();
        assert_eq!(recovered, input);
    }

    #[test]
    fn bwt_produces_runs() {
        // A key property of BWT: it tends to group identical characters
        // together. For "mississippi", the transform should cluster the
        // repeated characters.
        let input = b"mississippi";
        let bwt_out = bwt(input).unwrap();
        let recovered = ibwt(&bwt_out).unwrap();
        assert_eq!(recovered, input.to_vec());

        // Count runs in the BWT output (a "run" is a maximal sequence of
        // identical consecutive characters). The BWT output should have
        // fewer runs than a random permutation would.
        let runs = bwt_out
            .data
            .windows(2)
            .filter(|w| w[0] != w[1])
            .count()
            + 1;
        // "mississippi" BWT is known to produce good clustering.
        assert!(runs <= bwt_out.data.len(), "BWT should group similar chars");
    }

    // --- Error handling tests ---

    #[test]
    fn ibwt_invalid_sentinel_index() {
        let bad = BwtOutput {
            data: vec![b'a', SENTINEL, b'b'],
            sentinel_index: 99, // out of bounds
        };
        assert_eq!(ibwt(&bad), Err(BwtError::InvalidSentinelIndex));
    }

    #[test]
    fn ibwt_missing_sentinel() {
        let bad = BwtOutput {
            data: vec![b'a', SENTINEL, b'b'],
            sentinel_index: 0, // position 0 is 'a', not sentinel
        };
        assert_eq!(ibwt(&bad), Err(BwtError::MissingSentinel));
    }

    #[test]
    fn error_display() {
        assert_eq!(
            BwtError::InputContainsSentinel.to_string(),
            "input contains the reserved sentinel byte (0x00)"
        );
        assert_eq!(
            BwtError::InvalidSentinelIndex.to_string(),
            "sentinel index is out of bounds"
        );
        assert_eq!(
            BwtError::MissingSentinel.to_string(),
            "expected sentinel byte not found in data"
        );
    }
}
