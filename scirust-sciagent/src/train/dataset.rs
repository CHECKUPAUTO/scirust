use std::path::Path;

/// `<unk>` id in the BPE special-token table (see `bpe::SPECIAL_TOKENS`).
/// Out-of-vocab ids map here — NOT to `vocab_size - 1`, which is an ordinary
/// learned merge that would receive spurious probability mass.
const UNK_ID: usize = 3;

pub struct PretrainDataset {
    data: Vec<u32>,
    position: usize,
    seq_len: usize,
    vocab_size: usize,
}

impl PretrainDataset {
    pub fn from_slice(data: &[u32], seq_len: usize, vocab_size: usize) -> Self {
        Self {
            data: data.to_vec(),
            position: 0,
            seq_len,
            vocab_size,
        }
    }

    pub fn len(&self) -> usize {
        if self.data.len() <= self.seq_len + 1
        {
            0
        }
        else
        {
            (self.data.len() - 1) / self.seq_len
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn reset(&mut self) {
        self.position = 0;
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Option<(Vec<usize>, Vec<usize>)> {
        let total_needed = batch_size * (self.seq_len + 1);
        if self.position + total_needed > self.data.len()
        {
            self.position = 0;
            if self.data.len() < total_needed
            {
                return None;
            }
        }

        let mut inputs = Vec::with_capacity(batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(batch_size * self.seq_len);

        let vocab = self.vocab_size;
        let sanitize = move |tok: usize| {
            if tok < vocab
            {
                tok
            }
            else
            {
                UNK_ID.min(vocab - 1)
            }
        };
        for _ in 0..batch_size
        {
            let start = self.position;
            for j in 0..self.seq_len
            {
                inputs.push(sanitize(self.data[start + j] as usize));
                targets.push(sanitize(self.data[start + j + 1] as usize));
            }
            self.position += self.seq_len;
        }

        Some((inputs, targets))
    }

    pub fn shuffle(&mut self, seed: u64) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let n = self.len();
        if n <= 1
        {
            return;
        }
        let mut indices: Vec<usize> = (0..n).collect();

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let mut state = hasher.finish();
        for i in (1..n).rev()
        {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (state >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }

        let mut new_data = Vec::with_capacity(self.data.len());
        for &idx in &indices
        {
            let start = idx * self.seq_len;
            let end = (start + self.seq_len + 1).min(self.data.len());
            new_data.extend_from_slice(&self.data[start..end]);
        }
        self.data = new_data;
        self.position = 0;
    }
}

pub struct ShardLoader {
    buffer: Vec<u32>,
}

impl Default for ShardLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl ShardLoader {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub fn load_bin<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<()> {
        let bytes = std::fs::read(path.as_ref())?;
        let mut data = vec![0u32; bytes.len() / 4];
        for (i, chunk) in bytes.chunks_exact(4).enumerate()
        {
            data[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        self.buffer = data;
        Ok(())
    }

    pub fn load_dir<P: AsRef<Path>>(&mut self, dir: P) -> std::io::Result<()> {
        let mut all_data = Vec::new();
        let mut entries: Vec<_> = std::fs::read_dir(dir.as_ref())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "bin"))
            .collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in &entries
        {
            let bytes = std::fs::read(entry.path())?;
            for chunk in bytes.chunks_exact(4)
            {
                all_data.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        self.buffer = all_data;
        Ok(())
    }

    pub fn total_tokens(&self) -> usize {
        self.buffer.len()
    }

    /// The raw loaded token buffer (little-endian `u32` ids, unsanitised). Lets a
    /// caller inspect the id range — e.g. to detect a vocab / tokenizer mismatch
    /// before training — or stream the tokens directly.
    pub fn tokens(&self) -> &[u32] {
        &self.buffer
    }

    pub fn into_dataset(self, seq_len: usize, vocab_size: usize) -> PretrainDataset {
        PretrainDataset::from_slice(&self.buffer, seq_len, vocab_size)
    }
}

// ---- corpus-walk helpers (shared by collect-data / train-tokenizer) ----

/// Parse a comma-separated extension spec (`"rs,md,toml"` or `".rs, .md"`) into a
/// normalized lowercase list with any leading dots stripped. Empty entries drop.
pub fn parse_extensions(spec: &str) -> Vec<String> {
    spec.split(',')
        .map(|e| e.trim().trim_start_matches('.').to_ascii_lowercase())
        .filter(|e| !e.is_empty())
        .collect()
}

/// Whether `path`'s extension is one of `exts` (case-insensitive).
pub fn matches_extension(path: &Path, exts: &[String]) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .map(|e| exts.contains(&e))
        .unwrap_or(false)
}

/// Directories never worth walking for source-corpus collection — VCS internals,
/// build artifacts, vendored deps, caches. Skipping them keeps binary/generated
/// noise (and, for byte-level, `.git`'s packed objects) out of the corpus.
pub fn skip_source_dir(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | ".hg"
            | ".svn"
            | "target"
            | "node_modules"
            | ".cargo"
            | "dist"
            | "build"
            | ".venv"
            | "venv"
            | "__pycache__"
            | ".mypy_cache"
            | ".pytest_cache"
            | ".idea"
            | ".vscode"
    )
}

/// Heuristic corpus-quality gate: does `content` (from a file named `name`) look
/// like human-written **source**, or low-value bulk — generated code, lockfiles,
/// minified blobs, numeric/string **data tables** — that dilutes a code model?
/// Returns `Err(reason)` when the file should be **skipped**, `Ok(())` to keep it.
///
/// Deliberately **conservative**: it rejects only clearly-non-source files, so an
/// ordinary `.rs` full of macros, match arms, or a small lookup table still passes.
/// The step_20000 samples were heavy on quoted-string / test-fixture soup and leaked
/// generated data; this trims the egregious cases without hand-curating the corpus.
pub fn source_quality(name: &str, content: &str) -> Result<(), &'static str> {
    if content.trim().is_empty()
    {
        return Err("empty");
    }

    // Lockfiles / minified assets — by name (never useful training signal).
    let lname = name.to_ascii_lowercase();
    if lname == "cargo.lock"
        || lname.ends_with(".lock")
        || lname.ends_with("-lock.json")
        || lname.ends_with(".min.js")
        || lname.ends_with(".min.css")
    {
        return Err("lockfile/minified (by name)");
    }

    // Declared-generated files — a marker in the header comment.
    let head = content.get(..content.len().min(2048)).unwrap_or(content);
    let hl = head.to_ascii_lowercase();
    if hl.contains("@generated")
        || hl.contains("do not edit")
        || hl.contains("automatically generated")
        || hl.contains("auto-generated")
        || hl.contains("code generated by")
    {
        return Err("declared generated");
    }

    // Minified / single-giant-line blobs (data URIs, embedded assets, packed JS).
    let mut n_lines = 0usize;
    let mut max_line = 0usize;
    for l in content.lines()
    {
        n_lines += 1;
        max_line = max_line.max(l.len());
    }
    if max_line > 10_000
    {
        return Err("minified (giant line)");
    }
    if content.len() / n_lines.max(1) > 400
    {
        return Err("minified (long mean line)");
    }

    // Data-table dominated: over the non-whitespace characters, too few letters
    // (numeric tables, base64/hex blobs, string-literal soup) or too many digits.
    // Only judged on files big enough for the ratio to be meaningful.
    let (mut alpha, mut digit, mut nonws) = (0usize, 0usize, 0usize);
    for c in content.chars()
    {
        if c.is_whitespace()
        {
            continue;
        }
        nonws += 1;
        if c.is_alphabetic()
        {
            alpha += 1;
        }
        else if c.is_ascii_digit()
        {
            digit += 1;
        }
    }
    if nonws >= 200
    {
        if (alpha as f32 / nonws as f32) < 0.25
        {
            return Err("low letter density (data table)");
        }
        if (digit as f32 / nonws as f32) > 0.40
        {
            return Err("high digit density (numeric data)");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_basic() {
        let data: Vec<u32> = (0..20).collect();
        let mut ds = PretrainDataset::from_slice(&data, 4, 100);
        assert!(!ds.is_empty());
        let (inputs, targets) = ds.next_batch(1).unwrap();
        assert_eq!(inputs, vec![0, 1, 2, 3]);
        assert_eq!(targets, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_dataset_wraps_around() {
        let data: Vec<u32> = (0..10).collect();
        let mut ds = PretrainDataset::from_slice(&data, 4, 100);
        let _ = ds.next_batch(2);
        assert!(ds.next_batch(2).is_some() || ds.position == 0);
    }

    #[test]
    fn test_shuffle_changes_order() {
        let data: Vec<u32> = (0..50).collect();
        let mut ds1 = PretrainDataset::from_slice(&data, 5, 100);
        let mut ds2 = PretrainDataset::from_slice(&data, 5, 100);
        ds2.shuffle(12345);
        let b1 = ds1.next_batch(1).unwrap();
        let b2 = ds2.next_batch(1).unwrap();
        assert_ne!(b1.0, b2.0, "Shuffle should reorder data");
    }

    #[test]
    fn test_shuffle_deterministic() {
        let data: Vec<u32> = (0..50).collect();
        let mut ds1 = PretrainDataset::from_slice(&data, 5, 100);
        let mut ds2 = PretrainDataset::from_slice(&data, 5, 100);
        ds1.shuffle(42);
        ds2.shuffle(42);
        let b1 = ds1.next_batch(2).unwrap();
        let b2 = ds2.next_batch(2).unwrap();
        assert_eq!(b1.0, b2.0, "Same seed should produce same shuffle");
        assert_eq!(b1.1, b2.1);
    }

    #[test]
    fn test_oov_maps_to_unk() {
        let data = vec![0u32, 1, 2, 200, 4];
        let mut ds = PretrainDataset::from_slice(&data, 4, 10);
        let (inputs, targets) = ds.next_batch(1).unwrap();
        assert_eq!(inputs[3], 3, "OOV token 200 should map to <unk>=3");
        assert_eq!(targets[2], 3, "OOV target 200 should map to <unk>=3");
        assert_eq!(targets[3], 4, "Target 4 should be unchanged");
    }

    #[test]
    fn source_quality_keeps_real_rust() {
        // Ordinary source — including a file with a small numeric lookup table and
        // plenty of punctuation — must pass (the filter is conservative).
        let real = "\
//! A module doc comment.
use std::collections::HashMap;

/// Doc: computes something.
pub fn f(x: u32) -> u32 {
    let table = [1u32, 2, 3, 5, 8, 13, 21];
    table.iter().map(|&t| t * x).sum::<u32>() + x.wrapping_mul(31)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn t() { assert_eq!(f(0), 0); }
}
";
        assert!(
            source_quality("lib.rs", real).is_ok(),
            "real Rust must pass"
        );
    }

    #[test]
    fn source_quality_rejects_junk() {
        assert!(
            source_quality("Cargo.lock", "anything").is_err(),
            "lockfile"
        );
        assert!(source_quality("x.rs", "   \n\t  ").is_err(), "empty");
        assert!(
            source_quality("g.rs", "// @generated by prost\nstruct X;").is_err(),
            "declared generated"
        );
        // Minified: one enormous line.
        let big = format!("const D: &str = \"{}\";", "a".repeat(20_000));
        assert!(
            source_quality("blob.rs", &big).is_err(),
            "minified giant line"
        );
        // Numeric data table: mostly digits.
        let nums = (0..500).map(|i| format!("{i},")).collect::<String>();
        assert!(
            source_quality("data.rs", &nums).is_err(),
            "numeric data table"
        );
    }
}
