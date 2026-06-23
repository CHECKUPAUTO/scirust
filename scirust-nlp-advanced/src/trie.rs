//! Compact prefix trie for token-sequence deduplication and shared-prefix
//! compression.
//!
//! A trie stores each unique path once, so a set of strings that shares long
//! common prefixes (file paths, node ids, import trees, code identifiers) is
//! stored in *O(total distinct characters)* rather than *O(total characters)*.
//! This is the data-structure-level complement of MinHash/LSH: MinHash tells
//! you *which* chunks are near-duplicates; the trie *physically shares* their
//! storage.
//!
//! This implementation is a bitwise radix trie over UTF-8 bytes, with
//! deterministic child ordering (B-tree by byte value) so two tries built from
//! the same insertion order are structurally identical and serialize the same.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A byte-radix trie node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrieNode {
    /// Byte → child node.
    pub children: BTreeMap<u8, TrieNode>,
    /// True iff a complete string ends at this node.
    pub is_end: bool,
    /// Number of strings that pass through this node (for pruning decisions).
    pub pass_count: usize,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: BTreeMap::new(),
            is_end: false,
            pass_count: 0,
        }
    }
}

/// A deterministic byte-radix trie.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trie {
    pub root: TrieNode,
    /// Total number of complete strings stored.
    pub len: usize,
}

impl Default for Trie {
    fn default() -> Self {
        Self::new()
    }
}

impl Trie {
    /// Empty trie.
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(),
            len: 0,
        }
    }

    /// Insert a byte string. Returns `true` if it was new, `false` if already
    /// present (idempotent).
    pub fn insert(&mut self, data: &[u8]) -> bool {
        let mut node = &mut self.root;
        node.pass_count += 1;
        for &b in data {
            node = node.children.entry(b).or_insert_with(TrieNode::new);
            node.pass_count += 1;
        }
        let was_new = !node.is_end;
        if was_new {
            node.is_end = true;
            self.len += 1;
        }
        was_new
    }

    /// String variant of [`insert`](Self::insert).
    pub fn insert_str(&mut self, s: &str) -> bool {
        self.insert(s.as_bytes())
    }

    /// True iff the exact string has been inserted.
    pub fn contains(&self, data: &[u8]) -> bool {
        let mut node = &self.root;
        for &b in data {
            match node.children.get(&b) {
                Some(n) => node = n,
                None => return false,
            }
        }
        node.is_end
    }

    /// String variant of [`contains`](Self::contains).
    pub fn contains_str(&self, s: &str) -> bool {
        self.contains(s.as_bytes())
    }

    /// Longest stored string that is a prefix of `data`. Returns `""` if none.
    pub fn longest_prefix(&self, data: &[u8]) -> &[u8] {
        let mut node = &self.root;
        let mut last_end: usize = 0;
        for (i, &b) in data.iter().enumerate() {
            match node.children.get(&b) {
                Some(n) => {
                    node = n;
                    if n.is_end {
                        last_end = i + 1;
                    }
                }
                None => break,
            }
        }
        &data[..last_end]
    }

    /// Collect every stored string with the given prefix (lexicographic order).
    pub fn with_prefix(&self, prefix: &[u8]) -> Vec<Vec<u8>> {
        let mut node = &self.root;
        for &b in prefix {
            match node.children.get(&b) {
                Some(n) => node = n,
                None => return Vec::new(),
            }
        }
        let mut out = Vec::new();
        let mut buf = prefix.to_vec();
        Self::collect(node, &mut buf, &mut out);
        out
    }

    fn collect(node: &TrieNode, buf: &mut Vec<u8>, out: &mut Vec<Vec<u8>>) {
        if node.is_end {
            out.push(buf.clone());
        }
        for (&b, child) in &node.children {
            buf.push(b);
            Self::collect(child, buf, out);
            buf.pop();
        }
    }

    /// Number of distinct complete strings stored.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of nodes (incl. root). Useful to estimate the shared-prefix
    /// savings vs. storing each string separately.
    pub fn node_count(&self) -> usize {
        let mut n = 0usize;
        Self::count(&self.root, &mut n);
        n
    }

    fn count(node: &TrieNode, n: &mut usize) {
        *n += 1;
        for child in node.children.values() {
            Self::count(child, n);
        }
    }

    /// Total bytes that would be needed to store every inserted string
    /// independently (sum of lengths). Compare with [`node_count`] to gauge
    /// the trie's compression benefit.
    pub fn independent_bytes(&self) -> usize {
        let mut total = 0usize;
        let mut buf = Vec::new();
        Self::collect_lengths(&self.root, &mut buf, &mut total);
        total
    }

    fn collect_lengths(node: &TrieNode, buf: &mut Vec<u8>, total: &mut usize) {
        if node.is_end {
            *total += buf.len();
        }
        for (&b, child) in &node.children {
            buf.push(b);
            Self::collect_lengths(child, buf, total);
            buf.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_is_idempotent_and_unique() {
        let mut t = Trie::new();
        assert!(t.insert_str("src/db.rs"));
        assert!(t.insert_str("src/api.rs"));
        assert!(!t.insert_str("src/db.rs"), "second insert is a no-op");
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn contains_distinguishes_prefix_from_full() {
        let mut t = Trie::new();
        t.insert_str("src");
        assert!(t.contains_str("src"));
        assert!(!t.contains_str("src/db.rs"), "prefix only, not inserted");
    }

    #[test]
    fn longest_prefix_returns_stored_ancestor() {
        let mut t = Trie::new();
        t.insert_str("src");
        t.insert_str("src/db");
        let p = t.longest_prefix(b"src/db.rs/extra");
        assert_eq!(p, b"src/db");
    }

    #[test]
    fn with_prefix_lists_lexicographically() {
        let mut t = Trie::new();
        for s in ["src/a.rs", "src/b.rs", "src/c.rs", "tests/x.rs"] {
            t.insert_str(s);
        }
        let hits = t.with_prefix(b"src/");
        let strs: Vec<String> = hits
            .into_iter()
            .map(|v| String::from_utf8(v).unwrap())
            .collect();
        assert_eq!(strs, vec!["src/a.rs", "src/b.rs", "src/c.rs"]);
    }

    #[test]
    fn shared_prefix_is_compact() {
        let mut t = Trie::new();
        for i in 0..50 {
            t.insert_str(&format!("src/deeply/nested/path/to/file_{i}.rs"));
        }
        // Each string is ~38 bytes → 1900 bytes independent; the trie should
        // reuse the 30-byte shared prefix and store ~50 leaves for the suffix.
        let indep = t.independent_bytes();
        let nodes = t.node_count();
        assert!(nodes < indep, "trie ({nodes} nodes) beats {indep} independent bytes");
    }

    #[test]
    fn determinism_same_insertions_same_structure() {
        let mut a = Trie::new();
        let mut b = Trie::new();
        for s in ["x", "xy", "xyz", "xa", "xb"] {
            a.insert_str(s);
            b.insert_str(s);
        }
        // B-tree ordering → child order independent of insertion sequence.
        let mut c = Trie::new();
        for s in ["xb", "xa", "xyz", "xy", "x"] {
            c.insert_str(s);
        }
        assert_eq!(a.root.children.len(), b.root.children.len());
        assert_eq!(a.node_count(), c.node_count(), "order-independent structure");
    }
}