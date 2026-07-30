//! Cross-tier hardening for the COLD memory stack.
//!
//! The slices stack: resident eviction → content-spill (slice 3) → deep-spill
//! (slices 5/5b), each lossless and each with its own on-disk blobs and GC. Their
//! *interaction* under arbitrary operation sequences is the risky surface — a
//! page-in after a deep-spill after a re-ingest, a remove that should reclaim two
//! blobs, a cap tightened mid-stream. These property tests drive random sequences
//! with the lossless tiers active and assert the two guarantees that *are* the
//! product:
//!
//!   1. **Lossless round-trip** — every live node pages back with its exact label
//!      and content, whatever tier it sank to.
//!   2. **No leaked blobs** — once every node is removed, the on-disk store is
//!      empty: GC reclaimed every content and body blob, none orphaned.
//!
//! Compaction (slice 4) is deliberately left off here: it is *lossy by design*, so
//! it has its own observable-not-exact tests; this suite pins the lossless tiers.

use ccos_core::memory::{EdgeType, MemoryGraph, NodeId, NodeType};
use proptest::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static CASE: AtomicU64 = AtomicU64::new(0);

fn unique_dir() -> std::path::PathBuf {
    let n = CASE.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("ccos_cold_hard_{}_{}", std::process::id(), n))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Under a random op stream with content-spill and deep-spill both active and a
    /// tight resident cap, every live node must round-trip losslessly, and removing
    /// everything must leave no orphaned blob on disk.
    #[test]
    fn cold_tiers_round_trip_losslessly_and_leak_no_blobs(
        ops in prop::collection::vec((0u8..6u8, 0u16..32u16), 1..160)
    ) {
        let dir = unique_dir();
        let mut g = MemoryGraph::new(0.2, 6);
        // Lossless tiers on, tight budgets so content and metadata actually sink to
        // disk; no compaction (that tier is lossy).
        g.attach_cold_spill(&dir, 128).unwrap();
        g.set_cold_resident_budget(Some(512));

        // The source of truth: id → (label, content) for every node still alive.
        let mut expected: HashMap<String, (String, String)> = HashMap::new();

        for (op, key) in &ops {
            let id = format!("n{}", key % 32);
            match op % 6 {
                0 | 1 => {
                    // Upsert (weighted): a fresh revision replaces any prior content.
                    let label = format!("label::{id}");
                    let content =
                        format!("body[{id}] rev{key} {}", "payload ".repeat((key % 20) as usize));
                    g.upsert_node(id.clone().into(), label.clone(), content.clone(), NodeType::Symbol);
                    expected.insert(id, (label, content));
                }
                2 => {
                    // Edge between two nodes (both must be resident first).
                    let other = format!("n{}", key.wrapping_mul(7) % 32);
                    if id != other && expected.contains_key(&id) && expected.contains_key(&other) {
                        g.page_in(&id.clone().into());
                        g.page_in(&other.clone().into());
                        g.add_edge(id.into(), other.into(), 0.5, EdgeType::DependsOn);
                    }
                }
                3 => {
                    g.page_in(&id.into());
                }
                4 => {
                    g.remove_node(&id.clone().into());
                    expected.remove(&id);
                }
                _ => {
                    // Tighten the resident window and re-page, forcing demotion churn
                    // through the spill / deep-spill enforcers.
                    g.max_in_memory_nodes = (key % 8) as usize;
                    g.enforce_paging();
                }
            }
        }

        // Invariant 1 — lossless round-trip. Open the window wide so every node can
        // be made resident, then fault each one back and check it byte-for-byte.
        g.max_in_memory_nodes = expected.len() + 4;
        for (id, (label, content)) in &expected {
            let nid = NodeId(id.clone());
            g.page_in(&nid);
            match g.node(&nid) {
                Some(n) => {
                    prop_assert_eq!(&n.label, label, "label round-trip for {}", id);
                    prop_assert_eq!(&n.content, content, "content round-trip for {}", id);
                }
                None => prop_assert!(false, "live node {} vanished from every tier", id),
            }
        }

        // Invariant 2 — no leaked blobs. Remove every node still anywhere in the tier
        // hierarchy; a correct GC reclaims every content and body blob.
        let everywhere: Vec<NodeId> = g
            .node_ids()
            .cloned()
            .chain(g.cold_ids())
            .collect();
        for nid in everywhere {
            g.remove_node(&nid);
        }
        let leftover = std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0);
        std::fs::remove_dir_all(&dir).ok();
        prop_assert_eq!(leftover, 0, "{} orphaned blob(s) after removing all nodes", leftover);
    }

    /// The same op stream with *all four* tiers active — content-spill, deep-spill
    /// **and** lossy compaction — must still leave no orphaned blob once every node
    /// is removed, and every live node must stay retrievable (page back in, even if
    /// its content was lossily compacted). This pins the cross-tier GC: compaction
    /// orphans the full original blob, spill/deep-spill orphan theirs on page-in, and
    /// none may leak when they interleave.
    #[test]
    fn all_cold_tiers_leak_no_blobs_and_keep_nodes_retrievable(
        ops in prop::collection::vec((0u8..6u8, 0u16..32u16), 1..160)
    ) {
        let dir = unique_dir();
        let mut g = MemoryGraph::new(0.2, 6);
        g.attach_cold_spill(&dir, 128).unwrap();
        g.set_cold_content_budget(Some(256)); // lossy compaction floor — the deepest tier
        g.set_cold_resident_budget(Some(512)); // deep-spill

        let mut live: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (op, key) in &ops {
            let id = format!("n{}", key % 32);
            match op % 6 {
                0 | 1 => {
                    let content =
                        format!("body[{id}] rev{key} {}", "payload ".repeat((key % 20) as usize));
                    g.upsert_node(id.clone().into(), format!("label::{id}"), content, NodeType::Symbol);
                    live.insert(id);
                }
                2 => {
                    let other = format!("n{}", key.wrapping_mul(7) % 32);
                    if id != other && live.contains(&id) && live.contains(&other) {
                        g.page_in(&id.clone().into());
                        g.page_in(&other.clone().into());
                        g.add_edge(id.into(), other.into(), 0.5, EdgeType::DependsOn);
                    }
                }
                3 => { g.page_in(&id.into()); }
                4 => { g.remove_node(&id.clone().into()); live.remove(&id); }
                _ => { g.max_in_memory_nodes = (key % 8) as usize; g.enforce_paging(); }
            }
        }

        // Liveness — every live node pages back (content may be a lossy summary).
        g.max_in_memory_nodes = live.len() + 4;
        for id in &live {
            let nid = NodeId(id.clone());
            g.page_in(&nid);
            prop_assert!(g.node(&nid).is_some(), "live node {} vanished from every tier", id);
        }

        // No leaked blobs across all four tiers.
        let everywhere: Vec<NodeId> = g.node_ids().cloned().chain(g.cold_ids()).collect();
        for nid in everywhere {
            g.remove_node(&nid);
        }
        let leftover = std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0);
        std::fs::remove_dir_all(&dir).ok();
        prop_assert_eq!(leftover, 0, "{} orphaned blob(s) with all tiers active", leftover);
    }
}
