//! Premium licensing gate for the **Retrieval** module.
//!
//! Pure semantic retrieval is SciRust's "RAG-killer": a high-value, premium
//! add-on sold in the *Perception* and *Industrie 4.0* bundles. Access is gated
//! by [`RetrievalAccess`] — a zero-sized capability token whose only constructor,
//! [`RetrievalAccess::unlock`], requires a verified
//! [`Entitlements`](scirust_license::Entitlements) set covering
//! [`Module::Retrieval`](scirust_license::Module::Retrieval). The crate's
//! flagship retrievers hang off this token, so a caller without a valid license
//! cannot build one through the blessed path.
//!
//! The lower-level building blocks ([`DenseIndex`](crate::DenseIndex),
//! [`ProjectionHead`](crate::ProjectionHead), the vector and metric helpers)
//! stay open — they are primitives. What the token gates is the *product*: the
//! end-to-end retrievers a customer actually ships.
//!
//! The gate is deterministic and pure-Rust like everything else here: the
//! entitlement it checks comes from a hash-based signed license
//! (`scirust-license`) — no FFI, no network, no clock beyond the `now` the caller
//! passes to `verify_license`.
//!
//! ```
//! use scirust_license::{demo_root, demo_vendor, verify_license, License, Module};
//! use scirust_retrieval::RetrievalAccess;
//! use scirust_core::embed::EmbeddingEngine;
//!
//! // A license that covers the premium Retrieval module.
//! let signed = demo_vendor().issue_with_leaf(
//!     License::new("Acme", "L-RET-1", [Module::Retrieval], 0, None),
//!     0,
//! );
//! let ent = verify_license(&signed, &demo_root(), 1).unwrap();
//!
//! // Unlock once, then build a retriever through the gated entry point.
//! let access = RetrievalAccess::unlock(&ent).expect("entitled to Retrieval");
//! let mut r = access.semantic_retriever(EmbeddingEngine::new(&["hello world"]));
//! r.index_text(0, "hello world").unwrap();
//! assert_eq!(r.retrieve("hello world", 1)[0].id, 0);
//! ```

use crate::{ContrastiveConfig, Encoder, HybridRetriever, ImprovementLoop, SemanticRetriever};

scirust_license::module_gate! {
    /// Capability token proving entitlement to the premium **Retrieval** module
    /// ([`Module::Retrieval`](scirust_license::Module::Retrieval)).
    ///
    /// Obtain one with [`RetrievalAccess::unlock`] against a verified
    /// [`Entitlements`](scirust_license::Entitlements) set; the crate's flagship
    /// retrievers are then constructed through its methods. The token is `Copy`
    /// and zero-sized, so once unlocked it costs nothing to pass around.
    pub RetrievalAccess => Retrieval
}

impl RetrievalAccess {
    /// Build an end-to-end [`SemanticRetriever`] over `encoder` — the exact,
    /// deterministic dense retriever (cosine top-k). Gated premium entry point.
    pub fn semantic_retriever<E: Encoder>(self, encoder: E) -> SemanticRetriever<E> {
        SemanticRetriever::new(encoder)
    }

    /// Build a [`HybridRetriever`] (dense + BM25, fused by reciprocal rank
    /// fusion with constant `rrf_k`) over `encoder`. Gated premium entry point.
    pub fn hybrid_retriever<E: Encoder>(self, encoder: E, rrf_k: f32) -> HybridRetriever<E> {
        HybridRetriever::new(encoder, rrf_k)
    }

    /// Start a feedback-driven [`ImprovementLoop`] that re-trains a projection
    /// head (`dim_in` → `dim_out`, seeded `seed`, trained with `cfg`) as
    /// relevance pairs accumulate. Gated premium entry point.
    pub fn improvement_loop(
        self,
        dim_in: usize,
        dim_out: usize,
        seed: u64,
        cfg: ContrastiveConfig,
    ) -> ImprovementLoop {
        ImprovementLoop::new(dim_in, dim_out, seed, cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_core::embed::EmbeddingEngine;
    use scirust_license::{Entitlements, License, LicenseError, Module, Vendor, verify_license};

    // A self-contained test vendor (distinct from the demo vendor). Each call
    // signs one license under a distinct one-time `leaf`, so no Lamport leaf is
    // ever reused across the licenses a test mints.
    fn entitlements(modules: impl IntoIterator<Item = Module>, leaf: u32) -> Entitlements {
        let v = Vendor::from_seed(&[7u8; 32], 6);
        let signed = v.issue_with_leaf(License::new("Tester", "L-RET", modules, 0, None), leaf);
        verify_license(&signed, &v.root(), 1).expect("license verifies")
    }

    #[test]
    fn a_retrieval_license_unlocks_and_builds_a_working_retriever() {
        let ent = entitlements([Module::Retrieval], 0);
        let access = RetrievalAccess::unlock(&ent).expect("entitled to Retrieval");
        assert_eq!(RetrievalAccess::MODULE, Module::Retrieval);

        let mut r = access.semantic_retriever(EmbeddingEngine::new(&["rust is fast", "cats purr"]));
        r.index_text(0, "rust is fast").unwrap();
        r.index_text(1, "cats purr").unwrap();
        // The gated retriever behaves exactly like the ungated one.
        assert_eq!(r.retrieve("rust is fast", 2)[0].id, 0);
    }

    #[test]
    fn a_license_without_retrieval_is_refused() {
        // A perfectly valid license that simply does not include Retrieval.
        let ent = entitlements([Module::Core, Module::Vision], 0);
        assert_eq!(
            RetrievalAccess::unlock(&ent).err(),
            Some(LicenseError::NotEntitled(Module::Retrieval)),
        );
    }

    #[test]
    fn the_token_also_gates_hybrid_and_the_improvement_loop() {
        let ent = entitlements([Module::Retrieval], 0);
        let access = RetrievalAccess::unlock(&ent).expect("entitled");

        // Token is Copy, so it gates every flagship product without re-unlocking.
        let h = access.hybrid_retriever(EmbeddingEngine::new(&["alpha beta"]), 60.0);
        assert!(h.is_empty());

        let mut lp = access.improvement_loop(8, 4, 1, ContrastiveConfig::default());
        assert!(lp.is_empty());
        assert!(lp.train_cycle().is_empty()); // no feedback recorded yet
    }
}
