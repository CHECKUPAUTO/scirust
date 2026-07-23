//! [`CcosChain`] — a tamper-evident, hash-chained record of every cognitive act.
//!
//! Every act the cognitive backend performs (a proposal generated, a recall
//! served) is attested: its input and output are hashed and linked into an
//! append-only chain, `input_hash → output_hash → chain_hash`, each entry
//! committing to the previous entry's `chain_hash`. Altering, reordering, or
//! dropping any entry breaks the chain, which [`CcosChain::verify`] detects and
//! localizes. Only *hashes* are stored, so the record is tamper-evident without
//! retaining the (possibly large or sensitive) cognitive payloads. This is the
//! integrity role of the cognitive adapter — trusted, deterministic, and
//! anchorable into SOS provenance via a [`CcosChainRef`].

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Digest, HashAlgo};

use crate::error::{CcosError, Result};

/// Hash domains — distinct so an input digest can never be mistaken for an
/// output digest or a chain link.
const GENESIS_DOMAIN: &[u8] = b"sos-ccos:genesis:v1";
const INPUT_DOMAIN: &[u8] = b"sos-ccos:input:v1";
const OUTPUT_DOMAIN: &[u8] = b"sos-ccos:output:v1";
const LINK_DOMAIN: &[u8] = b"sos-ccos:link:v1";

/// The fixed genesis digest every chain's first entry links back to.
#[must_use]
fn genesis() -> Digest {
    HashAlgo::default().hash(GENESIS_DOMAIN, b"")
}

/// Compute an entry's chain hash from its position, payload hashes, and the
/// previous chain hash.
#[must_use]
fn link(seq: u64, input_hash: &Digest, output_hash: &Digest, prev: &Digest) -> Digest {
    let mut enc = CanonicalEncoder::new();
    enc.u64(seq);
    enc.bytes(input_hash.as_bytes());
    enc.bytes(output_hash.as_bytes());
    enc.bytes(prev.as_bytes());
    HashAlgo::default().hash(LINK_DOMAIN, &enc.finish())
}

/// One attested cognitive act: the hashes of its input and output, its link to
/// the previous entry, and the resulting chain hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attestation {
    /// The 0-based position of this act in the chain.
    pub seq: u64,
    /// Digest of the act's input.
    pub input_hash: Digest,
    /// Digest of the act's output.
    pub output_hash: Digest,
    /// The previous entry's `chain_hash` (or the genesis digest for `seq == 0`).
    pub prev: Digest,
    /// This entry's chain hash — the commitment over `(seq, input, output, prev)`.
    pub chain_hash: Digest,
}

impl Canonical for Attestation {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(self.seq);
        enc.bytes(self.input_hash.as_bytes());
        enc.bytes(self.output_hash.as_bytes());
        enc.bytes(self.prev.as_bytes());
        enc.bytes(self.chain_hash.as_bytes());
    }
}

/// A stable reference to a point in a [`CcosChain`] — its sequence number and
/// chain hash — for anchoring a cognitive act into SOS provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CcosChainRef {
    /// The attested act's sequence number.
    pub seq: u64,
    /// The chain hash at that point.
    pub chain_hash: Digest,
}

/// An append-only, hash-chained log of cognitive acts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CcosChain {
    entries: Vec<Attestation>,
}

impl CcosChain {
    /// An empty chain.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Attest an act with the given `input` and `output` bytes, appending a new
    /// entry and returning a reference to it. Only the hashes are retained.
    pub fn append(&mut self, input: &[u8], output: &[u8]) -> CcosChainRef {
        let algo = HashAlgo::default();
        let seq = self.entries.len() as u64;
        let input_hash = algo.hash(INPUT_DOMAIN, input);
        let output_hash = algo.hash(OUTPUT_DOMAIN, output);
        let prev = self.entries.last().map_or_else(genesis, |e| e.chain_hash);
        let chain_hash = link(seq, &input_hash, &output_hash, &prev);
        self.entries.push(Attestation {
            seq,
            input_hash,
            output_hash,
            prev,
            chain_hash,
        });
        CcosChainRef { seq, chain_hash }
    }

    /// The current chain head, if any acts have been attested.
    #[must_use]
    pub fn head(&self) -> Option<Digest> {
        self.entries.last().map(|e| e.chain_hash)
    }

    /// The attested entries, in order.
    #[must_use]
    pub fn entries(&self) -> &[Attestation] {
        &self.entries
    }

    /// How many acts have been attested.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the chain is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Verify the whole chain: every entry's sequence, link, and chain hash must
    /// be internally consistent and correctly chained from genesis.
    ///
    /// # Errors
    /// [`CcosError::ChainBroken`] at the sequence number of the first entry whose
    /// position, previous-link, or recomputed chain hash does not match — i.e.
    /// the first point tampering is detectable.
    pub fn verify(&self) -> Result<()> {
        let mut expected_prev = genesis();
        for (index, entry) in self.entries.iter().enumerate()
        {
            let seq = index as u64;
            let recomputed = link(seq, &entry.input_hash, &entry.output_hash, &entry.prev);
            if entry.seq != seq || entry.prev != expected_prev || entry.chain_hash != recomputed
            {
                return Err(CcosError::ChainBroken { seq });
            }
            expected_prev = entry.chain_hash;
        }
        Ok(())
    }
}
