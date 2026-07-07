//! Certification evidence pack (DO-178C / EN 50128 / ISO 26262 dossier).
//!
//! Bundles, per inference, the artefacts a certification authority asks for —
//! a model fingerprint, the input/output **determinism fingerprints**, an
//! attestation verdict, and the ASIL level — **chained by a tamper-evident
//! hash** (each entry hashes the previous one), so a whole run is a
//! reproducible, verifiable dossier. Changing any field anywhere breaks the
//! chain. The hash is a deterministic integrity digest (FNV-1a), not a
//! cryptographic signature — pair it with the runtime's finite-field
//! verifiable-inference argument for soundness.
//!
//! # Security boundary (important)
//!
//! This pack is **tamper-evident, not tamper-resistant**. The chaining hash is
//! FNV-1a — a public, non-cryptographic algorithm with no secret key. It
//! detects *naive* edits (a field changed without recomputing `entry`/`prev`)
//! because [`verify_chain`](fn@verify_chain) recomputes the chain and rejects
//! the mismatch. It does **not** resist an attacker who can *recompute the
//! whole chain*: such an attacker produces a self-consistent dossier that
//! passes `verify()`. There is no secret in the construction, so the
//! integrity guarantee rests entirely on **write access control to the
//! dossier file** and on pairing this pack with a soundness argument (e.g.
//! the runtime's finite-field verifiable-inference proof). Do **not** treat
//! `from_json(...).verify()` as authenticating an untrusted dossier on its
//! own — only as detecting accidental corruption or naive tampering. If a
//! forgery-resistant chain is required, wrap the records in an HMAC (keyed)
//! or a digital signature over the canonical encoding.

use crate::asil::AsilLevel;
use serde::{Deserialize, Serialize};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes
    {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Deterministic fingerprint of an `f32` output vector (fixed order, bit
/// patterns) — the determinism evidence for one inference.
pub fn fingerprint_f32(data: &[f32]) -> u64 {
    let mut h = FNV_OFFSET;
    for &x in data
    {
        h = fnv1a(h, &x.to_bits().to_le_bytes());
    }
    h
}

fn asil_code(a: AsilLevel) -> u8 {
    match a
    {
        AsilLevel::QM => 0,
        AsilLevel::A => 1,
        AsilLevel::B => 2,
        AsilLevel::C => 3,
        AsilLevel::D => 4,
    }
}

fn compute_entry(r: &EvidenceRecord) -> u64 {
    let mut h = FNV_OFFSET;
    for v in [r.prev, r.seq, r.model_hash, r.input_hash, r.output_hash]
    {
        h = fnv1a(h, &v.to_le_bytes());
    }
    fnv1a(h, &[asil_code(r.asil), r.verified as u8])
}

/// One inference's evidence, linked to the previous entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRecord {
    /// Position in the chain.
    pub seq: u64,
    /// Fingerprint of the model (weights / manifest).
    pub model_hash: u64,
    /// Fingerprint of the input.
    pub input_hash: u64,
    /// Determinism fingerprint of the output.
    pub output_hash: u64,
    /// ASIL level this inference is certified at.
    pub asil: AsilLevel,
    /// Attestation / verifiable-inference verdict.
    pub verified: bool,
    /// Previous entry hash (chain link).
    pub prev: u64,
    /// `H(prev ‖ seq ‖ fields)`.
    pub entry: u64,
}

/// Recompute the chain from scratch and check every entry and link.
pub fn verify_chain(records: &[EvidenceRecord]) -> bool {
    let mut prev = 0u64;
    for (i, r) in records.iter().enumerate()
    {
        if r.seq != i as u64 || r.prev != prev
        {
            return false;
        }
        if compute_entry(r) != r.entry
        {
            return false;
        }
        prev = r.entry;
    }
    true
}

/// An append-only, tamper-evident evidence dossier.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvidencePack {
    records: Vec<EvidenceRecord>,
}

impl EvidencePack {
    /// Empty pack.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one inference's evidence; returns the new chain root.
    pub fn record(
        &mut self,
        model_hash: u64,
        input_hash: u64,
        output_hash: u64,
        asil: AsilLevel,
        verified: bool,
    ) -> u64 {
        let mut r = EvidenceRecord {
            seq: self.records.len() as u64,
            model_hash,
            input_hash,
            output_hash,
            asil,
            verified,
            prev: self.root(),
            entry: 0,
        };
        r.entry = compute_entry(&r);
        let root = r.entry;
        self.records.push(r);
        root
    }

    /// Hash of the last entry (0 for an empty pack).
    pub fn root(&self) -> u64 {
        self.records.last().map(|r| r.entry).unwrap_or(0)
    }

    /// The evidence records.
    pub fn records(&self) -> &[EvidenceRecord] {
        &self.records
    }

    /// Number of records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the pack is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Verify the whole dossier.
    pub fn verify(&self) -> bool {
        verify_chain(&self.records)
    }

    /// Serialize the dossier to JSON (a portable, archivable certification
    /// artifact).
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| e.to_string())
    }

    /// Parse a dossier from JSON. Call [`verify`](Self::verify) on the result —
    /// a forged JSON is detected exactly as an in-memory tamper would be.
    pub fn from_json(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build() -> EvidencePack {
        let mut pack = EvidencePack::new();
        pack.record(
            0x1111,
            fingerprint_f32(&[1.0, 2.0, 3.0]),
            fingerprint_f32(&[0.5, 0.5]),
            AsilLevel::D,
            true,
        );
        pack.record(
            0x1111,
            fingerprint_f32(&[4.0, 5.0]),
            fingerprint_f32(&[0.9, 0.1]),
            AsilLevel::C,
            true,
        );
        pack.record(
            0x1111,
            fingerprint_f32(&[6.0]),
            fingerprint_f32(&[0.3, 0.7]),
            AsilLevel::B,
            false,
        );
        pack
    }

    #[test]
    fn fingerprint_is_deterministic_and_sensitive() {
        assert_eq!(fingerprint_f32(&[1.0, 2.0]), fingerprint_f32(&[1.0, 2.0]));
        assert_ne!(fingerprint_f32(&[1.0, 2.0]), fingerprint_f32(&[1.0, 2.001]));
        // -0.0 and +0.0 have different bit patterns -> different fingerprints.
        assert_ne!(fingerprint_f32(&[0.0]), fingerprint_f32(&[-0.0]));
    }

    #[test]
    fn intact_pack_verifies() {
        let pack = build();
        assert_eq!(pack.len(), 3);
        assert!(pack.verify());
        assert!(verify_chain(pack.records()));
    }

    #[test]
    fn tampering_any_field_breaks_the_chain() {
        let pack = build();
        // Flip an output fingerprint in the middle record, leaving its stored
        // entry hash unchanged — exactly what a forged dossier would look like.
        let mut tampered = pack.records().to_vec();
        tampered[1].output_hash ^= 1;
        assert!(!verify_chain(&tampered), "tamper not detected");

        // Re-ordering / dropping a record also breaks the links.
        let mut dropped = pack.records().to_vec();
        dropped.remove(1);
        assert!(!verify_chain(&dropped));
    }

    #[test]
    fn root_changes_when_evidence_changes() {
        let a = build().root();
        let mut p = EvidencePack::new();
        p.record(
            0x1111,
            fingerprint_f32(&[1.0, 2.0, 3.0]),
            fingerprint_f32(&[0.5, 0.5]),
            AsilLevel::D,
            true,
        );
        // Different second record -> different root.
        p.record(0x1111, 1, 2, AsilLevel::C, true);
        assert_ne!(a, p.root());
    }

    #[test]
    fn json_round_trips_and_still_verifies() {
        let pack = build();
        let json = pack.to_json().expect("serialize");
        let back = EvidencePack::from_json(&json).expect("parse");
        assert_eq!(back.len(), pack.len());
        assert_eq!(back.root(), pack.root());
        assert!(back.verify(), "round-tripped dossier must verify");

        // A forged JSON (an output fingerprint edited) fails verification.
        let forged = json.replacen(
            &pack.records()[0].output_hash.to_string(),
            &(pack.records()[0].output_hash ^ 1).to_string(),
            1,
        );
        let bad = EvidencePack::from_json(&forged).expect("parse");
        assert!(!bad.verify(), "forged dossier must not verify");
    }
}
