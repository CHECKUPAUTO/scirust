//! **Attestation log** — a hash-chained, deterministically-replayable record of
//! **verifiable** inference events: the bridge from scirust's verifiable inference
//! ([`crate::vinfer`], roadmap #80) into a CCOS-style `event_log`.
//!
//! Each recorded [`InferenceEvent`] fixes the model commitment, the input hash and
//! the output hash, and links to the previous entry by a SHA-256 **hash chain**
//! (`entryₙ = H(entryₙ₋₁ ‖ seq ‖ commitment ‖ input ‖ output)`) — exactly the
//! tamper-evident, append-only shape CCOS uses, so a scirust runtime's inferences
//! can be ingested into CCOS's audit trail. Recomputing the chain re-derives the
//! same head (deterministic replay); mutating or reordering any entry breaks it.
//!
//! [`attest_and_record`] additionally checks, before appending, that the claimed
//! `(input, output)` is a **genuine** inference of the committed model via the
//! finite-field Freivalds verifier — so the chain records only authentic inferences.

use crate::vinfer::{VModel, verify_inference};
use sha2::{Digest, Sha256};

/// A 32-byte SHA-256 digest.
pub type Hash = [u8; 32];

/// The genesis chain head (before any event).
pub const GENESIS: Hash = [0u8; 32];

fn sha256_concat(parts: &[&[u8]]) -> Hash {
    let mut h = Sha256::new();
    for p in parts
    {
        h.update(p);
    }
    h.finalize().into()
}

fn hash_i64(xs: &[i64]) -> Hash {
    let mut h = Sha256::new();
    for &x in xs
    {
        h.update(x.to_le_bytes());
    }
    h.finalize().into()
}

/// One attested inference event in the chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InferenceEvent {
    /// Position in the log (0-based).
    pub seq: u64,
    /// Commitment to the model that produced the output (e.g. [`VModel::commit`]).
    pub model_commitment: Hash,
    /// Hash of the input.
    pub input_hash: Hash,
    /// Hash of the claimed output.
    pub output_hash: Hash,
    /// Chain link: `H(prev ‖ seq ‖ commitment ‖ input ‖ output)`.
    pub entry_hash: Hash,
}

/// An append-only, hash-chained log of inference events.
#[derive(Clone, Debug, Default)]
pub struct AttestationLog {
    events: Vec<InferenceEvent>,
    head: Hash,
}

impl AttestationLog {
    /// A fresh log (head = [`GENESIS`]).
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            head: GENESIS,
        }
    }

    /// Append an inference event (commitment + input + output) and return the new
    /// chain head. Does **not** check authenticity — see [`attest_and_record`].
    pub fn record(&mut self, model_commitment: Hash, input: &[i64], output: &[i64]) -> Hash {
        let seq = self.events.len() as u64;
        let input_hash = hash_i64(input);
        let output_hash = hash_i64(output);
        let entry_hash = sha256_concat(&[
            &self.head,
            &seq.to_le_bytes(),
            &model_commitment,
            &input_hash,
            &output_hash,
        ]);
        self.events.push(InferenceEvent {
            seq,
            model_commitment,
            input_hash,
            output_hash,
            entry_hash,
        });
        self.head = entry_hash;
        entry_hash
    }

    /// The current chain head (the tamper-evident commitment to the whole log).
    pub fn head(&self) -> Hash {
        self.head
    }

    /// The recorded events.
    pub fn events(&self) -> &[InferenceEvent] {
        &self.events
    }

    /// Number of events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// **Verify the chain**: recompute every entry hash from the genesis head and
    /// check it matches what was recorded, and that the final hash equals the
    /// recorded [`head`](Self::head).
    /// Any tampering or reordering makes this `false`. Deterministic (replay).
    pub fn verify_chain(&self) -> bool {
        let mut h = GENESIS;
        for (i, e) in self.events.iter().enumerate()
        {
            if e.seq != i as u64
            {
                return false;
            }
            let expect = sha256_concat(&[
                &h,
                &e.seq.to_le_bytes(),
                &e.model_commitment,
                &e.input_hash,
                &e.output_hash,
            ]);
            if expect != e.entry_hash
            {
                return false;
            }
            h = e.entry_hash;
        }
        h == self.head
    }
}

/// Verify that the claimed `(input, output)` is a **genuine** batched inference of
/// `model` (Freivalds over `GF(p)`, [`verify_inference`]) and, if so, append it to
/// `log`, returning the new chain head. Returns `None` (and leaves the log unchanged)
/// if the inference is forged — so the chain attests only authentic inferences.
pub fn attest_and_record(
    log: &mut AttestationLog,
    model: &VModel,
    x: &[i64],
    batch: usize,
    y: &[i64],
    challenges: u32,
) -> Option<Hash> {
    let commitment = model.commit();
    if !verify_inference(model, x, batch, y, &commitment, challenges)
    {
        return None;
    }
    Some(log.record(commitment, x, y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vinfer::P;
    use scirust_core::nn::PcgEngine;

    fn rand_model(out: usize, inn: usize, rng: &mut PcgEngine) -> VModel {
        let w: Vec<i64> = (0..out * inn)
            .map(|_| (rng.next_u32() as i64) % P)
            .collect();
        VModel::new(w, out, inn)
    }

    /// A chain of recorded inferences verifies and is **deterministically replayable**
    /// (the same events re-derive the same head).
    #[test]
    fn chain_verifies_and_replays() {
        let mut rng = PcgEngine::new(1);
        let model = rand_model(4, 5, &mut rng);
        let c = model.commit();
        let build = || {
            let mut log = AttestationLog::new();
            for s in 0..6u64
            {
                let x: Vec<i64> = (0..5).map(|i| (s as i64 * 7 + i) % P).collect();
                let y: Vec<i64> = (0..4).map(|i| (s as i64 + i) % P).collect();
                log.record(c, &x, &y);
            }
            log
        };
        let log = build();
        assert!(log.verify_chain());
        assert_eq!(log.len(), 6);
        // Replay: identical events ⇒ identical head.
        assert_eq!(log.head(), build().head());
    }

    /// Tampering with a recorded event (or reordering) breaks the hash chain.
    #[test]
    fn tampering_breaks_chain() {
        let mut rng = PcgEngine::new(2);
        let model = rand_model(3, 4, &mut rng);
        let c = model.commit();
        let mut log = AttestationLog::new();
        for s in 0..5u64
        {
            log.record(c, &[s as i64, 1, 2, 3], &[s as i64, 9, 8]);
        }
        assert!(log.verify_chain());
        // Forge a stored output hash in the middle: the chain no longer recomputes.
        let mut tampered = log.clone();
        tampered.events[2].output_hash[0] ^= 0x01;
        assert!(!tampered.verify_chain(), "tampering not detected");
        // Reordering is detected too (seq / chain mismatch).
        let mut reordered = log.clone();
        reordered.events.swap(1, 3);
        assert!(!reordered.verify_chain(), "reordering not detected");
    }

    /// `attest_and_record` appends an **authentic** inference and rejects a forged
    /// output (leaving the log unchanged), bridging Freivalds verification (#80) and
    /// the hash chain.
    #[test]
    fn attest_records_authentic_and_rejects_forged() {
        let mut rng = PcgEngine::new(3);
        let (out, inn, batch) = (4usize, 6usize, 3usize);
        let model = rand_model(out, inn, &mut rng);
        let x: Vec<i64> = (0..inn * batch)
            .map(|_| (rng.next_u32() as i64) % P)
            .collect();
        let y = model.infer(&x, batch);

        let mut log = AttestationLog::new();
        // Authentic inference is attested and chained.
        let head = attest_and_record(&mut log, &model, &x, batch, &y, 2);
        assert!(head.is_some());
        assert_eq!(log.len(), 1);
        assert!(log.verify_chain());

        // A forged output is rejected and the log is not extended.
        let mut forged = y.clone();
        forged[0] = (forged[0] + 1) % P;
        let rejected = attest_and_record(&mut log, &model, &x, batch, &forged, 2);
        assert!(rejected.is_none(), "forged inference attested");
        assert_eq!(log.len(), 1, "log extended on rejection");
        assert!(log.verify_chain());
    }
}
