//! PLC program (ladder-logic) integrity and sabotage detection.
//!
//! A control program is a sequence of **rungs**. Two distinct threats:
//!
//! 1. *Integrity* — any change to a rung's compiled bytes. A golden baseline of
//!    per-rung digests, hash-chained, detects and localises the first altered
//!    rung (or a changed rung count).
//! 2. *Targeted sabotage* — the Stuxnet pattern: insert logic that drives a
//!    safety-critical output the legitimate program never wrote. Even without
//!    the golden bytes, comparing the program's *write-set* against the
//!    baseline's flags an output that is newly commanded.
//!
//! Integrity digest, not a signature: see [`crate::hashchain`] for the threat
//! model.

use crate::hashchain::{chain_all, digest, fnv1a};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// One compiled ladder rung: its program bytes and, if it drives a coil, the
/// output address it writes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlcRung {
    /// Compiled instruction bytes of the rung.
    pub bytes: Vec<u8>,
    /// Output coil/register this rung writes, if any.
    pub writes: Option<u16>,
}

impl PlcRung {
    /// Convenience constructor.
    pub fn new(bytes: impl Into<Vec<u8>>, writes: Option<u16>) -> Self {
        Self {
            bytes: bytes.into(),
            writes,
        }
    }

    fn digest(&self) -> u64 {
        let mut h = digest(&self.bytes);
        // Fold the write target so a redirected output changes the digest even
        // if the byte length is preserved.
        h = fnv1a(h, &self.writes.unwrap_or(u16::MAX).to_le_bytes());
        h
    }
}

/// Golden fingerprint of a PLC program.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlcBaseline {
    rung_digests: Vec<u64>,
    chain: u64,
    written: BTreeSet<u16>,
}

/// Outcome of verifying a program against a [`PlcBaseline`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlcVerdict {
    /// Program matches the baseline.
    Intact,
    /// Different number of rungs (logic inserted or removed).
    RungCountChanged { expected: usize, actual: usize },
    /// A rung's compiled logic changed; the first altered rung is reported.
    RungModified { rung: usize },
}

impl PlcBaseline {
    /// Capture a golden baseline from a trusted `program`.
    pub fn capture(program: &[PlcRung]) -> Self {
        let rung_digests: Vec<u64> = program.iter().map(PlcRung::digest).collect();
        let chain = chain_all(&rung_digests);
        let written = program.iter().filter_map(|r| r.writes).collect();
        Self {
            rung_digests,
            chain,
            written,
        }
    }

    /// The chained integrity digest of the golden program.
    pub fn chain_digest(&self) -> u64 {
        self.chain
    }

    /// Verify a `program`'s bit-level integrity against the baseline.
    pub fn verify(&self, program: &[PlcRung]) -> PlcVerdict {
        if program.len() != self.rung_digests.len()
        {
            return PlcVerdict::RungCountChanged {
                expected: self.rung_digests.len(),
                actual: program.len(),
            };
        }
        for (i, (base, r)) in self.rung_digests.iter().zip(program).enumerate()
        {
            if *base != r.digest()
            {
                return PlcVerdict::RungModified { rung: i };
            }
        }
        PlcVerdict::Intact
    }

    /// Safety-critical outputs in `critical` that `program` writes but the
    /// golden baseline never did — the targeted-sabotage indicator. Empty when
    /// the program commands only outputs the baseline already drove.
    pub fn unauthorized_critical_writes(
        &self,
        program: &[PlcRung],
        critical: &BTreeSet<u16>,
    ) -> Vec<u16> {
        let mut out: Vec<u16> = program
            .iter()
            .filter_map(|r| r.writes)
            .filter(|w| critical.contains(w) && !self.written.contains(w))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn golden() -> Vec<PlcRung> {
        vec![
            PlcRung::new(vec![0x01, 0x10, 0x20], Some(100)),
            PlcRung::new(vec![0x02, 0x11], Some(101)),
            PlcRung::new(vec![0x03, 0x12, 0x13, 0x14], None),
            PlcRung::new(vec![0x04, 0x22], Some(102)),
        ]
    }

    #[test]
    fn unmodified_program_verifies_intact() {
        let p = golden();
        let base = PlcBaseline::capture(&p);
        assert_eq!(base.verify(&p), PlcVerdict::Intact);
    }

    #[test]
    fn a_modified_rung_is_caught_and_localised() {
        let p = golden();
        let base = PlcBaseline::capture(&p);
        let mut bad = p.clone();
        bad[2].bytes[1] ^= 0xFF;
        assert_eq!(base.verify(&bad), PlcVerdict::RungModified { rung: 2 });
    }

    #[test]
    fn redirecting_an_output_changes_the_rung_digest() {
        let p = golden();
        let base = PlcBaseline::capture(&p);
        let mut bad = p.clone();
        bad[0].writes = Some(999); // same bytes, different target coil
        assert_eq!(base.verify(&bad), PlcVerdict::RungModified { rung: 0 });
    }

    #[test]
    fn inserted_rung_changes_the_count() {
        let p = golden();
        let base = PlcBaseline::capture(&p);
        let mut bad = p.clone();
        bad.insert(2, PlcRung::new(vec![0xEE], Some(500)));
        assert_eq!(
            base.verify(&bad),
            PlcVerdict::RungCountChanged {
                expected: 4,
                actual: 5
            }
        );
    }

    #[test]
    fn flags_a_write_to_a_critical_output_the_baseline_never_drove() {
        let p = golden();
        let base = PlcBaseline::capture(&p);
        // 200 = emergency-stop bypass coil; the golden program never writes it.
        let critical: BTreeSet<u16> = [200, 201].into_iter().collect();
        // Honest program: writes only its usual coils -> nothing flagged.
        assert!(base.unauthorized_critical_writes(&p, &critical).is_empty());
        // Sabotaged program inserts a write to coil 200.
        let mut bad = p.clone();
        bad.push(PlcRung::new(vec![0x09, 0x00], Some(200)));
        assert_eq!(
            base.unauthorized_critical_writes(&bad, &critical),
            vec![200]
        );
    }
}
