//! Firmware attestation for OT field devices.
//!
//! A PLC, RTU or IED runs firmware that must not change between commissioning
//! and operation. We capture a **golden baseline** — the firmware split into
//! fixed-size blocks, each digested, the digests hash-chained — and later
//! re-attest a read-back image against it. Any altered byte changes its block's
//! digest and the overall chain, so tampering is detected *and localised* to the
//! first changed block (useful when only a patched region matters).
//!
//! Integrity digest, not a signature: see [`crate::hashchain`] for the threat
//! model.

use crate::hashchain::{chain_all, digest};
use serde::{Deserialize, Serialize};

/// A golden firmware fingerprint: per-block digests plus the chained digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FirmwareBaseline {
    block_size: usize,
    len: usize,
    block_digests: Vec<u64>,
    chain: u64,
}

/// Outcome of attesting an image against a [`FirmwareBaseline`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Attestation {
    /// Image matches the baseline byte-for-byte.
    Intact,
    /// Image differs; the first differing block is reported.
    Tampered { first_bad_block: usize },
    /// Image length differs from the baseline (truncated/padded/replaced).
    SizeMismatch { expected: usize, actual: usize },
}

fn block_digests(firmware: &[u8], block_size: usize) -> Vec<u64> {
    firmware.chunks(block_size.max(1)).map(digest).collect()
}

impl FirmwareBaseline {
    /// Capture a golden baseline from trusted `firmware`, split into
    /// `block_size`-byte blocks.
    pub fn capture(firmware: &[u8], block_size: usize) -> Self {
        let bs = block_size.max(1);
        let block_digests = block_digests(firmware, bs);
        let chain = chain_all(&block_digests);
        Self {
            block_size: bs,
            len: firmware.len(),
            block_digests,
            chain,
        }
    }

    /// The chained integrity digest of the golden image.
    pub fn chain_digest(&self) -> u64 {
        self.chain
    }

    /// Attest a (possibly modified) `firmware` image against this baseline.
    pub fn attest(&self, firmware: &[u8]) -> Attestation {
        if firmware.len() != self.len
        {
            return Attestation::SizeMismatch {
                expected: self.len,
                actual: firmware.len(),
            };
        }
        let got = block_digests(firmware, self.block_size);
        for (i, (a, b)) in self.block_digests.iter().zip(&got).enumerate()
        {
            if a != b
            {
                return Attestation::Tampered { first_bad_block: i };
            }
        }
        // Chain check is redundant with the per-block scan but guards the digest
        // path itself.
        debug_assert_eq!(chain_all(&got), self.chain);
        Attestation::Intact
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_firmware() -> Vec<u8> {
        (0..1024u32).map(|i| (i.wrapping_mul(31) ^ 0xA5) as u8).collect()
    }

    #[test]
    fn unmodified_firmware_attests_intact() {
        let fw = sample_firmware();
        let base = FirmwareBaseline::capture(&fw, 64);
        assert_eq!(base.attest(&fw), Attestation::Intact);
    }

    #[test]
    fn a_single_flipped_byte_is_caught_and_localised() {
        let fw = sample_firmware();
        let base = FirmwareBaseline::capture(&fw, 64);
        let mut bad = fw.clone();
        bad[200] ^= 0x01; // block 200/64 = block 3
        assert_eq!(
            base.attest(&bad),
            Attestation::Tampered { first_bad_block: 3 }
        );
    }

    #[test]
    fn truncation_is_a_size_mismatch() {
        let fw = sample_firmware();
        let base = FirmwareBaseline::capture(&fw, 64);
        let short = &fw[..1000];
        assert_eq!(
            base.attest(short),
            Attestation::SizeMismatch {
                expected: 1024,
                actual: 1000
            }
        );
    }

    #[test]
    fn chain_digest_is_stable_and_change_sensitive() {
        let fw = sample_firmware();
        let base = FirmwareBaseline::capture(&fw, 64);
        let same = FirmwareBaseline::capture(&fw, 64);
        assert_eq!(base.chain_digest(), same.chain_digest());
        let mut bad = fw.clone();
        bad[0] ^= 0xFF;
        assert_ne!(
            FirmwareBaseline::capture(&bad, 64).chain_digest(),
            base.chain_digest()
        );
    }
}
