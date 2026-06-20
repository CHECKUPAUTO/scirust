//! Tamper-evident hash chaining (deterministic FNV-1a digests).
//!
//! Shared by the OT attestation guards ([`crate::ot_firmware`],
//! [`crate::ot_plc`]): a running chain value folds in one block digest at a
//! time, so changing any block — anywhere — changes the final chain. This is an
//! integrity digest, not a cryptographic signature: it detects accidental or
//! unsophisticated tampering and gives a stable fingerprint, but a determined
//! adversary who can recompute the chain could forge it. Pair it with a signed
//! baseline where that threat matters.

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

/// Fold `bytes` into the running FNV-1a hash `h`.
pub fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes
    {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Digest of a single byte block.
pub fn digest(bytes: &[u8]) -> u64 {
    fnv1a(FNV_OFFSET, bytes)
}

/// Extend a running chain `prev` with one block's `block_digest`.
pub fn chain(prev: u64, block_digest: u64) -> u64 {
    fnv1a(prev, &block_digest.to_le_bytes())
}

/// Chain a sequence of block digests into a single fingerprint (seeded from the
/// FNV offset).
pub fn chain_all(block_digests: &[u64]) -> u64 {
    block_digests.iter().fold(FNV_OFFSET, |acc, &d| chain(acc, d))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_is_sensitive_to_any_change() {
        let a = [digest(b"rung0"), digest(b"rung1"), digest(b"rung2")];
        let mut b = a;
        b[1] = digest(b"rung1-tampered");
        assert_ne!(chain_all(&a), chain_all(&b));
        // Reordering also changes the chain (order matters).
        let c = [a[0], a[2], a[1]];
        assert_ne!(chain_all(&a), chain_all(&c));
    }

    #[test]
    fn identical_inputs_chain_identically() {
        let a = [digest(b"x"), digest(b"y")];
        let b = [digest(b"x"), digest(b"y")];
        assert_eq!(chain_all(&a), chain_all(&b));
    }
}
