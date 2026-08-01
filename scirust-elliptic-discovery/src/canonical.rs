//! Canonical binary encoding and integrity fingerprints.

use sha2::{Digest, Sha256};

/// Explicit big-endian, length-delimited encoder.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CanonicalEncoder {
    bytes: Vec<u8>,
}

impl CanonicalEncoder {
    /// Begins an encoding with a domain-separation label.
    pub fn with_domain(domain: &[u8]) -> Self {
        let mut encoder = Self::default();
        encoder.bytes(domain);
        encoder
    }

    pub(crate) fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    pub(crate) fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    pub(crate) fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    pub(crate) fn bytes(&mut self, value: &[u8]) {
        let length = u64::try_from(value.len()).expect("canonical value length fits in u64");
        self.u64(length);
        self.bytes.extend_from_slice(value);
    }

    /// Final canonical bytes.
    pub fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

/// SHA-256 integrity fingerprint of canonical bytes.
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Lowercase hexadecimal encoding.
pub fn hex(bytes: &[u8]) -> String {
    use core::fmt::Write;

    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes
    {
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_answer_for_empty_input() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn lengths_prevent_concatenation_ambiguity() {
        let mut left = CanonicalEncoder::default();
        left.bytes(b"ab");
        left.bytes(b"c");
        let mut right = CanonicalEncoder::default();
        right.bytes(b"a");
        right.bytes(b"bc");
        assert_ne!(left.finish(), right.finish());
    }
}
