//! Deterministic content fingerprinting shared by this crate's typed
//! contracts (see [`crate::CausalCertificateBuilder::finalize`]).

use sha2::{Digest, Sha256};

/// Lowercase-hex SHA-256 of `bytes`. Deterministic: the same bytes always
/// produce the same digest, on any platform, in any process.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);

    let mut hex = String::with_capacity(64);

    for byte in digest
    {
        use core::fmt::Write;

        write!(hex, "{byte:02x}").expect("writing to a String is infallible");
    }

    hex
}
