//! Deterministic, domain-separated content digests for SciRust persistence and provenance.
//!
//! This crate is deliberately small: it provides one versioned SHA-256 construction,
//! a streaming state, stable lowercase-hex interchange, and a `Read` helper. It is
//! intended for content identity and corruption detection, not secrecy or authentication.

#![forbid(unsafe_code)]

use core::fmt;
use sha2::{Digest as _, Sha256};
use std::io::{self, Read};

const PREFIX: &[u8] = b"scirust-digest:v1\0";
pub const DIGEST_LEN: usize = 32;

/// A stable 32-byte content digest.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Digest32([u8; DIGEST_LEN]);

impl Digest32 {
    #[must_use]
    pub const fn from_bytes(bytes: [u8; DIGEST_LEN]) -> Self {
        Self(bytes)
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; DIGEST_LEN] {
        &self.0
    }

    #[must_use]
    pub fn to_hex(&self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(DIGEST_LEN * 2);
        for &b in &self.0
        {
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0x0f) as usize] as char);
        }
        out
    }

    pub fn from_hex(hex: &str) -> Result<Self, ParseDigestError> {
        if hex.len() != DIGEST_LEN * 2
        {
            return Err(ParseDigestError);
        }
        let bytes = hex.as_bytes();
        let mut out = [0u8; DIGEST_LEN];
        for (i, slot) in out.iter_mut().enumerate()
        {
            let hi = hex_value(bytes[i * 2]).ok_or(ParseDigestError)?;
            let lo = hex_value(bytes[i * 2 + 1]).ok_or(ParseDigestError)?;
            *slot = (hi << 4) | lo;
        }
        Ok(Self(out))
    }
}

impl fmt::Display for Digest32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for Digest32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Digest32({self})")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParseDigestError;

impl fmt::Display for ParseDigestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("digest must be exactly 64 hexadecimal characters")
    }
}

impl std::error::Error for ParseDigestError {}

/// Streaming SHA-256 state with an unambiguous domain prefix.
///
/// The preimage is `PREFIX || domain_len_le_u64 || domain || data...`, so the
/// domain/data boundary is fixed independently of update chunking.
pub struct DigestState(Sha256);

impl DigestState {
    #[must_use]
    pub fn new(domain: &[u8]) -> Self {
        let mut state = Sha256::new();
        state.update(PREFIX);
        state.update((domain.len() as u64).to_le_bytes());
        state.update(domain);
        Self(state)
    }

    pub fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    #[must_use]
    pub fn finalize(self) -> Digest32 {
        Digest32(self.0.finalize().into())
    }
}

#[must_use]
pub fn hash_bytes(domain: &[u8], bytes: &[u8]) -> Digest32 {
    let mut state = DigestState::new(domain);
    state.update(bytes);
    state.finalize()
}

/// Hashes a stream without buffering the whole artifact in memory.
pub fn hash_reader<R: Read>(domain: &[u8], reader: &mut R) -> io::Result<Digest32> {
    let mut state = DigestState::new(domain);
    let mut buf = [0u8; 64 * 1024];
    loop
    {
        let n = match reader.read(&mut buf)
        {
            Ok(n) => n,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        if n == 0
        {
            break;
        }
        state.update(&buf[..n]);
    }
    Ok(state.finalize())
}

fn hex_value(c: u8) -> Option<u8> {
    match c
    {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_shot_and_streaming_are_identical() {
        let data = b"persistent generation component";
        let expected = hash_bytes(b"octasoma-store-component", data);
        let mut state = DigestState::new(b"octasoma-store-component");
        state.update(&data[..7]);
        state.update(&data[7..19]);
        state.update(&data[19..]);
        assert_eq!(state.finalize(), expected);
    }

    #[test]
    fn domains_are_separated() {
        assert_ne!(hash_bytes(b"tree", b"same"), hash_bytes(b"sketch", b"same"));
    }

    #[test]
    fn reader_matches_bytes() {
        let data = vec![0xA5u8; 200_000];
        let mut reader = &data[..];
        assert_eq!(
            hash_reader(b"file", &mut reader).unwrap(),
            hash_bytes(b"file", &data)
        );
    }

    #[test]
    fn reader_retries_interrupted_reads_without_losing_state() {
        struct InterruptedOnce<'a> {
            first: bool,
            inner: &'a [u8],
        }

        impl Read for InterruptedOnce<'_> {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                if self.first
                {
                    self.first = false;
                    return Err(io::Error::from(io::ErrorKind::Interrupted));
                }
                self.inner.read(buf)
            }
        }

        let data = b"reader progress survives interrupted reads";
        let mut reader = InterruptedOnce {
            first: true,
            inner: data,
        };
        assert_eq!(
            hash_reader(b"file", &mut reader).unwrap(),
            hash_bytes(b"file", data)
        );
    }

    #[test]
    fn hex_roundtrip_is_stable() {
        let digest = hash_bytes(b"test", b"payload");
        let hex = digest.to_hex();
        assert_eq!(hex.len(), 64);
        assert_eq!(Digest32::from_hex(&hex).unwrap(), digest);
        assert_eq!(Digest32::from_hex(&hex.to_uppercase()).unwrap(), digest);
        assert!(Digest32::from_hex("xyz").is_err());
    }
}
