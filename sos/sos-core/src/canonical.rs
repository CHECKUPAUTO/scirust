//! Deterministic canonical byte encoding — the substrate of content addressing.
//!
//! Content addressing is only sound if encoding is **canonical**: two
//! semantically-equal values must produce byte-identical output on any machine
//! and in any process, and two semantically-different values must never
//! collide. This module provides a small, self-contained, unambiguous encoder
//! ([`CanonicalEncoder`]) and the [`Canonical`] trait that types implement to
//! feed it.
//!
//! ## Why not just use `serde_json`?
//!
//! JSON is used by SOS as a *human interchange* form, but it is not canonical:
//! map/key ordering, floating-point formatting, and whitespace are all
//! unspecified. Hashing must not depend on any of that. The encoding here is
//! **length-prefixed and type-tagged**, so it is self-delimiting (no
//! concatenation ambiguity) and independent of any serializer's choices.
//!
//! ## Encoding scheme
//!
//! Every value is written as a one-byte type tag followed by a fixed-form
//! payload:
//!
//! | Tag | Type | Payload |
//! |-----|------|---------|
//! | `0x01` | `u64` | 8 bytes, little-endian |
//! | `0x02` | `i64` | 8 bytes, little-endian (two's complement) |
//! | `0x03` | `bool` | 1 byte (`0`/`1`) |
//! | `0x04` | bytes | `u64` length (LE) then the raw bytes |
//! | `0x05` | `str` | `u64` byte-length (LE) then UTF-8 bytes |
//! | `0x06` | seq | `u64` element count (LE) then each element's encoding |
//! | `0x07` | `Some` | the inner value's encoding |
//! | `0x08` | `None` | (no payload) |
//!
//! Because integer width is fixed and every composite carries an explicit
//! length/count, the byte stream is unambiguous: it can be parsed back without
//! a schema, which is exactly the property that guarantees collision-freedom
//! for distinct inputs.
//!
//! ## Floating point
//!
//! The kernel envelope is deliberately float-free, so this encoder offers no
//! `f64` method. A domain [`crate::Body`] that carries floats and is only
//! *numerically* reproducible (determinism level `L2`) must encode a
//! **quantized** canonical form at a declared precision (e.g. via
//! [`CanonicalEncoder::i64`] over a fixed-point representation) and attach a
//! certificate — never hash a non-portable raw bit pattern. This keeps the
//! hash and the [`crate::DeterminismLevel`] taxonomy coherent.

/// Type tag: unsigned 64-bit integer.
const T_U64: u8 = 0x01;
/// Type tag: signed 64-bit integer.
const T_I64: u8 = 0x02;
/// Type tag: boolean.
const T_BOOL: u8 = 0x03;
/// Type tag: opaque byte string.
const T_BYTES: u8 = 0x04;
/// Type tag: UTF-8 string.
const T_STR: u8 = 0x05;
/// Type tag: homogeneous sequence.
const T_SEQ: u8 = 0x06;
/// Type tag: present optional (`Some`).
const T_SOME: u8 = 0x07;
/// Type tag: absent optional (`None`).
const T_NONE: u8 = 0x08;

/// A growable buffer that accumulates a value's canonical byte encoding.
///
/// Feed it via the typed methods (or via [`Canonical::encode`]), then take the
/// bytes with [`CanonicalEncoder::finish`]. The bytes are what gets hashed to
/// form a [`crate::ObjectId`].
#[derive(Debug, Clone, Default)]
pub struct CanonicalEncoder {
    buf: Vec<u8>,
}

impl CanonicalEncoder {
    /// Create an empty encoder.
    #[must_use]
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Consume the encoder and return the accumulated canonical bytes.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }

    /// Borrow the accumulated bytes without consuming the encoder.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    #[inline]
    fn tag(&mut self, t: u8) {
        self.buf.push(t);
    }

    #[inline]
    fn push_len(&mut self, len: usize) {
        // Lengths are encoded as u64 so the stream is platform-independent
        // (identical on 32- and 64-bit targets).
        self.buf.extend_from_slice(&(len as u64).to_le_bytes());
    }

    /// Encode an unsigned 64-bit integer.
    pub fn u64(&mut self, v: u64) {
        self.tag(T_U64);
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Encode a signed 64-bit integer (little-endian two's complement).
    pub fn i64(&mut self, v: i64) {
        self.tag(T_I64);
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Encode a boolean.
    pub fn bool(&mut self, v: bool) {
        self.tag(T_BOOL);
        self.buf.push(u8::from(v));
    }

    /// Encode an opaque byte string (length-prefixed).
    pub fn bytes(&mut self, b: &[u8]) {
        self.tag(T_BYTES);
        self.push_len(b.len());
        self.buf.extend_from_slice(b);
    }

    /// Encode a UTF-8 string (byte-length-prefixed).
    pub fn str(&mut self, s: &str) {
        self.tag(T_STR);
        let b = s.as_bytes();
        self.push_len(b.len());
        self.buf.extend_from_slice(b);
    }

    /// Encode a nested [`Canonical`] value.
    pub fn value<T: Canonical + ?Sized>(&mut self, v: &T) {
        v.encode(self);
    }

    /// Encode a homogeneous sequence of [`Canonical`] values (count-prefixed).
    pub fn seq<T: Canonical>(&mut self, items: &[T]) {
        self.tag(T_SEQ);
        self.push_len(items.len());
        for it in items
        {
            it.encode(self);
        }
    }

    /// Encode an optional [`Canonical`] value.
    pub fn option<T: Canonical>(&mut self, v: &Option<T>) {
        match v
        {
            Some(x) =>
            {
                self.tag(T_SOME);
                x.encode(self);
            },
            None => self.tag(T_NONE),
        }
    }
}

/// A type with a deterministic, canonical byte encoding.
///
/// Implementations MUST be **total and order-stable**: encode every field in a
/// fixed declaration order, never iterate a hash map (sort first), and never
/// depend on pointer values, wall-clock, or randomness. Two values that are
/// `==` must encode identically; two values that differ must encode
/// differently. These properties are what make [`crate::ObjectId`] a sound
/// content address.
pub trait Canonical {
    /// Append this value's canonical encoding to `enc`.
    fn encode(&self, enc: &mut CanonicalEncoder);

    /// Convenience: encode into a fresh buffer and return the bytes.
    #[must_use]
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut e = CanonicalEncoder::new();
        self.encode(&mut e);
        e.finish()
    }
}

impl Canonical for u64 {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(*self);
    }
}

impl Canonical for u32 {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(u64::from(*self));
    }
}

impl Canonical for u8 {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(u64::from(*self));
    }
}

impl Canonical for i64 {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.i64(*self);
    }
}

impl Canonical for bool {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.bool(*self);
    }
}

impl Canonical for str {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(self);
    }
}

impl Canonical for String {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(self);
    }
}

impl<T: Canonical> Canonical for Vec<T> {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.seq(self);
    }
}

impl<T: Canonical> Canonical for Option<T> {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.option(self);
    }
}

impl<T: Canonical + ?Sized> Canonical for &T {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        (*self).encode(enc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_values_encode_identically() {
        let a = String::from("hello");
        let b = String::from("hello");
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn distinct_values_differ() {
        assert_ne!(1u64.canonical_bytes(), 2u64.canonical_bytes());
        assert_ne!("a".canonical_bytes(), "b".canonical_bytes());
    }

    #[test]
    fn types_are_tagged_so_they_do_not_collide() {
        // The classic length-ambiguity attack: `["a","b"]` vs `["ab"]`,
        // and `1u64` vs the string "1". Tagging + length prefixes prevent it.
        let ab_split = vec![String::from("a"), String::from("b")];
        let ab_joined = vec![String::from("ab")];
        assert_ne!(ab_split.canonical_bytes(), ab_joined.canonical_bytes());
        assert_ne!(1u64.canonical_bytes(), String::from("1").canonical_bytes());
    }

    #[test]
    fn some_none_differ_and_are_stable() {
        let some = Some(7u64);
        let none: Option<u64> = None;
        assert_ne!(some.canonical_bytes(), none.canonical_bytes());
        assert_eq!(some.canonical_bytes(), Some(7u64).canonical_bytes());
    }

    #[test]
    fn empty_seq_is_distinct_from_none() {
        let empty: Vec<u64> = vec![];
        let none: Option<u64> = None;
        assert_ne!(empty.canonical_bytes(), none.canonical_bytes());
    }

    #[test]
    fn reference_forwards_to_inner() {
        let v = 42u64;
        let r = &v;
        assert_eq!(v.canonical_bytes(), r.canonical_bytes());
    }
}
