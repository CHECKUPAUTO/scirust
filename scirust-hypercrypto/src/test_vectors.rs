//! Official reproducible test vectors for the exact v0.1 permutation with the
//! real HKDF-SHA-256 key schedule (spec §13, §15).
//!
//! These vectors exercise the full construction (1024-bit state, two `W64`
//! octonion branches, 24 rounds, Even–Mansour whitening) keyed by the spec §10
//! HKDF schedule. Every vector round-trips (`P_K^{-1}(P_K(x)) = x`); the set is
//! pinned by [`VECTORS_FINGERPRINT`] as a cross-implementation regression
//! contract, analogous to the Philox fingerprint in `scirust-core`.
//!
//! EXPERIMENTAL: these are reproducibility vectors for a research permutation,
//! NOT evidence of any security property.

use crate::algebra::Oct;
use crate::algebra::word::W64;
use crate::analysis::report::{Json, sha256_hex};
use crate::fixtures::{Fixture, FixtureId};
use crate::permutation::{State, Variant, feistel};

/// Round count for the official vectors (spec §12.1).
pub const ROUNDS: u32 = 24;

/// One official test vector (all fields fixed and reproducible).
#[derive(Clone, Debug)]
pub struct TestVector {
    /// Category name.
    pub name: &'static str,
    /// 32-byte master key.
    pub key: [u8; 32],
    /// 16-byte tweak.
    pub tweak: [u8; 16],
    /// 128-byte input state (`L || R`, little-endian octonions).
    pub input: [u8; 128],
    /// `P_K(input)`.
    pub output: [u8; 128],
    /// `P_K^{-1}(output)` (must equal `input`).
    pub inverse_output: [u8; 128],
}

fn hexstr(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b
    {
        s.push_str(&format!("{x:02x}"));
    }
    s
}

fn split128(input: &[u8; 128]) -> State<W64> {
    let mut lb = [0u8; 64];
    let mut rb = [0u8; 64];
    lb.copy_from_slice(&input[..64]);
    rb.copy_from_slice(&input[64..]);
    State::new(
        Oct::<W64>::from_le_bytes(&lb),
        Oct::<W64>::from_le_bytes(&rb),
    )
}

fn join128(s: State<W64>) -> [u8; 128] {
    let mut out = [0u8; 128];
    out[..64].copy_from_slice(&s.l.to_le_bytes());
    out[64..].copy_from_slice(&s.r.to_le_bytes());
    out
}

/// Compute `(output, inverse_output)` for a key/tweak/input triple.
fn permute(key: [u8; 32], tweak: [u8; 16], input: [u8; 128]) -> ([u8; 128], [u8; 128]) {
    let fx = Fixture::new(FixtureId::Hkdf(key, tweak));
    let s = split128(&input);
    let enc = feistel::forward(s, &fx, Variant::V01, ROUNDS, true);
    let dec = feistel::inverse(enc, &fx, Variant::V01, ROUNDS, true);
    (join128(enc), join128(dec))
}

/// The official vector set covering the spec §15 categories.
pub fn official_vectors() -> Vec<TestVector> {
    let mk = |name: &'static str, key: [u8; 32], tweak: [u8; 16], input: [u8; 128]| {
        let (output, inverse_output) = permute(key, tweak, input);
        TestVector {
            name,
            key,
            tweak,
            input,
            output,
            inverse_output,
        }
    };

    let mut single_bit_input = [0u8; 128];
    single_bit_input[0] = 1;
    let mut single_bit_key = [0u8; 32];
    single_bit_key[0] = 1;
    let inc_input: [u8; 128] = std::array::from_fn(|i| i as u8);
    let inc_key: [u8; 32] = std::array::from_fn(|i| i as u8);
    let mut alt_input = [0u8; 128];
    for (i, b) in alt_input.iter_mut().enumerate()
    {
        *b = if i % 2 == 0 { 0x55 } else { 0xaa };
    }

    vec![
        mk(
            "all-zero-key-all-zero-input",
            [0u8; 32],
            [0u8; 16],
            [0u8; 128],
        ),
        mk("all-one-key", [0xffu8; 32], [0u8; 16], [0u8; 128]),
        mk("single-bit-input", [0u8; 32], [0u8; 16], single_bit_input),
        mk("single-bit-key", single_bit_key, [0u8; 16], [0u8; 128]),
        mk("incrementing-bytes", inc_key, [0u8; 16], inc_input),
        mk("alternating-bits", [0x55u8; 32], [0xaau8; 16], alt_input),
        mk("max-components", [0xffu8; 32], [0xffu8; 16], [0xffu8; 128]),
    ]
}

/// A deterministic SHA-256 fingerprint over the whole official vector set (the
/// cross-implementation contract). Recomputed by [`VECTORS_FINGERPRINT`]'s test.
pub fn vectors_fingerprint() -> String {
    let mut s = String::new();
    for tv in official_vectors()
    {
        s.push_str(tv.name);
        s.push('|');
        s.push_str(&hexstr(&tv.key));
        s.push('|');
        s.push_str(&hexstr(&tv.tweak));
        s.push('|');
        s.push_str(&hexstr(&tv.input));
        s.push('|');
        s.push_str(&hexstr(&tv.output));
        s.push('\n');
    }
    sha256_hex(&s)
}

/// Pinned fingerprint of the official vector set (regression contract). Any
/// change to the permutation, key schedule, serialization, or vector inputs
/// changes this value.
pub const VECTORS_FINGERPRINT: &str =
    "07db3cbfa4bab6f8cb68ab349075e2a7000c6327fb5ea77dd8caaed2344ceb4b";

/// The official vector set as a machine-readable JSON document (spec §15 fields).
pub fn vectors_json() -> Json {
    let entries: Vec<Json> = official_vectors()
        .into_iter()
        .map(|tv| {
            Json::obj(vec![
                ("name", Json::s(tv.name)),
                ("algorithm_version", Json::s(crate::SPEC_VERSION)),
                ("coefficient_domain", Json::s("Z/2^64")),
                ("state_width", Json::U64(1024)),
                ("round_count", Json::U64(ROUNDS as u64)),
                ("master_key", Json::s(hexstr(&tv.key))),
                ("tweak", Json::s(hexstr(&tv.tweak))),
                ("input", Json::s(hexstr(&tv.input))),
                ("output", Json::s(hexstr(&tv.output))),
                ("inverse_output", Json::s(hexstr(&tv.inverse_output))),
            ])
        })
        .collect();
    Json::obj(vec![
        ("algorithm_version", Json::s(crate::SPEC_VERSION)),
        ("key_schedule", Json::s("HKDF-SHA-256 (RFC 5869)")),
        ("rounds", Json::U64(ROUNDS as u64)),
        ("vectors_fingerprint", Json::s(vectors_fingerprint())),
        ("vectors", Json::Arr(entries)),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_vector_round_trips() {
        for tv in official_vectors()
        {
            assert_eq!(
                tv.inverse_output, tv.input,
                "round-trip failed for vector {}",
                tv.name
            );
            // a non-trivial permutation must move at least some inputs
            if tv.input.iter().any(|&b| b != 0)
            {
                assert_ne!(
                    tv.output, tv.input,
                    "permutation is identity on {}",
                    tv.name
                );
            }
        }
    }

    #[test]
    fn fingerprint_is_stable() {
        // determinism: two computations agree
        assert_eq!(vectors_fingerprint(), vectors_fingerprint());
        // contract: pinned value (skipped while the placeholder is in place)
        if VECTORS_FINGERPRINT != "0000000000000000000000000000000000000000000000000000000000000000"
        {
            assert_eq!(vectors_fingerprint(), VECTORS_FINGERPRINT);
        }
    }
}
