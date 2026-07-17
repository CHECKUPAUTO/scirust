//! The spec §10 deterministic key schedule: HKDF-SHA-256 (RFC 5869) with an
//! HMAC-SHA-256 counter-mode expansion, producing per-round subkeys, round
//! constants, and Even–Mansour whitening.
//!
//! This is the **real** derivation (unlike the `fixtures` test material). It is
//! still an EXPERIMENTAL research construction — HKDF is used here as a
//! standardized, well-reviewed KDF to key an unproven permutation; nothing about
//! that makes the permutation secure.
//!
//! `sha2` is used only for the KDF; the octonion permutation itself remains
//! integer-only.

use crate::algebra::Oct;
use crate::algebra::word::Word;
use crate::fixtures::{RoundMaterial, Whitening};
use sha2::{Digest, Sha256};

/// Domain-separation namespace (spec §8 domains).
const NS: &[u8] = b"SCIRUST-HYPERCRYPTO-V0.1";
/// State width bound into every `info` string (spec §10.3).
const STATE_WIDTH: u64 = 1024;

/// HMAC-SHA-256 (FIPS 198-1) over the RustCrypto `Sha256`.
pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut k = [0u8; 64];
    if key.len() > 64
    {
        let h = Sha256::digest(key);
        k[..32].copy_from_slice(&h);
    }
    else
    {
        k[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64
    {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }
    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(msg);
    let ih = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(ih);
    outer.finalize().into()
}

/// HKDF-Extract (RFC 5869): `PRK = HMAC(salt, IKM)`.
pub fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> [u8; 32] {
    hmac_sha256(salt, ikm)
}

/// HKDF-Expand (RFC 5869) counter-mode expansion. `out_len ≤ 255·32`.
pub fn hkdf_expand(prk: &[u8; 32], info: &[u8], out_len: usize) -> Vec<u8> {
    assert!(out_len <= 255 * 32, "HKDF-Expand output too long");
    let n = out_len.div_ceil(32);
    let mut t: Vec<u8> = Vec::new();
    let mut okm = Vec::with_capacity(n * 32);
    for i in 1..=n
    {
        let mut msg = Vec::with_capacity(t.len() + info.len() + 1);
        msg.extend_from_slice(&t);
        msg.extend_from_slice(info);
        msg.push(i as u8);
        t = hmac_sha256(prk, &msg).to_vec();
        okm.extend_from_slice(&t);
    }
    okm.truncate(out_len);
    okm
}

fn info(domain: &[u8], round: u32, slot: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(NS.len() + domain.len() + 16);
    v.extend_from_slice(NS);
    v.extend_from_slice(domain);
    v.extend_from_slice(&STATE_WIDTH.to_le_bytes());
    v.extend_from_slice(&round.to_le_bytes());
    v.extend_from_slice(&slot.to_le_bytes());
    v
}

fn oct_from_bytes<W: Word>(b: &[u8]) -> Oct<W> {
    debug_assert_eq!(b.len(), 64);
    let mut v = [0u64; 8];
    for i in 0..8
    {
        let mut w = [0u8; 8];
        w.copy_from_slice(&b[i * 8..i * 8 + 8]);
        v[i] = u64::from_le_bytes(w);
    }
    Oct::<W>::from_u64s(v)
}

/// The deterministic HKDF key schedule (spec §10).
#[derive(Clone, Debug)]
pub struct KeySchedule {
    prk: [u8; 32],
}

impl KeySchedule {
    /// Derive the PRK from a master key and tweak
    /// (`PRK = HKDF-Extract("…/EXTRACT", master_key || tweak)`).
    pub fn new(master_key: &[u8], tweak: &[u8]) -> Self {
        let mut salt = Vec::from(NS);
        salt.extend_from_slice(b"/EXTRACT");
        let mut ikm = Vec::with_capacity(master_key.len() + tweak.len());
        ikm.extend_from_slice(master_key);
        ikm.extend_from_slice(tweak);
        KeySchedule {
            prk: hkdf_extract(&salt, &ikm),
        }
    }

    fn oct<W: Word>(&self, domain: &[u8], round: u32, slot: u32) -> Oct<W> {
        let bytes = hkdf_expand(&self.prk, &info(domain, round, slot), 64);
        oct_from_bytes::<W>(&bytes)
    }

    /// Per-round subkeys `K0,K1,K2` (`/ROUNDKEY`) and constant `RC` (`/CONSTANT`).
    pub fn round_material<W: Word>(&self, r: u32) -> RoundMaterial<W> {
        RoundMaterial {
            k0: self.oct::<W>(b"/ROUNDKEY", r, 0),
            k1: self.oct::<W>(b"/ROUNDKEY", r, 1),
            k2: self.oct::<W>(b"/ROUNDKEY", r, 2),
            rc: self.oct::<W>(b"/CONSTANT", r, 0),
        }
    }

    /// Even–Mansour whitening (`/WHITENING`), at reserved round indices.
    pub fn whitening<W: Word>(&self) -> Whitening<W> {
        Whitening {
            in_l: self.oct::<W>(b"/WHITENING", 0xFFFF_FFFF, 0),
            in_r: self.oct::<W>(b"/WHITENING", 0xFFFF_FFFF, 1),
            out_l: self.oct::<W>(b"/WHITENING", 0xFFFF_FFFE, 0),
            out_r: self.oct::<W>(b"/WHITENING", 0xFFFF_FFFE, 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::word::W64;

    #[test]
    fn hmac_rfc4231_case1() {
        // RFC 4231 Test Case 1: key = 0x0b*20, data = "Hi There"
        let key = [0x0bu8; 20];
        let mac = hmac_sha256(&key, b"Hi There");
        let expect = "b0344c61d8db38535ca8afceaf0bf12b\
                      881dc200c9833da726e9376c2e32cff7";
        let got: String = mac.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(got, expect);
    }

    #[test]
    fn hkdf_rfc5869_case1() {
        // RFC 5869 Appendix A.1
        let ikm = [0x0bu8; 22];
        let salt: Vec<u8> = (0x00u8..=0x0c).collect();
        let info: Vec<u8> = (0xf0u8..=0xf9).collect();
        let prk = hkdf_extract(&salt, &ikm);
        let prk_hex: String = prk.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            prk_hex,
            "077709362c2e32df0ddc3f0dc47bba63\
             90b6c73bb50f9c3122ec844ad7c2b3e5"
        );
        let okm = hkdf_expand(&prk, &info, 42);
        let okm_hex: String = okm.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            okm_hex,
            "3cb25f25faacd57a90434f64d0362f2a\
             2d2d0a90cf1a5a4c5db02d56ecc4c5bf\
             34007208d5b887185865"
        );
    }

    #[test]
    fn schedule_is_deterministic() {
        let a = KeySchedule::new(&[0u8; 32], &[0u8; 16]);
        let b = KeySchedule::new(&[0u8; 32], &[0u8; 16]);
        assert_eq!(
            a.round_material::<W64>(3).k1.to_u64s(),
            b.round_material::<W64>(3).k1.to_u64s()
        );
        // distinct rounds give distinct material
        assert_ne!(
            a.round_material::<W64>(3).k1.to_u64s(),
            a.round_material::<W64>(4).k1.to_u64s()
        );
        // tweak changes the schedule
        let c = KeySchedule::new(&[0u8; 32], &[1u8; 16]);
        assert_ne!(
            a.round_material::<W64>(0).k0.to_u64s(),
            c.round_material::<W64>(0).k0.to_u64s()
        );
    }
}
