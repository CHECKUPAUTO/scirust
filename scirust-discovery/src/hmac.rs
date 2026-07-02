//! HMAC-SHA256 (RFC 2104 sur `scirust_sciagent::sha256`, FIPS 180-4).
//!
//! Sert à signer/vérifier une [`crate::scope::ScopeAuthorization`] : sans
//! signature vérifiable, une portée de découverte n'est qu'un fichier de
//! config que n'importe qui peut éditer pour s'auto-autoriser à sonder un
//! réseau qu'on ne lui a pas confié. Ce n'est pas une PKI complète (pas de
//! rotation de clé, pas de révocation) — une clé partagée pré-échangée entre
//! l'opérateur autorisant et l'agent qui exécute la découverte — mais c'est
//! vérifiable, hors-ligne, et sans dépendance externe.

use scirust_sciagent::sha256::sha256;

const BLOCK_SIZE: usize = 64;

pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut key_block = [0u8; BLOCK_SIZE];
    if key.len() > BLOCK_SIZE
    {
        let hashed = sha256(key);
        key_block[..32].copy_from_slice(&hashed);
    }
    else
    {
        key_block[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; BLOCK_SIZE];
    let mut opad = [0x5cu8; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE
    {
        ipad[i] ^= key_block[i];
        opad[i] ^= key_block[i];
    }

    let mut inner_input = ipad.to_vec();
    inner_input.extend_from_slice(message);
    let inner_hash = sha256(&inner_input);

    let mut outer_input = opad.to_vec();
    outer_input.extend_from_slice(&inner_hash);
    sha256(&outer_input)
}

pub fn hmac_sha256_hex(key: &[u8], message: &[u8]) -> String {
    let digest = hmac_sha256(key, message);
    let mut s = String::with_capacity(64);
    for b in digest
    {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc4231_test_case_1() {
        // RFC 4231 §4.2 : Key = 0x0b * 20, Data = "Hi There".
        let key = [0x0bu8; 20];
        let hash = hmac_sha256_hex(&key, b"Hi There");
        assert_eq!(
            hash,
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
        assert_eq!(hash.len(), 64);
    }

    #[test]
    fn rfc4231_test_case_2() {
        // RFC 4231 §4.3 : Key = "Jefe", Data = "what do ya want for nothing?".
        let hash = hmac_sha256_hex(b"Jefe", b"what do ya want for nothing?");
        assert_eq!(
            hash,
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn different_keys_produce_different_macs() {
        let a = hmac_sha256_hex(b"key-a", b"same message");
        let b = hmac_sha256_hex(b"key-b", b"same message");
        assert_ne!(a, b);
    }

    #[test]
    fn key_longer_than_block_size_is_hashed_first() {
        let long_key = vec![0x42u8; 200];
        // Ne doit pas paniquer, et doit être stable (déterministe).
        let a = hmac_sha256_hex(&long_key, b"msg");
        let b = hmac_sha256_hex(&long_key, b"msg");
        assert_eq!(a, b);
    }
}
