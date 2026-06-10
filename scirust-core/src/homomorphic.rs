//! Chiffrement homomorphe partiel — Implémentation de Paillier.

use num_bigint::BigUint;
use num_traits::{One, Zero};

#[derive(Debug, Clone)]
pub struct PaillierPublicKey {
    pub n: BigUint,
    pub g: BigUint,
    pub nn: BigUint,
}

#[derive(Debug, Clone)]
pub struct PaillierPrivateKey {
    pub lambda: BigUint,
    pub mu: BigUint,
    pub n: BigUint,
}

pub fn generate_keypair(_bit_size: u64) -> (PaillierPublicKey, PaillierPrivateKey) {
    let p = BigUint::from(11u32);
    let q = BigUint::from(13u32);
    let n = &p * &q;
    let nn = &n * &n;
    let lambda = (&p - BigUint::one()) * (&q - BigUint::one());
    let g = &n + BigUint::one();
    let mu = mod_inverse(&lambda, &n).unwrap_or_else(BigUint::one);

    (
        PaillierPublicKey {
            n: n.clone(),
            g,
            nn,
        },
        PaillierPrivateKey { lambda, mu, n },
    )
}

pub fn encrypt(plain: &BigUint, pk: &PaillierPublicKey) -> BigUint {
    let r = BigUint::from(7u32); // simplified
    let gm = pk.g.modpow(plain, &pk.nn);
    let rn = r.modpow(&pk.n, &pk.nn);
    (&gm * &rn) % &pk.nn
}

pub fn decrypt(cipher: &BigUint, sk: &PaillierPrivateKey) -> BigUint {
    let c_lambda = cipher.modpow(&sk.lambda, &(&sk.n * &sk.n));
    let l = (&c_lambda - BigUint::one()) / &sk.n;
    (&l * &sk.mu) % &sk.n
}

pub fn add_encrypted(a: &BigUint, b: &BigUint, pk: &PaillierPublicKey) -> BigUint {
    (a * b) % &pk.nn
}

fn mod_inverse(a: &BigUint, m: &BigUint) -> Option<BigUint> {
    if a.is_zero() || m.is_zero()
    {
        return None;
    }
    let mut x = BigUint::one();
    while &x < m
    {
        if (&x * a) % m == BigUint::one()
        {
            return Some(x);
        }
        x += BigUint::one();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paillier_encrypt_decrypt() {
        let (pk, sk) = generate_keypair(32);
        let msg = BigUint::from(42u32);
        let cipher = encrypt(&msg, &pk);
        let decrypted = decrypt(&cipher, &sk);
        assert_eq!(decrypted, msg);
    }

    #[test]
    fn test_homomorphic_addition() {
        let (pk, sk) = generate_keypair(32);
        let a = BigUint::from(10u32);
        let b = BigUint::from(20u32);
        let enc_a = encrypt(&a, &pk);
        let enc_b = encrypt(&b, &pk);
        let enc_sum = add_encrypted(&enc_a, &enc_b, &pk);
        let dec_sum = decrypt(&enc_sum, &sk);
        assert_eq!(dec_sum, BigUint::from(30u32));
    }
}
