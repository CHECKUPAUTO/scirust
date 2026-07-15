//! Partial homomorphic encryption — a from-scratch Paillier cryptosystem.
//!
//! # ⚠️ Security status — read before use
//!
//! This is an **educational, from-scratch** implementation. It generates real
//! random primes of the requested size and draws a fresh, cryptographically
//! secure random `r ∈ Z*_n` for **every** encryption, so encryption is
//! *probabilistic*: encrypting the same plaintext twice yields different
//! ciphertexts (textbook semantic security). This is a deliberate fix over the
//! previous toy, which hardcoded `p = 11, q = 13` and a fixed `r = 7` (making
//! encryption deterministic, hence trivially breakable).
//!
//! It is nonetheless **NOT production cryptography**: modular exponentiation is
//! not constant-time (side-channel exposure), there is no ciphertext integrity
//! or padding, and the code has not been independently audited. **Do not use it
//! to protect real secrets.** For production, use a vetted, maintained library.
//! For meaningful hardness pass `bit_size >= 2048`; smaller sizes exist only so
//! the tests run quickly and to keep the API honest about the parameter.
//!
//! Paillier is *additively* homomorphic:
//! `D(E(a) · E(b) mod n²) = a + b (mod n)`.

use num_bigint::{BigInt, BigUint};
use num_traits::{One, Signed, Zero};
use rand::RngCore;
use rand::rngs::OsRng;

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

/// Generate a Paillier keypair with primes of about `bit_size / 2` bits each
/// (so the modulus `n` has ~`bit_size` bits), seeding randomness from the OS
/// CSPRNG. See the module note: use `bit_size >= 2048` for any real hardness.
pub fn generate_keypair(bit_size: u64) -> (PaillierPublicKey, PaillierPrivateKey) {
    let mut rng = OsRng;
    generate_keypair_with_rng(bit_size, &mut rng)
}

/// Like [`generate_keypair`] but with a caller-supplied RNG. For real use the
/// RNG **must** be cryptographically secure (e.g. `OsRng`); a deterministic RNG
/// is acceptable only for reproducible tests.
pub fn generate_keypair_with_rng(
    bit_size: u64,
    rng: &mut impl RngCore,
) -> (PaillierPublicKey, PaillierPrivateKey) {
    let prime_bits = (bit_size / 2).max(8);
    // Pick distinct primes p, q such that gcd(n, (p-1)(q-1)) = 1 (required for
    // the standard g = n+1 variant so that λ is invertible mod n).
    let (p, q, n, lambda) = loop
    {
        let p = gen_prime(prime_bits, rng);
        let q = gen_prime(prime_bits, rng);
        if p == q
        {
            continue;
        }
        let n = &p * &q;
        let lambda = (&p - BigUint::one()) * (&q - BigUint::one());
        if gcd(&n, &lambda) == BigUint::one()
        {
            break (p, q, n, lambda);
        }
    };
    let _ = (&p, &q); // kept for clarity; not stored
    let nn = &n * &n;
    let g = &n + BigUint::one();
    // With g = n+1: L((n+1)^λ mod n²) = λ, hence μ = λ⁻¹ mod n.
    let mu = mod_inverse(&lambda, &n).expect("λ invertible mod n for admissible primes");

    (
        PaillierPublicKey {
            n: n.clone(),
            g,
            nn,
        },
        PaillierPrivateKey { lambda, mu, n },
    )
}

/// Encrypt `plain` (reduced mod `n`) with a fresh OS-random blinding factor.
pub fn encrypt(plain: &BigUint, pk: &PaillierPublicKey) -> BigUint {
    let mut rng = OsRng;
    encrypt_with_rng(plain, pk, &mut rng)
}

/// Like [`encrypt`] but with a caller-supplied RNG. The RNG **must** be
/// cryptographically secure for the ciphertext to hide the plaintext; a
/// deterministic RNG must only be used in tests.
pub fn encrypt_with_rng(
    plain: &BigUint,
    pk: &PaillierPublicKey,
    rng: &mut impl RngCore,
) -> BigUint {
    let m = plain % &pk.n;
    // Fresh random r ∈ Z*_n — this is what makes Paillier semantically secure.
    let r = loop
    {
        let cand = random_biguint_below(&pk.n, rng);
        if !cand.is_zero() && gcd(&cand, &pk.n) == BigUint::one()
        {
            break cand;
        }
    };
    let gm = pk.g.modpow(&m, &pk.nn);
    let rn = r.modpow(&pk.n, &pk.nn);
    (&gm * &rn) % &pk.nn
}

/// Decrypt a Paillier ciphertext back to `Z_n`.
pub fn decrypt(cipher: &BigUint, sk: &PaillierPrivateKey) -> BigUint {
    let nn = &sk.n * &sk.n;
    let c_lambda = cipher.modpow(&sk.lambda, &nn);
    // L(x) = (x - 1) / n
    let l = (&c_lambda - BigUint::one()) / &sk.n;
    (&l * &sk.mu) % &sk.n
}

/// Homomorphic addition of two ciphertexts: `D(add_encrypted(E(a), E(b))) = a + b`.
pub fn add_encrypted(a: &BigUint, b: &BigUint, pk: &PaillierPublicKey) -> BigUint {
    (a * b) % &pk.nn
}

/// Homomorphic multiplication of a ciphertext by a plaintext scalar `k`:
/// `D(mul_plain(E(m), k)) = k · m (mod n)`.
pub fn mul_plain(cipher: &BigUint, k: &BigUint, pk: &PaillierPublicKey) -> BigUint {
    cipher.modpow(k, &pk.nn)
}

// -------------------------------------------------------------------------
//  Number-theoretic helpers
// -------------------------------------------------------------------------

/// Modular inverse of `a` modulo `m` via the extended Euclidean algorithm.
/// Returns `None` when `gcd(a, m) ≠ 1`.
fn mod_inverse(a: &BigUint, m: &BigUint) -> Option<BigUint> {
    if m.is_zero()
    {
        return None;
    }
    let m_int = BigInt::from(m.clone());
    let mut t = BigInt::zero();
    let mut newt = BigInt::one();
    let mut r = m_int.clone();
    let mut newr = BigInt::from(a.clone()) % &m_int;
    while !newr.is_zero()
    {
        let q = &r / &newr;
        let tmp_t = &t - &q * &newt;
        t = std::mem::replace(&mut newt, tmp_t);
        let tmp_r = &r - &q * &newr;
        r = std::mem::replace(&mut newr, tmp_r);
    }
    if r > BigInt::one()
    {
        return None; // a and m are not coprime
    }
    if t.is_negative()
    {
        t += &m_int;
    }
    t.to_biguint()
}

/// Greatest common divisor (Euclid).
fn gcd(a: &BigUint, b: &BigUint) -> BigUint {
    let mut a = a.clone();
    let mut b = b.clone();
    while !b.is_zero()
    {
        let r = &a % &b;
        a = std::mem::replace(&mut b, r);
    }
    a
}

/// Uniform random `BigUint` in `[0, bound)` (rejection sampling). `bound` must
/// be > 0.
fn random_biguint_below(bound: &BigUint, rng: &mut impl RngCore) -> BigUint {
    let bits = bound.bits().max(1);
    let n_bytes = bits.div_ceil(8) as usize;
    loop
    {
        let mut bytes = vec![0u8; n_bytes];
        rng.fill_bytes(&mut bytes);
        let mut candidate = BigUint::from_bytes_le(&bytes);
        let excess = (n_bytes as u64) * 8 - bits;
        if excess > 0
        {
            candidate >>= excess as usize;
        }
        if &candidate < bound
        {
            return candidate;
        }
    }
}

/// Generate a probable prime with exactly `bits` bits (top bit set, odd).
fn gen_prime(bits: u64, rng: &mut impl RngCore) -> BigUint {
    let bits = bits.max(4);
    let n_bytes = bits.div_ceil(8) as usize;
    let top = BigUint::one() << ((bits - 1) as usize);
    loop
    {
        let mut bytes = vec![0u8; n_bytes];
        rng.fill_bytes(&mut bytes);
        let mut candidate = BigUint::from_bytes_le(&bytes);
        let excess = (n_bytes as u64) * 8 - bits;
        if excess > 0
        {
            candidate >>= excess as usize;
        }
        candidate |= &top; // force exact bit length
        candidate |= BigUint::one(); // force odd
        if is_probable_prime(&candidate, 40, rng)
        {
            return candidate;
        }
    }
}

/// Miller–Rabin probabilistic primality test with `rounds` random bases.
fn is_probable_prime(n: &BigUint, rounds: u32, rng: &mut impl RngCore) -> bool {
    let two = BigUint::from(2u32);
    let three = BigUint::from(3u32);
    if n < &two
    {
        return false;
    }
    if n == &two || n == &three
    {
        return true;
    }
    if !n.bit(0)
    {
        return false; // even
    }

    let one = BigUint::one();
    let n_minus_1 = n - &one;
    // Write n-1 = d · 2^s with d odd.
    let mut d = n_minus_1.clone();
    let mut s = 0u32;
    while !d.bit(0)
    {
        d >>= 1usize;
        s += 1;
    }

    let bound = n - &two; // sample a ∈ [2, n-2] ⇒ [0, n-4) + 2
    'witness: for _ in 0..rounds
    {
        let a = random_biguint_below(&(bound.clone()), rng) + &two;
        let mut x = a.modpow(&d, n);
        if x == one || x == n_minus_1
        {
            continue 'witness;
        }
        for _ in 0..s.saturating_sub(1)
        {
            x = x.modpow(&two, n);
            if x == n_minus_1
            {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    // Fixed-seed CSPRNG for reproducible tests; production paths use OsRng.
    fn test_rng() -> StdRng {
        StdRng::from_seed([7u8; 32])
    }

    #[test]
    fn test_paillier_encrypt_decrypt() {
        let mut rng = test_rng();
        let (pk, sk) = generate_keypair_with_rng(64, &mut rng);
        let msg = BigUint::from(42u32);
        let cipher = encrypt_with_rng(&msg, &pk, &mut rng);
        let decrypted = decrypt(&cipher, &sk);
        assert_eq!(decrypted, msg);
    }

    #[test]
    fn test_homomorphic_addition() {
        let mut rng = test_rng();
        let (pk, sk) = generate_keypair_with_rng(64, &mut rng);
        let a = BigUint::from(10u32);
        let b = BigUint::from(20u32);
        let enc_a = encrypt_with_rng(&a, &pk, &mut rng);
        let enc_b = encrypt_with_rng(&b, &pk, &mut rng);
        let enc_sum = add_encrypted(&enc_a, &enc_b, &pk);
        let dec_sum = decrypt(&enc_sum, &sk);
        assert_eq!(dec_sum, BigUint::from(30u32));
    }

    #[test]
    fn test_scalar_multiplication_homomorphism() {
        let mut rng = test_rng();
        let (pk, sk) = generate_keypair_with_rng(64, &mut rng);
        let m = BigUint::from(7u32);
        let k = BigUint::from(6u32);
        let enc = encrypt_with_rng(&m, &pk, &mut rng);
        let enc_km = mul_plain(&enc, &k, &pk);
        assert_eq!(decrypt(&enc_km, &sk), BigUint::from(42u32));
    }

    /// The security-critical property the old toy violated: encryption is
    /// probabilistic, so the same plaintext maps to different ciphertexts.
    #[test]
    fn encryption_is_probabilistic() {
        let mut rng = test_rng();
        let (pk, sk) = generate_keypair_with_rng(96, &mut rng);
        let msg = BigUint::from(12345u32);
        let c1 = encrypt_with_rng(&msg, &pk, &mut rng);
        let c2 = encrypt_with_rng(&msg, &pk, &mut rng);
        assert_ne!(c1, c2, "encryption must be randomized");
        // …yet both decrypt to the same plaintext.
        assert_eq!(decrypt(&c1, &sk), msg);
        assert_eq!(decrypt(&c2, &sk), msg);
    }

    /// `bit_size` is honored: the modulus has roughly the requested bit length
    /// (the toy ignored it and always produced n = 143).
    #[test]
    fn keypair_respects_bit_size() {
        let mut rng = test_rng();
        let (pk, _sk) = generate_keypair_with_rng(128, &mut rng);
        let bits = pk.n.bits();
        assert!(
            (120..=132).contains(&bits),
            "n has {bits} bits, expected ~128"
        );
    }

    #[test]
    fn mod_inverse_is_correct() {
        // 3 · 4 ≡ 1 (mod 11)
        let inv = mod_inverse(&BigUint::from(3u32), &BigUint::from(11u32)).unwrap();
        assert_eq!(inv, BigUint::from(4u32));
        // gcd ≠ 1 ⇒ no inverse
        assert!(mod_inverse(&BigUint::from(6u32), &BigUint::from(9u32)).is_none());
    }

    #[test]
    fn primality_test_agrees_with_known_values() {
        let mut rng = test_rng();
        for &p in &[2u32, 3, 5, 7, 97, 7919]
        {
            assert!(
                is_probable_prime(&BigUint::from(p), 20, &mut rng),
                "{p} prime"
            );
        }
        for &c in &[1u32, 4, 9, 100, 7917]
        {
            assert!(
                !is_probable_prime(&BigUint::from(c), 20, &mut rng),
                "{c} composite"
            );
        }
    }
}
