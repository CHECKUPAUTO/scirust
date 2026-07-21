//! Exact **discrete logarithms** in the multiplicative group `(ℤ/pℤ)*` of a
//! prime field: baby-step giant-step and the Pohlig–Hellman algorithm.
//!
//! The discrete logarithm problem is: given `g`, `h` and a modulus, find `x`
//! with `gˣ ≡ h`. Two exact, deterministic solvers are provided:
//!
//! - [`bsgs`] — **baby-step giant-step**, the generic `O(√n)` time / `O(√n)`
//!   space meet-in-the-middle search over any bounded exponent range;
//! - [`discrete_log`] — **Pohlig–Hellman**, which reduces the problem in
//!   `⟨g⟩` to one discrete log per prime-power factor of the group order (each
//!   solved by a small BSGS) and recombines them with the CRT. It is efficient
//!   exactly when the order of `g` is *smooth* (only small prime factors).
//!
//! Everything is exact `u64`/`u128` modular arithmetic — no floating point, no
//! OS entropy (the baby-step table is an ordered `BTreeMap`), reproducible
//! bit-for-bit. This module composes [`crate::numtheory`] for the modular
//! inverse/exponentiation, the factorization of the group order, and the CRT.

use crate::numtheory::{crt, factor, inv_mod, is_prime, isqrt, mulmod, pow_mod};
use std::collections::BTreeMap;

/// The multiplicative order of `a` modulo the prime `p` — the smallest `d > 0`
/// with `a^d ≡ 1 (mod p)`. Returns `None` if `a ≡ 0 (mod p)`.
///
/// Panics unless `p` is prime.
pub fn multiplicative_order(a: u64, p: u64) -> Option<u64> {
    assert!(p >= 2 && is_prime(p), "modulus must be prime");
    let a = a % p;
    if a == 0
    {
        return None;
    }
    // Start from p − 1 = |(ℤ/pℤ)*| and remove prime factors while the power
    // still reduces to 1.
    let mut ord = p - 1;
    for (q, _) in factor(p - 1)
    {
        while ord % q == 0 && pow_mod(a, ord / q, p) == 1
        {
            ord /= q;
        }
    }
    Some(ord)
}

/// **Baby-step giant-step**: the least `x` in `[0, bound)` with `gˣ ≡ h
/// (mod modulus)`, or `None` if there is none. Runs in `O(√bound)` time and
/// space.
///
/// `g` must be a unit modulo `modulus` (else there is no giant-step factor and
/// the search reports `None`).
pub fn bsgs(g: u64, h: u64, modulus: u64, bound: u64) -> Option<u64> {
    assert!(modulus >= 2, "modulus must be ≥ 2");
    if bound == 0
    {
        return None;
    }
    let g = g % modulus;
    let h = h % modulus;
    // m = ⌈√bound⌉
    let m = isqrt(bound as u128) as u64 + 1;

    // Baby steps: value g^j ↦ smallest j, for j = 0 … m−1.
    let mut table: BTreeMap<u64, u64> = BTreeMap::new();
    let mut cur = 1u64;
    for j in 0..m
    {
        table.entry(cur).or_insert(j);
        cur = mulmod(cur, g, modulus);
    }

    // Giant-step factor g^{-m}; absent iff g is not invertible.
    let g_inv = inv_mod(g, modulus)?;
    let factor = pow_mod(g_inv, m, modulus);

    let mut gamma = h;
    for i in 0..=m
    {
        if let Some(&j) = table.get(&gamma)
        {
            let x = i * m + j;
            if x < bound
            {
                return Some(x);
            }
        }
        gamma = mulmod(gamma, factor, modulus);
    }
    None
}

/// Solve `gamma^y ≡ target (mod p)` for `y` in `[0, q^e)`, where `gamma` has
/// order exactly `q^e` (Pohlig–Hellman's per-prime-power routine): recover the
/// base-`q` digits of `y` one at a time, each by a `BSGS` in the order-`q`
/// subgroup.
fn dlog_prime_power(gamma: u64, target: u64, q: u64, e: u32, p: u64) -> Option<u64> {
    let qe = q.pow(e);
    let gamma_inv = inv_mod(gamma, p)?;
    // gq generates the unique subgroup of order q.
    let gq = pow_mod(gamma, qe / q, p);
    let mut x = 0u64;
    for k in 0..e
    {
        // h_k = (gamma^{−x} · target)^{q^{e−1−k}} lies in ⟨gq⟩.
        let base = mulmod(pow_mod(gamma_inv, x, p), target, p);
        let hk = pow_mod(base, q.pow(e - 1 - k), p);
        let d = bsgs(gq, hk, p, q)?;
        x += d * q.pow(k);
    }
    Some(x)
}

/// The discrete logarithm of `h` to base `g` modulo the prime `p`: the least
/// `x` in `[0, ord(g))` with `gˣ ≡ h (mod p)`, or `None` if `h ∉ ⟨g⟩` (`h` is
/// not a power of `g`).
///
/// Uses **Pohlig–Hellman**: it factors `n = ord(g)`, solves one discrete log
/// per prime-power factor `q^e ‖ n` in the corresponding subgroup, and
/// recombines with the CRT — fast when `n` is smooth. Panics unless `p` is
/// prime.
pub fn discrete_log(g: u64, h: u64, p: u64) -> Option<u64> {
    assert!(p >= 2 && is_prime(p), "modulus must be prime");
    let g = g % p;
    let h = h % p;
    if g == 0
    {
        return None;
    }
    if h == 1
    {
        return Some(0);
    }
    let n = multiplicative_order(g, p)?;

    // For each prime power q^e ‖ n, project into the order-q^e subgroup and
    // solve there; collect the residues x ≡ x_i (mod q^e).
    let mut congruences: Vec<(u64, u64)> = Vec::new();
    for (q, e) in factor(n)
    {
        let qe = q.pow(e);
        let cofactor = n / qe;
        let gamma = pow_mod(g, cofactor, p);
        let target = pow_mod(h, cofactor, p);
        let xi = dlog_prime_power(gamma, target, q, e, p)?;
        congruences.push((xi, qe));
    }

    let (x, _modulus) = crt(&congruences)?;
    // Guard against `h ∉ ⟨g⟩`: the CRT residue is only a true logarithm if it
    // actually reproduces `h`.
    if pow_mod(g, x, p) == h { Some(x) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// Brute-force least `x` in `[0, bound)` with `g^x ≡ h (mod p)`.
    fn brute(g: u64, h: u64, p: u64, bound: u64) -> Option<u64> {
        let mut cur = 1u64;
        for x in 0..bound
        {
            if cur == h % p
            {
                return Some(x);
            }
            cur = mulmod(cur, g, p);
        }
        None
    }

    #[test]
    fn multiplicative_order_known_values() {
        assert_eq!(multiplicative_order(2, 11), Some(10)); // 2 is primitive mod 11
        assert_eq!(multiplicative_order(3, 7), Some(6)); // 3 is primitive mod 7
        assert_eq!(multiplicative_order(3, 13), Some(3)); // 3^3 = 27 ≡ 1 mod 13
        assert_eq!(multiplicative_order(1, 13), Some(1));
        assert_eq!(multiplicative_order(0, 13), None);
        // order always divides p − 1
        let mut s = 0x1234u64;
        for _ in 0..50
        {
            let a = 1 + xorshift(&mut s) % 100;
            let ord = multiplicative_order(a, 101).unwrap();
            assert_eq!(100 % ord, 0);
            assert_eq!(pow_mod(a, ord, 101), 1);
        }
    }

    #[test]
    fn bsgs_matches_brute_force() {
        let mut s = 0x9e37u64;
        for &p in &[11u64, 101, 1009]
        {
            for _ in 0..40
            {
                let g = 1 + xorshift(&mut s) % (p - 1);
                let ord = multiplicative_order(g, p).unwrap();
                // pick a genuine power of g so a solution exists
                let e = xorshift(&mut s) % ord;
                let h = pow_mod(g, e, p);
                let x = bsgs(g, h, p, ord).expect("solution exists");
                assert_eq!(pow_mod(g, x, p), h);
                assert_eq!(Some(x), brute(g, h, p, ord), "not the least solution");
            }
        }
    }

    #[test]
    fn bsgs_reports_none_when_unsolvable() {
        // 3 mod 13 has order 3 (subgroup {1, 3, 9}); 2 is not in it.
        assert_eq!(multiplicative_order(3, 13), Some(3));
        assert_eq!(bsgs(3, 2, 13, 3), None);
    }

    #[test]
    fn discrete_log_known_vector() {
        // log_2(9) mod 11: 2^6 = 64 ≡ 9, and 2 is primitive so the order is 10.
        assert_eq!(discrete_log(2, 9, 11), Some(6));
        assert_eq!(discrete_log(2, 1, 11), Some(0));
    }

    #[test]
    fn discrete_log_agrees_with_bsgs() {
        let mut s = 0xbeefu64;
        for &p in &[101u64, 1009, 7919]
        {
            for _ in 0..30
            {
                let g = 1 + xorshift(&mut s) % (p - 1);
                let ord = multiplicative_order(g, p).unwrap();
                let e = xorshift(&mut s) % ord;
                let h = pow_mod(g, e, p);
                let x = discrete_log(g, h, p).expect("h is a power of g");
                assert_eq!(pow_mod(g, x, p), h);
                // both solvers land on the least representative in [0, ord)
                assert_eq!(discrete_log(g, h, p), bsgs(g, h, p, ord));
            }
        }
    }

    #[test]
    fn discrete_log_none_outside_subgroup() {
        // 3 mod 13 has order 3; 2 is not a power of 3.
        assert_eq!(discrete_log(3, 2, 13), None);
    }

    #[test]
    fn pohlig_hellman_on_smooth_prime() {
        // p = 65537 is a Fermat prime, so p − 1 = 2^16 is fully smooth — the
        // regime where Pohlig–Hellman shines. 3 is a primitive root.
        let p = 65_537u64;
        let g = 3u64;
        assert_eq!(multiplicative_order(g, p), Some(p - 1));
        let mut s = 0xfeed_1234u64;
        for _ in 0..30
        {
            let e = xorshift(&mut s) % (p - 1);
            let h = pow_mod(g, e, p);
            let x = discrete_log(g, h, p).expect("primitive root: every h solvable");
            assert_eq!(x, e, "recovered exponent must match");
            assert_eq!(pow_mod(g, x, p), h);
        }
    }
}
