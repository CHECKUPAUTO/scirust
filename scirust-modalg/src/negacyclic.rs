//! Negacyclic convolution over `Z_q` — exact polynomial multiplication modulo
//! `x^n + 1` — plus Montgomery reduction, the two building blocks that sit at
//! the core of lattice-based (post-quantum) cryptography.
//!
//! Ring multiplication in `Z_q[x] / (x^n + 1)` is what Kyber, Dilithium and
//! Falcon spend most of their time doing. Over an "negacyclic-friendly" prime
//! (`2n | q − 1`) there is a primitive `2n`-th root of unity `ψ`, and the
//! product can be computed with a length-`n` [`crate::ntt::Ntt`] sandwiched
//! between pre- and post-scaling by powers of `ψ`. [`Montgomery`] provides the
//! fast exact modular reduction those implementations use.
//!
//! Everything here is an **exact reference**: integer-only, deterministic,
//! bit-reproducible — a correctness oracle for optimised code and a teaching
//! aid. It is **not** a hardened or constant-time cryptographic implementation.

use crate::ntt::Ntt;
use crate::numtheory::{inv_mod, mulmod, pow_mod};

/// A negacyclic number-theoretic transform: exact multiplication in
/// `Z_q[x] / (x^n + 1)`.
#[derive(Clone, Debug)]
pub struct NegacyclicNtt {
    ntt: Ntt,
    n: usize,
    q: u64,
    psi_pows: Vec<u64>,     // ψ^i, i = 0 … n-1
    psi_inv_pows: Vec<u64>, // ψ^{-i}, i = 0 … n-1
}

impl NegacyclicNtt {
    /// Build the transform for `Z_q[x]/(x^n + 1)` using primitive root `gen`
    /// of `Z_q`. Panics unless `q` is prime with `gen` a primitive root (both
    /// checked by [`crate::ntt::Ntt::new`]), `n` is a power of two, and
    /// `2n | q − 1` (so a primitive `2n`-th root of unity exists).
    pub fn new(q: u64, gen: u64, n: usize) -> Self {
        assert!(n.is_power_of_two() && n >= 1, "n must be a power of two");
        let ntt = Ntt::new(q, gen);
        assert!(n <= ntt.max_len(), "n exceeds the transform bound for q");
        assert!(
            (q - 1) % (2 * n as u64) == 0,
            "need 2n | q-1 for a 2n-th root"
        );
        // ψ = gen^{(q-1)/2n} is a primitive 2n-th root; ψ² is the n-th root the
        // inner NTT already uses, so the sandwich is consistent.
        let psi = pow_mod(gen, (q - 1) / (2 * n as u64), q);
        let psi_inv = inv_mod(psi, q).expect("ψ is a unit mod prime q");
        let psi_pows = pow_table(psi, n, q);
        let psi_inv_pows = pow_table(psi_inv, n, q);
        NegacyclicNtt {
            ntt,
            n,
            q,
            psi_pows,
            psi_inv_pows,
        }
    }

    /// The Falcon / NewHope prime `q = 12289 = 3·2^12 + 1` (primitive root `11`),
    /// which supports negacyclic transforms up to `n = 2048`.
    pub fn falcon(n: usize) -> Self {
        Self::new(12289, 11, n)
    }

    /// The ring degree `n`.
    pub fn degree(&self) -> usize {
        self.n
    }

    /// The modulus `q`.
    pub fn modulus(&self) -> u64 {
        self.q
    }

    /// Exact product of two length-`n` polynomials in `Z_q[x] / (x^n + 1)`
    /// (their **negacyclic** convolution): pre-scale by `ψ^i`, forward NTT,
    /// pointwise multiply, inverse NTT, post-scale by `ψ^{-i}`.
    pub fn mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        assert_eq!(a.len(), self.n, "a must have length n");
        assert_eq!(b.len(), self.n, "b must have length n");
        let q = self.q;
        let mut fa: Vec<u64> = (0..self.n)
            .map(|i| mulmod(a[i] % q, self.psi_pows[i], q))
            .collect();
        let mut fb: Vec<u64> = (0..self.n)
            .map(|i| mulmod(b[i] % q, self.psi_pows[i], q))
            .collect();
        self.ntt.transform(&mut fa, false);
        self.ntt.transform(&mut fb, false);
        for i in 0..self.n
        {
            fa[i] = mulmod(fa[i], fb[i], q);
        }
        self.ntt.transform(&mut fa, true);
        (0..self.n)
            .map(|i| mulmod(fa[i], self.psi_inv_pows[i], q))
            .collect()
    }
}

fn pow_table(base: u64, n: usize, q: u64) -> Vec<u64> {
    let mut v = Vec::with_capacity(n);
    let mut acc = 1u64 % q;
    for _ in 0..n
    {
        v.push(acc);
        acc = mulmod(acc, base, q);
    }
    v
}

/// Montgomery reduction for an odd modulus `q < 2^32`, with `R = 2^32`.
///
/// Montgomery form represents `a` as `a·R mod q`; multiplication of two such
/// values followed by [`Montgomery::redc`] yields the product in Montgomery
/// form using only a multiply, a shift and a conditional subtract — the exact
/// modular arithmetic that lattice-crypto inner loops rely on. This is a correct
/// reference, not a side-channel-hardened implementation.
#[derive(Copy, Clone, Debug)]
pub struct Montgomery {
    q: u64,
    qinv: u64, // −q^{-1} mod 2^32
    r2: u64,   // R^2 mod q = 2^64 mod q
}

impl Montgomery {
    /// Build the Montgomery context for an odd modulus `q` with `2 ≤ q < 2^32`.
    pub fn new(q: u64) -> Self {
        assert!(
            (2..(1u64 << 32)).contains(&q) && q & 1 == 1,
            "need odd q < 2^32"
        );
        // q^{-1} mod 2^32, then negate mod 2^32.
        let inv = inv_mod(q, 1u64 << 32).expect("odd q is invertible mod 2^32");
        let qinv = (1u64 << 32).wrapping_sub(inv) & 0xFFFF_FFFF;
        let r2 = ((1u128 << 64) % q as u128) as u64;
        Montgomery { q, qinv, r2 }
    }

    /// The modulus `q`.
    pub fn modulus(&self) -> u64 {
        self.q
    }

    /// Montgomery reduction: given `t < q·2^32`, return `t·R^{-1} mod q`.
    pub fn redc(&self, t: u128) -> u64 {
        let m = ((t as u64).wrapping_mul(self.qinv)) & 0xFFFF_FFFF;
        let u = ((t + m as u128 * self.q as u128) >> 32) as u64;
        if u >= self.q { u - self.q } else { u }
    }

    /// Convert `a` (`0 ≤ a < q`) into Montgomery form `a·R mod q`.
    pub fn to_mont(&self, a: u64) -> u64 {
        self.redc(a as u128 * self.r2 as u128)
    }

    /// Convert a Montgomery-form value back to an ordinary residue in `[0, q)`.
    pub fn from_mont(&self, a: u64) -> u64 {
        self.redc(a as u128)
    }

    /// Multiply two Montgomery-form values, returning the product in Montgomery
    /// form.
    pub fn mul(&self, a: u64, b: u64) -> u64 {
        self.redc(a as u128 * b as u128)
    }
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

    /// Independent O(n²) negacyclic convolution: c(x) = a(x)·b(x) mod (x^n + 1),
    /// so terms that wrap past degree n are **subtracted** (x^n = −1).
    fn naive_negacyclic(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
        let n = a.len();
        let mut c = vec![0i128; n];
        for i in 0..n
        {
            for j in 0..n
            {
                let prod = (a[i] as i128 * b[j] as i128) % q as i128;
                if i + j < n
                {
                    c[i + j] = (c[i + j] + prod) % q as i128;
                }
                else
                {
                    c[i + j - n] = (c[i + j - n] - prod).rem_euclid(q as i128);
                }
            }
        }
        c.iter().map(|&v| v.rem_euclid(q as i128) as u64).collect()
    }

    #[test]
    fn falcon_negacyclic_matches_naive() {
        let mut s = 0xfa1c0u64;
        for &log in &[3u32, 5, 7, 9]
        {
            let n = 1usize << log;
            let ring = NegacyclicNtt::falcon(n);
            let q = ring.modulus();
            for _ in 0..20
            {
                let a: Vec<u64> = (0..n).map(|_| xorshift(&mut s) % q).collect();
                let b: Vec<u64> = (0..n).map(|_| xorshift(&mut s) % q).collect();
                assert_eq!(ring.mul(&a, &b), naive_negacyclic(&a, &b, q), "n={n}");
            }
        }
    }

    #[test]
    fn negacyclic_wraparound_negates() {
        // x^{n-1} · x = x^n ≡ −1 (mod x^n + 1) → the constant −1 = q − 1.
        let n = 8usize;
        let ring = NegacyclicNtt::falcon(n);
        let q = ring.modulus();
        let mut a = vec![0u64; n];
        a[n - 1] = 1; // x^{n-1}
        let mut b = vec![0u64; n];
        b[1] = 1; // x
        let mut expect = vec![0u64; n];
        expect[0] = q - 1; // −1
        assert_eq!(ring.mul(&a, &b), expect);
        // multiplying by the ring unit 1 is the identity
        let mut one = vec![0u64; n];
        one[0] = 1;
        let poly: Vec<u64> = (0..n as u64).map(|i| i * 7 + 1).collect();
        assert_eq!(ring.mul(&poly, &one), poly);
    }

    #[test]
    fn custom_prime_ring() {
        // 7681 = 15·2^9 + 1, another lattice-crypto prime; primitive root 17.
        let ring = NegacyclicNtt::new(7681, 17, 256);
        let q = ring.modulus();
        let mut s = 0x7681u64;
        let a: Vec<u64> = (0..256).map(|_| xorshift(&mut s) % q).collect();
        let b: Vec<u64> = (0..256).map(|_| xorshift(&mut s) % q).collect();
        assert_eq!(ring.mul(&a, &b), naive_negacyclic(&a, &b, q));
    }

    #[test]
    fn montgomery_multiplication_is_exact() {
        for &q in &[12289u64, 7681, 3329, 0xFFFF_FFFBu64]
        {
            let m = Montgomery::new(q);
            let mut s = q ^ 0x9e37_79b9;
            for _ in 0..5000
            {
                let a = xorshift(&mut s) % q;
                let b = xorshift(&mut s) % q;
                // from_mont(mul(to_mont(a), to_mont(b))) == a·b mod q
                let got = m.from_mont(m.mul(m.to_mont(a), m.to_mont(b)));
                assert_eq!(got, mulmod(a, b, q), "q={q} a={a} b={b}");
                // round-trip through Montgomery form is the identity
                assert_eq!(m.from_mont(m.to_mont(a)), a);
            }
        }
    }
}
