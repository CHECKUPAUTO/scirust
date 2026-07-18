//! The **number-theoretic transform** (NTT): an exact FFT over the finite field
//! `Z/p`, and the `O(n log n)` exact integer convolution it enables.
//!
//! Over `Z/p` for an "NTT-friendly" prime (one with `2^k \| p − 1`) there is a
//! primitive `2^k`-th root of unity, so the Cooley–Tukey butterfly runs with
//! exact modular arithmetic — no floating point, no rounding, bit-exact on every
//! platform. This is the workhorse behind fast exact polynomial / big-integer
//! multiplication and lattice cryptography.
//!
//! The transform length must be a power of two dividing `p − 1`
//! (see [`Ntt::max_len`]). The bundled [`Ntt::new_default`] uses
//! `p = 998244353 = 119·2^23 + 1` with generator `3`, supporting lengths up to
//! `2^23`.
//!
//! This module composes [`crate::numtheory`]: the prime is checked with
//! `is_prime`, and the generator is verified to be a true primitive root via
//! `factor(p − 1)`.

use crate::numtheory::{factor, inv_mod, is_prime, mulmod, pow_mod};

/// An exact number-theoretic transform over `Z/prime`.
#[derive(Clone, Debug)]
pub struct Ntt {
    prime: u64,
    gen: u64,
}

impl Ntt {
    /// Construct an NTT over `Z/prime` using primitive root `gen`.
    ///
    /// Panics unless `prime` is an odd prime (checked with
    /// [`crate::numtheory::is_prime`]) and `gen` is a genuine primitive root
    /// modulo `prime` (checked exactly via [`crate::numtheory::factor`]).
    pub fn new(prime: u64, gen: u64) -> Self {
        assert!(prime > 2 && is_prime(prime), "modulus must be an odd prime");
        assert!(
            is_primitive_root(prime, gen),
            "gen is not a primitive root mod prime"
        );
        Ntt { prime, gen }
    }

    /// The standard NTT prime `998244353 = 119·2^23 + 1` with generator `3`,
    /// supporting transform lengths up to `2^23`.
    pub fn new_default() -> Self {
        Ntt {
            prime: 998_244_353,
            gen: 3,
        }
    }

    /// The field modulus `p`.
    pub fn prime(&self) -> u64 {
        self.prime
    }

    /// The largest power of two dividing `p − 1` — the maximum transform length.
    pub fn max_len(&self) -> usize {
        1usize << (self.prime - 1).trailing_zeros()
    }

    /// In-place forward (or `inverse`) NTT of a slice whose length is a power of
    /// two dividing `p − 1`. Every entry must already be a residue in `[0, p)`.
    pub fn transform(&self, a: &mut [u64], inverse: bool) {
        let n = a.len();
        assert!(n.is_power_of_two(), "length must be a power of two");
        assert!(n <= self.max_len(), "length exceeds max_len for this prime");
        let p = self.prime;

        // bit-reversal permutation
        let mut j = 0usize;
        for i in 1..n
        {
            let mut bit = n >> 1;
            while j & bit != 0
            {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j
            {
                a.swap(i, j);
            }
        }

        let mut len = 2usize;
        while len <= n
        {
            // primitive len-th root of unity (or its inverse for the INTT)
            let exp = (p - 1) / len as u64;
            let wlen = if inverse
            {
                pow_mod(self.gen, p - 1 - exp, p)
            }
            else
            {
                pow_mod(self.gen, exp, p)
            };
            let half = len / 2;
            let mut i = 0usize;
            while i < n
            {
                let mut w = 1u64;
                for k in 0..half
                {
                    let u = a[i + k];
                    let v = mulmod(a[i + k + half], w, p);
                    a[i + k] = if u + v >= p { u + v - p } else { u + v };
                    a[i + k + half] = if u >= v { u - v } else { u + p - v };
                    w = mulmod(w, wlen, p);
                }
                i += len;
            }
            len <<= 1;
        }

        if inverse
        {
            let n_inv = inv_mod(n as u64, p).expect("n is invertible mod p");
            for x in a.iter_mut()
            {
                *x = mulmod(*x, n_inv, p);
            }
        }
    }

    /// The linear convolution of two integer sequences (equivalently, the
    /// product of two polynomials given by their coefficients) computed exactly
    /// in `O(n log n)` via the NTT.
    ///
    /// Inputs are reduced mod `p`; the result is exact whenever every true
    /// convolution sum is `< p` (as it always is for, e.g., non-negative inputs
    /// whose pairwise products summed over the overlap stay below `p`).
    /// The output length is `a.len() + b.len() − 1`.
    pub fn convolve(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        if a.is_empty() || b.is_empty()
        {
            return Vec::new();
        }
        let result_len = a.len() + b.len() - 1;
        let mut n = 1usize;
        while n < result_len
        {
            n <<= 1;
        }
        assert!(n <= self.max_len(), "inputs too long for this prime");
        let p = self.prime;
        let mut fa = vec![0u64; n];
        let mut fb = vec![0u64; n];
        for (i, &x) in a.iter().enumerate()
        {
            fa[i] = x % p;
        }
        for (i, &x) in b.iter().enumerate()
        {
            fb[i] = x % p;
        }
        self.transform(&mut fa, false);
        self.transform(&mut fb, false);
        for i in 0..n
        {
            fa[i] = mulmod(fa[i], fb[i], p);
        }
        self.transform(&mut fa, true);
        fa.truncate(result_len);
        fa
    }
}

/// Is `g` a primitive root modulo the prime `p`? Exact, via factoring `p − 1`.
fn is_primitive_root(p: u64, g: u64) -> bool {
    if g == 0 || g % p == 0
    {
        return false;
    }
    factor(p - 1)
        .into_iter()
        .all(|(q, _)| pow_mod(g, (p - 1) / q, p) != 1)
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

    /// Independent O(n²) convolution reference over Z/p.
    fn naive_convolve(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
        if a.is_empty() || b.is_empty()
        {
            return Vec::new();
        }
        let mut out = vec![0u64; a.len() + b.len() - 1];
        for (i, &x) in a.iter().enumerate()
        {
            for (j, &y) in b.iter().enumerate()
            {
                out[i + j] = (out[i + j] + mulmod(x % p, y % p, p)) % p;
            }
        }
        out
    }

    #[test]
    fn default_prime_is_ntt_friendly() {
        let ntt = Ntt::new_default();
        assert_eq!(ntt.prime(), 998_244_353);
        // 998244353 - 1 = 119 · 2^23
        assert_eq!(ntt.max_len(), 1 << 23);
        assert!(is_primitive_root(998_244_353, 3));
    }

    #[test]
    fn forward_then_inverse_is_identity() {
        let ntt = Ntt::new_default();
        let mut s = 0x1234_5678u64;
        for log in 0..=10u32
        {
            let n = 1usize << log;
            let orig: Vec<u64> = (0..n).map(|_| xorshift(&mut s) % ntt.prime()).collect();
            let mut buf = orig.clone();
            ntt.transform(&mut buf, false);
            ntt.transform(&mut buf, true);
            assert_eq!(buf, orig, "roundtrip failed at n={n}");
        }
    }

    #[test]
    fn convolution_matches_naive() {
        let ntt = Ntt::new_default();
        let mut s = 0x9e37_79b9u64;
        for _ in 0..100
        {
            let la = 1 + (xorshift(&mut s) % 64) as usize;
            let lb = 1 + (xorshift(&mut s) % 64) as usize;
            let a: Vec<u64> = (0..la).map(|_| xorshift(&mut s) % ntt.prime()).collect();
            let b: Vec<u64> = (0..lb).map(|_| xorshift(&mut s) % ntt.prime()).collect();
            assert_eq!(ntt.convolve(&a, &b), naive_convolve(&a, &b, ntt.prime()));
        }
    }

    #[test]
    fn exact_small_integer_convolution() {
        let ntt = Ntt::new_default();
        // [1,2,3] * [4,5,6] = [4, 13, 28, 27, 18] (exact integer convolution)
        assert_eq!(
            ntt.convolve(&[1, 2, 3], &[4, 5, 6]),
            vec![4, 13, 28, 27, 18]
        );
        // polynomial (1 + x)^2 = 1 + 2x + x^2
        assert_eq!(ntt.convolve(&[1, 1], &[1, 1]), vec![1, 2, 1]);
        // multiply by 1 is the identity
        assert_eq!(ntt.convolve(&[7, 8, 9], &[1]), vec![7, 8, 9]);
    }

    #[test]
    fn large_scale_convolution() {
        let ntt = Ntt::new_default();
        let mut s = 0xdead_beefu64;
        // small coefficients keep every sum well below p, so the result is the
        // exact integer convolution
        let a: Vec<u64> = (0..1000).map(|_| xorshift(&mut s) % 1000).collect();
        let b: Vec<u64> = (0..1000).map(|_| xorshift(&mut s) % 1000).collect();
        assert_eq!(ntt.convolve(&a, &b), naive_convolve(&a, &b, ntt.prime()));
    }

    #[test]
    fn custom_prime_works() {
        // 2013265921 = 15·2^27 + 1, a primitive root is 31
        let ntt = Ntt::new(2_013_265_921, 31);
        assert_eq!(ntt.max_len(), 1 << 27);
        let a = [5u64, 0, 3, 9];
        let b = [2u64, 1, 0, 4];
        assert_eq!(ntt.convolve(&a, &b), naive_convolve(&a, &b, ntt.prime()));
    }

    #[test]
    #[should_panic(expected = "not a primitive root")]
    fn rejects_non_primitive_root() {
        // 4 = 2^2 is a quadratic residue, hence never a primitive root mod an
        // odd prime.
        let _ = Ntt::new(998_244_353, 4);
    }
}
