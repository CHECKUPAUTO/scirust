//! Exact eight-component octonion `Oct<W>` over a coefficient ring `W: Word`.
//!
//! Multiplication is an authoritative 64-term bilinear oracle (the cyclic/Fano
//! sign convention); no Cayley–Dickson recursion and no algebraic shortcut is
//! used, and it is cross-checked against an independent table rebuilt from the
//! Fano triples. Octonions are non-commutative and non-associative but
//! alternative; the modular norm is multiplicative (Degen's eight-square
//! identity). Over `Z/2^k` the algebra is split (it has zero divisors), unlike
//! the real octonions.

use super::table::{IDX, SIGN, table_from_triples};
use crate::ring::Word;

/// An octonion: coefficients `c[0..8]`, with `c[0]` the scalar unit `e0` and
/// `c[1..8]` the imaginary units `e1..e7`.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Oct<W: Word> {
    /// Coefficients in basis-index order.
    pub c: [W; 8],
}

// Algebra types deliberately expose inherent `add`/`sub`/`mul`/`neg` methods
// (the octonion product is non-commutative and non-associative — an operator
// `*` would mislead), so the operator-trait lint is intentionally relaxed here.
#[allow(clippy::should_implement_trait)]
impl<W: Word> Oct<W> {
    /// Additive identity (all-zero).
    pub const fn from_coeffs(c: [W; 8]) -> Self {
        Oct { c }
    }

    /// Zero octonion.
    pub fn zero() -> Self {
        Oct { c: [W::ZERO; 8] }
    }

    /// Multiplicative identity `e0 = 1`.
    pub fn one() -> Self {
        let mut c = [W::ZERO; 8];
        c[0] = W::ONE;
        Oct { c }
    }

    /// Basis unit `e_k` (`k in 0..8`).
    pub fn e(k: usize) -> Self {
        let mut c = [W::ZERO; 8];
        c[k] = W::ONE;
        Oct { c }
    }

    /// Build from eight `u64`s (each masked to the width).
    pub fn from_u64s(v: [u64; 8]) -> Self {
        let mut c = [W::ZERO; 8];
        for i in 0..8
        {
            c[i] = W::from_u64(v[i]);
        }
        Oct { c }
    }

    /// Coefficients as canonical `u64`s.
    pub fn to_u64s(self) -> [u64; 8] {
        let mut v = [0u64; 8];
        for i in 0..8
        {
            v[i] = self.c[i].to_u64();
        }
        v
    }

    /// Component-wise wrapping addition `⊞`.
    pub fn add(self, o: Self) -> Self {
        let mut c = [W::ZERO; 8];
        for i in 0..8
        {
            c[i] = self.c[i].wadd(o.c[i]);
        }
        Oct { c }
    }

    /// Component-wise wrapping subtraction `⊟`.
    pub fn sub(self, o: Self) -> Self {
        let mut c = [W::ZERO; 8];
        for i in 0..8
        {
            c[i] = self.c[i].wsub(o.c[i]);
        }
        Oct { c }
    }

    /// Component-wise wrapping negation.
    pub fn neg(self) -> Self {
        let mut c = [W::ZERO; 8];
        for i in 0..8
        {
            c[i] = self.c[i].wneg();
        }
        Oct { c }
    }

    /// Conjugation `x̄`: negate imaginary coefficients.
    pub fn conj(self) -> Self {
        let mut c = self.c;
        for i in 1..8
        {
            c[i] = c[i].wneg();
        }
        Oct { c }
    }

    /// Bitwise XOR `⊕` of the coefficient words.
    pub fn xor(self, o: Self) -> Self {
        let mut c = [W::ZERO; 8];
        for i in 0..8
        {
            c[i] = self.c[i].xor(o.c[i]);
        }
        Oct { c }
    }

    /// **Authoritative** octonion product `⊗` — the 64-term bilinear oracle.
    /// Fixed loop bounds, public `SIGN`/`IDX` tables, no secret indexing, no
    /// associativity assumption, no Cayley–Dickson recursion.
    pub fn mul(self, o: Self) -> Self {
        let x = self.c;
        let y = o.c;
        let mut z = [W::ZERO; 8];
        for i in 0..8
        {
            for j in 0..8
            {
                let p = x[i].wmul(y[j]);
                let k = IDX[i][j];
                if SIGN[i][j] > 0
                {
                    z[k] = z[k].wadd(p);
                }
                else
                {
                    z[k] = z[k].wsub(p);
                }
            }
        }
        Oct { c: z }
    }

    /// Independent cross-check oracle: identical product computed from a table
    /// rebuilt from the Fano triples at runtime. NOT the authoritative path —
    /// used only to validate [`Oct::mul`].
    pub fn mul_via_triples(self, o: Self) -> Self {
        let (idx, sign) = table_from_triples();
        let x = self.c;
        let y = o.c;
        let mut z = [W::ZERO; 8];
        for i in 0..8
        {
            for j in 0..8
            {
                let p = x[i].wmul(y[j]);
                let k = idx[i][j];
                if sign[i][j] > 0
                {
                    z[k] = z[k].wadd(p);
                }
                else
                {
                    z[k] = z[k].wsub(p);
                }
            }
        }
        Oct { c: z }
    }

    /// Modular norm `N(x) = Σ_i x_i²  (mod 2^k)`.
    pub fn norm(self) -> W {
        let mut acc = W::ZERO;
        for i in 0..8
        {
            acc = acc.wadd(self.c[i].wmul(self.c[i]));
        }
        acc
    }

    /// Rotate each coefficient (lane) `j` left within its `BITS`-bit word by
    /// `amounts[j]` bits. A general per-lane bit rotation.
    pub fn rotate_lanes(self, amounts: &[u32; 8]) -> Self {
        let mut c = [W::ZERO; 8];
        for j in 0..8
        {
            c[j] = self.c[j].rotl(amounts[j]);
        }
        Oct { c }
    }

    /// Permute the coefficient slots: `(permute_slots(x))_i = x_{perm[i]}`.
    /// `perm` must be a permutation of `0..8`.
    pub fn permute_slots(self, perm: &[usize; 8]) -> Self {
        let mut c = [W::ZERO; 8];
        for i in 0..8
        {
            c[i] = self.c[perm[i]];
        }
        Oct { c }
    }
}

impl Oct<crate::ring::W64> {
    /// Canonical little-endian serialization: 64 bytes, `c[0..8]` each as a
    /// little-endian `u64`.
    pub fn to_le_bytes(self) -> [u8; 64] {
        let mut out = [0u8; 64];
        for i in 0..8
        {
            out[i * 8..i * 8 + 8].copy_from_slice(&self.c[i].to_u64().to_le_bytes());
        }
        out
    }

    /// Parse 64 little-endian bytes into an octonion. Every 64-byte
    /// string is valid (no rejection over `Z/2^64`).
    pub fn from_le_bytes(b: &[u8; 64]) -> Self {
        let mut v = [0u64; 8];
        for i in 0..8
        {
            let mut w = [0u8; 8];
            w.copy_from_slice(&b[i * 8..i * 8 + 8]);
            v[i] = u64::from_le_bytes(w);
        }
        Oct::from_u64s(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::{W8, W64};

    #[test]
    fn identity_and_zero() {
        let a = Oct::<W64>::from_u64s([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(a.mul(Oct::one()), a);
        assert_eq!(Oct::<W64>::one().mul(a), a);
        assert_eq!(a.add(Oct::zero()), a);
    }

    #[test]
    fn oracle_matches_triple_oracle_random() {
        // deterministic pseudo-random cross-check of the two oracles
        let mut s: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..2000
        {
            let x = Oct::<W64>::from_u64s(std::array::from_fn(|_| next()));
            let y = Oct::<W64>::from_u64s(std::array::from_fn(|_| next()));
            assert_eq!(x.mul(y), x.mul_via_triples(y));
        }
    }

    #[test]
    fn all_64_basis_products() {
        // e_i * e_j must match the triple-derived table for every basis pair.
        let (idx, sign) = super::table_from_triples();
        for i in 0..8
        {
            for j in 0..8
            {
                let prod = Oct::<W8>::e(i).mul(Oct::<W8>::e(j));
                let mut expect = [0u64; 8];
                expect[idx[i][j]] = if sign[i][j] > 0
                {
                    1
                }
                else
                {
                    W8::from_u64(0).wsub(W8::ONE).to_u64()
                };
                assert_eq!(prod.to_u64s(), expect, "e{i}*e{j} wrong");
            }
        }
    }

    #[test]
    fn le_roundtrip() {
        let a = Oct::<W64>::from_u64s([
            0x0011223344556677,
            0x8899aabbccddeeff,
            1,
            2,
            u64::MAX,
            0,
            0xdead_beef,
            0xcafe_babe,
        ]);
        assert_eq!(Oct::from_le_bytes(&a.to_le_bytes()), a);
    }
}
