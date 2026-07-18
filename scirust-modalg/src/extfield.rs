//! Exact **finite extension fields `GF(p^k)`** built as `GF(p)[x] / (m)` for a
//! monic irreducible modulus `m` of degree `k`.
//!
//! Elements are [`crate::poly::Poly`] residues of degree `< k`, and every field
//! operation is exact modular polynomial arithmetic — addition, subtraction,
//! multiplication, powering, and inversion (through the extended Euclidean
//! algorithm modulo `m`). The modulus is verified irreducible on construction
//! (that is exactly the condition for the quotient to be a field), and a modulus
//! of a requested degree can be found automatically with [`ExtField::build`].
//!
//! This generalises [`crate::gf2`], which packs the special case `p = 2` into
//! machine words: building `GF(2^8)` here with the AES/Rijndael modulus
//! `x^8 + x^4 + x^3 + x + 1` reproduces `gf2::Gf2Field::rijndael8` exactly (see
//! the tests). It composes [`crate::poly`] for the underlying arithmetic and
//! irreducibility test.

use crate::poly::Poly;

/// A finite field `GF(p^k) = GF(p)[x] / (m)` with `m` monic irreducible of
/// degree `k`. Field elements are `Poly` residues of degree `< k` over `GF(p)`.
#[derive(Clone, Debug)]
pub struct ExtField {
    p: u64,
    k: usize,
    modulus: Poly,
}

impl ExtField {
    /// Build `GF(p^k)` from a monic irreducible modulus polynomial.
    ///
    /// Panics unless `modulus` is monic, of degree `≥ 1`, and irreducible over
    /// its prime field (checked with [`crate::poly::Poly::is_irreducible`]).
    pub fn new(modulus: Poly) -> Self {
        let k = modulus.degree().expect("modulus must be nonzero");
        assert!(k >= 1, "modulus must have degree ≥ 1");
        assert!(modulus.is_monic(), "modulus must be monic");
        assert!(modulus.is_irreducible(), "modulus must be irreducible");
        ExtField {
            p: modulus.modulus(),
            k,
            modulus,
        }
    }

    /// Build `GF(p^k)` by searching for the lexicographically-first monic
    /// irreducible modulus of degree `k` over `GF(p)`. Deterministic; intended
    /// for small fields (the search is `O(p^k)` candidates).
    pub fn build(p: u64, k: usize) -> Self {
        assert!(k >= 1, "degree must be ≥ 1");
        // Enumerate the p^k choices for the low k coefficients (leading = 1) in
        // increasing order and return the first irreducible.
        let total = (p as u128)
            .checked_pow(k as u32)
            .expect("field too large to search");
        for code in 0..total
        {
            let mut coeffs = vec![0u64; k + 1];
            let mut c = code;
            for slot in coeffs.iter_mut().take(k)
            {
                *slot = (c % p as u128) as u64;
                c /= p as u128;
            }
            coeffs[k] = 1;
            let cand = Poly::from_coeffs(p, &coeffs);
            if cand.is_irreducible()
            {
                return ExtField::new(cand);
            }
        }
        unreachable!("an irreducible polynomial of every degree exists over GF(p)")
    }

    /// The prime `p` (characteristic).
    pub fn prime(&self) -> u64 {
        self.p
    }

    /// The extension degree `k`.
    pub fn degree(&self) -> usize {
        self.k
    }

    /// The field size `p^k` (number of elements).
    pub fn order(&self) -> u128 {
        (self.p as u128).pow(self.k as u32)
    }

    /// The modulus polynomial `m`.
    pub fn modulus(&self) -> &Poly {
        &self.modulus
    }

    /// Reduce an arbitrary polynomial into the canonical residue of degree `< k`.
    pub fn reduce(&self, a: &Poly) -> Poly {
        a.rem(&self.modulus)
    }

    /// The additive identity `0`.
    pub fn zero(&self) -> Poly {
        Poly::zero(self.p)
    }

    /// The multiplicative identity `1`.
    pub fn one(&self) -> Poly {
        Poly::one(self.p)
    }

    /// The class of the indeterminate `x` — a root of the modulus, and a
    /// natural multiplicative generator candidate.
    pub fn generator(&self) -> Poly {
        self.reduce(&Poly::x(self.p))
    }

    /// The element with the given coefficients (low-degree-first), reduced into
    /// the field.
    pub fn element(&self, coeffs: &[u64]) -> Poly {
        self.reduce(&Poly::from_coeffs(self.p, coeffs))
    }

    /// Field addition.
    pub fn add(&self, a: &Poly, b: &Poly) -> Poly {
        self.reduce(&a.add(b))
    }

    /// Field subtraction.
    pub fn sub(&self, a: &Poly, b: &Poly) -> Poly {
        self.reduce(&a.sub(b))
    }

    /// Additive inverse.
    pub fn neg(&self, a: &Poly) -> Poly {
        self.reduce(&a.neg())
    }

    /// Field multiplication (polynomial product reduced modulo `m`).
    pub fn mul(&self, a: &Poly, b: &Poly) -> Poly {
        self.reduce(&a.mul(b))
    }

    /// Exponentiation `a^e` by square-and-multiply in the field.
    pub fn pow(&self, a: &Poly, e: u64) -> Poly {
        let mut result = self.one();
        let mut base = self.reduce(a);
        let mut e = e;
        while e > 0
        {
            if e & 1 == 1
            {
                result = self.mul(&result, &base);
            }
            base = self.mul(&base, &base);
            e >>= 1;
        }
        result
    }

    /// Multiplicative inverse, or `None` for `0`.
    ///
    /// Uses the extended Euclidean algorithm: since `m` is irreducible and
    /// `a ≠ 0` has degree `< k`, `gcd(a, m) = 1`, and the Bézout coefficient of
    /// `a` is `a^{-1}` modulo `m`.
    pub fn inv(&self, a: &Poly) -> Option<Poly> {
        let a = self.reduce(a);
        if a.is_zero()
        {
            return None;
        }
        let (g, s, _t) = a.egcd(&self.modulus);
        debug_assert_eq!(
            g.degree(),
            Some(0),
            "gcd with an irreducible must be a unit"
        );
        Some(self.reduce(&s))
    }

    /// The Frobenius endomorphism `a ↦ a^p` (an automorphism of `GF(p^k)`
    /// fixing the prime field `GF(p)`).
    pub fn frobenius(&self, a: &Poly) -> Poly {
        self.pow(a, self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf2::Gf2Field;

    fn xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// A random field element of `GF(p^k)`.
    fn rand_elem(f: &ExtField, s: &mut u64) -> Poly {
        let coeffs: Vec<u64> = (0..f.degree()).map(|_| xorshift(s) % f.prime()).collect();
        f.element(&coeffs)
    }

    #[test]
    fn build_finds_irreducible_modulus_of_the_right_degree() {
        for &(p, k) in &[
            (2u64, 1usize),
            (2, 3),
            (2, 8),
            (3, 2),
            (3, 4),
            (5, 3),
            (7, 2),
        ]
        {
            let f = ExtField::build(p, k);
            assert_eq!(f.degree(), k);
            assert_eq!(f.prime(), p);
            assert_eq!(f.order(), (p as u128).pow(k as u32));
            assert!(f.modulus().is_irreducible());
            assert!(f.modulus().is_monic());
        }
    }

    #[test]
    fn field_axioms_hold_on_random_elements() {
        let mut s = 0x1234_abcdu64;
        for &(p, k) in &[(2u64, 5usize), (3, 3), (5, 2), (7, 3), (11, 2)]
        {
            let f = ExtField::build(p, k);
            for _ in 0..60
            {
                let a = rand_elem(&f, &mut s);
                let b = rand_elem(&f, &mut s);
                let c = rand_elem(&f, &mut s);
                // commutativity
                assert_eq!(f.add(&a, &b), f.add(&b, &a));
                assert_eq!(f.mul(&a, &b), f.mul(&b, &a));
                // associativity
                assert_eq!(f.mul(&f.mul(&a, &b), &c), f.mul(&a, &f.mul(&b, &c)));
                // distributivity
                assert_eq!(
                    f.mul(&a, &f.add(&b, &c)),
                    f.add(&f.mul(&a, &b), &f.mul(&a, &c))
                );
                // additive inverse
                assert!(f.add(&a, &f.neg(&a)).is_zero());
                // multiplicative inverse
                if !a.is_zero()
                {
                    assert_eq!(f.mul(&a, &f.inv(&a).unwrap()), f.one());
                }
            }
            // only zero has no inverse
            assert!(f.inv(&f.zero()).is_none());
        }
    }

    #[test]
    fn multiplicative_group_and_frobenius() {
        let mut s = 0x9e37_beefu64;
        for &(p, k) in &[(2u64, 4usize), (3, 3), (5, 2), (7, 2)]
        {
            let f = ExtField::build(p, k);
            let order = (f.order() - 1) as u64; // |GF(p^k)*|
            let size = f.order() as u64; // p^k
            for _ in 0..40
            {
                let a = rand_elem(&f, &mut s);
                if !a.is_zero()
                {
                    // Lagrange: a^(p^k − 1) = 1 for every nonzero a.
                    assert_eq!(f.pow(&a, order), f.one());
                }
                // Frobenius^k = identity: a^(p^k) = a for every a.
                assert_eq!(f.pow(&a, size), a);
                // Frobenius is additive and multiplicative.
                let b = rand_elem(&f, &mut s);
                assert_eq!(
                    f.frobenius(&f.add(&a, &b)),
                    f.add(&f.frobenius(&a), &f.frobenius(&b))
                );
                assert_eq!(
                    f.frobenius(&f.mul(&a, &b)),
                    f.mul(&f.frobenius(&a), &f.frobenius(&b))
                );
            }
            // Frobenius fixes the prime field GF(p).
            for c in 0..p
            {
                let e = f.element(&[c]);
                assert_eq!(f.frobenius(&e), e);
            }
        }
    }

    // Map a byte to/from a GF(2^8) element (bit i ↔ coefficient of x^i).
    fn byte_to_poly(f: &ExtField, b: u8) -> Poly {
        let coeffs: Vec<u64> = (0..8).map(|i| ((b >> i) & 1) as u64).collect();
        f.element(&coeffs)
    }
    fn poly_to_byte(e: &Poly) -> u8 {
        let mut b = 0u8;
        for i in 0..8
        {
            if e.coeff(i) == 1
            {
                b |= 1 << i;
            }
        }
        b
    }

    #[test]
    fn matches_the_aes_field_gf256() {
        // GF(2^8) via the AES/Rijndael modulus x^8+x^4+x^3+x+1 must agree with
        // the packed `gf2::Gf2Field::rijndael8` on every product and inverse.
        let m = Poly::from_coeffs(2, &[1, 1, 0, 1, 1, 0, 0, 0, 1]);
        let f = ExtField::new(m);
        let g = Gf2Field::rijndael8();

        // FIPS-197's worked example {57}·{83} = {c1}.
        let prod = f.mul(&byte_to_poly(&f, 0x57), &byte_to_poly(&f, 0x83));
        assert_eq!(poly_to_byte(&prod), 0xc1);

        let mut s = 0x0055_aa11u64;
        for _ in 0..300
        {
            let a = (xorshift(&mut s) & 0xff) as u8;
            let b = (xorshift(&mut s) & 0xff) as u8;
            let prod = f.mul(&byte_to_poly(&f, a), &byte_to_poly(&f, b));
            assert_eq!(poly_to_byte(&prod), g.mul(a as u64, b as u64) as u8);
            if a != 0
            {
                let inv = f.inv(&byte_to_poly(&f, a)).unwrap();
                assert_eq!(poly_to_byte(&inv), g.inv(a as u64).unwrap() as u8);
            }
        }
    }

    #[test]
    fn degree_one_is_the_prime_field() {
        // GF(p^1) = GF(p): arithmetic is just modular integer arithmetic.
        let f = ExtField::build(7, 1);
        assert_eq!(f.order(), 7);
        assert_eq!(poly_val(&f.mul(&f.element(&[3]), &f.element(&[5]))), 1); // 15 ≡ 1
        assert_eq!(poly_val(&f.inv(&f.element(&[3])).unwrap()), 5); // 3·5 = 15 ≡ 1
    }

    fn poly_val(e: &Poly) -> u64 {
        e.coeff(0)
    }

    #[test]
    #[should_panic(expected = "modulus must be irreducible")]
    fn rejects_reducible_modulus() {
        // x^2 + 1 = (x+1)^2 over GF(2) is reducible, so the quotient is not a
        // field.
        let _ = ExtField::new(Poly::from_coeffs(2, &[1, 0, 1]));
    }
}
