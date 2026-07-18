//! Exact univariate polynomials over a **prime field `GF(p)`** — the polynomial
//! ring `GF(p)[x]` — with everything its Euclidean-domain structure gives you:
//! long division, (extended) GCD, modular exponentiation `f^e mod g`, Lagrange
//! interpolation, the formal derivative, and an exact **irreducibility test**
//! (Rabin's algorithm).
//!
//! Coefficients are reduced residues in `[0, p)`, stored low-degree-first with
//! no trailing zeros, so the representation is canonical and derived `==` is
//! exactly mathematical equality. Every operation is exact modular integer
//! arithmetic — no floating point, deterministic and bit-identical on every
//! platform.
//!
//! This is the field-generic companion to [`crate::gf2`] (which packs the
//! special case `p = 2` into machine words) and the ordinary-polynomial
//! companion to [`crate::ntt`] (cyclic convolution over `Z/p`). It composes
//! [`crate::numtheory`] for the modular inverse, the primality check, and the
//! factorization of the degree used by the irreducibility test.

use crate::numtheory::{factor, inv_mod, is_prime, mulmod};

/// A univariate polynomial over the prime field `GF(p)`.
///
/// Coefficients are kept reduced mod `p`, low-degree-first, with no trailing
/// zeros — the zero polynomial is the empty coefficient vector. This canonical
/// form makes the derived [`PartialEq`] exactly polynomial equality (two
/// polynomials compare equal iff they share the modulus and all coefficients).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly {
    p: u64,
    coeffs: Vec<u64>,
}

/// Strip trailing zero coefficients so the representation is canonical.
fn trim(mut coeffs: Vec<u64>) -> Vec<u64> {
    while coeffs.last() == Some(&0)
    {
        coeffs.pop();
    }
    coeffs
}

impl Poly {
    /// Build from raw coefficients, reducing mod `p` and trimming. Does **not**
    /// re-check primality — used internally once the modulus is known good.
    fn raw(p: u64, coeffs: Vec<u64>) -> Self {
        let reduced = coeffs.into_iter().map(|c| c % p).collect();
        Poly {
            p,
            coeffs: trim(reduced),
        }
    }

    /// The polynomial with the given coefficients (low-degree-first) over
    /// `GF(p)`. Coefficients are reduced mod `p`.
    ///
    /// Panics unless `p` is prime (checked with [`crate::numtheory::is_prime`]).
    pub fn from_coeffs(p: u64, coeffs: &[u64]) -> Self {
        assert!(p >= 2 && is_prime(p), "modulus must be prime");
        Poly::raw(p, coeffs.to_vec())
    }

    /// The zero polynomial over `GF(p)`.
    pub fn zero(p: u64) -> Self {
        assert!(p >= 2 && is_prime(p), "modulus must be prime");
        Poly {
            p,
            coeffs: Vec::new(),
        }
    }

    /// The constant polynomial `c` over `GF(p)`.
    pub fn constant(p: u64, c: u64) -> Self {
        assert!(p >= 2 && is_prime(p), "modulus must be prime");
        Poly::raw(p, vec![c])
    }

    /// The unit polynomial `1` over `GF(p)`.
    pub fn one(p: u64) -> Self {
        Poly::constant(p, 1)
    }

    /// The polynomial `x` (the indeterminate) over `GF(p)`.
    pub fn x(p: u64) -> Self {
        assert!(p >= 2 && is_prime(p), "modulus must be prime");
        Poly::raw(p, vec![0, 1])
    }

    /// The monomial `coeff · x^deg` over `GF(p)`.
    pub fn monomial(p: u64, coeff: u64, deg: usize) -> Self {
        assert!(p >= 2 && is_prime(p), "modulus must be prime");
        let mut c = vec![0u64; deg + 1];
        c[deg] = coeff % p;
        Poly::raw(p, c)
    }

    /// The field modulus `p`.
    pub fn modulus(&self) -> u64 {
        self.p
    }

    /// The coefficients (low-degree-first, reduced, no trailing zeros).
    pub fn coeffs(&self) -> &[u64] {
        &self.coeffs
    }

    /// Is this the zero polynomial?
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// The degree, or `None` for the zero polynomial.
    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.is_empty()
        {
            None
        }
        else
        {
            Some(self.coeffs.len() - 1)
        }
    }

    /// The coefficient of `x^i` (`0` beyond the stored degree).
    pub fn coeff(&self, i: usize) -> u64 {
        self.coeffs.get(i).copied().unwrap_or(0)
    }

    /// The leading coefficient (`0` for the zero polynomial).
    pub fn leading_coeff(&self) -> u64 {
        self.coeffs.last().copied().unwrap_or(0)
    }

    /// Is the leading coefficient `1`?
    pub fn is_monic(&self) -> bool {
        self.leading_coeff() == 1
    }

    fn same_modulus(&self, other: &Self) {
        assert_eq!(self.p, other.p, "polynomials over different fields");
    }

    /// Sum of two polynomials over the same field.
    pub fn add(&self, other: &Self) -> Self {
        self.same_modulus(other);
        let p = self.p;
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut out = vec![0u64; n];
        for (i, &c) in self.coeffs.iter().enumerate()
        {
            out[i] = c;
        }
        for (i, &c) in other.coeffs.iter().enumerate()
        {
            out[i] = (out[i] + c) % p;
        }
        Poly::raw(p, out)
    }

    /// Difference of two polynomials over the same field.
    pub fn sub(&self, other: &Self) -> Self {
        self.same_modulus(other);
        let p = self.p;
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut out = vec![0u64; n];
        for (i, &c) in self.coeffs.iter().enumerate()
        {
            out[i] = c;
        }
        for (i, &c) in other.coeffs.iter().enumerate()
        {
            out[i] = (out[i] + p - c) % p;
        }
        Poly::raw(p, out)
    }

    /// Additive inverse `−self`.
    pub fn neg(&self) -> Self {
        let p = self.p;
        Poly::raw(p, self.coeffs.iter().map(|&c| (p - c) % p).collect())
    }

    /// Scalar multiple `c · self` (`c` reduced mod `p`).
    pub fn scale(&self, c: u64) -> Self {
        let p = self.p;
        let c = c % p;
        Poly::raw(p, self.coeffs.iter().map(|&a| mulmod(a, c, p)).collect())
    }

    /// Product of two polynomials (schoolbook, exact mod `p`).
    pub fn mul(&self, other: &Self) -> Self {
        self.same_modulus(other);
        let p = self.p;
        if self.is_zero() || other.is_zero()
        {
            return Poly::zero(p);
        }
        let mut out = vec![0u64; self.coeffs.len() + other.coeffs.len() - 1];
        for (i, &a) in self.coeffs.iter().enumerate()
        {
            if a == 0
            {
                continue;
            }
            for (j, &b) in other.coeffs.iter().enumerate()
            {
                out[i + j] = (out[i + j] + mulmod(a, b, p)) % p;
            }
        }
        Poly::raw(p, out)
    }

    /// Evaluate at `x` (Horner's method), returning a residue in `[0, p)`.
    pub fn eval(&self, x: u64) -> u64 {
        let p = self.p;
        let x = x % p;
        let mut acc = 0u64;
        for &c in self.coeffs.iter().rev()
        {
            acc = (mulmod(acc, x, p) + c) % p;
        }
        acc
    }

    /// The formal derivative `d/dx`.
    pub fn derivative(&self) -> Self {
        let p = self.p;
        if self.coeffs.len() <= 1
        {
            return Poly::zero(p);
        }
        let mut out = vec![0u64; self.coeffs.len() - 1];
        for i in 1..self.coeffs.len()
        {
            out[i - 1] = mulmod(self.coeffs[i], (i as u64) % p, p);
        }
        Poly::raw(p, out)
    }

    /// The monic associate `self / leading_coeff` (panics on the zero
    /// polynomial).
    pub fn make_monic(&self) -> Self {
        assert!(!self.is_zero(), "zero polynomial has no monic associate");
        let inv = inv_mod(self.leading_coeff(), self.p).expect("prime field: units invert");
        self.scale(inv)
    }

    /// Euclidean division: returns `(quotient, remainder)` with
    /// `self == quotient · divisor + remainder` and `deg(remainder) <
    /// deg(divisor)`. Panics if `divisor` is zero.
    pub fn divmod(&self, divisor: &Self) -> (Self, Self) {
        self.same_modulus(divisor);
        assert!(!divisor.is_zero(), "division by the zero polynomial");
        let p = self.p;
        let (n, m) = (self.coeffs.len(), divisor.coeffs.len());
        if n < m
        {
            return (Poly::zero(p), self.clone());
        }
        let inv = inv_mod(divisor.leading_coeff(), p).expect("prime field: units invert");
        let mut rem = self.coeffs.clone();
        let mut quot = vec![0u64; n - m + 1];
        // Synthetic division from the highest quotient degree downward.
        for k in (0..=n - m).rev()
        {
            let coeff = mulmod(rem[k + m - 1], inv, p);
            quot[k] = coeff;
            if coeff != 0
            {
                for j in 0..m
                {
                    let s = mulmod(coeff, divisor.coeffs[j], p);
                    rem[k + j] = (rem[k + j] + p - s) % p;
                }
            }
        }
        rem.truncate(m - 1);
        (Poly::raw(p, quot), Poly::raw(p, rem))
    }

    /// The quotient `self / divisor` (truncated, over the field).
    pub fn div(&self, divisor: &Self) -> Self {
        self.divmod(divisor).0
    }

    /// The remainder `self mod divisor`.
    pub fn rem(&self, divisor: &Self) -> Self {
        self.divmod(divisor).1
    }

    /// The **monic** greatest common divisor (`0` if both inputs are zero).
    pub fn gcd(&self, other: &Self) -> Self {
        self.same_modulus(other);
        let mut a = self.clone();
        let mut b = other.clone();
        while !b.is_zero()
        {
            let r = a.rem(&b);
            a = b;
            b = r;
        }
        if a.is_zero() { a } else { a.make_monic() }
    }

    /// Extended Euclidean algorithm: returns `(g, s, t)` with
    /// `s · self + t · other == g`, where `g` is the **monic** GCD.
    pub fn egcd(&self, other: &Self) -> (Self, Self, Self) {
        self.same_modulus(other);
        let p = self.p;
        let (mut old_r, mut r) = (self.clone(), other.clone());
        let (mut old_s, mut s) = (Poly::one(p), Poly::zero(p));
        let (mut old_t, mut t) = (Poly::zero(p), Poly::one(p));
        while !r.is_zero()
        {
            let q = old_r.div(&r);
            let nr = old_r.sub(&q.mul(&r));
            old_r = r;
            r = nr;
            let ns = old_s.sub(&q.mul(&s));
            old_s = s;
            s = ns;
            let nt = old_t.sub(&q.mul(&t));
            old_t = t;
            t = nt;
        }
        if old_r.is_zero()
        {
            return (old_r, old_s, old_t);
        }
        // Normalize so g is monic; the same field scalar keeps s·a + t·b = g.
        let inv = inv_mod(old_r.leading_coeff(), p).expect("prime field: units invert");
        (old_r.scale(inv), old_s.scale(inv), old_t.scale(inv))
    }

    /// Modular exponentiation `self^exp mod modulus` (square-and-multiply).
    /// Panics if `modulus` is zero.
    pub fn pow_mod(&self, exp: u64, modulus: &Self) -> Self {
        self.same_modulus(modulus);
        assert!(!modulus.is_zero(), "modulus must be nonzero");
        let p = self.p;
        let mut result = Poly::one(p).rem(modulus);
        let mut base = self.rem(modulus);
        let mut e = exp;
        while e > 0
        {
            if e & 1 == 1
            {
                result = result.mul(&base).rem(modulus);
            }
            base = base.mul(&base).rem(modulus);
            e >>= 1;
        }
        result
    }

    /// Is this polynomial **irreducible** over `GF(p)`? Exact, via Rabin's test.
    ///
    /// Constants (degree 0) and the zero polynomial are not irreducible. A
    /// degree-`n` polynomial `f` is irreducible iff, writing `q = p`,
    /// `x^(q^n) ≡ x (mod f)` and `gcd(x^(q^(n/d)) − x, f) = 1` for every prime
    /// divisor `d` of `n`.
    pub fn is_irreducible(&self) -> bool {
        let n = match self.degree()
        {
            Some(d) => d,
            None => return false,
        };
        if n == 0
        {
            return false;
        }
        if n == 1
        {
            return true;
        }
        let p = self.p;
        let f = self.make_monic();
        let x = Poly::x(p);
        // For each prime d | n: gcd(x^(p^(n/d)) − x, f) must be 1.
        for (d, _) in factor(n as u64)
        {
            let h = frobenius_pow(&f, (n as u64 / d) as u32);
            let g = h.sub(&x).gcd(&f);
            if g.degree() != Some(0)
            {
                return false;
            }
        }
        // and x^(p^n) ≡ x (mod f).
        frobenius_pow(&f, n as u32) == x.rem(&f)
    }
}

/// `x^(p^k) mod f`, built by applying the Frobenius map (raising to the `p`-th
/// power modulo `f`) `k` times to `x`.
fn frobenius_pow(f: &Poly, k: u32) -> Poly {
    let p = f.modulus();
    let mut cur = Poly::x(p).rem(f);
    for _ in 0..k
    {
        cur = cur.pow_mod(p, f);
    }
    cur
}

/// The unique polynomial of degree `< points.len()` passing through the given
/// `(x, y)` samples over `GF(p)` (Lagrange interpolation).
///
/// Panics unless `p` is prime and the `x`-coordinates are distinct mod `p`.
pub fn interpolate(p: u64, points: &[(u64, u64)]) -> Poly {
    assert!(p >= 2 && is_prime(p), "modulus must be prime");
    let xs: Vec<u64> = points.iter().map(|&(x, _)| x % p).collect();
    for i in 0..xs.len()
    {
        for j in (i + 1)..xs.len()
        {
            assert!(xs[i] != xs[j], "interpolation nodes must be distinct mod p");
        }
    }
    let mut acc = Poly::zero(p);
    for i in 0..points.len()
    {
        // Basis polynomial L_i(x) = Π_{j≠i} (x − x_j)/(x_i − x_j).
        let mut num = Poly::one(p);
        let mut den = 1u64;
        for j in 0..points.len()
        {
            if j == i
            {
                continue;
            }
            num = num.mul(&Poly::raw(p, vec![(p - xs[j]) % p, 1]));
            den = mulmod(den, (xs[i] + p - xs[j]) % p, p);
        }
        let w = mulmod(points[i].1 % p, inv_mod(den, p).expect("distinct nodes"), p);
        acc = acc.add(&num.scale(w));
    }
    acc
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

    /// A random polynomial over `GF(p)` of degree `< max_len`.
    fn rand_poly(p: u64, max_len: usize, s: &mut u64) -> Poly {
        let len = 1 + (xorshift(s) % max_len as u64) as usize;
        let coeffs: Vec<u64> = (0..len).map(|_| xorshift(s) % p).collect();
        Poly::from_coeffs(p, &coeffs)
    }

    #[test]
    fn eval_is_a_ring_homomorphism() {
        // (a+b)(x) = a(x)+b(x) and (a·b)(x) = a(x)·b(x) at random points —
        // an independent check of add/sub/mul against evaluation.
        let mut s = 0x1234_5678u64;
        for &p in &[2u64, 3, 5, 97, 998_244_353]
        {
            for _ in 0..40
            {
                let a = rand_poly(p, 8, &mut s);
                let b = rand_poly(p, 8, &mut s);
                let x = xorshift(&mut s) % p;
                assert_eq!(a.add(&b).eval(x), (a.eval(x) + b.eval(x)) % p);
                assert_eq!(a.sub(&b).eval(x), (a.eval(x) + p - b.eval(x)) % p);
                assert_eq!(a.mul(&b).eval(x), mulmod(a.eval(x), b.eval(x), p));
            }
        }
    }

    #[test]
    fn divmod_reconstructs_the_dividend() {
        let mut s = 0x9e37_79b9u64;
        for &p in &[2u64, 5, 97]
        {
            for _ in 0..80
            {
                let a = rand_poly(p, 12, &mut s);
                let b = rand_poly(p, 6, &mut s);
                if b.is_zero()
                {
                    continue;
                }
                let (q, r) = a.divmod(&b);
                // a == q·b + r
                assert_eq!(q.mul(&b).add(&r), a);
                // deg r < deg b
                if let Some(dr) = r.degree()
                {
                    assert!(dr < b.degree().unwrap());
                }
            }
        }
    }

    #[test]
    fn gcd_divides_both_and_egcd_is_bezout() {
        let mut s = 0xdead_beefu64;
        for &p in &[2u64, 7, 97]
        {
            for _ in 0..60
            {
                let a = rand_poly(p, 8, &mut s);
                let b = rand_poly(p, 8, &mut s);
                if a.is_zero() && b.is_zero()
                {
                    continue;
                }
                let g = a.gcd(&b);
                assert!(g.is_monic(), "gcd must be monic");
                assert!(a.rem(&g).is_zero(), "gcd must divide a");
                assert!(b.rem(&g).is_zero(), "gcd must divide b");
                let (g2, u, v) = a.egcd(&b);
                assert_eq!(g2, g, "egcd and gcd disagree");
                // u·a + v·b == g
                assert_eq!(u.mul(&a).add(&v.mul(&b)), g);
            }
        }
    }

    #[test]
    fn derivative_obeys_the_product_rule() {
        // (f·g)' == f'·g + f·g' — an independent check of `derivative`.
        let mut s = 0x0f0f_1234u64;
        for &p in &[3u64, 5, 97]
        {
            for _ in 0..40
            {
                let f = rand_poly(p, 8, &mut s);
                let g = rand_poly(p, 8, &mut s);
                let lhs = f.mul(&g).derivative();
                let rhs = f.derivative().mul(&g).add(&f.mul(&g.derivative()));
                assert_eq!(lhs, rhs);
            }
        }
    }

    #[test]
    fn pow_mod_matches_naive_repeated_multiply() {
        let mut s = 0x5a5a_00f1u64;
        for &p in &[2u64, 5, 97]
        {
            for _ in 0..40
            {
                let base = rand_poly(p, 6, &mut s);
                let mut modulus = rand_poly(p, 5, &mut s);
                if modulus.degree().unwrap_or(0) == 0
                {
                    modulus = Poly::from_coeffs(p, &[1, 0, 1]); // ensure deg ≥ 1
                }
                let e = xorshift(&mut s) % 20;
                let mut naive = Poly::one(p).rem(&modulus);
                for _ in 0..e
                {
                    naive = naive.mul(&base).rem(&modulus);
                }
                assert_eq!(base.pow_mod(e, &modulus), naive);
            }
        }
    }

    #[test]
    fn interpolation_passes_through_the_samples() {
        let mut s = 0x1357_9bdfu64;
        for &p in &[97u64, 998_244_353]
        {
            for _ in 0..30
            {
                let k = 1 + (xorshift(&mut s) % 6) as usize;
                // distinct nodes 0..k, random values
                let pts: Vec<(u64, u64)> =
                    (0..k).map(|i| (i as u64, xorshift(&mut s) % p)).collect();
                let poly = interpolate(p, &pts);
                assert!(poly.degree().map(|d| d < k).unwrap_or(true));
                for &(x, y) in &pts
                {
                    assert_eq!(poly.eval(x), y);
                }
            }
        }
    }

    #[test]
    fn interpolation_recovers_a_known_polynomial() {
        // Sampling a known degree-2 poly at ≥3 points recovers it exactly.
        let p = 97u64;
        let f = Poly::from_coeffs(p, &[5, 3, 1]); // 5 + 3x + x²
        let pts: Vec<(u64, u64)> = (0..4).map(|x| (x, f.eval(x))).collect();
        assert_eq!(interpolate(p, &pts), f);
    }

    // Brute-force root search: a degree-2 or -3 polynomial over GF(p) is
    // reducible iff it has a root in the field — an independent oracle for
    // `is_irreducible` on low degrees.
    fn has_root(f: &Poly) -> bool {
        (0..f.modulus()).any(|x| f.eval(x) == 0)
    }

    #[test]
    fn irreducibility_matches_root_search_on_low_degree() {
        let mut s = 0x2468_ace0u64;
        for &p in &[2u64, 3, 5, 7, 11]
        {
            for deg in 2..=3usize
            {
                for _ in 0..40
                {
                    // random monic polynomial of exact degree `deg`
                    let c: Vec<u64> = (0..deg)
                        .map(|_| xorshift(&mut s) % p)
                        .chain(std::iter::once(1))
                        .collect();
                    let f = Poly::from_coeffs(p, &c);
                    assert_eq!(
                        f.is_irreducible(),
                        !has_root(&f),
                        "mismatch for p={p} f={:?}",
                        f.coeffs()
                    );
                }
            }
        }
    }

    #[test]
    fn irreducibility_known_vectors() {
        // GF(2): x²+x+1 irreducible; x²+1=(x+1)² reducible; x⁴+x+1 irreducible.
        assert!(Poly::from_coeffs(2, &[1, 1, 1]).is_irreducible());
        assert!(!Poly::from_coeffs(2, &[1, 0, 1]).is_irreducible());
        assert!(Poly::from_coeffs(2, &[1, 1, 0, 0, 1]).is_irreducible());
        // GF(3): x²+1 irreducible (−1 is a non-residue); x²−1 reducible.
        assert!(Poly::from_coeffs(3, &[1, 0, 1]).is_irreducible());
        assert!(!Poly::from_coeffs(3, &[2, 0, 1]).is_irreducible()); // x²+2 = x²−1
        // GF(5): x²+2 irreducible (3 is a non-residue); x²+1 reducible.
        assert!(Poly::from_coeffs(5, &[2, 0, 1]).is_irreducible());
        assert!(!Poly::from_coeffs(5, &[1, 0, 1]).is_irreducible());
    }

    #[test]
    fn aes_field_polynomial_is_irreducible() {
        // The AES/Rijndael reduction polynomial x⁸+x⁴+x³+x+1 over GF(2) must be
        // irreducible — that is exactly why GF(2)[x]/(m) is the field GF(2^8).
        let m = Poly::from_coeffs(2, &[1, 1, 0, 1, 1, 0, 0, 0, 1]);
        assert!(m.is_irreducible());
    }

    #[test]
    fn degree_and_accessors() {
        let p = 97u64;
        assert_eq!(Poly::zero(p).degree(), None);
        assert!(Poly::zero(p).is_zero());
        let f = Poly::from_coeffs(p, &[3, 0, 7]);
        assert_eq!(f.degree(), Some(2));
        assert_eq!(f.coeff(2), 7);
        assert_eq!(f.coeff(1), 0);
        assert_eq!(f.coeff(5), 0);
        assert_eq!(f.leading_coeff(), 7);
        assert!(!f.is_monic());
        assert!(f.make_monic().is_monic());
        // trailing zeros are trimmed away
        assert_eq!(Poly::from_coeffs(p, &[1, 2, 0, 0]).degree(), Some(1));
    }

    #[test]
    #[should_panic(expected = "modulus must be prime")]
    fn rejects_composite_modulus() {
        let _ = Poly::from_coeffs(10, &[1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "division by the zero polynomial")]
    fn rejects_division_by_zero() {
        let p = 7u64;
        let _ = Poly::from_coeffs(p, &[1, 2]).divmod(&Poly::zero(p));
    }
}
