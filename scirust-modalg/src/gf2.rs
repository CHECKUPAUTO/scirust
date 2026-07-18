//! Exact binary-polynomial algebra: carryless multiplication over `GF(2)[x]`
//! and finite fields `GF(2^n)` defined by a reduction polynomial.
//!
//! Polynomials are packed little-endian into an integer, so bit `i` is the
//! coefficient of `x^i`. Addition is `XOR`; multiplication is *carryless*
//! (`clmul`). A [`Gf2Field`] wraps a reduction polynomial (degree `n`, with its
//! leading `x^n` bit set) and provides the field operations, including the
//! multiplicative inverse via the extended Euclidean algorithm in `GF(2)[x]`.
//!
//! These are the standard reusable primitives behind CRCs, LFSRs, Reed–Solomon
//! and AES-style diffusion — here in an exact, dependency-free, integer-only
//! form. Everything is deterministic and platform-independent.

/// Carryless (XOR) multiplication of two `GF(2)[x]` polynomials packed in
/// `u64`, returning the full `≤ 127`-degree product in a `u128`.
pub fn clmul(a: u64, b: u64) -> u128 {
    let mut r = 0u128;
    let aw = a as u128;
    for i in 0..64
    {
        if (b >> i) & 1 == 1
        {
            r ^= aw << i;
        }
    }
    r
}

/// Degree of a `GF(2)[x]` polynomial (`deg(0) = -1`).
pub fn poly_degree(a: u128) -> i32 {
    if a == 0
    {
        -1
    }
    else
    {
        127 - a.leading_zeros() as i32
    }
}

/// Polynomial division in `GF(2)[x]`: returns `(quotient, remainder)` with
/// `a = quotient·b XOR remainder` and `deg(remainder) < deg(b)`. Panics if
/// `b == 0`.
pub fn poly_divmod(a: u128, b: u128) -> (u128, u128) {
    assert!(b != 0, "division by the zero polynomial");
    let db = poly_degree(b);
    let mut r = a;
    let mut q = 0u128;
    while poly_degree(r) >= db
    {
        let shift = poly_degree(r) - db;
        q ^= 1u128 << shift;
        r ^= b << shift;
    }
    (q, r)
}

/// Greatest common divisor in `GF(2)[x]` of two polynomials packed in `u64`.
pub fn poly_gcd(a: u64, b: u64) -> u64 {
    let mut x = a as u128;
    let mut y = b as u128;
    while y != 0
    {
        let (_, r) = poly_divmod(x, y);
        x = y;
        y = r;
    }
    x as u64
}

/// A finite field `GF(2^n)` given by a monic reduction polynomial of degree
/// `n` (the `x^n` bit must be set). Elements are the `2^n` polynomials of
/// degree `< n`, packed in a `u64` (so `n ≤ 63`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Gf2Field {
    degree: u32,
    modulus: u64,
}

impl Gf2Field {
    /// A field from a reduction polynomial of the given `degree` (`1 ≤ degree ≤
    /// 63`). The polynomial must be monic — its `x^degree` bit set — which is
    /// asserted. Irreducibility is the caller's responsibility (see the crate
    /// tests, which verify the bundled constructors give genuine fields).
    pub const fn new(degree: u32, modulus: u64) -> Self {
        assert!(degree >= 1 && degree <= 63, "degree out of range");
        assert!(
            modulus & (1u64 << degree) != 0,
            "reduction polynomial must be monic of the stated degree"
        );
        Gf2Field { degree, modulus }
    }

    /// `GF(2^8)` with the AES/Rijndael polynomial `x^8 + x^4 + x^3 + x + 1`
    /// (`0x11B`). Note this polynomial is irreducible but **not** primitive:
    /// `x` (`0x02`) does not generate the multiplicative group.
    pub const fn rijndael8() -> Self {
        Self::new(8, 0x11B)
    }

    /// `GF(2^8)` with the primitive polynomial `x^8 + x^4 + x^3 + x^2 + 1`
    /// (`0x11D`), for which `x` (`0x02`) generates the multiplicative group.
    /// This is the field used by QR-code and CD Reed–Solomon codes.
    pub const fn primitive8() -> Self {
        Self::new(8, 0x11D)
    }

    /// `GF(2^16)` with the primitive polynomial
    /// `x^16 + x^5 + x^3 + x^2 + 1` (`0x1002D`).
    pub const fn gf2_16() -> Self {
        Self::new(16, 0x1_002D)
    }

    /// The field's extension degree `n`.
    pub fn degree(&self) -> u32 {
        self.degree
    }

    /// The field order `2^n`.
    pub fn order(&self) -> u64 {
        1u64 << self.degree
    }

    /// The reduction polynomial (with its leading `x^n` bit).
    pub fn modulus(&self) -> u64 {
        self.modulus
    }

    /// Reduce an arbitrary `GF(2)[x]` polynomial modulo the reduction
    /// polynomial into the canonical representative of degree `< n`.
    pub fn reduce(&self, a: u128) -> u64 {
        let m = self.modulus as u128;
        let n = self.degree as i32;
        let mut r = a;
        while poly_degree(r) >= n
        {
            let shift = poly_degree(r) - n;
            r ^= m << shift;
        }
        r as u64
    }

    /// Field addition (`XOR`). Inputs are assumed already reduced.
    pub fn add(&self, a: u64, b: u64) -> u64 {
        a ^ b
    }

    /// Field multiplication (carryless multiply followed by reduction).
    pub fn mul(&self, a: u64, b: u64) -> u64 {
        self.reduce(clmul(a, b))
    }

    /// `a^e` in the field by square-and-multiply (`a^0 = 1`).
    pub fn pow(&self, mut a: u64, mut e: u64) -> u64 {
        let mut acc = 1u64;
        while e > 0
        {
            if e & 1 == 1
            {
                acc = self.mul(acc, a);
            }
            a = self.mul(a, a);
            e >>= 1;
        }
        acc
    }

    /// Multiplicative inverse via the extended Euclidean algorithm in
    /// `GF(2)[x]`, or `None` for `0` (or for a non-coprime element when the
    /// reduction polynomial is reducible).
    pub fn inv(&self, a: u64) -> Option<u64> {
        if a == 0
        {
            return None;
        }
        // Track only the Bezout coefficient of `a`.
        let mut old_r = self.modulus as u128;
        let mut r = a as u128;
        let mut old_s = 0u128;
        let mut s = 1u128;
        while r != 0
        {
            let (q, rem) = poly_divmod(old_r, r);
            old_r = r;
            r = rem;
            let ns = old_s ^ clmul_u128(q, s);
            old_s = s;
            s = ns;
        }
        if old_r == 1
        {
            Some(self.reduce(old_s))
        }
        else
        {
            None
        }
    }

    /// `a / b = a · b^{-1}`, or `None` when `b` has no inverse.
    pub fn div(&self, a: u64, b: u64) -> Option<u64> {
        self.inv(b).map(|bi| self.mul(a, bi))
    }
}

/// Carryless multiply of two `u128` polynomials whose product still fits in a
/// `u128` (used internally for small Euclid coefficients).
fn clmul_u128(a: u128, b: u128) -> u128 {
    let mut r = 0u128;
    for i in 0..128
    {
        if (b >> i) & 1 == 1
        {
            r ^= a << i;
        }
    }
    r
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

    #[test]
    fn clmul_matches_schoolbook() {
        // independent reference: convolution of coefficient vectors over GF(2)
        let reference = |a: u64, b: u64| -> u128 {
            let mut coeffs = [0u8; 128];
            for i in 0..64
            {
                if (a >> i) & 1 == 1
                {
                    for j in 0..64
                    {
                        if (b >> j) & 1 == 1
                        {
                            coeffs[i + j] ^= 1;
                        }
                    }
                }
            }
            let mut r = 0u128;
            for (k, &c) in coeffs.iter().enumerate()
            {
                if c == 1
                {
                    r |= 1u128 << k;
                }
            }
            r
        };
        let mut s = 0xabcd_1234u64;
        for _ in 0..2000
        {
            let a = xorshift(&mut s);
            let b = xorshift(&mut s);
            assert_eq!(clmul(a, b), reference(a, b));
        }
    }

    #[test]
    fn divmod_reconstructs() {
        let mut s = 0x1111u64;
        for _ in 0..2000
        {
            let a = xorshift(&mut s) as u128;
            let b = (xorshift(&mut s) | 1) as u128; // nonzero
            let (q, r) = poly_divmod(a, b);
            assert_eq!(clmul_u128(q, b) ^ r, a);
            assert!(poly_degree(r) < poly_degree(b));
        }
    }

    #[test]
    fn rijndael_known_products() {
        let f = Gf2Field::rijndael8();
        // FIPS-197 worked example: {57} · {83} = {c1}
        assert_eq!(f.mul(0x57, 0x83), 0xc1);
        // classic inverse pair {53}^{-1} = {ca}
        assert_eq!(f.inv(0x53), Some(0xca));
        assert_eq!(f.mul(0x53, 0xca), 1);
    }

    #[test]
    fn rijndael_is_a_field() {
        let f = Gf2Field::rijndael8();
        // every nonzero element has an inverse and a·a^{-1} = 1
        for a in 1u64..256
        {
            let inv = f.inv(a).expect("nonzero element of a field is invertible");
            assert_eq!(f.mul(a, inv), 1, "bad inverse for {a:#x}");
        }
        assert_eq!(f.inv(0), None);
        // Fermat: a^(2^8 - 1) = 1 for a != 0
        for a in 1u64..256
        {
            assert_eq!(f.pow(a, 255), 1);
            assert_eq!(f.pow(a, 256), a);
        }
    }

    #[test]
    fn gf2_16_is_a_field() {
        let f = Gf2Field::gf2_16();
        let n = f.order();
        // sample: inverses round-trip, division inverts multiplication
        let mut s = 0x2c2cu64;
        for _ in 0..20_000
        {
            let a = 1 + xorshift(&mut s) % (n - 1);
            let inv = f.inv(a).expect("field element invertible");
            assert_eq!(f.mul(a, inv), 1);
            let b = 1 + xorshift(&mut s) % (n - 1);
            assert_eq!(f.div(f.mul(a, b), b), Some(a));
        }
        // the polynomial is primitive, so 2 (i.e. x) has full order 2^16 - 1
        assert_eq!(f.pow(2, n - 1), 1);
        for &d in &[3u64, 5, 15, 17, 51, 85, 255, 257, 4369, 13107]
        // proper divisors of 65535
        {
            assert_ne!(f.pow(2, d), 1, "x had order dividing {d}, not primitive");
        }
    }

    #[test]
    fn field_axioms_sampled() {
        let f = Gf2Field::rijndael8();
        let mut s = 0x9e37u64;
        for _ in 0..5000
        {
            let a = xorshift(&mut s) & 0xff;
            let b = xorshift(&mut s) & 0xff;
            let c = xorshift(&mut s) & 0xff;
            // commutativity
            assert_eq!(f.mul(a, b), f.mul(b, a));
            // associativity
            assert_eq!(f.mul(f.mul(a, b), c), f.mul(a, f.mul(b, c)));
            // distributivity over XOR
            assert_eq!(f.mul(a, f.add(b, c)), f.add(f.mul(a, b), f.mul(a, c)));
        }
    }

    #[test]
    fn primitive8_generator_full_order() {
        let f = Gf2Field::primitive8();
        // it is a field: every nonzero element inverts
        for a in 1u64..256
        {
            assert_eq!(f.mul(a, f.inv(a).unwrap()), 1);
        }
        // x (=2) is a generator: order exactly 255 (3·5·17)
        assert_eq!(f.pow(2, 255), 1);
        for &d in &[1u64, 3, 5, 15, 17, 51, 85]
        {
            assert_ne!(f.pow(2, d), 1, "x had order dividing {d}, not primitive");
        }
        // the 255 powers of x enumerate every nonzero element exactly once
        let mut seen = std::collections::BTreeSet::new();
        let mut acc = 1u64;
        for _ in 0..255
        {
            seen.insert(acc);
            acc = f.mul(acc, 2);
        }
        assert_eq!(seen.len(), 255);
    }

    #[test]
    fn poly_gcd_basics() {
        // gcd(x^2 + 1, x + 1) over GF(2): x^2+1 = (x+1)^2, so gcd = x+1 (0b11)
        assert_eq!(poly_gcd(0b101, 0b11), 0b11);
        // gcd of a poly with itself is itself
        assert_eq!(poly_gcd(0b10110, 0b10110), 0b10110);
        // coprime example: gcd(x, x+1) = 1
        assert_eq!(poly_gcd(0b10, 0b11), 1);
    }
}
