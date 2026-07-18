//! Arbitrary-precision signed integers — exact, dependency-free, deterministic.
//!
//! [`BigInt`] lifts the crate's exactness from the machine-word ceiling
//! (`i128`) to unbounded precision: sign-and-magnitude with base-`2^32` limbs,
//! decimal parse/format, comparison, `+ − ×`, truncated `divmod`, `pow`, and
//! `gcd`. Multiplication is schoolbook and division is bit-by-bit — a correct
//! **reference**, not a performance-tuned bignum library, in keeping with the
//! rest of the crate.
//!
//! Truncated division matches Rust's `/` and `%` on machine integers (quotient
//! rounds toward zero, the remainder takes the dividend's sign), so results
//! agree bit-for-bit with `i128` wherever both are defined.

use core::cmp::Ordering;

// ---- magnitude helpers (little-endian base-2^32 limbs, no high-zero limbs) ---

fn mag_trim(m: &mut Vec<u32>) {
    while m.last() == Some(&0)
    {
        m.pop();
    }
}

fn mag_is_zero(m: &[u32]) -> bool {
    m.is_empty()
}

fn mag_cmp(a: &[u32], b: &[u32]) -> Ordering {
    if a.len() != b.len()
    {
        return a.len().cmp(&b.len());
    }
    for i in (0..a.len()).rev()
    {
        if a[i] != b[i]
        {
            return a[i].cmp(&b[i]);
        }
    }
    Ordering::Equal
}

fn mag_add(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len().max(b.len()) + 1);
    let mut carry = 0u64;
    for i in 0..a.len().max(b.len())
    {
        let av = *a.get(i).unwrap_or(&0) as u64;
        let bv = *b.get(i).unwrap_or(&0) as u64;
        let s = av + bv + carry;
        out.push((s & 0xFFFF_FFFF) as u32);
        carry = s >> 32;
    }
    if carry != 0
    {
        out.push(carry as u32);
    }
    out
}

/// `a − b`, requires `a ≥ b`.
fn mag_sub(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len());
    let mut borrow = 0i64;
    for i in 0..a.len()
    {
        let av = a[i] as i64;
        let bv = *b.get(i).unwrap_or(&0) as i64;
        let mut d = av - bv - borrow;
        if d < 0
        {
            d += 1 << 32;
            borrow = 1;
        }
        else
        {
            borrow = 0;
        }
        out.push(d as u32);
    }
    mag_trim(&mut out);
    out
}

fn mag_mul(a: &[u32], b: &[u32]) -> Vec<u32> {
    if a.is_empty() || b.is_empty()
    {
        return Vec::new();
    }
    let mut out = vec![0u32; a.len() + b.len()];
    for (i, &av) in a.iter().enumerate()
    {
        let mut carry = 0u64;
        for (j, &bv) in b.iter().enumerate()
        {
            let cur = out[i + j] as u64 + av as u64 * bv as u64 + carry;
            out[i + j] = (cur & 0xFFFF_FFFF) as u32;
            carry = cur >> 32;
        }
        out[i + b.len()] += carry as u32;
    }
    mag_trim(&mut out);
    out
}

fn mag_bitlen(a: &[u32]) -> usize {
    match a.last()
    {
        None => 0,
        Some(&top) => (a.len() - 1) * 32 + (32 - top.leading_zeros() as usize),
    }
}

fn mag_bit(a: &[u32], i: usize) -> bool {
    let limb = i / 32;
    limb < a.len() && (a[limb] >> (i % 32)) & 1 == 1
}

/// Shift left by one bit (multiply by two).
fn mag_shl1(a: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + 1);
    let mut carry = 0u32;
    for &limb in a
    {
        out.push((limb << 1) | carry);
        carry = limb >> 31;
    }
    if carry != 0
    {
        out.push(carry);
    }
    out
}

/// Bit-by-bit long division of magnitudes: returns `(quotient, remainder)` with
/// `a = quotient·b + remainder`, `0 ≤ remainder < b`. Requires `b ≠ 0`.
fn mag_divmod(a: &[u32], b: &[u32]) -> (Vec<u32>, Vec<u32>) {
    debug_assert!(!mag_is_zero(b));
    if mag_cmp(a, b) == Ordering::Less
    {
        return (Vec::new(), a.to_vec());
    }
    let bits = mag_bitlen(a);
    let mut q = vec![0u32; a.len()];
    let mut r: Vec<u32> = Vec::new();
    for i in (0..bits).rev()
    {
        r = mag_shl1(&r);
        if mag_bit(a, i)
        {
            if r.is_empty()
            {
                r.push(1);
            }
            else
            {
                r[0] |= 1;
            }
        }
        if mag_cmp(&r, b) != Ordering::Less
        {
            r = mag_sub(&r, b);
            q[i / 32] |= 1u32 << (i % 32);
        }
    }
    mag_trim(&mut q);
    (q, r)
}

/// Divide a magnitude by a small `u32`, returning `(quotient, remainder)`.
fn mag_divmod_small(a: &[u32], d: u32) -> (Vec<u32>, u32) {
    let mut out = vec![0u32; a.len()];
    let mut rem = 0u64;
    for i in (0..a.len()).rev()
    {
        let cur = (rem << 32) | a[i] as u64;
        out[i] = (cur / d as u64) as u32;
        rem = cur % d as u64;
    }
    mag_trim(&mut out);
    (out, rem as u32)
}

/// An arbitrary-precision signed integer.
#[derive(Clone, Debug)]
pub struct BigInt {
    sign: i8,      // -1, 0, or 1
    mag: Vec<u32>, // little-endian base-2^32, no high-zero limbs; empty iff zero
}

impl BigInt {
    /// The integer `0`.
    pub fn zero() -> Self {
        BigInt {
            sign: 0,
            mag: Vec::new(),
        }
    }

    /// The integer `1`.
    pub fn one() -> Self {
        BigInt {
            sign: 1,
            mag: vec![1],
        }
    }

    fn from_parts(sign: i8, mut mag: Vec<u32>) -> Self {
        mag_trim(&mut mag);
        if mag.is_empty()
        {
            BigInt { sign: 0, mag }
        }
        else
        {
            BigInt { sign, mag }
        }
    }

    /// Construct from a signed 128-bit integer.
    pub fn from_i128(v: i128) -> Self {
        if v == 0
        {
            return Self::zero();
        }
        let sign = if v < 0 { -1 } else { 1 };
        let mut u = v.unsigned_abs(); // correct even for i128::MIN
        let mut mag = Vec::new();
        while u != 0
        {
            mag.push((u & 0xFFFF_FFFF) as u32);
            u >>= 32;
        }
        BigInt { sign, mag }
    }

    /// Parse a decimal string (optional leading `+`/`-`). Returns `None` on any
    /// non-digit content or an empty digit sequence.
    pub fn parse(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.is_empty()
        {
            return None;
        }
        let (sign_char, digits) = match bytes[0]
        {
            b'-' => (-1i8, &s[1..]),
            b'+' => (1i8, &s[1..]),
            _ => (1i8, s),
        };
        if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit())
        {
            return None;
        }
        // consume in chunks of up to 9 digits (10^9 fits a u32)
        let db = digits.as_bytes();
        // leading chunk so subsequent chunks are exactly 9 digits
        let first = db.len() % 9;
        let start = if first == 0 { 9 } else { first };
        let head: u32 = digits[..start].parse().ok()?;
        let mut mag: Vec<u32> = if head == 0 { Vec::new() } else { vec![head] };
        let mut idx = start;
        while idx < db.len()
        {
            let chunk: u32 = digits[idx..idx + 9].parse().ok()?;
            // mag = mag·10^9 + chunk
            let prod = mag_mul(&mag, &[1_000_000_000]);
            mag = mag_add(&prod, &[chunk]);
            mag_trim(&mut mag);
            idx += 9;
        }
        Some(Self::from_parts(sign_char, mag))
    }

    /// `true` iff this is zero.
    pub fn is_zero(&self) -> bool {
        self.sign == 0
    }
    /// `true` iff this is negative.
    pub fn is_negative(&self) -> bool {
        self.sign < 0
    }

    /// The absolute value.
    pub fn abs(&self) -> Self {
        BigInt {
            sign: if self.sign == 0 { 0 } else { 1 },
            mag: self.mag.clone(),
        }
    }

    /// The negation.
    pub fn neg(&self) -> Self {
        BigInt {
            sign: -self.sign,
            mag: self.mag.clone(),
        }
    }

    /// Sum `self + other`.
    pub fn add(&self, o: &BigInt) -> Self {
        if self.sign == 0
        {
            return o.clone();
        }
        if o.sign == 0
        {
            return self.clone();
        }
        if self.sign == o.sign
        {
            Self::from_parts(self.sign, mag_add(&self.mag, &o.mag))
        }
        else
        {
            match mag_cmp(&self.mag, &o.mag)
            {
                Ordering::Equal => Self::zero(),
                Ordering::Greater => Self::from_parts(self.sign, mag_sub(&self.mag, &o.mag)),
                Ordering::Less => Self::from_parts(o.sign, mag_sub(&o.mag, &self.mag)),
            }
        }
    }

    /// Difference `self − other`.
    pub fn sub(&self, o: &BigInt) -> Self {
        self.add(&o.neg())
    }

    /// Product `self · other`.
    pub fn mul(&self, o: &BigInt) -> Self {
        if self.sign == 0 || o.sign == 0
        {
            return Self::zero();
        }
        Self::from_parts(self.sign * o.sign, mag_mul(&self.mag, &o.mag))
    }

    /// Truncated division: returns `(quotient, remainder)` with
    /// `self = quotient·divisor + remainder`, the quotient rounded toward zero
    /// and the remainder taking `self`'s sign (matching `i128`'s `/` and `%`).
    /// Panics on division by zero.
    pub fn divmod(&self, divisor: &BigInt) -> (BigInt, BigInt) {
        assert!(divisor.sign != 0, "division by zero");
        if self.sign == 0
        {
            return (Self::zero(), Self::zero());
        }
        let (q, r) = mag_divmod(&self.mag, &divisor.mag);
        let quo = Self::from_parts(self.sign * divisor.sign, q);
        let rem = Self::from_parts(self.sign, r);
        (quo, rem)
    }

    /// Truncated quotient `self / divisor`.
    pub fn div(&self, divisor: &BigInt) -> BigInt {
        self.divmod(divisor).0
    }
    /// Remainder `self % divisor` (takes `self`'s sign).
    pub fn rem(&self, divisor: &BigInt) -> BigInt {
        self.divmod(divisor).1
    }

    /// `self` raised to a non-negative power by square-and-multiply.
    pub fn pow(&self, mut exp: u64) -> BigInt {
        let mut acc = Self::one();
        let mut base = self.clone();
        while exp > 0
        {
            if exp & 1 == 1
            {
                acc = acc.mul(&base);
            }
            base = base.mul(&base);
            exp >>= 1;
        }
        acc
    }

    /// Greatest common divisor (non-negative; `gcd(0, 0) = 0`).
    pub fn gcd(&self, o: &BigInt) -> BigInt {
        let mut a = self.abs();
        let mut b = o.abs();
        while !b.is_zero()
        {
            let r = a.rem(&b);
            a = b;
            b = r;
        }
        a
    }

    /// The decimal string representation.
    pub fn to_decimal(&self) -> String {
        if self.sign == 0
        {
            return "0".to_string();
        }
        let mut chunks: Vec<u32> = Vec::new();
        let mut m = self.mag.clone();
        while !m.is_empty()
        {
            let (q, r) = mag_divmod_small(&m, 1_000_000_000);
            chunks.push(r);
            m = q;
        }
        let mut s = String::new();
        if self.sign < 0
        {
            s.push('-');
        }
        // most-significant chunk without padding, the rest zero-padded to 9
        s.push_str(&chunks.last().unwrap().to_string());
        for &c in chunks.iter().rev().skip(1)
        {
            s.push_str(&format!("{c:09}"));
        }
        s
    }
}

impl PartialEq for BigInt {
    fn eq(&self, o: &Self) -> bool {
        self.sign == o.sign && self.mag == o.mag
    }
}
impl Eq for BigInt {}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for BigInt {
    fn cmp(&self, o: &Self) -> Ordering {
        match self.sign.cmp(&o.sign)
        {
            Ordering::Equal =>
            {
                let m = mag_cmp(&self.mag, &o.mag);
                if self.sign < 0 { m.reverse() } else { m }
            },
            other => other,
        }
    }
}

impl core::fmt::Display for BigInt {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.to_decimal())
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

    fn big(v: i128) -> BigInt {
        BigInt::from_i128(v)
    }

    #[test]
    fn decimal_roundtrip() {
        for v in [
            0i128,
            1,
            -1,
            9,
            10,
            -10,
            999_999_999,
            1_000_000_000,
            i128::MAX,
            i128::MIN,
        ]
        {
            let b = big(v);
            assert_eq!(b.to_decimal(), v.to_string(), "to_decimal {v}");
            assert_eq!(BigInt::parse(&v.to_string()).unwrap(), b, "parse {v}");
        }
        assert_eq!(BigInt::parse("+42").unwrap(), big(42));
        assert_eq!(BigInt::parse("007").unwrap(), big(7));
        assert!(BigInt::parse("").is_none());
        assert!(BigInt::parse("-").is_none());
        assert!(BigInt::parse("12a3").is_none());
    }

    #[test]
    fn arithmetic_matches_i128() {
        let mut s = 0xb161_00b7u64;
        for _ in 0..5000
        {
            // keep operands small enough that i128 results don't overflow
            let a = (xorshift(&mut s) as i64 as i128) / 3;
            let b = (xorshift(&mut s) as i64 as i128) / 3;
            let (ba, bb) = (big(a), big(b));
            assert_eq!(ba.add(&bb), big(a + b), "add {a}+{b}");
            assert_eq!(ba.sub(&bb), big(a - b), "sub {a}-{b}");
            assert_eq!(ba.mul(&bb), big(a * b), "mul {a}*{b}");
            if b != 0
            {
                let (q, r) = ba.divmod(&bb);
                assert_eq!(q, big(a / b), "div {a}/{b}");
                assert_eq!(r, big(a % b), "rem {a}%{b}");
                // reconstruction: a = q·b + r
                assert_eq!(q.mul(&bb).add(&r), ba);
            }
            assert_eq!(ba.cmp(&bb), a.cmp(&b), "cmp {a} {b}");
        }
    }

    #[test]
    fn big_values_and_pow() {
        // 2^128 exactly (beyond i128)
        let two = big(2);
        assert_eq!(
            two.pow(128).to_decimal(),
            "340282366920938463463374607431768211456"
        );
        // 20! = 2432902008176640000 (fits i128, cross-check)
        let mut fact = BigInt::one();
        for k in 1..=20i128
        {
            fact = fact.mul(&big(k));
        }
        assert_eq!(fact.to_decimal(), "2432902008176640000");
        // 30! is far beyond i128
        let mut f30 = BigInt::one();
        for k in 1..=30i128
        {
            f30 = f30.mul(&big(k));
        }
        assert_eq!(f30.to_decimal(), "265252859812191058636308480000000");
    }

    #[test]
    fn divmod_large_reconstruction() {
        // (10^50 + 7) divided by (10^17 - 3): check a = q·b + r, 0 ≤ r < b
        let a = BigInt::parse("100000000000000000000000000000000000000000000000007").unwrap();
        let b = BigInt::parse("99999999999999997").unwrap();
        let (q, r) = a.divmod(&b);
        assert_eq!(q.mul(&b).add(&r), a);
        assert!(!r.is_negative() && r < b);
    }

    #[test]
    fn gcd_matches_and_bezout_scale() {
        // gcd of two large numbers with a known common factor
        let g = BigInt::parse("123456789").unwrap();
        let a = g.mul(&BigInt::parse("98765432100000001").unwrap());
        let b = g.mul(&BigInt::parse("1000000000000000003").unwrap());
        let d = a.gcd(&b);
        // the known factor divides d, and d divides both
        assert!(a.rem(&d).is_zero() && b.rem(&d).is_zero());
        assert!(d.rem(&g).is_zero());
        // small sanity: gcd(48, 36) = 12
        assert_eq!(big(48).gcd(&big(36)), big(12));
    }
}
