//! Exact rational arithmetic over [`crate::bigint::BigInt`] and the
//! rounding-free linear algebra it makes possible **without an overflow
//! ceiling** — the scalable counterpart of [`crate::rational`].
//!
//! [`BigRational`] is a `num / den` pair of arbitrary-precision integers, always
//! in lowest terms with a positive denominator. [`solve`] and [`determinant`]
//! run exact Gaussian elimination over `BigRational`, so systems whose exact
//! solution has entries far beyond `i128` (a Hilbert matrix, for instance) are
//! solved with no rounding and no overflow.
//!
//! Deterministic and dependency-free, like the rest of the crate. Exactness is
//! unbounded (limited only by memory), at the cost of the schoolbook/​bitwise
//! `BigInt` performance.

use crate::bigint::BigInt;
use core::cmp::Ordering;

/// An exact rational `num / den` in lowest terms with `den > 0`.
#[derive(Clone, Debug)]
pub struct BigRational {
    num: BigInt,
    den: BigInt,
}

impl BigRational {
    /// The rational `0`.
    pub fn zero() -> Self {
        BigRational {
            num: BigInt::zero(),
            den: BigInt::one(),
        }
    }

    /// The rational `1`.
    pub fn one() -> Self {
        BigRational {
            num: BigInt::one(),
            den: BigInt::one(),
        }
    }

    /// Construct `num / den` reduced to lowest terms. Panics if `den == 0`.
    pub fn new(num: BigInt, den: BigInt) -> Self {
        assert!(!den.is_zero(), "zero denominator");
        if num.is_zero()
        {
            return Self::zero();
        }
        let (mut num, mut den) = (num, den);
        if den.is_negative()
        {
            num = num.neg();
            den = den.neg();
        }
        let g = num.gcd(&den); // positive; divides both exactly
        BigRational {
            num: num.div(&g),
            den: den.div(&g),
        }
    }

    /// The integer `n` (from an `i128`) as a rational.
    pub fn from_i128(n: i128) -> Self {
        BigRational {
            num: BigInt::from_i128(n),
            den: BigInt::one(),
        }
    }

    /// A `BigInt` as a rational.
    pub fn from_bigint(n: BigInt) -> Self {
        BigRational {
            num: n,
            den: BigInt::one(),
        }
    }

    /// The numerator (carries the sign; the denominator is positive).
    pub fn numerator(&self) -> &BigInt {
        &self.num
    }
    /// The (positive) denominator.
    pub fn denominator(&self) -> &BigInt {
        &self.den
    }
    /// `true` iff exactly zero.
    pub fn is_zero(&self) -> bool {
        self.num.is_zero()
    }
    /// `true` iff this is an integer (`den == 1`).
    pub fn is_integer(&self) -> bool {
        self.den == BigInt::one()
    }

    /// The reciprocal `den / num`, or `None` for zero.
    pub fn recip(&self) -> Option<BigRational> {
        if self.num.is_zero()
        {
            None
        }
        else
        {
            Some(BigRational::new(self.den.clone(), self.num.clone()))
        }
    }

    /// Sum `self + other`.
    pub fn add(&self, o: &BigRational) -> BigRational {
        let num = self.num.mul(&o.den).add(&o.num.mul(&self.den));
        BigRational::new(num, self.den.mul(&o.den))
    }
    /// Difference `self − other`.
    pub fn sub(&self, o: &BigRational) -> BigRational {
        let num = self.num.mul(&o.den).sub(&o.num.mul(&self.den));
        BigRational::new(num, self.den.mul(&o.den))
    }
    /// Product `self · other`.
    pub fn mul(&self, o: &BigRational) -> BigRational {
        BigRational::new(self.num.mul(&o.num), self.den.mul(&o.den))
    }
    /// Quotient `self / other`. Panics if `other` is zero.
    pub fn div(&self, o: &BigRational) -> BigRational {
        assert!(!o.num.is_zero(), "division by zero");
        BigRational::new(self.num.mul(&o.den), self.den.mul(&o.num))
    }
    /// Negation.
    pub fn neg(&self) -> BigRational {
        BigRational {
            num: self.num.neg(),
            den: self.den.clone(),
        }
    }

    /// Decimal-ish string `num/den` (or just `num` when integral).
    pub fn to_string_frac(&self) -> String {
        if self.is_integer()
        {
            self.num.to_decimal()
        }
        else
        {
            format!("{}/{}", self.num.to_decimal(), self.den.to_decimal())
        }
    }
}

impl PartialEq for BigRational {
    fn eq(&self, o: &Self) -> bool {
        self.num == o.num && self.den == o.den
    }
}
impl Eq for BigRational {}

impl PartialOrd for BigRational {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for BigRational {
    fn cmp(&self, o: &Self) -> Ordering {
        // den > 0 for both, so compare num·o.den against o.num·self.den
        self.num.mul(&o.den).cmp(&o.num.mul(&self.den))
    }
}

impl core::fmt::Display for BigRational {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.to_string_frac())
    }
}

/// Solve the square linear system `A·x = b` exactly over `ℚ`, or `None` if `A`
/// is singular or the dimensions are inconsistent. Exact and overflow-free.
pub fn solve(a: &[Vec<BigRational>], b: &[BigRational]) -> Option<Vec<BigRational>> {
    let n = a.len();
    if n == 0 || b.len() != n || a.iter().any(|r| r.len() != n)
    {
        return None;
    }
    // augmented matrix [A | b]
    let mut m: Vec<Vec<BigRational>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i].clone());
            r
        })
        .collect();

    for col in 0..n
    {
        let pr = (col..n).find(|&r| !m[r][col].is_zero())?;
        m.swap(col, pr);
        let pv = m[col][col].clone();
        for j in col..=n
        {
            m[col][j] = m[col][j].div(&pv);
        }
        for r in 0..n
        {
            if r != col && !m[r][col].is_zero()
            {
                let f = m[r][col].clone();
                for j in col..=n
                {
                    let t = m[r][j].sub(&f.mul(&m[col][j]));
                    m[r][j] = t;
                }
            }
        }
    }
    Some((0..n).map(|i| m[i][n].clone()).collect())
}

/// The exact determinant of a square matrix over `ℚ`, by Gaussian elimination.
pub fn determinant(a: &[Vec<BigRational>]) -> BigRational {
    let n = a.len();
    assert!(a.iter().all(|r| r.len() == n), "matrix must be square");
    if n == 0
    {
        return BigRational::one();
    }
    let mut m: Vec<Vec<BigRational>> = a.to_vec();
    let mut det = BigRational::one();
    for col in 0..n
    {
        let pr = match (col..n).find(|&r| !m[r][col].is_zero())
        {
            Some(r) => r,
            None => return BigRational::zero(),
        };
        if pr != col
        {
            m.swap(pr, col);
            det = det.neg();
        }
        let pv = m[col][col].clone();
        det = det.mul(&pv);
        for r in (col + 1)..n
        {
            if !m[r][col].is_zero()
            {
                let f = m[r][col].div(&pv);
                for j in col..n
                {
                    let t = m[r][j].sub(&f.mul(&m[col][j]));
                    m[r][j] = t;
                }
            }
        }
    }
    det
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rational::Fraction;

    fn r(n: i128, d: i128) -> BigRational {
        BigRational::new(BigInt::from_i128(n), BigInt::from_i128(d))
    }

    #[test]
    fn arithmetic_and_reduction() {
        assert_eq!(r(1, 2).add(&r(1, 3)), r(5, 6));
        assert_eq!(r(2, 4), r(1, 2)); // reduced
        assert_eq!(r(3, 4).sub(&r(1, 4)), r(1, 2));
        assert_eq!(r(2, 3).mul(&r(3, 4)), r(1, 2));
        assert_eq!(r(1, 2).div(&r(1, 4)), BigRational::from_i128(2));
        assert_eq!(r(1, -2), r(-1, 2)); // denominator normalised positive
        assert!(r(1, 3) < r(1, 2));
        assert!(BigRational::zero().is_zero());
        assert_eq!(r(6, 3).to_string_frac(), "2");
        assert_eq!(r(1, 3).to_string_frac(), "1/3");
    }

    #[test]
    fn agrees_with_i128_fraction() {
        // cross-check against the bounded rational for in-range values
        let mut s = 0x6a11_0000u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 200) as i128 - 100
        };
        for _ in 0..1000
        {
            let (a, b, c, d) = (next(), next() | 1, next(), next() | 1);
            let big = r(a, b).add(&r(c, d));
            let small = Fraction::new(a, b) + Fraction::new(c, d);
            assert_eq!(big.numerator().to_decimal(), small.numerator().to_string());
            assert_eq!(
                big.denominator().to_decimal(),
                small.denominator().to_string()
            );
        }
    }

    #[test]
    fn solve_small_exact() {
        // 2x + y = 3 ; x + 3y = 5  →  x = 4/5, y = 7/5
        let a = vec![vec![r(2, 1), r(1, 1)], vec![r(1, 1), r(3, 1)]];
        let b = vec![r(3, 1), r(5, 1)];
        let x = solve(&a, &b).unwrap();
        assert_eq!(x, vec![r(4, 5), r(7, 5)]);
    }

    /// Build the `n×n` Hilbert matrix `H[i][j] = 1/(i+j+1)`.
    fn hilbert(n: usize) -> Vec<Vec<BigRational>> {
        (0..n)
            .map(|i| (0..n).map(|j| r(1, (i + j + 1) as i128)).collect())
            .collect()
    }

    #[test]
    fn hilbert_solve_is_exact_beyond_i128() {
        // The Hilbert system has an exact solution with astronomically large
        // integer entries (it overflows any fixed-width rational by n ≈ 8), yet
        // BigRational recovers it and H·x == b holds exactly.
        for n in 2..=8usize
        {
            let h = hilbert(n);
            // target b = H · ones, so the true solution is all-ones
            let ones = vec![BigRational::one(); n];
            let b: Vec<BigRational> = (0..n)
                .map(|i| {
                    (0..n).fold(BigRational::zero(), |acc, j| {
                        acc.add(&h[i][j].mul(&ones[j]))
                    })
                })
                .collect();
            let x = solve(&h, &b).expect("nonsingular");
            assert_eq!(x, ones, "Hilbert solve wrong at n={n}");
            // the Hilbert determinant is a tiny but nonzero rational
            assert!(!determinant(&h).is_zero());
        }
    }

    #[test]
    fn hilbert_determinant_matches_known() {
        // det(H_3) = 1/2160 exactly.
        let d = determinant(&hilbert(3));
        assert_eq!(d, r(1, 2160));
        // det(H_4) = 1/6048000.
        let d4 = determinant(&hilbert(4));
        assert_eq!(d4, r(1, 6048000));
    }
}
