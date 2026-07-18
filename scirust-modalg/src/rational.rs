//! Exact rational linear algebra and the integer **Hermite normal form** —
//! certified, floating-point-free linear algebra for verification and
//! computer-algebra settings.
//!
//! - [`Fraction`] — an exact rational number over `i128`, always in lowest
//!   terms with a positive denominator.
//! - [`RatMatrix`] — dense matrices over `Fraction` with exact Gaussian
//!   elimination: `solve`, `determinant`, `inverse`, and `rank` over `ℚ`, all
//!   without rounding.
//! - [`hermite_normal_form`] — the row Hermite normal form `H = U·A` of an
//!   integer matrix together with the **unimodular certificate** `U`
//!   (`det U = ±1`), so the reduction is independently checkable.
//!
//! Arithmetic is `i128`; it is exact until a value exceeds that range (large
//! problems would need bignum). Everything is deterministic and platform
//! independent.

use core::cmp::Ordering;
use core::ops::{Add, Div, Mul, Neg, Sub};

fn igcd(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0
    {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// An exact rational number `num / den`, stored in lowest terms with `den > 0`.
#[derive(Copy, Clone, Debug)]
pub struct Fraction {
    num: i128,
    den: i128,
}

impl Fraction {
    /// The rational `0`.
    pub const ZERO: Fraction = Fraction { num: 0, den: 1 };
    /// The rational `1`.
    pub const ONE: Fraction = Fraction { num: 1, den: 1 };

    /// Construct `num / den` reduced to lowest terms. Panics if `den == 0`.
    pub fn new(num: i128, den: i128) -> Self {
        assert!(den != 0, "zero denominator");
        let sign = if den < 0 { -1 } else { 1 };
        let g = igcd(num, den).max(1);
        Fraction {
            num: sign * num / g,
            den: sign * den / g,
        }
    }

    /// The integer `n` as a rational.
    pub fn from_int(n: i128) -> Self {
        Fraction { num: n, den: 1 }
    }

    /// The numerator (sign lives here; the denominator is positive).
    pub fn numerator(&self) -> i128 {
        self.num
    }
    /// The (positive) denominator.
    pub fn denominator(&self) -> i128 {
        self.den
    }
    /// `true` iff this is exactly zero.
    pub fn is_zero(&self) -> bool {
        self.num == 0
    }
    /// `true` iff this rational is an integer.
    pub fn is_integer(&self) -> bool {
        self.den == 1
    }

    /// The reciprocal `den / num`, or `None` for zero.
    pub fn recip(&self) -> Option<Fraction> {
        if self.num == 0
        {
            None
        }
        else
        {
            Some(Fraction::new(self.den, self.num))
        }
    }
}

impl PartialEq for Fraction {
    fn eq(&self, o: &Self) -> bool {
        // both are canonical, so component equality suffices
        self.num == o.num && self.den == o.den
    }
}
impl Eq for Fraction {}

impl PartialOrd for Fraction {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Fraction {
    fn cmp(&self, o: &Self) -> Ordering {
        // den > 0 for both, so compare num*o.den vs o.num*self.den
        (self.num * o.den).cmp(&(o.num * self.den))
    }
}

impl Add for Fraction {
    type Output = Fraction;
    fn add(self, o: Fraction) -> Fraction {
        Fraction::new(self.num * o.den + o.num * self.den, self.den * o.den)
    }
}
impl Sub for Fraction {
    type Output = Fraction;
    fn sub(self, o: Fraction) -> Fraction {
        Fraction::new(self.num * o.den - o.num * self.den, self.den * o.den)
    }
}
impl Mul for Fraction {
    type Output = Fraction;
    fn mul(self, o: Fraction) -> Fraction {
        Fraction::new(self.num * o.num, self.den * o.den)
    }
}
impl Div for Fraction {
    type Output = Fraction;
    fn div(self, o: Fraction) -> Fraction {
        assert!(o.num != 0, "division by zero");
        Fraction::new(self.num * o.den, self.den * o.num)
    }
}
impl Neg for Fraction {
    type Output = Fraction;
    fn neg(self) -> Fraction {
        Fraction {
            num: -self.num,
            den: self.den,
        }
    }
}

/// A dense `rows × cols` matrix over [`Fraction`], stored row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RatMatrix {
    rows: usize,
    cols: usize,
    data: Vec<Fraction>,
}

impl RatMatrix {
    /// A `rows × cols` zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        RatMatrix {
            rows,
            cols,
            data: vec![Fraction::ZERO; rows * cols],
        }
    }

    /// Build from integer rows.
    pub fn from_int_rows(rows_in: &[Vec<i128>]) -> Self {
        let rows = rows_in.len();
        let cols = if rows == 0 { 0 } else { rows_in[0].len() };
        let mut m = Self::zeros(rows, cols);
        for (i, r) in rows_in.iter().enumerate()
        {
            assert_eq!(r.len(), cols, "ragged rows");
            for (j, &v) in r.iter().enumerate()
            {
                m.set(i, j, Fraction::from_int(v));
            }
        }
        m
    }

    /// Number of rows / columns.
    pub fn nrows(&self) -> usize {
        self.rows
    }
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Entry `(r, c)`.
    pub fn get(&self, r: usize, c: usize) -> Fraction {
        self.data[r * self.cols + c]
    }
    /// Set entry `(r, c)`.
    pub fn set(&mut self, r: usize, c: usize, v: Fraction) {
        self.data[r * self.cols + c] = v;
    }

    /// Matrix–vector product `M · x`.
    pub fn matvec(&self, x: &[Fraction]) -> Vec<Fraction> {
        assert_eq!(x.len(), self.cols, "dimension mismatch");
        let mut y = vec![Fraction::ZERO; self.rows];
        for r in 0..self.rows
        {
            let mut acc = Fraction::ZERO;
            for c in 0..self.cols
            {
                acc = acc + self.get(r, c) * x[c];
            }
            y[r] = acc;
        }
        y
    }

    /// Matrix product `self · other`.
    pub fn matmul(&self, o: &RatMatrix) -> RatMatrix {
        assert_eq!(self.cols, o.rows, "dimension mismatch");
        let mut out = RatMatrix::zeros(self.rows, o.cols);
        for i in 0..self.rows
        {
            for j in 0..o.cols
            {
                let mut acc = Fraction::ZERO;
                for k in 0..self.cols
                {
                    acc = acc + self.get(i, k) * o.get(k, j);
                }
                out.set(i, j, acc);
            }
        }
        out
    }

    /// Rank over `ℚ`, by exact Gaussian elimination.
    pub fn rank(&self) -> usize {
        let mut m = self.data.clone();
        let cols = self.cols;
        let at = |m: &[Fraction], r: usize, c: usize| m[r * cols + c];
        let mut rank = 0usize;
        let mut prow = 0usize;
        for col in 0..self.cols
        {
            if prow >= self.rows
            {
                break;
            }
            // find a pivot in this column at row >= prow
            let piv = (prow..self.rows).find(|&r| !at(&m, r, col).is_zero());
            if let Some(pr) = piv
            {
                for c in 0..cols
                {
                    m.swap(prow * cols + c, pr * cols + c);
                }
                let pv = at(&m, prow, col);
                for r in 0..self.rows
                {
                    if r != prow && !at(&m, r, col).is_zero()
                    {
                        let f = at(&m, r, col) / pv;
                        for c in col..cols
                        {
                            let nv = at(&m, r, c) - f * at(&m, prow, c);
                            m[r * cols + c] = nv;
                        }
                    }
                }
                rank += 1;
                prow += 1;
            }
        }
        rank
    }

    /// Exact determinant (panics if not square).
    pub fn determinant(&self) -> Fraction {
        assert_eq!(self.rows, self.cols, "determinant of a non-square matrix");
        let n = self.rows;
        let mut m = self.data.clone();
        let at = |m: &[Fraction], r: usize, c: usize| m[r * n + c];
        let mut det = Fraction::ONE;
        for col in 0..n
        {
            let piv = (col..n).find(|&r| !at(&m, r, col).is_zero());
            let pr = match piv
            {
                Some(r) => r,
                None => return Fraction::ZERO,
            };
            if pr != col
            {
                for c in 0..n
                {
                    m.swap(col * n + c, pr * n + c);
                }
                det = -det;
            }
            let pv = at(&m, col, col);
            det = det * pv;
            for r in (col + 1)..n
            {
                if !at(&m, r, col).is_zero()
                {
                    let f = at(&m, r, col) / pv;
                    for c in col..n
                    {
                        let nv = at(&m, r, c) - f * at(&m, col, c);
                        m[r * n + c] = nv;
                    }
                }
            }
        }
        det
    }

    /// Solve `self · x = b` exactly when `self` is square and nonsingular.
    pub fn solve(&self, b: &[Fraction]) -> Option<Vec<Fraction>> {
        if self.rows != self.cols || b.len() != self.rows
        {
            return None;
        }
        let n = self.rows;
        // augmented [A | b]
        let mut a = RatMatrix::zeros(n, n + 1);
        for i in 0..n
        {
            for j in 0..n
            {
                a.set(i, j, self.get(i, j));
            }
            a.set(i, n, b[i]);
        }
        for col in 0..n
        {
            let pr = (col..n).find(|&r| !a.get(r, col).is_zero())?;
            if pr != col
            {
                for j in 0..=n
                {
                    let t = a.get(col, j);
                    a.set(col, j, a.get(pr, j));
                    a.set(pr, j, t);
                }
            }
            let pv = a.get(col, col);
            for j in col..=n
            {
                a.set(col, j, a.get(col, j) / pv);
            }
            for r in 0..n
            {
                if r != col && !a.get(r, col).is_zero()
                {
                    let f = a.get(r, col);
                    for j in col..=n
                    {
                        let nv = a.get(r, j) - f * a.get(col, j);
                        a.set(r, j, nv);
                    }
                }
            }
        }
        Some((0..n).map(|i| a.get(i, n)).collect())
    }

    /// Exact inverse when square and nonsingular, else `None`.
    pub fn inverse(&self) -> Option<RatMatrix> {
        if self.rows != self.cols
        {
            return None;
        }
        let n = self.rows;
        let mut a = RatMatrix::zeros(n, 2 * n);
        for i in 0..n
        {
            for j in 0..n
            {
                a.set(i, j, self.get(i, j));
            }
            a.set(i, n + i, Fraction::ONE);
        }
        for col in 0..n
        {
            let pr = (col..n).find(|&r| !a.get(r, col).is_zero())?;
            if pr != col
            {
                for j in 0..2 * n
                {
                    let t = a.get(col, j);
                    a.set(col, j, a.get(pr, j));
                    a.set(pr, j, t);
                }
            }
            let pv = a.get(col, col);
            for j in col..2 * n
            {
                a.set(col, j, a.get(col, j) / pv);
            }
            for r in 0..n
            {
                if r != col && !a.get(r, col).is_zero()
                {
                    let f = a.get(r, col);
                    for j in col..2 * n
                    {
                        let nv = a.get(r, j) - f * a.get(col, j);
                        a.set(r, j, nv);
                    }
                }
            }
        }
        let mut inv = RatMatrix::zeros(n, n);
        for i in 0..n
        {
            for j in 0..n
            {
                inv.set(i, j, a.get(i, n + j));
            }
        }
        Some(inv)
    }
}

/// The **row Hermite normal form** of an integer matrix `a`: returns `(H, U)`
/// with `H = U·A`, `H` upper-triangular with positive pivots and every entry
/// above a pivot reduced into `[0, pivot)`, and `U` **unimodular** (`det U =
/// ±1`) — an independently verifiable certificate of the reduction.
///
/// Arithmetic is `i128`; intermediate values must stay in range.
pub fn hermite_normal_form(a: &[Vec<i128>]) -> (Vec<Vec<i128>>, Vec<Vec<i128>>) {
    let m = a.len();
    let n = if m == 0 { 0 } else { a[0].len() };
    let mut h: Vec<Vec<i128>> = a.to_vec();
    let mut u: Vec<Vec<i128>> = (0..m)
        .map(|i| (0..m).map(|j| if i == j { 1 } else { 0 }).collect())
        .collect();

    // Apply the 2×2 unimodular transform [[p, q], [r, s]] to rows `top` and
    // `bot` of both H and U: row_top ← p·row_top + q·row_bot, row_bot ← r·row_top
    // + s·row_bot (using the original rows).
    let combine = |mat: &mut Vec<Vec<i128>>, top: usize, bot: usize, p, q, r, s| {
        let width = mat[top].len();
        for c in 0..width
        {
            let a0 = mat[top][c];
            let b0 = mat[bot][c];
            mat[top][c] = p * a0 + q * b0;
            mat[bot][c] = r * a0 + s * b0;
        }
    };

    let mut row = 0usize;
    for col in 0..n
    {
        if row >= m
        {
            break;
        }
        // Bring a pivot into (row, col) and zero out the rest of the column
        // below it using gcd-based unimodular row operations.
        for r in (row + 1)..m
        {
            if h[r][col] == 0
            {
                continue;
            }
            if h[row][col] == 0
            {
                h.swap(row, r);
                u.swap(row, r);
                continue;
            }
            // egcd(h[row][col], h[r][col]) = g = x·h[row][col] + y·h[r][col]
            let (mut old_r, mut rr) = (h[row][col], h[r][col]);
            let (mut old_x, mut x) = (1i128, 0i128);
            let (mut old_y, mut y) = (0i128, 1i128);
            while rr != 0
            {
                let quo = old_r.div_euclid(rr);
                let nr = old_r - quo * rr;
                old_r = rr;
                rr = nr;
                let nx = old_x - quo * x;
                old_x = x;
                x = nx;
                let ny = old_y - quo * y;
                old_y = y;
                y = ny;
            }
            let g = old_r;
            let a2 = h[row][col] / g;
            let b2 = h[r][col] / g;
            // T = [[x, y], [-b2, a2]], det = x·a2 + y·b2 = 1
            combine(&mut h, row, r, old_x, old_y, -b2, a2);
            combine(&mut u, row, r, old_x, old_y, -b2, a2);
        }
        if h[row][col] == 0
        {
            continue; // column has no pivot; do not advance `row`
        }
        // make the pivot positive
        if h[row][col] < 0
        {
            for c in 0..n
            {
                h[row][c] = -h[row][c];
            }
            for c in 0..m
            {
                u[row][c] = -u[row][c];
            }
        }
        // reduce entries above the pivot into [0, pivot)
        let piv = h[row][col];
        for r in 0..row
        {
            let quo = h[r][col].div_euclid(piv);
            if quo != 0
            {
                for c in 0..n
                {
                    h[r][c] -= quo * h[row][c];
                }
                for c in 0..m
                {
                    u[r][c] -= quo * u[row][c];
                }
            }
        }
        row += 1;
    }
    (h, u)
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

    fn frac(n: i128, d: i128) -> Fraction {
        Fraction::new(n, d)
    }

    #[test]
    fn fraction_arithmetic() {
        assert_eq!(frac(1, 2) + frac(1, 3), frac(5, 6));
        assert_eq!(frac(2, 4), frac(1, 2)); // reduction
        assert_eq!(frac(3, 4) - frac(1, 4), frac(1, 2));
        assert_eq!(frac(2, 3) * frac(3, 4), frac(1, 2));
        assert_eq!(frac(1, 2) / frac(1, 4), Fraction::from_int(2));
        assert_eq!(-frac(1, 3), frac(-1, 3));
        assert_eq!(frac(1, -2), frac(-1, 2)); // denominator normalised positive
        assert!(frac(1, 3) < frac(1, 2));
        assert!(Fraction::ZERO.is_zero());
        assert!(frac(6, 3).is_integer());
    }

    #[test]
    fn solve_is_exact() {
        // 2x + y = 3 ; x + 3y = 5  →  x = 4/5, y = 7/5
        let a = RatMatrix::from_int_rows(&[vec![2, 1], vec![1, 3]]);
        let b = [Fraction::from_int(3), Fraction::from_int(5)];
        let x = a.solve(&b).unwrap();
        assert_eq!(x, vec![frac(4, 5), frac(7, 5)]);
        // A·x recovers b exactly
        assert_eq!(a.matvec(&x), b.to_vec());
    }

    #[test]
    fn solve_random_systems_exactly() {
        let mut s = 0x501e_1234u64;
        for _ in 0..200
        {
            let n = 4usize;
            let rows: Vec<Vec<i128>> = (0..n)
                .map(|_| {
                    (0..n)
                        .map(|_| (xorshift(&mut s) % 21) as i128 - 10)
                        .collect()
                })
                .collect();
            let a = RatMatrix::from_int_rows(&rows);
            let x_true: Vec<Fraction> = (0..n)
                .map(|_| Fraction::from_int((xorshift(&mut s) % 11) as i128 - 5))
                .collect();
            let b = a.matvec(&x_true);
            match a.solve(&b)
            {
                Some(x) =>
                {
                    // A·x == b exactly (x may differ from x_true only if singular)
                    assert_eq!(a.matvec(&x), b);
                    assert!(!a.determinant().is_zero());
                },
                None => assert!(a.determinant().is_zero()),
            }
        }
    }

    #[test]
    fn determinant_and_inverse() {
        let a = RatMatrix::from_int_rows(&[vec![1, 2], vec![3, 4]]);
        assert_eq!(a.determinant(), Fraction::from_int(-2));
        let inv = a.inverse().unwrap();
        let prod = a.matmul(&inv);
        // A·A^{-1} == I exactly
        for i in 0..2
        {
            for j in 0..2
            {
                let want = if i == j
                {
                    Fraction::ONE
                }
                else
                {
                    Fraction::ZERO
                };
                assert_eq!(prod.get(i, j), want);
            }
        }
        // singular matrix has no inverse and zero determinant
        let sing = RatMatrix::from_int_rows(&[vec![1, 2], vec![2, 4]]);
        assert!(sing.determinant().is_zero());
        assert!(sing.inverse().is_none());
        assert_eq!(sing.rank(), 1);
    }

    fn imatmul(u: &[Vec<i128>], a: &[Vec<i128>]) -> Vec<Vec<i128>> {
        let m = u.len();
        let inner = a.len();
        let n = a[0].len();
        (0..m)
            .map(|i| {
                (0..n)
                    .map(|j| (0..inner).map(|k| u[i][k] * a[k][j]).sum())
                    .collect()
            })
            .collect()
    }

    fn idet(a: &[Vec<i128>]) -> i128 {
        // exact integer determinant via the rational engine
        let rows: Vec<Vec<i128>> = a.to_vec();
        let d = RatMatrix::from_int_rows(&rows).determinant();
        assert!(d.is_integer());
        d.numerator()
    }

    #[test]
    fn hnf_certificate_holds() {
        let mut s = 0x11f0_0099u64;
        for _ in 0..200
        {
            let m = 4usize;
            let a: Vec<Vec<i128>> = (0..m)
                .map(|_| {
                    (0..m)
                        .map(|_| (xorshift(&mut s) % 15) as i128 - 7)
                        .collect()
                })
                .collect();
            let (h, u) = hermite_normal_form(&a);
            // U·A == H
            assert_eq!(imatmul(&u, &a), h, "U·A ≠ H");
            // U is unimodular
            assert_eq!(idet(&u).abs(), 1, "U not unimodular");
            // H is upper triangular with nonnegative diagonal, and entries above
            // each positive pivot are reduced into [0, pivot)
            for i in 0..m
            {
                for j in 0..i
                {
                    assert_eq!(h[i][j], 0, "H not upper triangular at ({i},{j})");
                }
                assert!(h[i][i] >= 0);
                if h[i][i] > 0
                {
                    for r in 0..i
                    {
                        assert!(h[r][i] >= 0 && h[r][i] < h[i][i], "above-pivot not reduced");
                    }
                }
            }
        }
    }

    #[test]
    fn hnf_known_small() {
        // A classic 2×2 example: HNF of [[2,3],[4,5]].
        let a = vec![vec![2i128, 3], vec![4, 5]];
        let (h, u) = hermite_normal_form(&a);
        assert_eq!(imatmul(&u, &a), h);
        assert_eq!(idet(&u).abs(), 1);
        // determinant of A is preserved up to the sign of U in H's diagonal product
        let diag_prod: i128 = (0..2).map(|i| h[i][i]).product();
        assert_eq!(diag_prod, idet(&a).abs());
    }
}
