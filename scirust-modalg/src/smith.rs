//! Exact **Smith normal form** of an integer matrix, computed **over `BigInt`
//! so there is no overflow ceiling** — every intermediate value is exact.
//!
//! For any integer matrix `A` (`m × n`) there exist unimodular integer matrices
//! `U` (`m × m`) and `V` (`n × n`) with
//!
//! ```text
//! U · A · V = D = diag(d₁, d₂, …, dᵣ, 0, …, 0),   d₁ | d₂ | … | dᵣ,
//! ```
//!
//! the `dᵢ` being the **invariant factors** of `A`. They determine the rank, the
//! structure of the cokernel `ℤⁿ / A·ℤⁿ` (`⊕ ℤ/dᵢ` plus a free part), and — for a
//! square matrix — `∏ dᵢ = |det A|`. [`smith_normal_form`] returns `D`'s diagonal
//! together with the two unimodular certificates `U` and `V`, so `U · A · V = D`
//! is independently checkable.
//!
//! This is the **overflow-free companion** to the `i128` Hermite normal form in
//! [`crate::rational`]; it composes [`crate::bigint`] for its exact arithmetic.

use crate::bigint::BigInt;

/// The Smith normal form of an integer matrix: the invariant factors together
/// with the unimodular transforms `U`, `V` satisfying `U · A · V = D`.
#[derive(Clone, Debug)]
pub struct SmithNormalForm {
    /// The diagonal of `D`, length `min(rows, cols)`: the nonneg invariant
    /// factors `d₁ | d₂ | … | dᵣ` followed by zeros for the rank deficiency.
    pub invariants: Vec<BigInt>,
    /// The left unimodular transform `U` (`rows × rows`).
    pub u: Vec<Vec<BigInt>>,
    /// The right unimodular transform `V` (`cols × cols`).
    pub v: Vec<Vec<BigInt>>,
    /// Row count of the original matrix.
    pub rows: usize,
    /// Column count of the original matrix.
    pub cols: usize,
}

impl SmithNormalForm {
    /// The rank of the matrix — the number of nonzero invariant factors.
    pub fn rank(&self) -> usize {
        self.invariants.iter().filter(|d| !d.is_zero()).count()
    }

    /// The diagonal matrix `D` (`rows × cols`) reconstructed from the invariant
    /// factors.
    pub fn diagonal(&self) -> Vec<Vec<BigInt>> {
        let mut d = zeros(self.rows, self.cols);
        for (i, di) in self.invariants.iter().enumerate()
        {
            d[i][i] = di.clone();
        }
        d
    }
}

fn zeros(m: usize, n: usize) -> Vec<Vec<BigInt>> {
    (0..m)
        .map(|_| (0..n).map(|_| BigInt::zero()).collect())
        .collect()
}

fn identity(n: usize) -> Vec<Vec<BigInt>> {
    let mut m = zeros(n, n);
    for (i, row) in m.iter_mut().enumerate()
    {
        row[i] = BigInt::one();
    }
    m
}

/// `row_i ← row_i − q · row_t` (also used on the transform `U`).
fn row_axpy(mat: &mut [Vec<BigInt>], i: usize, t: usize, q: &BigInt) {
    let rt = mat[t].clone();
    for (c, x) in mat[i].iter_mut().enumerate()
    {
        *x = x.sub(&q.mul(&rt[c]));
    }
}

/// `col_j ← col_j − q · col_t` (also used on the transform `V`).
fn col_axpy(mat: &mut [Vec<BigInt>], j: usize, t: usize, q: &BigInt) {
    for row in mat.iter_mut()
    {
        let term = q.mul(&row[t]);
        row[j] = row[j].sub(&term);
    }
}

fn swap_cols(mat: &mut [Vec<BigInt>], a: usize, b: usize) {
    for row in mat.iter_mut()
    {
        row.swap(a, b);
    }
}

/// `row_t ← row_t + row_i` (used to pull an indivisible entry into the pivot
/// row before shrinking the pivot).
fn row_add(mat: &mut [Vec<BigInt>], t: usize, i: usize) {
    let ri = mat[i].clone();
    for (c, x) in mat[t].iter_mut().enumerate()
    {
        *x = x.add(&ri[c]);
    }
}

fn negate_row(mat: &mut [Vec<BigInt>], t: usize) {
    for x in mat[t].iter_mut()
    {
        *x = x.neg();
    }
}

/// The position of the smallest-magnitude nonzero entry in the submatrix
/// `d[t.., t..]`, or `None` if that submatrix is entirely zero. Choosing the
/// smallest pivot keeps the intermediate integers small.
fn find_pivot(d: &[Vec<BigInt>], t: usize) -> Option<(usize, usize)> {
    let (m, n) = (d.len(), d[0].len());
    let mut best: Option<(usize, usize)> = None;
    for i in t..m
    {
        for j in t..n
        {
            if !d[i][j].is_zero() && best.is_none_or(|(bi, bj)| d[i][j].abs() < d[bi][bj].abs())
            {
                best = Some((i, j));
            }
        }
    }
    best
}

/// Compute the Smith normal form of an integer matrix.
///
/// Returns the invariant factors `d₁ | d₂ | … ` (nonneg, with trailing zeros for
/// the rank deficiency) and the unimodular transforms `U`, `V` with
/// `U · A · V = D`. Exact and deterministic (pivoting is fully determined by the
/// entries). Panics on an empty or ragged matrix.
pub fn smith_normal_form(a: &[Vec<BigInt>]) -> SmithNormalForm {
    let m = a.len();
    assert!(m >= 1, "empty matrix");
    let n = a[0].len();
    assert!(
        n >= 1 && a.iter().all(|r| r.len() == n),
        "empty or ragged matrix"
    );

    let mut d: Vec<Vec<BigInt>> = a.to_vec();
    let mut u = identity(m);
    let mut v = identity(n);
    let limit = m.min(n);

    for t in 0..limit
    {
        // Bring the smallest-magnitude nonzero entry of the remaining submatrix
        // to the pivot (t, t); stop once the submatrix is all zero.
        match find_pivot(&d, t)
        {
            Some((pi, pj)) =>
            {
                if pi != t
                {
                    d.swap(t, pi);
                    u.swap(t, pi);
                }
                if pj != t
                {
                    swap_cols(&mut d, t, pj);
                    swap_cols(&mut v, t, pj);
                }
            },
            None => break,
        }

        loop
        {
            let mut progressed = false;
            // Clear column t below the pivot by repeated division + swap (Euclid
            // on each entry against the pivot).
            for i in (t + 1)..m
            {
                while !d[i][t].is_zero()
                {
                    let q = d[i][t].div(&d[t][t]);
                    if !q.is_zero()
                    {
                        row_axpy(&mut d, i, t, &q);
                        row_axpy(&mut u, i, t, &q);
                    }
                    if !d[i][t].is_zero()
                    {
                        d.swap(t, i);
                        u.swap(t, i);
                        progressed = true;
                    }
                }
            }
            // Clear row t to the right of the pivot.
            for j in (t + 1)..n
            {
                while !d[t][j].is_zero()
                {
                    let q = d[t][j].div(&d[t][t]);
                    if !q.is_zero()
                    {
                        col_axpy(&mut d, j, t, &q);
                        col_axpy(&mut v, j, t, &q);
                    }
                    if !d[t][j].is_zero()
                    {
                        swap_cols(&mut d, t, j);
                        swap_cols(&mut v, t, j);
                        progressed = true;
                    }
                }
            }
            if progressed
            {
                continue; // a swap may have re-filled the pivot row/column
            }
            // Enforce the divisibility chain: the pivot must divide every entry
            // of the remaining submatrix. If some entry does not divide, add its
            // row into the pivot row and shrink the pivot on the next pass.
            let mut indivisible = None;
            'scan: for i in (t + 1)..m
            {
                for j in (t + 1)..n
                {
                    if !d[i][j].rem(&d[t][t]).is_zero()
                    {
                        indivisible = Some(i);
                        break 'scan;
                    }
                }
            }
            match indivisible
            {
                Some(i) =>
                {
                    row_add(&mut d, t, i);
                    row_add(&mut u, t, i);
                },
                None => break,
            }
        }

        if d[t][t].is_negative()
        {
            negate_row(&mut d, t);
            negate_row(&mut u, t);
        }
    }

    let invariants = (0..limit).map(|i| d[i][i].clone()).collect();
    SmithNormalForm {
        invariants,
        u,
        v,
        rows: m,
        cols: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bigrational::{BigRational, determinant};

    fn xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    fn big(v: i128) -> BigInt {
        BigInt::from_i128(v)
    }

    fn mat(rows: &[&[i128]]) -> Vec<Vec<BigInt>> {
        rows.iter()
            .map(|r| r.iter().map(|&v| big(v)).collect())
            .collect()
    }

    fn matmul(a: &[Vec<BigInt>], b: &[Vec<BigInt>]) -> Vec<Vec<BigInt>> {
        let (m, k, n) = (a.len(), b.len(), b[0].len());
        (0..m)
            .map(|i| {
                (0..n)
                    .map(|c| (0..k).fold(BigInt::zero(), |acc, t| acc.add(&a[i][t].mul(&b[t][c]))))
                    .collect()
            })
            .collect()
    }

    /// `|det|` of an integer matrix, computed independently over `BigRational`.
    fn abs_det(a: &[Vec<BigInt>]) -> BigInt {
        let rows: Vec<Vec<BigRational>> = a
            .iter()
            .map(|r| {
                r.iter()
                    .map(|x| BigRational::from_bigint(x.clone()))
                    .collect()
            })
            .collect();
        let det = determinant(&rows);
        assert!(
            det.denominator() == &BigInt::one(),
            "integer det must be integral"
        );
        det.numerator().abs()
    }

    fn reconstructs(a: &[Vec<BigInt>], snf: &SmithNormalForm) {
        // U · A · V == D
        let uav = matmul(&matmul(&snf.u, a), &snf.v);
        assert_eq!(uav, snf.diagonal(), "U·A·V ≠ D");
    }

    fn check_unimodular(sq: &[Vec<BigInt>]) {
        let d = abs_det(sq);
        assert_eq!(d, BigInt::one(), "transform is not unimodular");
    }

    fn check_divisibility(inv: &[BigInt]) {
        for w in inv.windows(2)
        {
            if w[0].is_zero()
            {
                assert!(w[1].is_zero(), "a zero invariant must be followed by zeros");
            }
            else
            {
                assert!(w[1].rem(&w[0]).is_zero(), "d_i must divide d_{{i+1}}");
            }
        }
    }

    #[test]
    fn known_invariant_factors() {
        // [[1,2],[3,4]]: gcd of entries 1, det −2 ⇒ diag(1, 2).
        let s = smith_normal_form(&mat(&[&[1, 2], &[3, 4]]));
        assert_eq!(s.invariants, vec![big(1), big(2)]);

        // diag(2, 3): coprime, so CRT merges them ⇒ diag(1, 6), not diag(2, 3).
        let s = smith_normal_form(&mat(&[&[2, 0], &[0, 3]]));
        assert_eq!(s.invariants, vec![big(1), big(6)]);

        // diag(2, 4): already a divisibility chain ⇒ unchanged.
        let s = smith_normal_form(&mat(&[&[2, 0], &[0, 4]]));
        assert_eq!(s.invariants, vec![big(2), big(4)]);

        // A singular matrix: rank 1, one zero invariant.
        let s = smith_normal_form(&mat(&[&[1, 2], &[2, 4]]));
        assert_eq!(s.rank(), 1);
        assert_eq!(s.invariants, vec![big(1), big(0)]);
    }

    #[test]
    fn random_square_matrices() {
        let mut s = 0xabcd_1234u64;
        for _ in 0..80
        {
            let n = 1 + (xorshift(&mut s) % 4) as usize; // 1..4
            let a: Vec<Vec<BigInt>> = (0..n)
                .map(|_| {
                    (0..n)
                        .map(|_| big((xorshift(&mut s) % 21) as i128 - 10))
                        .collect()
                })
                .collect();
            let snf = smith_normal_form(&a);
            reconstructs(&a, &snf);
            check_unimodular(&snf.u);
            check_unimodular(&snf.v);
            check_divisibility(&snf.invariants);
            // ∏ dᵢ == |det A| (both zero exactly when A is singular).
            let prod = snf
                .invariants
                .iter()
                .fold(BigInt::one(), |acc, d| acc.mul(d));
            assert_eq!(prod.abs(), abs_det(&a), "∏ dᵢ ≠ |det A|");
        }
    }

    #[test]
    fn random_rectangular_matrices() {
        let mut s = 0x7777_0f0fu64;
        for _ in 0..60
        {
            let m = 1 + (xorshift(&mut s) % 4) as usize;
            let n = 1 + (xorshift(&mut s) % 4) as usize;
            let a: Vec<Vec<BigInt>> = (0..m)
                .map(|_| {
                    (0..n)
                        .map(|_| big((xorshift(&mut s) % 15) as i128 - 7))
                        .collect()
                })
                .collect();
            let snf = smith_normal_form(&a);
            assert_eq!(snf.invariants.len(), m.min(n));
            reconstructs(&a, &snf);
            check_unimodular(&snf.u);
            check_unimodular(&snf.v);
            check_divisibility(&snf.invariants);
        }
    }

    #[test]
    fn zero_matrix_has_trivial_form() {
        let a = zeros(2, 3);
        let snf = smith_normal_form(&a);
        assert_eq!(snf.rank(), 0);
        assert!(snf.invariants.iter().all(|d| d.is_zero()));
        reconstructs(&a, &snf);
        assert_eq!(snf.u, identity(2));
        assert_eq!(snf.v, identity(3));
    }

    #[test]
    fn overflow_free_large_entries() {
        // Entries near 10^18 whose products exceed i128 in the intermediate
        // determinant would overflow fixed-width arithmetic; BigInt stays exact.
        let a = mat(&[
            &[1_000_000_000_000_000_003, 999_999_999_999_999_989],
            &[999_999_999_999_999_937, 1_000_000_000_000_000_037],
        ]);
        let snf = smith_normal_form(&a);
        reconstructs(&a, &snf);
        check_divisibility(&snf.invariants);
        let prod = snf
            .invariants
            .iter()
            .fold(BigInt::one(), |acc, d| acc.mul(d));
        assert_eq!(prod.abs(), abs_det(&a));
    }

    #[test]
    #[should_panic(expected = "empty or ragged matrix")]
    fn rejects_ragged_matrix() {
        let _ = smith_normal_form(&[vec![big(1), big(2)], vec![big(3)]]);
    }
}
