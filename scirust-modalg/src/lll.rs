//! Exact **LLL lattice-basis reduction** (Lenstra–Lenstra–Lovász) over the
//! integers, with rational Gram–Schmidt so there is **no floating point and no
//! overflow** — the reduction and every intermediate value are exact.
//!
//! Given a basis of an integer lattice (linearly independent rows), [`reduce`]
//! returns an equivalent basis that is *LLL-reduced* (size-reduced and
//! Lovász-condition-satisfying), together with the **unimodular transform** `U`
//! (`det U = ±1`) mapping the input to the output — an independently checkable
//! certificate that the two bases span the same lattice.
//!
//! Gram–Schmidt runs over [`crate::bigrational::BigRational`], so this is an
//! exact reference (arbitrary precision, single-threaded), not a fast
//! floating-point LLL. Input rows must be linearly independent.

use crate::bigint::BigInt;
use crate::bigrational::BigRational;

/// The result of an LLL reduction.
#[derive(Clone, Debug)]
pub struct LllResult {
    /// The LLL-reduced basis (rows).
    pub basis: Vec<Vec<BigInt>>,
    /// The unimodular transform `U` with `reduced = U · input` (rows).
    pub transform: Vec<Vec<BigInt>>,
}

fn brat(n: &BigInt) -> BigRational {
    BigRational::from_bigint(n.clone())
}

/// Round a rational to the nearest integer (ties toward `+∞`, i.e.
/// `⌊x + 1/2⌋`).
fn round_to_int(r: &BigRational) -> BigInt {
    let two = BigInt::from_i128(2);
    // ⌊(2·num + den) / (2·den)⌋ with den > 0
    let a = r.numerator().mul(&two).add(r.denominator());
    let b = r.denominator().mul(&two);
    let q = a.div(&b);
    let rem = a.rem(&b);
    if rem.is_negative()
    {
        q.sub(&BigInt::one())
    }
    else
    {
        q
    }
}

/// Exact Gram–Schmidt: returns the coefficients `μ[i][j]` (`j < i`) and the
/// squared norms `B[i] = ‖b*_i‖²`, all as rationals.
fn gram_schmidt(basis: &[Vec<BigInt>]) -> (Vec<Vec<BigRational>>, Vec<BigRational>) {
    let n = basis.len();
    let d = basis[0].len();
    let mut bstar: Vec<Vec<BigRational>> = vec![vec![BigRational::zero(); d]; n];
    let mut mu = vec![vec![BigRational::zero(); n]; n];
    let mut b_sq = vec![BigRational::zero(); n];

    for i in 0..n
    {
        for t in 0..d
        {
            bstar[i][t] = brat(&basis[i][t]);
        }
        for j in 0..i
        {
            // μ_{i,j} = ⟨b_i, b*_j⟩ / B_j
            let mut dot = BigRational::zero();
            for t in 0..d
            {
                dot = dot.add(&brat(&basis[i][t]).mul(&bstar[j][t]));
            }
            let m = dot.div(&b_sq[j]);
            // b*_i -= μ_{i,j} · b*_j
            for t in 0..d
            {
                bstar[i][t] = bstar[i][t].sub(&m.mul(&bstar[j][t]));
            }
            mu[i][j] = m;
        }
        let mut norm = BigRational::zero();
        for t in 0..d
        {
            norm = norm.add(&bstar[i][t].mul(&bstar[i][t]));
        }
        b_sq[i] = norm;
    }
    (mu, b_sq)
}

/// LLL-reduce an integer basis with the standard parameter `δ = 3/4`.
pub fn reduce(basis: &[Vec<BigInt>]) -> LllResult {
    reduce_with_delta(
        basis,
        &BigRational::new(BigInt::from_i128(3), BigInt::from_i128(4)),
    )
}

/// LLL-reduce with an explicit Lovász parameter `delta ∈ (1/4, 1)`.
///
/// Panics if the basis is empty, ragged, or (during Gram–Schmidt) linearly
/// dependent.
pub fn reduce_with_delta(basis: &[Vec<BigInt>], delta: &BigRational) -> LllResult {
    let n = basis.len();
    assert!(n >= 1, "empty basis");
    let d = basis[0].len();
    assert!(basis.iter().all(|r| r.len() == d), "ragged basis");

    let mut b: Vec<Vec<BigInt>> = basis.to_vec();
    // transform U, tracking `reduced = U · input`
    let mut u: Vec<Vec<BigInt>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j
                    {
                        BigInt::one()
                    }
                    else
                    {
                        BigInt::zero()
                    }
                })
                .collect()
        })
        .collect();

    let (mut mu, mut b_sq) = gram_schmidt(&b);
    let mut k = 1usize;
    while k < n
    {
        // size-reduce row k against rows k-1 … 0
        for j in (0..k).rev()
        {
            let q = round_to_int(&mu[k][j]);
            if !q.is_zero()
            {
                let qr = brat(&q);
                for t in 0..d
                {
                    b[k][t] = b[k][t].sub(&q.mul(&b[j][t]));
                }
                for t in 0..n
                {
                    u[k][t] = u[k][t].sub(&q.mul(&u[j][t]));
                }
                // μ_{k,i} -= q·μ_{j,i} (i < j); μ_{k,j} -= q
                for i in 0..j
                {
                    mu[k][i] = mu[k][i].sub(&qr.mul(&mu[j][i]));
                }
                mu[k][j] = mu[k][j].sub(&qr);
            }
        }
        // Lovász condition: B_k ≥ (δ − μ_{k,k-1}²)·B_{k-1}
        let mkk = &mu[k][k - 1];
        let rhs = delta.sub(&mkk.mul(mkk)).mul(&b_sq[k - 1]);
        if b_sq[k] >= rhs
        {
            k += 1;
        }
        else
        {
            b.swap(k, k - 1);
            u.swap(k, k - 1);
            let (m2, bs2) = gram_schmidt(&b);
            mu = m2;
            b_sq = bs2;
            k = if k > 1 { k - 1 } else { 1 };
        }
    }

    LllResult {
        basis: b,
        transform: u,
    }
}

/// Convenience: LLL-reduce a basis given as `i128` rows.
pub fn reduce_i128(basis: &[Vec<i128>]) -> LllResult {
    let big: Vec<Vec<BigInt>> = basis
        .iter()
        .map(|r| r.iter().map(|&v| BigInt::from_i128(v)).collect())
        .collect();
    reduce(&big)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bigrational::determinant;

    fn xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    fn norm_sq(v: &[BigInt]) -> BigInt {
        v.iter().fold(BigInt::zero(), |acc, x| acc.add(&x.mul(x)))
    }

    // integer matrix product U·A
    fn matmul(u: &[Vec<BigInt>], a: &[Vec<BigInt>]) -> Vec<Vec<BigInt>> {
        let n = u.len();
        let inner = a.len();
        let d = a[0].len();
        (0..n)
            .map(|i| {
                (0..d)
                    .map(|c| {
                        (0..inner).fold(BigInt::zero(), |acc, k| acc.add(&u[i][k].mul(&a[k][c])))
                    })
                    .collect()
            })
            .collect()
    }

    fn det_i128(m: &[Vec<BigInt>]) -> BigRational {
        let rows: Vec<Vec<BigRational>> = m.iter().map(|r| r.iter().map(brat).collect()).collect();
        determinant(&rows)
    }

    fn is_lll_reduced(basis: &[Vec<BigInt>], delta: &BigRational) -> bool {
        let (mu, b_sq) = gram_schmidt(basis);
        let n = basis.len();
        let half = BigRational::new(BigInt::one(), BigInt::from_i128(2));
        // size-reduced: |μ_{i,j}| ≤ 1/2
        for i in 0..n
        {
            for j in 0..i
            {
                let m = &mu[i][j];
                let mag = if m < &BigRational::zero()
                {
                    m.neg()
                }
                else
                {
                    m.clone()
                };
                if mag > half
                {
                    return false;
                }
            }
        }
        // Lovász: B_k ≥ (δ − μ_{k,k-1}²)·B_{k-1}
        for k in 1..n
        {
            let mkk = &mu[k][k - 1];
            let rhs = delta.sub(&mkk.mul(mkk)).mul(&b_sq[k - 1]);
            if b_sq[k] < rhs
            {
                return false;
            }
        }
        true
    }

    #[test]
    fn reduces_skewed_basis_to_unit_lattice() {
        // (100,1) and (99,1) generate Z² (their difference is (1,0)); LLL must
        // find a basis of two unit vectors.
        let res = reduce_i128(&[vec![100, 1], vec![99, 1]]);
        for row in &res.basis
        {
            assert_eq!(norm_sq(row), BigInt::one(), "expected unit vectors");
        }
        // reduced = U · input, U unimodular
        let input = [
            vec![BigInt::from_i128(100), BigInt::from_i128(1)],
            vec![BigInt::from_i128(99), BigInt::from_i128(1)],
        ];
        assert_eq!(matmul(&res.transform, &input), res.basis);
        let du = det_i128(&res.transform);
        assert!(du == BigRational::one() || du == BigRational::one().neg());
    }

    #[test]
    fn properties_hold_on_random_bases() {
        let delta = BigRational::new(BigInt::from_i128(3), BigInt::from_i128(4));
        let mut s = 0x111e_2222u64;
        for _ in 0..60
        {
            let n = 2 + (xorshift(&mut s) % 3) as usize; // 2..4
            let rows: Vec<Vec<i128>> = (0..n)
                .map(|_| {
                    (0..n)
                        .map(|_| (xorshift(&mut s) % 41) as i128 - 20)
                        .collect()
                })
                .collect();
            // skip singular bases (need linear independence)
            let big: Vec<Vec<BigInt>> = rows
                .iter()
                .map(|r| r.iter().map(|&v| BigInt::from_i128(v)).collect())
                .collect();
            if det_i128(&big).is_zero()
            {
                continue;
            }
            let res = reduce(&big);
            // LLL-reduced
            assert!(is_lll_reduced(&res.basis, &delta), "output not LLL-reduced");
            // reduced = U · input, U unimodular
            assert_eq!(matmul(&res.transform, &big), res.basis);
            let du = det_i128(&res.transform);
            assert!(du == BigRational::one() || du == BigRational::one().neg());
            // lattice volume invariant: |det(reduced)| == |det(input)|
            let di = det_i128(&big);
            let dr = det_i128(&res.basis);
            let abs = |x: BigRational| if x < BigRational::zero() { x.neg() } else { x };
            assert_eq!(abs(dr), abs(di), "lattice volume changed");
        }
    }

    #[test]
    fn first_vector_is_not_longer() {
        // a deliberately skewed 3D basis; the reduced first vector must be no
        // longer than the shortest input row.
        let input = vec![vec![1i128, 1, 1], vec![-1, 0, 2], vec![3, 5, 6]];
        let res = reduce_i128(&input);
        let big: Vec<Vec<BigInt>> = input
            .iter()
            .map(|r| r.iter().map(|&v| BigInt::from_i128(v)).collect())
            .collect();
        let min_in = big.iter().map(|r| norm_sq(r)).min().unwrap();
        assert!(norm_sq(&res.basis[0]) <= min_in);
        // and the output is genuinely reduced
        let delta = BigRational::new(BigInt::from_i128(3), BigInt::from_i128(4));
        assert!(is_lll_reduced(&res.basis, &delta));
    }
}
