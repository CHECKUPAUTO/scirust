//! Dense matrices over a ring `Z/2^k` (`W: Word`) with the exact operations that
//! are awkward or unavailable in generic linear-algebra libraries because
//! `Z/2^k` is not a field:
//!
//! * [`ModMatrix::det_mod`] — determinant **mod `2^k`**, by 2-adic Gaussian
//!   elimination (`O(n³)`), exact.
//! * [`ModMatrix::is_unit`] — invertibility over `Z/2^k` (determinant odd).
//! * [`ModMatrix::gf2_rank`] — rank of the matrix **reduced mod 2**, over
//!   `GF(2)`. Kept strictly distinct from any rank over `Z/2^k`.
//! * [`ModMatrix::smith_valuations`] — the 2-adic elementary-divisor valuations
//!   (Smith normal form over the discrete valuation ring `Z_2`), hence exact
//!   [`ModMatrix::kernel_log2`] / [`ModMatrix::image_log2`].
//! * [`ModMatrix::inverse`] — the exact inverse when the determinant is a unit.
//!
//! A matrix over `Z/2^k` is invertible **iff** its determinant is odd, which is
//! equivalent to its mod-2 reduction being invertible over `GF(2)`.

use crate::ring::Word;

/// A dense `rows × cols` matrix over `W`, stored row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModMatrix<W: Word> {
    rows: usize,
    cols: usize,
    data: Vec<W>,
}

impl<W: Word> ModMatrix<W> {
    /// A `rows × cols` zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        ModMatrix {
            rows,
            cols,
            data: vec![W::ZERO; rows * cols],
        }
    }

    /// The `n × n` identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n
        {
            m.set(i, i, W::ONE);
        }
        m
    }

    /// Build from a vector of rows.
    pub fn from_rows(rows_in: &[Vec<W>]) -> Self {
        let rows = rows_in.len();
        let cols = if rows == 0 { 0 } else { rows_in[0].len() };
        let mut data = Vec::with_capacity(rows * cols);
        for r in rows_in
        {
            assert_eq!(r.len(), cols, "ragged rows");
            data.extend_from_slice(r);
        }
        ModMatrix { rows, cols, data }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.rows
    }
    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Entry `(r, c)`.
    pub fn get(&self, r: usize, c: usize) -> W {
        self.data[r * self.cols + c]
    }
    /// Set entry `(r, c)`.
    pub fn set(&mut self, r: usize, c: usize, v: W) {
        self.data[r * self.cols + c] = v;
    }
    /// Overwrite column `c` from a slice of length `rows`.
    pub fn set_col(&mut self, c: usize, col: &[W]) {
        assert_eq!(col.len(), self.rows, "column length mismatch");
        for (r, &v) in col.iter().enumerate()
        {
            self.set(r, c, v);
        }
    }

    /// Matrix–vector product `M · x` (wrapping in `Z/2^k`). `x.len() == cols`.
    pub fn matvec(&self, x: &[W]) -> Vec<W> {
        assert_eq!(x.len(), self.cols, "dimension mismatch");
        let mut y = vec![W::ZERO; self.rows];
        for r in 0..self.rows
        {
            let mut acc = W::ZERO;
            for c in 0..self.cols
            {
                acc = acc.wadd(self.get(r, c).wmul(x[c]));
            }
            y[r] = acc;
        }
        y
    }

    /// Matrix product `self · other`. Inner dimensions must match.
    pub fn matmul(&self, o: &ModMatrix<W>) -> ModMatrix<W> {
        assert_eq!(self.cols, o.rows, "dimension mismatch");
        let mut out = ModMatrix::zeros(self.rows, o.cols);
        for i in 0..self.rows
        {
            for j in 0..o.cols
            {
                let mut acc = W::ZERO;
                for k in 0..self.cols
                {
                    acc = acc.wadd(self.get(i, k).wmul(o.get(k, j)));
                }
                out.set(i, j, acc);
            }
        }
        out
    }

    /// `true` iff square and equal to the identity.
    pub fn is_identity(&self) -> bool {
        if self.rows != self.cols
        {
            return false;
        }
        for r in 0..self.rows
        {
            for c in 0..self.cols
            {
                let want = if r == c { W::ONE } else { W::ZERO };
                if self.get(r, c) != want
                {
                    return false;
                }
            }
        }
        true
    }

    /// Determinant **mod `2^k`**, exact, by 2-adic Gaussian elimination.
    ///
    /// Pivoting selects the entry of minimum 2-adic valuation in the current
    /// column (which then divides all entries below it), so row elimination is
    /// exact in the ring. Determinant = sign · ∏ (diagonal). Panics if not square.
    pub fn det_mod(&self) -> W {
        assert_eq!(self.rows, self.cols, "determinant of a non-square matrix");
        let n = self.rows;
        let mut w = self.data.clone(); // row-major working copy
        let at = |w: &[W], r: usize, c: usize| w[r * n + c];
        let mut sign_pos = true;

        for col in 0..n
        {
            // pivot: row >= col minimizing valuation of column `col`
            let mut best: Option<(usize, u32)> = None;
            for r in col..n
            {
                let v = at(&w, r, col).valuation();
                if v < W::BITS
                {
                    match best
                    {
                        Some((_, bv)) if bv <= v =>
                        {},
                        _ => best = Some((r, v)),
                    }
                }
            }
            let (pr, vp) = match best
            {
                Some(x) => x,
                None => return W::ZERO, // entire column is 0 mod 2^k
            };
            if pr != col
            {
                for c in 0..n
                {
                    w.swap(col * n + c, pr * n + c);
                }
                sign_pos = !sign_pos;
            }
            let pivot = at(&w, col, col);
            let unit_inv = W::from_u64(pivot.to_u64() >> vp).inv().expect("odd unit");
            for r in (col + 1)..n
            {
                let e = at(&w, r, col);
                if e.to_u64() == 0
                {
                    continue;
                }
                let factor = W::from_u64(e.to_u64() >> vp).wmul(unit_inv);
                for c in col..n
                {
                    let nv = at(&w, r, c).wsub(factor.wmul(at(&w, col, c)));
                    w[r * n + c] = nv;
                }
            }
        }
        let mut det = W::ONE;
        for i in 0..n
        {
            det = det.wmul(at(&w, i, i));
        }
        if sign_pos { det } else { det.wneg() }
    }

    /// `true` iff invertible over `Z/2^k` (determinant is a unit / odd).
    pub fn is_unit(&self) -> bool {
        self.det_mod().is_unit()
    }

    /// Rank of the matrix **reduced mod 2**, over `GF(2)`. NOT a rank over
    /// `Z/2^k`.
    pub fn gf2_rank(&self) -> u32 {
        // boolean working copy, Gaussian elimination
        let mut rows: Vec<Vec<bool>> = (0..self.rows)
            .map(|r| {
                (0..self.cols)
                    .map(|c| self.get(r, c).to_u64() & 1 == 1)
                    .collect()
            })
            .collect();
        let mut rank = 0usize;
        let mut pivot_col = 0usize;
        let mut row = 0usize;
        while pivot_col < self.cols && row < self.rows
        {
            let sel = (row..self.rows).find(|&rr| rows[rr][pivot_col]);
            if let Some(sr) = sel
            {
                rows.swap(row, sr);
                for rr in 0..self.rows
                {
                    if rr != row && rows[rr][pivot_col]
                    {
                        for c in pivot_col..self.cols
                        {
                            rows[rr][c] ^= rows[row][c];
                        }
                    }
                }
                rank += 1;
                row += 1;
            }
            pivot_col += 1;
        }
        rank as u32
    }

    /// 2-adic elementary-divisor valuations (Smith normal form over `Z_2`),
    /// each capped at `k`. For a square `n × n` matrix this returns `n` values;
    /// a valuation of `k` marks an elementary divisor `≡ 0`.
    pub fn smith_valuations(&self) -> Vec<u32> {
        let bits = W::BITS;
        let n = self.rows.min(self.cols);
        let (r, c) = (self.rows, self.cols);
        let mut w = self.data.clone();
        let at = |w: &[W], i: usize, j: usize| w[i * c + j];
        let mut vals = vec![bits; n];

        for t in 0..n
        {
            // pivot of minimum valuation in submatrix [t..r][t..c]
            let mut best: Option<(usize, usize, u32)> = None;
            for i in t..r
            {
                for j in t..c
                {
                    let v = at(&w, i, j).valuation();
                    if v < bits
                    {
                        match best
                        {
                            Some((_, _, bv)) if bv <= v =>
                            {},
                            _ => best = Some((i, j, v)),
                        }
                    }
                }
            }
            let (pi, pj, vp) = match best
            {
                Some(x) => x,
                None => break,
            };
            if pi != t
            {
                for j in 0..c
                {
                    w.swap(t * c + j, pi * c + j);
                }
            }
            if pj != t
            {
                for i in 0..r
                {
                    w.swap(i * c + t, i * c + pj);
                }
            }
            vals[t] = vp;
            let pivot = at(&w, t, t);
            let unit_inv = W::from_u64(pivot.to_u64() >> vp).inv().expect("odd unit");
            // clear rest of column t
            for i in (t + 1)..r
            {
                let e = at(&w, i, t);
                if e.to_u64() == 0
                {
                    continue;
                }
                let factor = W::from_u64(e.to_u64() >> vp).wmul(unit_inv);
                for j in t..c
                {
                    let nv = at(&w, i, j).wsub(factor.wmul(at(&w, t, j)));
                    w[i * c + j] = nv;
                }
            }
            // clear rest of row t
            for j in (t + 1)..c
            {
                let e = at(&w, t, j);
                if e.to_u64() == 0
                {
                    continue;
                }
                let factor = W::from_u64(e.to_u64() >> vp).wmul(unit_inv);
                for i in t..r
                {
                    let nv = at(&w, i, j).wsub(factor.wmul(at(&w, i, t)));
                    w[i * c + j] = nv;
                }
            }
        }
        vals
    }

    /// `log2` of the exact kernel size over `Z/2^k`
    /// (`|ker| = 2^{Σ min(valuation_i, k)}`, for a square matrix).
    pub fn kernel_log2(&self) -> u32 {
        self.smith_valuations()
            .iter()
            .map(|&v| v.min(W::BITS))
            .sum()
    }

    /// `log2` of the exact image size over `Z/2^k` for a square `n × n` matrix
    /// (`|image| = 2^{n·k − kernel_log2}`).
    pub fn image_log2(&self) -> u32 {
        debug_assert_eq!(self.rows, self.cols);
        self.rows as u32 * W::BITS - self.kernel_log2()
    }

    /// Exact inverse over `Z/2^k` when the determinant is a unit, else `None`.
    /// Gauss–Jordan with unit pivots (which exist at every step precisely because
    /// the determinant is odd).
    pub fn inverse(&self) -> Option<ModMatrix<W>> {
        if self.rows != self.cols
        {
            return None;
        }
        let n = self.rows;
        // augmented [A | I] as row-major n × 2n
        let mut a = ModMatrix::<W>::zeros(n, 2 * n);
        for i in 0..n
        {
            for j in 0..n
            {
                a.set(i, j, self.get(i, j));
            }
            a.set(i, n + i, W::ONE);
        }
        for col in 0..n
        {
            // find a unit (odd) pivot in column `col`, rows >= col
            let pr = (col..n).find(|&r| a.get(r, col).is_unit())?;
            if pr != col
            {
                for j in 0..2 * n
                {
                    let tmp = a.get(col, j);
                    a.set(col, j, a.get(pr, j));
                    a.set(pr, j, tmp);
                }
            }
            let pinv = a.get(col, col).inv()?;
            for j in 0..2 * n
            {
                a.set(col, j, a.get(col, j).wmul(pinv));
            }
            for r in 0..n
            {
                if r == col
                {
                    continue;
                }
                let f = a.get(r, col);
                if f.to_u64() == 0
                {
                    continue;
                }
                for j in 0..2 * n
                {
                    let nv = a.get(r, j).wsub(f.wmul(a.get(col, j)));
                    a.set(r, j, nv);
                }
            }
        }
        let mut inv = ModMatrix::<W>::zeros(n, n);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::{W2, W4, W8};

    // Independent Leibniz determinant (n! terms) for small-n cross-checks.
    fn leibniz_det<W: Word>(m: &ModMatrix<W>) -> W {
        let n = m.nrows();
        let mut perm: Vec<usize> = (0..n).collect();
        let mut det = W::ZERO;
        let mut sign_pos = true;
        let term = |perm: &[usize], m: &ModMatrix<W>| {
            let mut t = W::ONE;
            for (r, &c) in perm.iter().enumerate()
            {
                t = t.wmul(m.get(r, c));
            }
            t
        };
        let acc = |perm: &[usize], sp: bool, det: &mut W| {
            let t = term(perm, m);
            *det = if sp { det.wadd(t) } else { det.wsub(t) };
        };
        acc(&perm, sign_pos, &mut det);
        // Heap's algorithm
        let mut c = vec![0usize; n];
        let mut i = 0;
        while i < n
        {
            if c[i] < i
            {
                if i % 2 == 0
                {
                    perm.swap(0, i);
                }
                else
                {
                    perm.swap(c[i], i);
                }
                sign_pos = !sign_pos;
                acc(&perm, sign_pos, &mut det);
                c[i] += 1;
                i = 0;
            }
            else
            {
                c[i] = 0;
                i += 1;
            }
        }
        det
    }

    #[test]
    fn identity_props() {
        let id = ModMatrix::<W8>::identity(5);
        assert!(id.is_identity());
        assert_eq!(id.det_mod().to_u64(), 1);
        assert!(id.is_unit());
        assert_eq!(id.kernel_log2(), 0);
        assert_eq!(id.gf2_rank(), 5);
    }

    #[test]
    fn det_matches_leibniz() {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for n in 1..=6
        {
            for _ in 0..40
            {
                let mut m = ModMatrix::<W8>::zeros(n, n);
                for i in 0..n
                {
                    for j in 0..n
                    {
                        m.set(i, j, W8::from_u64(next()));
                    }
                }
                assert_eq!(m.det_mod(), leibniz_det(&m), "det mismatch n={n}");
            }
        }
    }

    #[test]
    fn inverse_roundtrips_when_unit() {
        let mut s = 0x1000_0001u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut inverted = 0;
        for _ in 0..200
        {
            let n = 4;
            let mut m = ModMatrix::<W8>::zeros(n, n);
            for i in 0..n
            {
                for j in 0..n
                {
                    m.set(i, j, W8::from_u64(next()));
                }
            }
            match m.inverse()
            {
                Some(inv) =>
                {
                    assert!(m.is_unit());
                    assert!(m.matmul(&inv).is_identity());
                    assert!(inv.matmul(&m).is_identity());
                    inverted += 1;
                },
                None => assert!(!m.is_unit()),
            }
        }
        assert!(inverted > 0, "expected some invertible matrices");
    }

    #[test]
    fn kernel_size_matches_brute_force_w2() {
        let mut s = 0x9999u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..20
        {
            let n = 4usize;
            let mut m = ModMatrix::<W2>::zeros(n, n);
            for i in 0..n
            {
                for j in 0..n
                {
                    m.set(i, j, W2::from_u64(next()));
                }
            }
            let mut count = 0u64;
            let total = 1u64 << (2 * n); // (2^2)^n
            for code in 0..total
            {
                let x: Vec<W2> = (0..n)
                    .map(|i| W2::from_u64((code >> (2 * i)) & 3))
                    .collect();
                if m.matvec(&x).iter().all(|w| w.to_u64() == 0)
                {
                    count += 1;
                }
            }
            assert_eq!(count, 1u64 << m.kernel_log2(), "kernel size mismatch");
        }
    }

    #[test]
    fn even_scaling_is_singular() {
        // 2*I over Z/2^4: det = 2^4 = 0 mod 16, kernel_log2 = 4 (each 2 has val 1)
        let mut m = ModMatrix::<W4>::zeros(4, 4);
        for i in 0..4
        {
            m.set(i, i, W4::from_u64(2));
        }
        assert!(!m.is_unit());
        assert_eq!(m.det_mod().to_u64(), 0);
        assert_eq!(m.kernel_log2(), 4);
        assert_eq!(m.gf2_rank(), 0);
        assert!(m.inverse().is_none());
    }
}
