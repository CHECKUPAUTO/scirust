//! Exact `8×8` linear algebra over `Z/2^k` for the matrix-lifting experiments
//! (spec §Experiment 1–2).
//!
//! Two notions of "rank" are kept strictly separate, per the spec's warning:
//!
//! * [`Mat8::det_mod`] / [`Mat8::is_unit`] — the determinant **mod `2^k`** and
//!   the unit (invertibility) criterion over `Z/2^k`. A matrix over `Z/2^k` is
//!   invertible iff its determinant is odd.
//! * [`Mat8::gf2_rank`] — rank of the matrix **reduced mod 2**, over `GF(2)`.
//!   This is explicitly *not* a rank over `Z/2^k`.
//! * [`Mat8::smith_valuations`] / [`Mat8::kernel_log2`] — the exact 2-adic
//!   elementary-divisor valuations and hence the exact kernel size over `Z/2^k`.

use crate::algebra::word::Word;

/// A dense `8×8` matrix over a coefficient ring `W`. `m[row][col]`.
#[derive(Clone, Debug)]
pub struct Mat8<W: Word> {
    /// Entries, row-major.
    pub m: [[W; 8]; 8],
}

impl<W: Word> Mat8<W> {
    /// Zero matrix.
    pub fn zero() -> Self {
        Mat8 {
            m: [[W::ZERO; 8]; 8],
        }
    }

    /// Column `c` equals `f(e_j)` for a linear `f`: set column from a vector.
    pub fn set_col(&mut self, c: usize, col: [W; 8]) {
        for r in 0..8
        {
            self.m[r][c] = col[r];
        }
    }

    /// Matrix–vector product `M · x` (rows dot `x`), wrapping in `Z/2^k`.
    pub fn matvec(&self, x: &[W; 8]) -> [W; 8] {
        let mut y = [W::ZERO; 8];
        for r in 0..8
        {
            let mut acc = W::ZERO;
            for c in 0..8
            {
                acc = acc.wadd(self.m[r][c].wmul(x[c]));
            }
            y[r] = acc;
        }
        y
    }

    /// Matrix product `self · other`.
    pub fn matmul(&self, o: &Mat8<W>) -> Mat8<W> {
        let mut out = Mat8::zero();
        for i in 0..8
        {
            for j in 0..8
            {
                let mut acc = W::ZERO;
                for k in 0..8
                {
                    acc = acc.wadd(self.m[i][k].wmul(o.m[k][j]));
                }
                out.m[i][j] = acc;
            }
        }
        out
    }

    /// Determinant **mod `2^k`**, computed exactly by the Leibniz formula
    /// (sum over the `8!` permutations with sign). Because the determinant is an
    /// integer polynomial in the entries and reduction mod `2^k` is a ring
    /// homomorphism, wrapping arithmetic yields the exact `det mod 2^k`.
    pub fn det_mod(&self) -> W {
        let mut det = W::ZERO;
        let mut perm = [0usize, 1, 2, 3, 4, 5, 6, 7];
        let mut sign_pos = true;

        // accumulate the identity permutation first (sign +).
        let term = |perm: &[usize; 8], m: &[[W; 8]; 8]| {
            let mut t = W::ONE;
            for (r, &c) in perm.iter().enumerate()
            {
                t = t.wmul(m[r][c]);
            }
            t
        };
        let t0 = term(&perm, &self.m);
        det = det.wadd(t0);

        // Heap's algorithm: each step is one transposition, so parity flips.
        let mut c = [0usize; 8];
        let mut i = 0usize;
        while i < 8
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
                let t = term(&perm, &self.m);
                if sign_pos
                {
                    det = det.wadd(t);
                }
                else
                {
                    det = det.wsub(t);
                }
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

    /// `true` iff the matrix is invertible over `Z/2^k` (determinant is a unit,
    /// i.e. odd).
    pub fn is_unit(&self) -> bool {
        self.det_mod().is_unit()
    }

    /// Rank of the matrix **reduced mod 2**, computed over `GF(2)`. Rows are
    /// packed into `u8` bitmasks. This is NOT a rank over `Z/2^k`.
    pub fn gf2_rank(&self) -> u32 {
        let mut rows: [u16; 8] = [0; 8];
        for r in 0..8
        {
            let mut bits = 0u16;
            for c in 0..8
            {
                if self.m[r][c].to_u64() & 1 == 1
                {
                    bits |= 1 << c;
                }
            }
            rows[r] = bits;
        }
        let mut rank = 0u32;
        let mut pivot_col = 0usize;
        let mut row = 0usize;
        while pivot_col < 8 && row < 8
        {
            // find a row >= `row` with bit `pivot_col` set
            let mut sel = None;
            for rr in row..8
            {
                if rows[rr] & (1 << pivot_col) != 0
                {
                    sel = Some(rr);
                    break;
                }
            }
            if let Some(sr) = sel
            {
                rows.swap(row, sr);
                for rr in 0..8
                {
                    if rr != row && rows[rr] & (1 << pivot_col) != 0
                    {
                        rows[rr] ^= rows[row];
                    }
                }
                rank += 1;
                row += 1;
            }
            pivot_col += 1;
        }
        rank
    }

    /// 2-adic elementary-divisor valuations (Smith normal form over the DVR
    /// `Z/2^k`), each capped at `k`. A valuation of `k` marks a divisor `≡ 0`.
    ///
    /// Pivoting always chooses the entry of minimum 2-adic valuation, which then
    /// divides every remaining entry, so the pivot row/column clear exactly.
    pub fn smith_valuations(&self) -> [u32; 8] {
        let bits = W::BITS;
        let mut w = self.m;
        let mut vals = [bits; 8];

        for t in 0..8
        {
            // find pivot of minimum valuation in submatrix [t..8][t..8]
            let mut best: Option<(usize, usize, u32)> = None;
            for i in t..8
            {
                for j in t..8
                {
                    let v = w[i][j].valuation();
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
                None => break, // remaining submatrix is entirely 0 mod 2^k
            };
            // move pivot to (t,t)
            if pi != t
            {
                w.swap(pi, t);
            }
            if pj != t
            {
                for row in w.iter_mut()
                {
                    row.swap(pj, t);
                }
            }
            vals[t] = vp;
            let pivot = w[t][t];
            // pivot = 2^vp * u ; unit part = pivot >> vp (odd), invertible.
            let unit = W::from_u64(pivot.to_u64() >> vp);
            let unit_inv = unit.inv().expect("odd unit is invertible");

            // clear the rest of column t (rows t+1..8)
            for i in (t + 1)..8
            {
                let e = w[i][t];
                if e.to_u64() == 0
                {
                    continue;
                }
                let factor = W::from_u64(e.to_u64() >> vp).wmul(unit_inv);
                for col in t..8
                {
                    w[i][col] = w[i][col].wsub(factor.wmul(w[t][col]));
                }
            }
            // clear the rest of row t (cols t+1..8)
            for j in (t + 1)..8
            {
                let e = w[t][j];
                if e.to_u64() == 0
                {
                    continue;
                }
                let factor = W::from_u64(e.to_u64() >> vp).wmul(unit_inv);
                for row in t..8
                {
                    w[row][j] = w[row][j].wsub(factor.wmul(w[row][t]));
                }
            }
        }
        vals
    }

    /// `log2` of the exact kernel size over `Z/2^k`
    /// (`|ker| = 2^{Σ min(valuation_i, k)}`).
    pub fn kernel_log2(&self) -> u32 {
        self.smith_valuations()
            .iter()
            .map(|&v| v.min(W::BITS))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::word::{W2, W4, W8};

    fn identity8<W: Word>() -> Mat8<W> {
        let mut m = Mat8::<W>::zero();
        for i in 0..8
        {
            m.m[i][i] = W::ONE;
        }
        m
    }

    #[test]
    fn identity_det_and_rank() {
        let id = identity8::<W8>();
        assert_eq!(id.det_mod().to_u64(), 1);
        assert!(id.is_unit());
        assert_eq!(id.gf2_rank(), 8);
        assert_eq!(id.kernel_log2(), 0);
    }

    #[test]
    fn even_scale_is_singular_over_ring() {
        // 2*I over Z/2^4: det = 2^8 mod 16 = 0, not a unit; kernel is large.
        let mut m = Mat8::<W4>::zero();
        for i in 0..8
        {
            m.m[i][i] = W4::from_u64(2);
        }
        assert!(!m.is_unit());
        // each diagonal 2 has valuation 1 -> kernel_log2 = 8
        assert_eq!(m.kernel_log2(), 8);
        // gf2 rank of (2*I mod 2) = 0
        assert_eq!(m.gf2_rank(), 0);
    }

    #[test]
    fn kernel_size_matches_brute_force_w2() {
        // Build a few W2 matrices and check kernel_log2 against enumeration.
        let mats = [
            {
                let mut m = Mat8::<W2>::zero();
                for i in 0..8
                {
                    m.m[i][i] = W2::from_u64(if i % 2 == 0 { 1 } else { 2 });
                }
                m
            },
            {
                let mut m = Mat8::<W2>::zero();
                for i in 0..8
                {
                    for j in 0..8
                    {
                        m.m[i][j] = W2::from_u64(((i * 3 + j * 5) as u64) & 3);
                    }
                }
                m
            },
        ];
        for m in &mats
        {
            // brute kernel count over (Z/2^2)^8 = 4^8 = 65536
            let mut count = 0u64;
            for code in 0u32..65536
            {
                let mut x = [W2::ZERO; 8];
                for i in 0..8
                {
                    x[i] = W2::from_u64(((code >> (2 * i)) & 3) as u64);
                }
                let y = m.matvec(&x);
                if y.iter().all(|w| w.to_u64() == 0)
                {
                    count += 1;
                }
            }
            let expect = 1u64 << m.kernel_log2();
            assert_eq!(count, expect, "kernel size mismatch");
        }
    }
}
