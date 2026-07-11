//! Block/batch QRD-RLS: absorb `B` new samples per call via a single
//! Householder-QR reduction of the augmented system, instead of `B`
//! sequential single-row Givens rotations ([`crate::squared_givens`]).
//!
//! ## Scope — what this is, and what it deliberately is not
//!
//! The research brief that motivated [`crate::squared_givens`] also asked for
//! "block-channel FQRD-RLS for BLAS-3 throughput at scale." That literature
//! (multichannel fast QRD-RLS) covers two genuinely distinct ideas, and this
//! module delivers only the first, precisely scoped:
//!
//! 1. **Block processing**: absorb a block of `B` new time-samples in one
//!    shot, by QR-reducing the `(n+B)×n` matrix formed by stacking the
//!    λ-weighted existing factor on top of the `B` new (correctly
//!    λ-weighted-by-recency) rows — instead of `B` sequential single-row
//!    updates. **This is what [`BlockQrdRls`] implements**, via classical
//!    Householder reflectors (Golub & Van Loan, *Matrix Computations*, Alg.
//!    5.1.1), generalized to apply every reflector to `n_out` right-hand-side
//!    columns as well as the trailing factor columns.
//! 2. **"Fast" (order-recursive) QRD-RLS**: an *asymptotically* different
//!    algorithm family (Cioffi–Kailath / Proudler-style lattice recursions)
//!    that updates in `O(n)` instead of `O(n²)` per sample by propagating
//!    forward/backward prediction errors instead of the triangular factor
//!    directly. **Not implemented here** — it is a different derivation, not
//!    a block-size generalization of Gentleman's rotations, and porting it
//!    correctly would need its own from-scratch cross-oracle validation.
//!
//! Within scope (1), "BLAS-3 throughput" is also intentionally not fully
//! claimed: each Householder reflector below is applied to the trailing
//! matrix one column at a time (a rank-1 update per reflector — BLAS-2
//! shaped: one dot product + one `axpy` per trailing column). A genuine
//! BLAS-3 restructuring would accumulate several reflectors into the compact
//! WY representation (`Q = I − Y·T·Yᵀ`) and apply them to the trailing block
//! via two matrix-matrix products — that further optimization is not done
//! here and is future work. What *is* delivered and measured: whether
//! grouping `B` samples into one QR reduction, using only local dense loops
//! (no external BLAS dependency — [`crate::squared_givens`]'s zero-dependency
//! boundary is preserved), beats `B` sequential per-sample updates on this
//! machine. See `bench_rls` for the measured answer; do not assume block
//! processing wins without checking those numbers — Householder reflectors
//! reintroduce the `√` and `÷` that Gentleman's substitution eliminated from
//! [`crate::squared_givens::SquaredGivensRls`]'s hot loop, so the comparison
//! is not a foregone conclusion.
//!
//! ## Recency weighting inside a block
//!
//! `λ` still means "one sample old, one factor of `λ` forgotten." Absorbing
//! `B` samples sequentially scales the *whole* existing factor by `√λ` on
//! *every* call (see [`crate::squared_givens`]'s module docs), so `B`
//! sequential calls scale it by `λ^(B/2)` in total, while the oldest new
//! sample in the block (index `0`) ends up scaled by `λ^((B-1)/2)` relative
//! to the newest (index `B-1`, scaled by `λ⁰ = 1`). [`BlockQrdRls::update_block`]
//! reproduces this exactly by construction, which is what the cross-oracle
//! tests below verify: one `update_block` call with `block_size = 1` must
//! match [`crate::squared_givens::GivensQrdRls::update`] to tight tolerance,
//! and grouping the same stream into blocks of `B` must match processing it
//! one sample at a time.

use serde::{Deserialize, Serialize};

/// Block/batch QRD-RLS via Householder reflectors on the augmented system.
/// See the module docs for exactly what "block" and "BLAS-3" mean here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockQrdRls {
    n_in: usize,
    n_out: usize,
    lambda: f64,
    /// Upper-triangular information factor, row-major n_in×n_in (same
    /// convention as [`crate::squared_givens::GivensQrdRls`]: `r` is the
    /// square root of `P⁻¹`, real diagonal, division needed to solve).
    r: Vec<f64>,
    /// Right-hand-side columns, row-major n_in×n_out.
    z: Vec<f64>,
    #[serde(skip, default)]
    scratch_a: Vec<f64>,
    #[serde(skip, default)]
    scratch_z: Vec<f64>,
    #[serde(skip, default)]
    scratch_v: Vec<f64>,
}

impl BlockQrdRls {
    /// `lambda ∈ (0, 1]`; `r` starts at `(1/√delta)·I`, `z = 0` — the same
    /// information-factor convention as `GivensQrdRls::new`.
    pub fn new(n_in: usize, n_out: usize, lambda: f64, delta: f64) -> Self {
        assert!(lambda > 0.0 && lambda <= 1.0, "lambda must be in (0, 1]");
        assert!(delta > 0.0, "delta must be positive");
        let mut r = vec![0.0; n_in * n_in];
        let s = 1.0 / delta.sqrt();
        for i in 0..n_in
        {
            r[i * n_in + i] = s;
        }
        Self {
            n_in,
            n_out,
            lambda,
            r,
            z: vec![0.0; n_in * n_out],
            scratch_a: Vec::new(),
            scratch_z: Vec::new(),
            scratch_v: Vec::new(),
        }
    }

    /// Absorb a block of `block_size` new samples in one QR reduction.
    /// `u_block` is `block_size × n_in` row-major, oldest sample first;
    /// `d_block` is `block_size × n_out` row-major, same order. `O((n_in +
    /// block_size)·n_in²)`; resizes its scratch buffers to the block's shape
    /// (not zero-allocation across varying block sizes, unlike the
    /// per-sample filters in [`crate::squared_givens`]).
    #[allow(clippy::needless_range_loop)]
    pub fn update_block(&mut self, u_block: &[f64], d_block: &[f64], block_size: usize) {
        assert!(block_size >= 1, "block_size must be at least 1");
        let n = self.n_in;
        let no = self.n_out;
        assert_eq!(u_block.len(), block_size * n);
        assert_eq!(d_block.len(), block_size * no);

        let total = n + block_size;
        if self.scratch_a.len() != total * n
        {
            self.scratch_a.resize(total * n, 0.0);
        }
        if self.scratch_z.len() != total * no
        {
            self.scratch_z.resize(total * no, 0.0);
        }
        if self.scratch_v.len() != total
        {
            self.scratch_v.resize(total, 0.0);
        }

        // Top block: the existing factor, forgotten by one factor of λ^(B/2)
        // total (matching B sequential per-sample calls, each of which scales
        // the whole factor by √λ). Bottom block: the B new rows, oldest
        // scaled down the most (λ^((B-1)/2)), newest unscaled (λ⁰ = 1) — see
        // module docs for why this exactly reproduces sequential absorption.
        let lambda_half = self.lambda.sqrt();
        let r_scale = lambda_half.powi(block_size as i32);
        for i in 0..n
        {
            let row = i * n;
            for j in 0..n
            {
                self.scratch_a[row + j] = r_scale * self.r[row + j];
            }
            let zrow = i * no;
            for o in 0..no
            {
                self.scratch_z[zrow + o] = r_scale * self.z[zrow + o];
            }
        }
        for k in 0..block_size
        {
            let sample_scale = lambda_half.powi((block_size - 1 - k) as i32);
            let arow = (n + k) * n;
            let urow = k * n;
            for j in 0..n
            {
                self.scratch_a[arow + j] = sample_scale * u_block[urow + j];
            }
            let zrow = (n + k) * no;
            let drow = k * no;
            for o in 0..no
            {
                self.scratch_z[zrow + o] = sample_scale * d_block[drow + o];
            }
        }

        self.householder_reduce(n, no, total);

        for i in 0..n
        {
            let row = i * n;
            self.r[row..row + n].copy_from_slice(&self.scratch_a[row..row + n]);
            let zrow = i * no;
            self.z[zrow..zrow + no].copy_from_slice(&self.scratch_z[zrow..zrow + no]);
        }
    }

    /// Reduce `scratch_a` (`total × n`) to upper-triangular via `n`
    /// Householder reflectors, applying each to the trailing factor columns
    /// and to every `scratch_z` column. Local dense loops only — see the
    /// module docs for why this is not a BLAS-3 (compact-WY) restructuring.
    #[allow(clippy::needless_range_loop)]
    fn householder_reduce(&mut self, n: usize, no: usize, total: usize) {
        for k in 0..n
        {
            let m = total - k;
            for i in 0..m
            {
                self.scratch_v[i] = self.scratch_a[(k + i) * n + k];
            }
            let norm: f64 = self.scratch_v[..m]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            if norm <= 0.0
            {
                continue;
            }
            let alpha = if self.scratch_v[0] >= 0.0
            {
                -norm
            }
            else
            {
                norm
            };
            self.scratch_v[0] -= alpha;
            let vnorm: f64 = self.scratch_v[..m]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            if vnorm <= 0.0
            {
                self.scratch_a[k * n + k] = alpha;
                for i in 1..m
                {
                    self.scratch_a[(k + i) * n + k] = 0.0;
                }
                continue;
            }
            for i in 0..m
            {
                self.scratch_v[i] /= vnorm;
            }

            for c in (k + 1)..n
            {
                let mut dot = 0.0;
                for i in 0..m
                {
                    dot += self.scratch_v[i] * self.scratch_a[(k + i) * n + c];
                }
                for i in 0..m
                {
                    self.scratch_a[(k + i) * n + c] -= 2.0 * self.scratch_v[i] * dot;
                }
            }
            for c in 0..no
            {
                let mut dot = 0.0;
                for i in 0..m
                {
                    dot += self.scratch_v[i] * self.scratch_z[(k + i) * no + c];
                }
                for i in 0..m
                {
                    self.scratch_z[(k + i) * no + c] -= 2.0 * self.scratch_v[i] * dot;
                }
            }

            self.scratch_a[k * n + k] = alpha;
            for i in 1..m
            {
                self.scratch_a[(k + i) * n + k] = 0.0;
            }
        }
    }

    /// Extract the weight vector for output `o` by back-substitution.
    /// `O(n_in²)`.
    #[allow(clippy::needless_range_loop)]
    pub fn weights_for(&self, o: usize) -> Vec<f64> {
        assert!(o < self.n_out);
        let n = self.n_in;
        let mut w = vec![0.0; n];
        for i in (0..n).rev()
        {
            let row = i * n;
            let mut acc = self.z[i * self.n_out + o];
            for j in (i + 1)..n
            {
                acc -= self.r[row + j] * w[j];
            }
            let diag = self.r[row + i];
            w[i] = if diag.abs() > 1.0e-300
            {
                acc / diag
            }
            else
            {
                0.0
            };
        }
        w
    }

    /// The upper-triangular information factor (row-major n_in×n_in), for
    /// cross-checks.
    pub fn factor(&self) -> &[f64] {
        &self.r
    }

    pub fn n_in(&self) -> usize {
        self.n_in
    }

    pub fn n_out(&self) -> usize {
        self.n_out
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::squared_givens::{GivensQrdRls, SquaredGivensRls};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    #[test]
    fn block_size_one_matches_givens_qrd_reference() {
        let n = 4;
        let mut block = BlockQrdRls::new(n, 1, 0.98, 100.0);
        let mut givens = GivensQrdRls::new(n, 0.98, 100.0);
        let mut rng = Lcg(211);
        for _ in 0..1000
        {
            let u: Vec<f64> = (0..n).map(|_| rng.next()).collect();
            let d = 1.5 * u[0] - 0.7 * u[1] + 2.0 * u[3] + 0.02 * rng.next();
            block.update_block(&u, &[d], 1);
            givens.update(&u, d);

            let w_b = block.weights_for(0);
            let w_g = givens.weights();
            for (a, b) in w_b.iter().zip(&w_g)
            {
                assert!((a - b).abs() < 1.0e-6, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn grouping_into_blocks_matches_processing_one_at_a_time() {
        let n = 3;
        let no = 2;
        let lambda = 0.985;
        let mut singles = BlockQrdRls::new(n, no, lambda, 80.0);
        let mut blocked = BlockQrdRls::new(n, no, lambda, 80.0);
        let mut rng = Lcg(223);

        let block_size = 5;
        let n_blocks = 80;
        for _ in 0..n_blocks
        {
            let mut u_block = vec![0.0; block_size * n];
            let mut d_block = vec![0.0; block_size * no];
            for k in 0..block_size
            {
                let u: Vec<f64> = (0..n).map(|_| rng.next()).collect();
                let d: Vec<f64> = (0..no)
                    .map(|o| (o as f64 + 1.0) * u[0] - 0.5 * u[1] + 0.3 * u[2] + 0.01 * rng.next())
                    .collect();
                u_block[k * n..(k + 1) * n].copy_from_slice(&u);
                d_block[k * no..(k + 1) * no].copy_from_slice(&d);
                singles.update_block(&u, &d, 1);
            }
            blocked.update_block(&u_block, &d_block, block_size);
        }

        for o in 0..no
        {
            let w_s = singles.weights_for(o);
            let w_b = blocked.weights_for(o);
            for (a, b) in w_s.iter().zip(&w_b)
            {
                assert!(
                    (a - b).abs() < 1.0e-6 * (1.0 + a.abs()),
                    "output {o}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn block_qrd_mimo_matches_squared_givens_oracle() {
        let (n_in, n_out) = (3, 2);
        let lambda = 0.99;
        let mut block = BlockQrdRls::new(n_in, n_out, lambda, 100.0);
        let mut sg = SquaredGivensRls::new(n_in, n_out, lambda, 100.0);
        let true_w = [[2.0, -1.0, 0.5], [0.3, 1.2, -0.7]];
        let mut rng = Lcg(227);

        let block_size = 8;
        for _ in 0..250
        {
            let mut u_block = vec![0.0; block_size * n_in];
            let mut d_block = vec![0.0; block_size * n_out];
            for k in 0..block_size
            {
                let u: Vec<f64> = (0..n_in).map(|_| rng.next()).collect();
                let d: Vec<f64> = true_w
                    .iter()
                    .map(|row| row.iter().zip(&u).map(|(a, b)| a * b).sum())
                    .collect();
                u_block[k * n_in..(k + 1) * n_in].copy_from_slice(&u);
                d_block[k * n_out..(k + 1) * n_out].copy_from_slice(&d);
                sg.update(&u, &d);
            }
            block.update_block(&u_block, &d_block, block_size);
        }

        for o in 0..n_out
        {
            let w_b = block.weights_for(o);
            let w_sg = sg.weights_for(o);
            for (a, b) in w_b.iter().zip(&w_sg)
            {
                assert!((a - b).abs() < 1.0e-6, "output {o}: {a} vs {b}");
            }
            for (j, &t) in true_w[o].iter().enumerate()
            {
                assert!(
                    (w_b[j] - t).abs() < 1.0e-6,
                    "output {o} tap {j}: {} vs {t}",
                    w_b[j]
                );
            }
        }
    }

    #[test]
    fn tracks_a_drifting_system() {
        let n = 2;
        let mut block = BlockQrdRls::new(n, 1, 0.95, 100.0);
        let mut rng = Lcg(229);
        let mut w0 = 1.0;
        let block_size = 4;
        for _ in 0..750
        {
            let mut u_block = vec![0.0; block_size * n];
            let mut d_block = vec![0.0; block_size];
            for k in 0..block_size
            {
                w0 += 0.001;
                let u = [rng.next(), rng.next()];
                let d = w0 * u[0] - u[1];
                u_block[k * n..(k + 1) * n].copy_from_slice(&u);
                d_block[k] = d;
            }
            block.update_block(&u_block, &d_block, block_size);
        }
        let w = block.weights_for(0);
        assert!((w[0] - w0).abs() < 0.05, "lagging drift: {} vs {w0}", w[0]);
    }

    #[test]
    fn degenerate_inputs() {
        let mut block = BlockQrdRls::new(2, 1, 0.98, 50.0);
        block.update_block(&[0.0, 0.0, 0.0, 0.0], &[0.0, 0.0], 2);
        let w = block.weights_for(0);
        assert!(w.iter().all(|x| x.is_finite()));

        // A block that is entirely rank-deficient along one axis (every
        // sample has u[1] == 0) must not poison the other axis or panic.
        let mut block2 = BlockQrdRls::new(2, 1, 0.98, 50.0);
        block2.update_block(&[1.0, 0.0, -1.0, 0.0, 2.0, 0.0], &[1.0, -1.0, 2.0], 3);
        let w2 = block2.weights_for(0);
        assert!(w2.iter().all(|x| x.is_finite()));
        assert!((w2[0] - 1.0).abs() < 0.01, "w2[0]: {}", w2[0]);
    }
}
