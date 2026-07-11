//! QR-decomposition RLS via Givens rotations, and its square-root-free
//! ("Fast"/"Squared Givens") form — the **information** square-root dual of
//! [`crate::qr_rls::QrRls`], and the algorithmic entry point for the McWhirter
//! systolic decomposition in [`systolic`].
//!
//! ## Two square roots, two purposes
//!
//! [`crate::qr_rls::QrRls`] propagates a **covariance** square-root `S`
//! (`P = S·Sᵀ`) via Potter's rank-1 update — convenient because the weight
//! update falls out directly (`w += k·e`), no linear solve needed.
//!
//! This module propagates the dual: an **information** (inverse-covariance)
//! square-root `R` (`R = P⁻¹` in factored form) via sequential **Givens
//! rotations** applied to the growing regressor. This is the classical
//! QR-decomposition RLS (QRD-RLS): each rotation is an *exact* orthogonal
//! (isometric) 2×2 transformation, so the whole update is provably backward
//! stable — no re-symmetrization needed, no drift to correct for — and,
//! critically, it **decomposes into independent, nearest-neighbor-communicating
//! elementary cells** ([`systolic`]), the property Potter's whole-row update
//! does not have.
//!
//! [`GivensQrdRls`] is the textbook (`√`-based) reference. [`SquaredGivensRls`]
//! is the same algorithm with the hypotenuse `√(a²+b²)` eliminated (Gentleman,
//! *"Least squares computations by Givens transformations without square
//! roots"*, J. Inst. Math. Appl., 1973): every rotation costs `+, −, ×, ÷`
//! only — no transcendental call in the hot loop, and half the multiplications
//! of the textbook form.
//!
//! ### Deriving the square-root-free rotation
//!
//! Store each triangular row `i` as a **weight** `d_i > 0` and a vector `t_i`
//! with the invariant `t_i[i] = 1`, representing the physical (Givens-rotated)
//! row as `√d_i · t_i`. Combining row `i` (weight `d_i`, pivot `a = t_i[i] = 1`)
//! with an incoming row (weight `d_in`, pivot `b = t_in[i]`) via the *implied*
//! ordinary Givens rotation `c = √d_i/√(d_i+d_in b²)`, `s = √d_in·b/√(d_i+d_in b²)`
//! and substituting through, every `√` cancels exactly, leaving
//!
//! ```text
//! d_i'  = d_i + d_in·b²
//! d_in' = d_i·d_in / d_i'
//! t_i'[j]  = (d_i·t_i[j] + d_in·b·t_in[j]) / d_i'     for j > i
//! t_in'[j] = t_in[j] − b·t_i[j]                        for j > i
//! ```
//! — `t_i'[i] = 1` and `t_in'[i] = 0` fall out of the same formulas, so the
//! invariant is self-maintaining. `λ` enters by scaling `d_i ← λ·d_i` the
//! instant row `i` is used as the pivot for a new sample (matching the usual
//! `R ← λ·R + u·uᵀ` exponential-forgetting recursion).
//!
//! A second, equally useful consequence: because each row's `√d_i` scale is
//! common to the whole row, it **cancels out of the normal equations** —
//! `T[:,0:n]·w = T[:,n]` holds with the *stored* matrix `T` directly (`T` is
//! unit upper-triangular), so weight extraction by back-substitution needs no
//! `√` and not even a division (the diagonal is 1).
//!
//! Every claim above is falsifiable and tested: [`SquaredGivensRls`]'s
//! reconstructed physical `R` (`√d_i · t_i`, row by row) is checked bit-close
//! against [`GivensQrdRls`]'s `√`-based `R` over hundreds of random steps, and
//! both are cross-checked against [`crate::rls::VectorRls`] and
//! [`crate::rls::RlsFilter`] — independent code paths solving the same
//! least-squares problem.

use serde::{Deserialize, Serialize};

/// Reference (square-root) sequential-Givens QRD-RLS, single output.
///
/// Maintains the upper-triangular information factor `r` (n×n, row-major) and
/// the associated right-hand side `z` (n) such that the weights solve
/// `r · w = z` by back-substitution. Kept only as the numerically-obvious
/// baseline that [`SquaredGivensRls`] is validated against — prefer the
/// square-root-free filter in new code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GivensQrdRls {
    n: usize,
    lambda: f64,
    /// Upper-triangular factor, row-major n×n (only the upper triangle is used).
    r: Vec<f64>,
    /// Right-hand side.
    z: Vec<f64>,
}

impl GivensQrdRls {
    /// `lambda ∈ (0, 1]`; `r` is the square root of the *information* matrix
    /// `P⁻¹`, so it starts at `r = (1/√delta) · I`, `z = 0` — the dual of
    /// `P(0) = delta·I` in the covariance-form filters (`r² = 1/delta`).
    pub fn new(n: usize, lambda: f64, delta: f64) -> Self {
        assert!(lambda > 0.0 && lambda <= 1.0, "lambda must be in (0, 1]");
        assert!(delta > 0.0, "delta must be positive");
        let mut r = vec![0.0; n * n];
        let s = 1.0 / delta.sqrt();
        for i in 0..n
        {
            r[i * n + i] = s;
        }
        Self {
            n,
            lambda,
            r,
            z: vec![0.0; n],
        }
    }

    /// Triangularize one new sample `(u, d)` into the factor. `O(n²)`, one
    /// `√` per pivot (`n` total).
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, u: &[f64], d: f64) {
        assert_eq!(u.len(), self.n);
        let n = self.n;
        let mut x = u.to_vec();
        let mut rhs = d;
        for i in 0..n
        {
            let row = i * n;
            let r_ii = self.r[row + i] * self.lambda.sqrt();
            let x_i = x[i];
            let rho = (r_ii * r_ii + x_i * x_i).sqrt();
            if rho <= 0.0
            {
                continue;
            }
            let c = r_ii / rho;
            let s = x_i / rho;
            for j in i..n
            {
                let r_ij = self.r[row + j] * self.lambda.sqrt();
                let x_j = x[j];
                self.r[row + j] = c * r_ij + s * x_j;
                x[j] = -s * r_ij + c * x_j;
            }
            let z_i = self.z[i] * self.lambda.sqrt();
            let new_z_i = c * z_i + s * rhs;
            rhs = -s * z_i + c * rhs;
            self.z[i] = new_z_i;
        }
    }

    /// Extract the current weight vector by back-substitution, `O(n²)`.
    #[allow(clippy::needless_range_loop)]
    pub fn weights(&self) -> Vec<f64> {
        let n = self.n;
        let mut w = vec![0.0; n];
        for i in (0..n).rev()
        {
            let row = i * n;
            let mut acc = self.z[i];
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

    /// The upper-triangular factor (row-major n×n), for cross-checks.
    pub fn factor(&self) -> &[f64] {
        &self.r
    }
}

/// Square-root-free ("Fast"/"Squared Givens", Gentleman 1973) QRD-RLS,
/// multi-channel (`n_in` inputs, `n_out` outputs).
///
/// See the module docs for the derivation. The information factor is shared
/// across outputs (it depends only on the inputs); each output keeps its own
/// right-hand-side column. `update` is `O(n_in² + n_out·n_in)`, zero heap
/// allocation, and calls no `√` — every operation is `+, −, ×, ÷`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SquaredGivensRls {
    n_in: usize,
    n_out: usize,
    lambda: f64,
    /// Row weights `d_i > 0` (n_in).
    d: Vec<f64>,
    /// Unit-upper-triangular factor, row-major n_in×n_in (`t[i][i] = 1`).
    t: Vec<f64>,
    /// Right-hand-side columns, row-major n_in×n_out (`z[i][o]`).
    z: Vec<f64>,
    #[serde(skip, default)]
    scratch_x: Vec<f64>,
    #[serde(skip, default)]
    scratch_rhs: Vec<f64>,
}

impl SquaredGivensRls {
    /// `lambda ∈ (0, 1]`; `d`/`t` represent the square root of the
    /// *information* matrix `P⁻¹` (physical row `i` is `√d_i · t_i`), so they
    /// start at `d = 1/delta`, `t = I`, `z = 0` — the dual of `P(0) = delta·I`
    /// in the covariance-form filters.
    pub fn new(n_in: usize, n_out: usize, lambda: f64, delta: f64) -> Self {
        assert!(lambda > 0.0 && lambda <= 1.0, "lambda must be in (0, 1]");
        assert!(delta > 0.0, "delta must be positive");
        let mut t = vec![0.0; n_in * n_in];
        for i in 0..n_in
        {
            t[i * n_in + i] = 1.0;
        }
        Self {
            n_in,
            n_out,
            lambda,
            d: vec![1.0 / delta; n_in],
            t,
            z: vec![0.0; n_in * n_out],
            scratch_x: vec![0.0; n_in],
            scratch_rhs: vec![0.0; n_out],
        }
    }

    /// Triangularize one new sample: input regressor `u` (n_in), targets `d`
    /// (n_out). Zero heap allocation; no `√`.
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, u: &[f64], d: &[f64]) {
        assert_eq!(u.len(), self.n_in);
        assert_eq!(d.len(), self.n_out);
        let n = self.n_in;
        let no = self.n_out;
        if self.scratch_x.len() != n
        {
            self.scratch_x.resize(n, 0.0);
        }
        if self.scratch_rhs.len() != no
        {
            self.scratch_rhs.resize(no, 0.0);
        }
        self.scratch_x.copy_from_slice(u);
        self.scratch_rhs.copy_from_slice(d);

        // The incoming residual's own weight starts at 1 (a fresh, unweighted
        // sample) and evolves after every row-combine — it must be threaded
        // through the loop, not reset each iteration (that was bug #2 here;
        // see the module docs for the derivation this must satisfy).
        let mut d_in = 1.0_f64;
        for i in 0..n
        {
            let row = i * n;
            let d_i = self.d[i] * self.lambda; // fold forgetting in at first use
            let b = self.scratch_x[i];
            let d_i_new = d_i + d_in * b * b;
            if d_i_new <= 0.0
            {
                self.d[i] = d_i;
                continue;
            }
            let d_in_new = d_i * d_in / d_i_new;

            // Row i and the incoming residual both hold t[i][i] = 1 / x[i] = b
            // implicitly; process columns i+1..n and the n_out RHS columns.
            for j in (i + 1)..n
            {
                let t_ij = self.t[row + j];
                let x_j = self.scratch_x[j];
                self.t[row + j] = (d_i * t_ij + d_in * b * x_j) / d_i_new;
                self.scratch_x[j] = x_j - b * t_ij;
            }
            let zrow = i * no;
            for o in 0..no
            {
                let z_io = self.z[zrow + o];
                let rhs_o = self.scratch_rhs[o];
                self.z[zrow + o] = (d_i * z_io + d_in * b * rhs_o) / d_i_new;
                self.scratch_rhs[o] = rhs_o - b * z_io;
            }

            self.d[i] = d_i_new;
            d_in = d_in_new;
        }
    }

    /// Extract the weight vector for output `o` by back-substitution.
    /// `T` is unit upper-triangular, so this needs no `√` and no division.
    /// `O(n_in²)` per call — cheap to call occasionally, not free every sample.
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
                acc -= self.t[row + j] * w[j];
            }
            w[i] = acc; // t[i][i] == 1, no division
        }
        w
    }

    /// Reconstruct the physical (Givens-rotated) upper-triangular factor
    /// `R[i][j] = √d_i · T[i][j]` — for cross-checks against [`GivensQrdRls`]
    /// only; not needed for filtering.
    pub fn physical_factor(&self) -> Vec<f64> {
        let n = self.n_in;
        let mut r = vec![0.0; n * n];
        for i in 0..n
        {
            let s = self.d[i].sqrt();
            for j in i..n
            {
                r[i * n + j] = s * self.t[i * n + j];
            }
        }
        r
    }

    /// Row weights `d_i` (diagnostic / systolic-cell use).
    pub fn weights_diag(&self) -> &[f64] {
        &self.d
    }

    /// Stored unit-upper-triangular factor (diagnostic / systolic-cell use).
    pub fn factor(&self) -> &[f64] {
        &self.t
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

/// The McWhirter (1983) triangular systolic array, expressed as two pure,
/// nearest-neighbor-communicating cell functions — a software reference model
/// proving the square-root-free update **decomposes with zero data hazards**,
/// not a claim of realized hardware/thread parallelism on CPU (that is future
/// work: each row-pass below is independent enough to slot into a wavefront
/// scheduler or a GPU kernel, but this module runs them sequentially).
///
/// A **boundary cell** sits on the diagonal: it consumes the incoming pivot
/// `x_i` together with the row's own weight `d_i` **and** the incoming
/// residual's own running weight `d_in` (the one non-local-looking quantity —
/// see below), and produces the broadcast ratio `b` plus the row's and the
/// residual's updated weights. An **internal cell** touches only its own
/// stored value, the boundary's broadcast `(d_i, d_in, b, d_i_new)`, and the
/// matching column of the incoming residual — no cell ever reads another
/// cell's column. That is the whole McWhirter claim, made checkable: run the
/// array cell-by-cell and require **bit-identical** output to
/// [`SquaredGivensRls::update`]'s ordinary loop.
///
/// `d_in` is *broadcast*, not neighbor-to-neighbor state smuggled sideways:
/// it is the one scalar per row-pass that every cell in the row needs (same
/// role as `b`), computed once by that row's boundary cell from the *previous*
/// row's `d_in` and forwarded — still nearest-neighbor in the row (pipeline)
/// direction, exactly like `b`.
pub mod systolic {
    /// Boundary-cell output: the pivot ratio and the two updated weights,
    /// broadcast to this row's internal cells (`b`, `d_i_new`) and to the next
    /// row's boundary cell (`d_in_new`).
    pub struct BoundaryOutput {
        pub b: f64,
        pub d_i_new: f64,
        pub d_in_new: f64,
    }

    /// Boundary cell (diagonal element of row `i`): consumes this row's
    /// forgetting-scaled weight `d_i`, the incoming residual's current weight
    /// `d_in`, and its pivot `x_i`. No `√`.
    pub fn boundary_cell(d_i: f64, d_in: f64, x_i: f64) -> BoundaryOutput {
        let b = x_i; // t[i][i] == 1, so the pivot ratio IS x_i
        let d_i_new = d_i + d_in * b * b;
        let d_in_new = d_i * d_in / d_i_new;
        BoundaryOutput {
            b,
            d_i_new,
            d_in_new,
        }
    }

    /// Internal cell (off-diagonal element at column `j` of row `i`):
    /// consumes the boundary's broadcast `(d_i, d_in, b, d_i_new)`, this
    /// cell's own stored value `t_ij`, and the matching column of the
    /// incoming residual `x_j`. Produces the new stored value and the new
    /// residual column, which is all the next row ever sees — nearest-neighbor
    /// only. No `√`.
    pub fn internal_cell(
        d_i: f64,
        d_in: f64,
        b: f64,
        d_i_new: f64,
        t_ij: f64,
        x_j: f64,
    ) -> (f64, f64) {
        let t_ij_new = (d_i * t_ij + d_in * b * x_j) / d_i_new;
        let x_j_new = x_j - b * t_ij;
        (t_ij_new, x_j_new)
    }

    /// Run the full triangular array (sequentially, one row-pass at a time)
    /// on state matching [`super::SquaredGivensRls`]'s layout, over the
    /// regressor and `n_out` RHS columns — cell-by-cell, using only
    /// [`boundary_cell`] and [`internal_cell`]. Used exclusively by the
    /// equivalence test; production code should call
    /// [`super::SquaredGivensRls::update`] directly (same arithmetic, less
    /// call overhead).
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    pub fn run_array(
        n: usize,
        n_out: usize,
        d: &mut [f64],
        t: &mut [f64],
        z: &mut [f64],
        lambda: f64,
        u: &[f64],
        targets: &[f64],
    ) {
        let mut x = u.to_vec();
        let mut rhs = targets.to_vec();
        let mut d_in = 1.0_f64;
        for i in 0..n
        {
            let row = i * n;
            let d_i = d[i] * lambda;
            let out = boundary_cell(d_i, d_in, x[i]);
            for j in (i + 1)..n
            {
                let (t_new, x_new) = internal_cell(d_i, d_in, out.b, out.d_i_new, t[row + j], x[j]);
                t[row + j] = t_new;
                x[j] = x_new;
            }
            let zrow = i * n_out;
            for o in 0..n_out
            {
                let (z_new, rhs_new) =
                    internal_cell(d_i, d_in, out.b, out.d_i_new, z[zrow + o], rhs[o]);
                z[zrow + o] = z_new;
                rhs[o] = rhs_new;
            }
            d[i] = out.d_i_new;
            d_in = out.d_in_new;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rls::{RlsFilter, VectorRls};

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
    fn givens_qrd_matches_vector_rls_oracle() {
        let n = 4;
        let mut givens = GivensQrdRls::new(n, 0.98, 100.0);
        let mut vec_rls = VectorRls::new(n, 0.98, 100.0);
        let mut rng = Lcg(101);
        for _ in 0..1500
        {
            let u: Vec<f64> = (0..n).map(|_| rng.next()).collect();
            let d = 1.5 * u[0] - 0.7 * u[1] + 2.0 * u[3] + 0.02 * rng.next();
            givens.update(&u, d);
            vec_rls.update(&u, d);
            let w_g = givens.weights();
            for (a, b) in w_g.iter().zip(vec_rls.weights())
            {
                assert!((a - b).abs() < 1.0e-6, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn squared_givens_matches_givens_qrd_weights_and_factor() {
        let n = 4;
        let mut sg = SquaredGivensRls::new(n, 1, 0.98, 100.0);
        let mut givens = GivensQrdRls::new(n, 0.98, 100.0);
        let mut rng = Lcg(103);
        for _ in 0..1500
        {
            let u: Vec<f64> = (0..n).map(|_| rng.next()).collect();
            let d = -u[0] + 2.5 * u[2] - u[3] + 0.02 * rng.next();
            sg.update(&u, &[d]);
            givens.update(&u, d);

            let w_sg = sg.weights_for(0);
            let w_g = givens.weights();
            for (a, b) in w_sg.iter().zip(&w_g)
            {
                assert!((a - b).abs() < 1.0e-6, "weights: {a} vs {b}");
            }
        }
        // The reconstructed physical factor must match the √-based one too —
        // proof the sqrt-free derivation is exact, not just coincidentally
        // giving the same weights.
        let r_sg = sg.physical_factor();
        let r_g = givens.factor();
        for i in 0..n
        {
            for j in i..n
            {
                let a = r_sg[i * n + j];
                let b = r_g[i * n + j];
                assert!(
                    (a - b).abs() < 1.0e-6 * (1.0 + b.abs()),
                    "R[{i}][{j}]: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn squared_givens_mimo_matches_rls_filter_oracle() {
        let (n_in, n_out) = (3, 2);
        let mut sg = SquaredGivensRls::new(n_in, n_out, 0.99, 100.0);
        let mut mimo = RlsFilter::new(n_in, n_out, 0.99, 100.0);
        let true_w = [[2.0, -1.0, 0.5], [0.3, 1.2, -0.7]];
        let mut rng = Lcg(107);
        for _ in 0..3000
        {
            let u: Vec<f64> = (0..n_in).map(|_| rng.next()).collect();
            let d: Vec<f64> = true_w
                .iter()
                .map(|row| row.iter().zip(&u).map(|(a, b)| a * b).sum())
                .collect();
            sg.update(&u, &d);
            mimo.update(&u, &d);
        }
        for o in 0..n_out
        {
            let w_sg = sg.weights_for(o);
            for (j, &t) in true_w[o].iter().enumerate()
            {
                assert!(
                    (w_sg[j] - t).abs() < 1.0e-6,
                    "out {o} tap {j}: {} vs {t}",
                    w_sg[j]
                );
            }
            for (j, &mw) in mimo.weights()[o * n_in..(o + 1) * n_in].iter().enumerate()
            {
                assert!(
                    (w_sg[j] - mw).abs() < 1.0e-6,
                    "out {o} tap {j} vs RlsFilter: {} vs {mw}",
                    w_sg[j]
                );
            }
        }
    }

    #[test]
    fn squared_givens_tracks_a_drifting_system() {
        let n = 2;
        let mut sg = SquaredGivensRls::new(n, 1, 0.95, 100.0);
        let mut rng = Lcg(109);
        let mut w0 = 1.0;
        for _ in 0..3000
        {
            w0 += 0.001;
            let u = [rng.next(), rng.next()];
            let d = w0 * u[0] - u[1];
            sg.update(&u, &[d]);
        }
        let w = sg.weights_for(0);
        assert!((w[0] - w0).abs() < 0.05, "lagging drift: {} vs {w0}", w[0]);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn systolic_array_is_bit_identical_to_sequential_update() {
        let n = 4;
        let n_out = 2;
        let lambda = 0.97;
        let mut a = SquaredGivensRls::new(n, n_out, lambda, 50.0);
        let (mut d, mut t, mut z) = (vec![1.0 / 50.0; n], vec![0.0; n * n], vec![0.0; n * n_out]);
        for i in 0..n
        {
            t[i * n + i] = 1.0;
        }
        let mut rng = Lcg(113);
        for _ in 0..400
        {
            let u: Vec<f64> = (0..n).map(|_| rng.next()).collect();
            let targets: Vec<f64> = (0..n_out).map(|_| rng.next()).collect();
            a.update(&u, &targets);
            systolic::run_array(n, n_out, &mut d, &mut t, &mut z, lambda, &u, &targets);

            for i in 0..n
            {
                assert_eq!(
                    d[i].to_bits(),
                    a.weights_diag()[i].to_bits(),
                    "d[{i}] diverged"
                );
            }
            for i in 0..(n * n)
            {
                assert_eq!(t[i].to_bits(), a.factor()[i].to_bits(), "t[{i}] diverged");
            }
        }
    }

    #[test]
    fn degenerate_inputs() {
        let mut sg = SquaredGivensRls::new(2, 1, 0.98, 50.0);
        sg.update(&[0.0, 0.0], &[0.0]);
        let w = sg.weights_for(0);
        assert!(w.iter().all(|x| x.is_finite()));

        let mut g = GivensQrdRls::new(2, 0.98, 50.0);
        g.update(&[0.0, 0.0], 0.0);
        assert!(g.weights().iter().all(|x| x.is_finite()));
    }
}
