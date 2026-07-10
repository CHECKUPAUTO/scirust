//! Const-generic, stack-allocated RLS — the embedded-friendly twin of
//! [`crate::rls::RlsFilter`].
//!
//! Dimensions are compile-time constants, every buffer is a plain array on the
//! stack (or inside the struct, wherever the caller puts it), and the module
//! uses **`core` only** — no `Vec`, no allocator, no `std` — so the type can be
//! lifted verbatim into a `no_std` firmware build. Because LLVM knows `N_IN`
//! and `N_OUT` at compile time, the update loops are fully unrollable and
//! auto-vectorizable, which the heap version's runtime dimensions preclude.
//!
//! The arithmetic is **operation-for-operation identical** to
//! [`crate::rls::RlsFilter::update`] (same accumulation order, same on-the-fly
//! gain, same forced symmetrization), so the two produce **bit-identical**
//! weight trajectories — verified by test.

/// Multi-channel RLS with compile-time dimensions, entirely stack-resident.
///
/// `N_IN` input channels, `N_OUT` output channels; learns the `N_OUT × N_IN`
/// weight matrix online with forgetting factor `λ`.
#[derive(Debug, Clone)]
pub struct RlsFilterConst<const N_IN: usize, const N_OUT: usize> {
    lambda: f64,
    /// Weight matrix, `w[i][j]` = output `i`, input `j`.
    w: [[f64; N_IN]; N_OUT],
    /// Inverse input covariance.
    p: [[f64; N_IN]; N_IN],
}

impl<const N_IN: usize, const N_OUT: usize> RlsFilterConst<N_IN, N_OUT> {
    /// Create the filter with forgetting factor `lambda ∈ (0, 1]` and initial
    /// covariance `P(0) = delta·I`.
    pub fn new(lambda: f64, delta: f64) -> Self {
        assert!(lambda > 0.0 && lambda <= 1.0, "lambda must be in (0, 1]");
        let mut p = [[0.0; N_IN]; N_IN];
        let mut i = 0;
        while i < N_IN
        {
            p[i][i] = delta;
            i += 1;
        }
        Self {
            lambda,
            w: [[0.0; N_IN]; N_OUT],
            p,
        }
    }

    /// Filter one sample. Returns the a-priori error `e = d − w·u`.
    /// No heap, no allocator: everything lives on the stack.
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, u: &[f64; N_IN], d: &[f64; N_OUT]) -> [f64; N_OUT] {
        // Error: e = d - w·u (prediction folded in).
        let mut e = [0.0; N_OUT];
        for i in 0..N_OUT
        {
            let mut d_hat = 0.0;
            for j in 0..N_IN
            {
                d_hat += self.w[i][j] * u[j];
            }
            e[i] = d[i] - d_hat;
        }

        // Gain numerator pu = P·u and denominator λ + uᵀ·P·u.
        let mut pu = [0.0; N_IN];
        for i in 0..N_IN
        {
            let mut acc = 0.0;
            for j in 0..N_IN
            {
                acc += self.p[i][j] * u[j];
            }
            pu[i] = acc;
        }
        let mut upu = 0.0;
        for i in 0..N_IN
        {
            upu += u[i] * pu[i];
        }
        let denom = self.lambda + upu;

        // Weight update: w += k ⊗ e with k[j] = pu[j]/denom.
        for i in 0..N_OUT
        {
            let ei = e[i];
            for j in 0..N_IN
            {
                self.w[i][j] += pu[j] / denom * ei;
            }
        }

        // Covariance update + forced symmetrization, same as the heap version.
        for i in 0..N_IN
        {
            let ki = pu[i] / denom;
            for j in 0..N_IN
            {
                self.p[i][j] = (self.p[i][j] - ki * pu[j]) / self.lambda;
            }
        }
        for i in 0..N_IN
        {
            for j in (i + 1)..N_IN
            {
                let avg = (self.p[i][j] + self.p[j][i]) * 0.5;
                self.p[i][j] = avg;
                self.p[j][i] = avg;
            }
        }

        e
    }

    /// Current weight matrix.
    pub fn weights(&self) -> &[[f64; N_IN]; N_OUT] {
        &self.w
    }

    /// Current inverse covariance matrix.
    pub fn covariance_inv(&self) -> &[[f64; N_IN]; N_IN] {
        &self.p
    }

    /// Forgetting factor.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rls::RlsFilter;

    /// Deterministic LCG for reproducible pseudo-random inputs, `core`-only.
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
    fn const_variant_is_bit_identical_to_heap_variant() {
        // Same math, same order ⇒ the trajectories must match to the last bit.
        const N_IN: usize = 3;
        const N_OUT: usize = 2;
        let mut heap = RlsFilter::new(N_IN, N_OUT, 0.97, 50.0);
        let mut stack: RlsFilterConst<N_IN, N_OUT> = RlsFilterConst::new(0.97, 50.0);
        let mut rng = Lcg(12345);
        for _ in 0..500
        {
            let u = [rng.next(), rng.next(), rng.next()];
            let d = [
                2.0 * u[0] - u[1] + 0.5 * u[2] + 0.01 * rng.next(),
                -u[0] + 3.0 * u[2] + 0.01 * rng.next(),
            ];
            let e_heap: Vec<f64> = heap.update(&u, &d).to_vec();
            let e_stack = stack.update(&u, &d);
            for (a, b) in e_heap.iter().zip(e_stack.iter())
            {
                assert_eq!(a.to_bits(), b.to_bits(), "error diverged");
            }
        }
        // Weights bit-identical too.
        let wh = heap.weights();
        let ws = stack.weights();
        for i in 0..N_OUT
        {
            for j in 0..N_IN
            {
                assert_eq!(
                    wh[i * N_IN + j].to_bits(),
                    ws[i][j].to_bits(),
                    "weight [{i}][{j}] diverged"
                );
            }
        }
    }

    #[test]
    fn const_variant_converges_to_true_system() {
        let mut rls: RlsFilterConst<2, 1> = RlsFilterConst::new(0.99, 100.0);
        let inputs = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-1.0, 2.0], [2.0, -1.0]];
        for _ in 0..200
        {
            for u in &inputs
            {
                let d = [3.0 * u[0] - 2.0 * u[1]];
                rls.update(u, &d);
            }
        }
        let w = rls.weights();
        assert!((w[0][0] - 3.0).abs() < 0.15, "w00 = {}", w[0][0]);
        assert!((w[0][1] + 2.0).abs() < 0.15, "w01 = {}", w[0][1]);
    }
}
