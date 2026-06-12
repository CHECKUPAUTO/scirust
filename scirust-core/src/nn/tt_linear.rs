//! TT-Linear layer: a drop-in replacement for [`scirust_core::nn::Linear`]
//! whose weight matrix is stored as a Tensor-Train (TT) decomposition.
//!
//! ## Memory savings
//!
//! For a `Linear(in, out)` with `in = ∏ I_k` and `out = ∏ O_k` (each of length `d`),
//! a TTLinear stores `d` cores of shape `(r_k, I_k * O_k, r_{k+1})` plus an
//! optional bias. With balanced factorizations and small ranks this can
//! reduce the parameter count from `in * out` down to `O(d * I * O * r^2)`,
//! often a 10-100× compression on transformer-style FFN projections.
//!
//! ## Phase 2: on-tape contraction with gradient flow
//!
//! The forward pass reconstructs the dense weight matrix `W` from the cores
//! and computes `x @ W + b` through a single fused `Op::TtContract` node.
//! The backward computes exact gradients through the reconstruction, flowing
//! back into each core via interleaved gradient projection.

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::Linear;
use crate::nn::Module;

use crate::tn::factorize::{auto_factorize, check_factorization};
use crate::tn::tt_decompose::{TTCores, reconstruct_matrix, tt_decompose_matrix};

/// A `Linear` layer compressed as a Tensor-Train decomposition.
///
/// Implements [`Module`] so it is a drop-in replacement for `Linear` in any
/// `Sequential` or custom forward chain.
#[derive(Clone)]
pub struct TTLinear {
    /// Factorization of `in_features = ∏ in_dims`.
    pub in_dims: Vec<usize>,
    /// Factorization of `out_features = ∏ out_dims`.
    pub out_dims: Vec<usize>,
    /// Bond dimensions, length `d + 1`. `ranks[0] = ranks[d] = 1`.
    pub ranks: Vec<usize>,
    /// TT-cores. Each core `k` has logical shape `(r_k, I_k * O_k, r_{k+1})`
    /// stored as a 2D `Tensor` of shape `(r_k * I_k * O_k, r_{k+1})`.
    pub cores: Vec<Tensor>,
    /// Tape indices for the cores, populated on first forward.
    pub core_indices: Vec<usize>,
    /// Optional bias of shape `(1, out_features)`.
    pub bias: Option<Tensor>,
    /// Tape index for the bias, populated on first forward.
    pub bias_idx: Option<usize>,
    /// Total in_features = ∏ in_dims.
    pub in_features: usize,
    /// Total out_features = ∏ out_dims.
    pub out_features: usize,
}

impl TTLinear {
    /// Total parameter count across all cores plus bias.
    pub fn num_params(&self) -> usize {
        let core_params: usize = self.cores.iter().map(|c| c.data.len()).sum();
        let bias_params = self.bias.as_ref().map_or(0, |b| b.data.len());
        core_params + bias_params
    }

    /// Parameter count of the equivalent dense `Linear` (for compression
    /// ratio reporting).
    pub fn dense_params(&self) -> usize {
        self.in_features * self.out_features + self.out_features
    }

    /// Ratio `dense_params / num_params`. Values > 1 indicate compression.
    pub fn compression_ratio(&self) -> f32 {
        self.dense_params() as f32 / self.num_params().max(1) as f32
    }

    /// Reconstruct the dense weight matrix (used by the Phase 1 forward and
    /// for diagnostics).
    pub fn reconstruct_weight(&self) -> Tensor {
        let core_tnd: Vec<crate::tensor::TensorND> = self
            .cores
            .iter()
            .enumerate()
            .map(|(k, c)| {
                let r_k = self.ranks[k];
                let n_k = self.in_dims[k] * self.out_dims[k];
                let r_next = self.ranks[k + 1];
                crate::tensor::TensorND::new(c.data.clone(), vec![r_k, n_k, r_next])
            })
            .collect();
        let mode_dims: Vec<usize> = (0..self.in_dims.len())
            .map(|k| self.in_dims[k] * self.out_dims[k])
            .collect();
        let tt = TTCores {
            cores: core_tnd,
            ranks: self.ranks.clone(),
            mode_dims,
        };
        let w_flat = reconstruct_matrix(&tt, &self.in_dims, &self.out_dims);
        Tensor {
            rows: self.in_features,
            cols: self.out_features,
            data: w_flat,
        }
    }

    /// Build a `TTLinear` directly from already-computed cores (advanced API).
    /// `cores[k]` must have shape `(ranks[k] * in_dims[k] * out_dims[k], ranks[k+1])`.
    pub fn from_cores(
        in_dims: Vec<usize>,
        out_dims: Vec<usize>,
        ranks: Vec<usize>,
        cores: Vec<Tensor>,
        bias: Option<Tensor>,
    ) -> Self {
        let d = in_dims.len();
        assert_eq!(
            out_dims.len(),
            d,
            "in_dims and out_dims must have the same length"
        );
        assert_eq!(ranks.len(), d + 1, "ranks must have length d+1");
        assert_eq!(ranks[0], 1, "ranks[0] must be 1");
        assert_eq!(ranks[d], 1, "ranks[d] must be 1");
        assert_eq!(cores.len(), d, "need {d} cores");
        for k in 0..d
        {
            let expected_rows = ranks[k] * in_dims[k] * out_dims[k];
            let expected_cols = ranks[k + 1];
            assert_eq!(
                cores[k].rows, expected_rows,
                "core {k} rows = {} but expected {expected_rows}",
                cores[k].rows
            );
            assert_eq!(
                cores[k].cols, expected_cols,
                "core {k} cols = {} but expected {expected_cols}",
                cores[k].cols
            );
        }

        let in_features = in_dims.iter().product();
        let out_features = out_dims.iter().product();
        Self {
            in_dims,
            out_dims,
            ranks,
            cores,
            core_indices: Vec::new(),
            bias,
            bias_idx: None,
            in_features,
            out_features,
        }
    }

    /// Register cores and bias as fresh inputs on the tape.
    /// Called every forward() to ensure indices are valid for the current tape.
    fn register_params(&mut self, tape: &Tape) {
        self.core_indices = self
            .cores
            .iter()
            .map(|c| tape.input(c.clone()).idx())
            .collect();
        self.bias_idx = self.bias.as_ref().map(|b| tape.input(b.clone()).idx());
    }
}

impl Module for TTLinear {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        self.register_params(tape);

        let mut core_vars: Vec<Var<'t>> = Vec::new();
        for &idx in &self.core_indices
        {
            core_vars.push(Var::new(tape, idx));
        }
        let b_var = self.bias_idx.map(|idx| Var::new(tape, idx));

        input.tt_contract(
            core_vars,
            b_var,
            self.in_dims.clone(),
            self.out_dims.clone(),
            self.ranks.clone(),
        )
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut idx = self.core_indices.clone();
        if let Some(b) = self.bias_idx
        {
            idx.push(b);
        }
        idx
    }

    fn sync(&mut self, tape: &Tape) {
        // Pull updated values back from the tape into our local cores so the
        // next forward sees the post-optimizer-step parameters.
        if self.core_indices.len() == self.cores.len()
        {
            for (k, &idx) in self.core_indices.iter().enumerate()
            {
                self.cores[k] = tape.value(idx);
            }
        }
        if let (Some(b_idx), Some(_)) = (self.bias_idx, &self.bias)
        {
            self.bias = Some(tape.value(b_idx));
        }
    }
}

// ---------------------------------------------------------------------------
// Public decomposition API
// ---------------------------------------------------------------------------

/// Decompose a `Linear` into TT-Linear form with explicit mode dimensions.
///
/// # Parameters
/// - `linear`: source layer to compress.
/// - `in_dims`: factorization of `linear.in_features`. Must satisfy
///   `in_dims.iter().product() == linear.in_features`.
/// - `out_dims`: factorization of `linear.out_features`. Same length as `in_dims`.
/// - `max_rank`: cap on each bond dimension `r_k`. Smaller = more compression
///   but more reconstruction error.
/// - `tolerance`: relative SVD truncation threshold. Singular values below
///   `tolerance * sigma_max` are dropped. Use `0.0` to truncate only by rank.
///
/// # Returns
/// A `TTLinear` with cores carrying the decomposed weight. The bias from
/// `linear` is copied unchanged.
pub fn tt_decompose(
    linear: &Linear,
    in_dims: &[usize],
    out_dims: &[usize],
    max_rank: usize,
    tolerance: f32,
) -> TTLinear {
    assert!(
        check_factorization(in_dims, linear.in_features),
        "in_dims product {} != linear.in_features {}",
        in_dims.iter().product::<usize>(),
        linear.in_features
    );
    assert!(
        check_factorization(out_dims, linear.out_features),
        "out_dims product {} != linear.out_features {}",
        out_dims.iter().product::<usize>(),
        linear.out_features
    );
    assert_eq!(
        in_dims.len(),
        out_dims.len(),
        "in/out_dims must have the same length"
    );

    let tt = tt_decompose_matrix(&linear.weight.data, in_dims, out_dims, max_rank, tolerance);

    // Convert each TensorND core (shape (r_k, I_k * O_k, r_{k+1})) into the
    // 2D Tensor representation (r_k * I_k * O_k, r_{k+1}) expected by TTLinear.
    let cores: Vec<Tensor> = tt
        .cores
        .iter()
        .enumerate()
        .map(|(k, c)| {
            let r_k = tt.ranks[k];
            let n_k = in_dims[k] * out_dims[k];
            let r_next = tt.ranks[k + 1];
            Tensor {
                rows: r_k * n_k,
                cols: r_next,
                data: c.data.clone(),
            }
        })
        .collect();

    TTLinear::from_cores(
        in_dims.to_vec(),
        out_dims.to_vec(),
        tt.ranks,
        cores,
        Some(linear.bias.clone()),
    )
}

/// Decompose a `Linear` into TT-Linear form with **automatic balanced
/// factorization** of in/out features.
///
/// Convenience wrapper around [`tt_decompose`] that calls [`auto_factorize`]
/// on both `linear.in_features` and `linear.out_features`.
///
/// # Parameters
/// - `linear`: source layer to compress.
/// - `n_factors`: number of modes `d`. Typical: 2 or 3.
/// - `max_rank`: cap on each bond dimension.
/// - `tolerance`: relative SVD truncation threshold.
pub fn tt_decompose_auto(
    linear: &Linear,
    n_factors: usize,
    max_rank: usize,
    tolerance: f32,
) -> TTLinear {
    let in_dims = auto_factorize(linear.in_features, n_factors);
    let out_dims = auto_factorize(linear.out_features, n_factors);
    tt_decompose(linear, &in_dims, &out_dims, max_rank, tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frob_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn frob_norm(a: &[f32]) -> f32 {
        a.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn test_tt_decompose_reconstructs_weight() {
        let mut rng = crate::nn::rng::PcgEngine::new(42);
        let mut linear = Linear::new(
            6,
            4,
            &crate::nn::init::Zeros,
            &crate::nn::init::Zeros,
            &mut rng,
        );
        // Fill weight with a non-trivial pattern.
        for i in 0..6
        {
            for j in 0..4
            {
                linear.weight.data[i * 4 + j] = ((i * 4 + j) as f32).sin();
            }
        }
        let tt = tt_decompose(&linear, &[2, 3], &[2, 2], 100, 0.0);
        let w_back = tt.reconstruct_weight();
        let rel = frob_err(&linear.weight.data, &w_back.data) / frob_norm(&linear.weight.data);
        assert!(rel < 1e-3, "rel err = {rel}");
    }

    #[test]
    fn test_tt_decompose_auto() {
        let mut rng = crate::nn::rng::PcgEngine::new(42);
        let mut linear = Linear::new(
            8,
            16,
            &crate::nn::init::Zeros,
            &crate::nn::init::Zeros,
            &mut rng,
        );
        for i in 0..(8 * 16)
        {
            linear.weight.data[i] = ((i as f32) * 0.13).cos();
        }
        let tt = tt_decompose_auto(&linear, 2, 100, 0.0);
        assert_eq!(tt.in_dims.iter().product::<usize>(), 8);
        assert_eq!(tt.out_dims.iter().product::<usize>(), 16);
        let w_back = tt.reconstruct_weight();
        let rel = frob_err(&linear.weight.data, &w_back.data) / frob_norm(&linear.weight.data);
        assert!(rel < 1e-3);
    }

    #[test]
    fn test_compression_ratio() {
        let mut rng = crate::nn::rng::PcgEngine::new(42);
        let linear = Linear::new(
            16,
            16,
            &crate::nn::init::Zeros,
            &crate::nn::init::Zeros,
            &mut rng,
        );
        // The default weight is all-zeros so compression won't be meaningful
        // but the formula must be well-defined.
        let tt = tt_decompose_auto(&linear, 2, 4, 0.0);
        let ratio = tt.compression_ratio();
        assert!(ratio.is_finite());
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_parameter_indices() {
        let mut rng = crate::nn::rng::PcgEngine::new(42);
        let linear = Linear::new(
            6,
            4,
            &crate::nn::init::Zeros,
            &crate::nn::init::Zeros,
            &mut rng,
        );
        let mut tt = tt_decompose(&linear, &[2, 3], &[2, 2], 100, 0.0);
        let tape = Tape::new();
        let _ = tt.forward(
            &tape,
            tape.input(Tensor {
                rows: 1,
                cols: 6,
                data: vec![0.0; 6],
            }),
        );
        let idx = tt.parameter_indices();
        assert_eq!(idx.len(), tt.cores.len() + 1);
    }

    #[test]
    fn tt_forward_matches_dense_structured() {
        // Use structured (low TT-rank) weights: separable function that
        // the rank-3 TT decomposition can reproduce accurately.
        let outer = 12;
        let inner = 8;
        let in_dims = vec![3, 4];
        let out_dims = vec![2, 4];
        let tt_rank = 3;

        let mut linear = Linear::new(
            outer,
            inner,
            &crate::nn::init::Zeros,
            &crate::nn::init::Zeros,
            &mut crate::nn::rng::PcgEngine::new(42),
        );
        for i in 0..outer
        {
            for j in 0..inner
            {
                linear.weight.data[i * inner + j] = ((i * inner + j) as f32).sin();
            }
        }

        let mut tt = tt_decompose(&linear, &in_dims, &out_dims, tt_rank, 0.0);
        let tape = Tape::new();
        let batch = 4;

        for batch_idx in 0..5
        {
            let input_data: Vec<f32> = (0..batch * outer)
                .map(|k| ((k + batch_idx * 37) as f32).cos())
                .collect();
            let input_t = Tensor {
                rows: batch,
                cols: outer,
                data: input_data.clone(),
            };

            let dense_out = linear.forward(&tape, tape.input(input_t.clone()));
            let tt_out = tt.forward(&tape, tape.input(input_t));

            let dense_t = tape.value(dense_out.idx());
            let tt_t = tape.value(tt_out.idx());
            let rel = frob_err(&dense_t.data, &tt_t.data) / (frob_norm(&dense_t.data) + 1e-12);
            assert!(rel < 1e-3, "TT forward rel error exceeds threshold: {rel}");
        }
    }

    #[test]
    fn tt_backward_gradient_flows() {
        let mut linear = Linear::new(
            6,
            4,
            &crate::nn::init::Zeros,
            &crate::nn::init::Zeros,
            &mut crate::nn::rng::PcgEngine::new(42),
        );
        for i in 0..(6 * 4)
        {
            linear.weight.data[i] = ((i as f32) * 0.13).cos();
        }
        let mut tt = tt_decompose(&linear, &[2, 3], &[2, 2], 100, 0.0);
        let tape = Tape::new();

        let x = tape.input(Tensor {
            rows: 2,
            cols: 6,
            data: vec![0.5; 12],
        });
        let y = tt.forward(&tape, x);
        let loss = y.sum();
        tape.backward(loss.idx());

        for (i, idx) in tt.parameter_indices().iter().enumerate()
        {
            let grad = tape.grad(*idx);
            assert!(
                grad.data.iter().any(|&g| g.abs() > 1e-6),
                "core {i} (idx {idx}) has zero gradient"
            );
        }
    }
}
