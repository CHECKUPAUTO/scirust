//! Model pruning — structured and unstructured weight pruning.
//!
//! Supports:
//! - **Magnitude pruning**: remove weights with smallest absolute values.
//! - **Structured pruning**: remove entire rows/columns (neurons).
//! - **Lottery Ticket pruning**: iterative magnitude pruning with rewinding.
//!
//! # Example
//!
//! ```ignore
//! use scirust_core::pruning::prune_magnitude;
//!
//! let mut weights = vec![0.5, 0.01, -0.02, 0.8, -0.001, 0.3];
//! prune_magnitude(&mut weights, 0.5); // prune 50% smallest
//! // weights ≈ [0.5, 0.0, 0.0, 0.8, 0.0, 0.3]
//! ```

/// Pruning strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PruningMethod {
    /// Keep top-k weights by absolute magnitude, zero the rest.
    Magnitude,
    /// Structured: remove entire output neurons (columns in weight matrix).
    StructuredColumns,
    /// Structured: remove entire input features (rows in weight matrix).
    StructuredRows,
}

/// Prune a flat weight vector using magnitude-based pruning.
///
/// `sparsity` is the fraction of weights to zero out (0.0 = none, 0.9 = 90% pruned).
pub fn prune_magnitude(weights: &mut [f32], sparsity: f32) {
    if sparsity <= 0.0 || weights.is_empty()
    {
        return;
    }

    let n_prune = ((weights.len() as f32) * sparsity) as usize;
    if n_prune == 0
    {
        return;
    }

    // Sort indices by absolute weight value
    let mut indexed: Vec<(usize, f32)> = weights
        .iter()
        .enumerate()
        .map(|(i, &w)| (i, w.abs()))
        .collect();

    // Sort by magnitude ascending (smallest first)
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Zero out the smallest n_prune weights
    for (idx, _) in indexed.iter().take(n_prune)
    {
        weights[*idx] = 0.0;
    }
}

/// Prune a weight matrix using structured column pruning.
///
/// Removes columns with smallest L2 norm. `sparsity` fraction of columns
/// are zeroed out entirely.
pub fn prune_structured_columns(weights: &mut [f32], rows: usize, cols: usize, sparsity: f32) {
    if sparsity <= 0.0 || cols == 0
    {
        return;
    }

    let n_prune = ((cols as f32) * sparsity) as usize;
    if n_prune == 0
    {
        return;
    }

    // Compute L2 norm of each column
    let mut col_norms: Vec<(usize, f32)> = (0..cols)
        .map(|c| {
            let norm: f32 = (0..rows)
                .map(|r| {
                    let v = weights[r * cols + c];
                    v * v
                })
                .sum::<f32>()
                .sqrt();
            (c, norm)
        })
        .collect();

    col_norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Zero out smallest columns
    for (col, _) in col_norms.iter().take(n_prune)
    {
        for r in 0..rows
        {
            weights[r * cols + *col] = 0.0;
        }
    }
}

/// **Wanda** one-shot pruning (Sun et al., 2023): prune by the importance
/// `|W_ij| · ‖X_j‖₂` — weight magnitude scaled by the L2 norm of input feature
/// `j`'s calibration activations — comparing **per output row**. No retraining,
/// no gradients; just a forward pass to gather `input_norms`.
///
/// `weights` is `(out × in)` row-major; `input_norms.len() == in_features`;
/// `sparsity` is the fraction of weights zeroed **within each row**.
pub fn prune_wanda(
    weights: &mut [f32],
    out: usize,
    in_features: usize,
    input_norms: &[f32],
    sparsity: f32,
) {
    assert_eq!(weights.len(), out * in_features, "prune_wanda: weight size");
    assert_eq!(
        input_norms.len(),
        in_features,
        "prune_wanda: input_norms len"
    );
    if sparsity <= 0.0 || in_features == 0
    {
        return;
    }
    let n_prune = ((in_features as f32) * sparsity) as usize;
    if n_prune == 0
    {
        return;
    }
    for r in 0..out
    {
        let row = &mut weights[r * in_features..(r + 1) * in_features];
        // Importance per input feature; deterministic tie-break by index.
        let mut scored: Vec<(usize, f32)> = row
            .iter()
            .zip(input_norms)
            .enumerate()
            .map(|(j, (&w, &xn))| (j, w.abs() * xn))
            .collect();
        scored.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        for (j, _) in scored.iter().take(n_prune)
        {
            row[*j] = 0.0;
        }
    }
}

/// Compute current sparsity ratio (fraction of exactly zero weights).
pub fn sparsity_ratio(weights: &[f32]) -> f32 {
    if weights.is_empty()
    {
        return 0.0;
    }
    let zeros = weights.iter().filter(|&&w| w == 0.0).count();
    zeros as f32 / weights.len() as f32
}

/// Iterative Lottery Ticket pruning with rewinding.
///
/// 1. Train to convergence
/// 2. Prune p% of smallest weights
/// 3. Rewind remaining weights to their initial values
/// 4. Repeat
pub struct LotteryTicketPruner {
    /// Fraction to prune each iteration.
    pub prune_fraction: f32,
    /// Number of pruning iterations.
    pub iterations: usize,
    /// Initial weights snapshot (for rewinding).
    initial_weights: Option<Vec<f32>>,
}

impl LotteryTicketPruner {
    pub fn new(prune_fraction: f32, iterations: usize) -> Self {
        Self {
            prune_fraction,
            iterations,
            initial_weights: None,
        }
    }

    /// Save initial weights for rewinding.
    pub fn save_initial(&mut self, weights: &[f32]) {
        self.initial_weights = Some(weights.to_vec());
    }

    /// Prune and rewind: zero smallest weights, restore others to initial values.
    pub fn prune_and_rewind(&self, weights: &mut [f32]) {
        let initial = match &self.initial_weights
        {
            Some(w) => w,
            None => return,
        };

        let current_sparsity = sparsity_ratio(weights);
        if current_sparsity >= 1.0 - (1.0 - self.prune_fraction).powi(self.iterations as i32)
        {
            return; // Already reached target sparsity
        }

        // Prune smallest
        prune_magnitude(weights, self.prune_fraction);

        // Rewind non-zero weights to initial values
        for (w, &init) in weights.iter_mut().zip(initial.iter())
        {
            if *w != 0.0
            {
                *w = init;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magnitude_pruning() {
        let mut weights = vec![0.5, 0.01, -0.02, 0.8, -0.001, 0.3];
        prune_magnitude(&mut weights, 0.5); // Prune 50% = 3 weights
        let zeros = weights.iter().filter(|&&w| w == 0.0).count();
        assert_eq!(zeros, 3);
        assert_eq!(weights[0], 0.5); // Large, kept
        assert_eq!(weights[3], 0.8); // Large, kept
        assert_eq!(weights[5], 0.3); // Medium, kept
    }

    #[test]
    fn test_no_pruning() {
        let mut weights = vec![0.1, 0.2, 0.3];
        let original = weights.clone();
        prune_magnitude(&mut weights, 0.0);
        assert_eq!(weights, original);
    }

    #[test]
    fn test_full_pruning() {
        let mut weights = vec![0.1, 0.2];
        prune_magnitude(&mut weights, 1.0);
        assert!(weights.iter().all(|&w| w == 0.0));
    }

    #[test]
    fn test_structured_column_pruning() {
        // 2x3 matrix: rows=2, cols=3
        // [1.0, 0.1, 0.5]
        // [2.0, 0.2, 0.3]
        let mut weights = vec![1.0, 0.1, 0.5, 2.0, 0.2, 0.3];
        prune_structured_columns(&mut weights, 2, 3, 0.34); // prune 1/3 cols ~33%
        // Column 1 (0.1, 0.2) has lowest L2 norm, should be zeroed
        assert_eq!(weights[1], 0.0);
        assert_eq!(weights[4], 0.0);
        // Column 0 should be kept
        assert_ne!(weights[0], 0.0);
    }

    #[test]
    fn test_sparsity_ratio() {
        let weights = vec![0.0, 1.0, 0.0, 2.0, 0.0, 0.0];
        assert!((sparsity_ratio(&weights) - 4.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_lottery_ticket_rewind() {
        let initial = vec![0.5, 0.01, 0.8, 0.02];
        let mut pruner = LotteryTicketPruner::new(0.5, 1);
        pruner.save_initial(&initial);

        let mut weights = initial.clone();
        // Simulate training: change values
        weights[1] = 0.03;
        weights[3] = 0.04;

        pruner.prune_and_rewind(&mut weights);

        // Weight 1 and 3 are smallest, should be zeroed
        assert_eq!(weights[1], 0.0);
        assert_eq!(weights[3], 0.0);
        // Weight 0 and 2 kept, rewound to initial
        assert_eq!(weights[0], 0.5);
        assert_eq!(weights[2], 0.8);
    }

    /// Wanda prunes by `|w|·‖x‖`, so a *large* weight on a *quiet* input is
    /// dropped while a smaller weight on a loud input is kept — the opposite of
    /// pure magnitude pruning.
    #[test]
    fn test_wanda_differs_from_magnitude() {
        let input_norms = [0.1f32, 10.0];

        // Wanda: metric = [1.0·0.1, 0.5·10] = [0.1, 5.0] → drop col 0.
        let mut w = [1.0f32, 0.5];
        prune_wanda(&mut w, 1, 2, &input_norms, 0.5);
        assert_eq!(w, [0.0, 0.5]);

        // Pure magnitude on the same weights drops the *other* one.
        let mut wm = [1.0f32, 0.5];
        prune_magnitude(&mut wm, 0.5);
        assert_eq!(wm, [1.0, 0.0]);
    }

    /// Wanda zeroes the requested fraction within each output row.
    #[test]
    fn test_wanda_respects_sparsity_per_row() {
        let input_norms = [1.0f32, 1.0, 1.0, 1.0];
        let mut w: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect(); // 2 rows × 4
        prune_wanda(&mut w, 2, 4, &input_norms, 0.5); // drop 2 of 4 per row
        for r in 0..2
        {
            let zeros = w[r * 4..(r + 1) * 4].iter().filter(|&&v| v == 0.0).count();
            assert_eq!(zeros, 2, "row {r} should have 2 zeros");
        }
    }

    /// With uniform input norms, Wanda reduces to magnitude pruning (per row).
    #[test]
    fn test_wanda_uniform_norms_is_magnitude() {
        let input_norms = [1.0f32; 4];
        let mut w = [4.0f32, 1.0, 3.0, 2.0];
        prune_wanda(&mut w, 1, 4, &input_norms, 0.5); // drop the 2 smallest: 1.0, 2.0
        assert_eq!(w, [4.0, 0.0, 3.0, 0.0]);
    }
}
