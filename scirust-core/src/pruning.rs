//! Model pruning — structured and unstructured weight pruning.
//!
//! > ⚠️ **Experimental / no consumers**: this module is not used by any
//! > crate in the workspace. The API may change or be removed; open an
//! > issue if you depend on it.
//!
//! Supports:
//! - **Magnitude pruning**: remove weights with smallest absolute values.
//! - **Structured pruning**: remove entire rows/columns (neurons).
//! - **Lottery Ticket pruning**: iterative magnitude pruning with rewinding.
//!
//! # Example
//!
//! ```
//! use scirust_core::pruning::try_prune_magnitude;
//!
//! let mut weights = vec![0.5, 0.01, -0.02, 0.8, -0.001, 0.3];
//! try_prune_magnitude(&mut weights, 0.5).unwrap();
//! assert_eq!(weights, vec![0.5, 0.0, 0.0, 0.8, 0.0, 0.3]);
//! ```

use crate::error::{Result, SciRustError};

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

fn validate_sparsity(op: &'static str, sparsity: f32) -> Result<()> {
    if !sparsity.is_finite() || !(0.0..=1.0).contains(&sparsity) {
        return Err(SciRustError::InvalidConfig(format!(
            "{op}: sparsity must be finite and in [0, 1], got {sparsity}"
        )));
    }
    Ok(())
}

fn validate_finite_weights(op: &'static str, weights: &[f32]) -> Result<()> {
    if let Some((index, value)) = weights.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(SciRustError::InvalidConfig(format!(
            "{op}: weight at index {index} must be finite, got {value}"
        )));
    }
    Ok(())
}

fn validate_matrix_shape(
    op: &'static str,
    weights_len: usize,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        SciRustError::InvalidConfig(format!("{op}: rows * cols overflows usize"))
    })?;
    if weights_len != expected {
        return Err(SciRustError::InvalidConfig(format!(
            "{op}: expected {expected} weights for shape ({rows}, {cols}), got {weights_len}"
        )));
    }
    Ok(())
}

fn prune_count(len: usize, sparsity: f32) -> usize {
    ((len as f64) * f64::from(sparsity)).floor() as usize
}

/// Fallible magnitude pruning.
///
/// `sparsity` is the fraction of weights to zero out and must be finite and in
/// `[0, 1]`. All weights must be finite so ordering is deterministic.
pub fn try_prune_magnitude(weights: &mut [f32], sparsity: f32) -> Result<()> {
    validate_sparsity("prune_magnitude", sparsity)?;
    validate_finite_weights("prune_magnitude", weights)?;
    if sparsity == 0.0 || weights.is_empty() {
        return Ok(());
    }

    let n_prune = prune_count(weights.len(), sparsity);
    if n_prune == 0 {
        return Ok(());
    }

    let mut indexed: Vec<(usize, f32)> = weights
        .iter()
        .enumerate()
        .map(|(i, &w)| (i, w.abs()))
        .collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));

    for (idx, _) in indexed.iter().take(n_prune) {
        weights[*idx] = 0.0;
    }
    Ok(())
}

/// Prune a flat weight vector using magnitude-based pruning.
///
/// # Panics
///
/// Panics when `sparsity` is non-finite/outside `[0, 1]` or a weight is
/// non-finite. Prefer [`try_prune_magnitude`] at public error boundaries.
pub fn prune_magnitude(weights: &mut [f32], sparsity: f32) {
    try_prune_magnitude(weights, sparsity).expect("invalid magnitude-pruning input");
}

/// Prune a fraction of weights that are still active (non-zero).
fn prune_active_magnitude(weights: &mut [f32], fraction: f32) {
    let mut active: Vec<(usize, f32)> = weights
        .iter()
        .enumerate()
        .filter(|(_, w)| **w != 0.0)
        .map(|(i, &w)| (i, w.abs()))
        .collect();

    let n_prune = prune_count(active.len(), fraction);
    if n_prune == 0 {
        return;
    }

    active.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    for (idx, _) in active.iter().take(n_prune) {
        weights[*idx] = 0.0;
    }
}

/// Fallible structured column pruning for a row-major `(rows, cols)` matrix.
///
/// Removes columns with the smallest L2 norm. The flat weight length must equal
/// `rows * cols`, and `sparsity` must be finite and in `[0, 1]`.
pub fn try_prune_structured_columns(
    weights: &mut [f32],
    rows: usize,
    cols: usize,
    sparsity: f32,
) -> Result<()> {
    validate_sparsity("prune_structured_columns", sparsity)?;
    validate_matrix_shape("prune_structured_columns", weights.len(), rows, cols)?;
    validate_finite_weights("prune_structured_columns", weights)?;
    if sparsity == 0.0 || cols == 0 {
        return Ok(());
    }

    let n_prune = prune_count(cols, sparsity);
    if n_prune == 0 {
        return Ok(());
    }

    let mut col_norms: Vec<(usize, f64)> = (0..cols)
        .map(|c| {
            let sum_sq = (0..rows)
                .map(|r| f64::from(weights[r * cols + c]).powi(2))
                .sum::<f64>();
            (c, sum_sq.sqrt())
        })
        .collect();
    col_norms.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));

    for (col, _) in col_norms.iter().take(n_prune) {
        for r in 0..rows {
            weights[r * cols + *col] = 0.0;
        }
    }
    Ok(())
}

/// Prune a row-major weight matrix by structured columns.
///
/// # Panics
///
/// Panics on invalid matrix dimensions, invalid sparsity, or non-finite weights.
/// Prefer [`try_prune_structured_columns`] at public error boundaries.
pub fn prune_structured_columns(weights: &mut [f32], rows: usize, cols: usize, sparsity: f32) {
    try_prune_structured_columns(weights, rows, cols, sparsity)
        .expect("invalid structured-column pruning input");
}

/// Fallible structured row pruning for a row-major `(rows, cols)` matrix.
///
/// Removes rows with the smallest L2 norm. This is the structured input-feature
/// pruning counterpart to [`try_prune_structured_columns`].
pub fn try_prune_structured_rows(
    weights: &mut [f32],
    rows: usize,
    cols: usize,
    sparsity: f32,
) -> Result<()> {
    validate_sparsity("prune_structured_rows", sparsity)?;
    validate_matrix_shape("prune_structured_rows", weights.len(), rows, cols)?;
    validate_finite_weights("prune_structured_rows", weights)?;
    if sparsity == 0.0 || rows == 0 {
        return Ok(());
    }

    let n_prune = prune_count(rows, sparsity);
    if n_prune == 0 {
        return Ok(());
    }

    let mut row_norms: Vec<(usize, f64)> = (0..rows)
        .map(|r| {
            let sum_sq = weights[r * cols..(r + 1) * cols]
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>();
            (r, sum_sq.sqrt())
        })
        .collect();
    row_norms.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));

    for (row, _) in row_norms.iter().take(n_prune) {
        weights[*row * cols..(*row + 1) * cols].fill(0.0);
    }
    Ok(())
}

/// Prune a row-major weight matrix by structured rows.
///
/// # Panics
///
/// Panics on invalid matrix dimensions, invalid sparsity, or non-finite weights.
/// Prefer [`try_prune_structured_rows`] at public error boundaries.
pub fn prune_structured_rows(weights: &mut [f32], rows: usize, cols: usize, sparsity: f32) {
    try_prune_structured_rows(weights, rows, cols, sparsity)
        .expect("invalid structured-row pruning input");
}

/// Fallible **Wanda** one-shot pruning (Sun et al., 2023).
pub fn try_prune_wanda(
    weights: &mut [f32],
    out: usize,
    in_features: usize,
    input_norms: &[f32],
    sparsity: f32,
) -> Result<()> {
    validate_sparsity("prune_wanda", sparsity)?;
    validate_matrix_shape("prune_wanda", weights.len(), out, in_features)?;
    validate_finite_weights("prune_wanda", weights)?;
    if input_norms.len() != in_features {
        return Err(SciRustError::InvalidConfig(format!(
            "prune_wanda: expected {in_features} input norms, got {}",
            input_norms.len()
        )));
    }
    if let Some((index, value)) = input_norms
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || **value < 0.0)
    {
        return Err(SciRustError::InvalidConfig(format!(
            "prune_wanda: input norm at index {index} must be finite and non-negative, got {value}"
        )));
    }
    if sparsity == 0.0 || in_features == 0 {
        return Ok(());
    }

    let n_prune = prune_count(in_features, sparsity);
    if n_prune == 0 {
        return Ok(());
    }
    for r in 0..out {
        let row = &mut weights[r * in_features..(r + 1) * in_features];
        let mut scored: Vec<(usize, f32)> = row
            .iter()
            .zip(input_norms)
            .enumerate()
            .map(|(j, (&w, &xn))| (j, w.abs() * xn))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        for (j, _) in scored.iter().take(n_prune) {
            row[*j] = 0.0;
        }
    }
    Ok(())
}

/// **Wanda** one-shot pruning (Sun et al., 2023).
///
/// # Panics
///
/// Panics on invalid matrix dimensions, invalid sparsity, non-finite weights,
/// or invalid calibration norms. Prefer [`try_prune_wanda`] at public error
/// boundaries.
pub fn prune_wanda(
    weights: &mut [f32],
    out: usize,
    in_features: usize,
    input_norms: &[f32],
    sparsity: f32,
) {
    try_prune_wanda(weights, out, in_features, input_norms, sparsity)
        .expect("invalid Wanda pruning input");
}

/// Compute current sparsity ratio (fraction of exactly zero weights).
pub fn sparsity_ratio(weights: &[f32]) -> f32 {
    if weights.is_empty() {
        return 0.0;
    }
    let zeros = weights.iter().filter(|&&w| w == 0.0).count();
    zeros as f32 / weights.len() as f32
}

/// Iterative Lottery Ticket pruning with rewinding.
///
/// 1. Train to convergence
/// 2. Prune p% of smallest active weights
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
    /// Fallible constructor validating the per-round pruning fraction.
    pub fn try_new(prune_fraction: f32, iterations: usize) -> Result<Self> {
        validate_sparsity("LotteryTicketPruner::new", prune_fraction)?;
        Ok(Self {
            prune_fraction,
            iterations,
            initial_weights: None,
        })
    }

    /// Construct a Lottery Ticket pruner.
    ///
    /// # Panics
    ///
    /// Panics when `prune_fraction` is non-finite or outside `[0, 1]`.
    pub fn new(prune_fraction: f32, iterations: usize) -> Self {
        Self::try_new(prune_fraction, iterations).expect("invalid Lottery Ticket pruning fraction")
    }

    /// Save initial weights for rewinding.
    pub fn save_initial(&mut self, weights: &[f32]) {
        self.initial_weights = Some(weights.to_vec());
    }

    /// Fallible prune-and-rewind operation.
    pub fn try_prune_and_rewind(&self, weights: &mut [f32]) -> Result<()> {
        validate_sparsity("LotteryTicketPruner::prune_and_rewind", self.prune_fraction)?;
        validate_finite_weights("LotteryTicketPruner::prune_and_rewind", weights)?;
        let initial = self.initial_weights.as_ref().ok_or_else(|| {
            SciRustError::InvalidConfig(
                "LotteryTicketPruner::prune_and_rewind: initial weights were not saved".into(),
            )
        })?;
        if initial.len() != weights.len() {
            return Err(SciRustError::InvalidConfig(format!(
                "LotteryTicketPruner::prune_and_rewind: initial length {} does not match current length {}",
                initial.len(),
                weights.len()
            )));
        }
        validate_finite_weights("LotteryTicketPruner::initial_weights", initial)?;

        let target = 1.0 - (1.0 - self.prune_fraction).powi(self.iterations as i32);
        if sparsity_ratio(weights) >= target {
            return Ok(());
        }

        prune_active_magnitude(weights, self.prune_fraction);
        for (w, &init) in weights.iter_mut().zip(initial.iter()) {
            if *w != 0.0 {
                *w = init;
            }
        }
        Ok(())
    }

    /// Prune and rewind: zero smallest active weights, restore survivors.
    ///
    /// # Panics
    ///
    /// Panics when configuration, current weights, or the saved rewind snapshot
    /// are invalid. Prefer [`Self::try_prune_and_rewind`] at public error boundaries.
    pub fn prune_and_rewind(&self, weights: &mut [f32]) {
        self.try_prune_and_rewind(weights)
            .expect("invalid Lottery Ticket prune-and-rewind input");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magnitude_pruning() {
        let mut weights = vec![0.5, 0.01, -0.02, 0.8, -0.001, 0.3];
        prune_magnitude(&mut weights, 0.5);
        assert_eq!(weights.iter().filter(|&&w| w == 0.0).count(), 3);
        assert_eq!(weights[0], 0.5);
        assert_eq!(weights[3], 0.8);
        assert_eq!(weights[5], 0.3);
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
        let mut weights = vec![1.0, 0.1, 0.5, 2.0, 0.2, 0.3];
        prune_structured_columns(&mut weights, 2, 3, 0.34);
        assert_eq!(weights[1], 0.0);
        assert_eq!(weights[4], 0.0);
        assert_ne!(weights[0], 0.0);
    }

    #[test]
    fn structured_row_pruning_is_implemented() {
        // Row norms: 5, sqrt(2), 10. The middle row must be removed.
        let mut weights = vec![3.0, 4.0, 1.0, 1.0, 6.0, 8.0];
        try_prune_structured_rows(&mut weights, 3, 2, 0.34).unwrap();
        assert_eq!(weights, vec![3.0, 4.0, 0.0, 0.0, 6.0, 8.0]);
    }

    #[test]
    fn structured_pruning_rejects_bad_shape_and_sparsity() {
        let mut weights = vec![1.0, 2.0, 3.0];
        assert!(try_prune_structured_columns(&mut weights, 2, 2, 0.5).is_err());
        assert!(try_prune_structured_rows(&mut weights, 1, 3, 1.1).is_err());
        assert!(try_prune_structured_rows(&mut weights, 1, 3, f32::NAN).is_err());
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
        weights[1] = 0.03;
        weights[3] = 0.04;
        pruner.prune_and_rewind(&mut weights);

        assert_eq!(weights[1], 0.0);
        assert_eq!(weights[3], 0.0);
        assert_eq!(weights[0], 0.5);
        assert_eq!(weights[2], 0.8);
    }

    #[test]
    fn lottery_ticket_prunes_survivors_on_later_rounds() {
        let initial = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut pruner = LotteryTicketPruner::new(0.5, 2);
        pruner.save_initial(&initial);
        let mut weights = initial.clone();

        pruner.prune_and_rewind(&mut weights);
        assert_eq!(weights.iter().filter(|&&w| w == 0.0).count(), 4);
        pruner.prune_and_rewind(&mut weights);
        assert_eq!(weights.iter().filter(|&&w| w == 0.0).count(), 6);
        assert!((sparsity_ratio(&weights) - 0.75).abs() < 1e-6);
        assert_eq!(weights[0], initial[0]);
        assert_eq!(weights[1], initial[1]);
    }

    #[test]
    fn lottery_ticket_rejects_rewind_length_mismatch() {
        let mut pruner = LotteryTicketPruner::new(0.5, 2);
        pruner.save_initial(&[1.0, 2.0, 3.0]);
        let mut weights = [1.0, 2.0];
        assert!(pruner.try_prune_and_rewind(&mut weights).is_err());
    }

    #[test]
    fn lottery_ticket_rejects_missing_snapshot() {
        let pruner = LotteryTicketPruner::new(0.5, 2);
        let mut weights = [1.0, 2.0];
        assert!(pruner.try_prune_and_rewind(&mut weights).is_err());
    }

    #[test]
    fn invalid_sparsity_is_rejected() {
        let mut weights = [1.0, 2.0];
        assert!(try_prune_magnitude(&mut weights, -0.1).is_err());
        assert!(try_prune_magnitude(&mut weights, 1.1).is_err());
        assert!(try_prune_magnitude(&mut weights, f32::INFINITY).is_err());
    }

    #[test]
    fn test_wanda_differs_from_magnitude() {
        let input_norms = [0.1f32, 10.0];
        let mut w = [1.0f32, 0.5];
        prune_wanda(&mut w, 1, 2, &input_norms, 0.5);
        assert_eq!(w, [0.0, 0.5]);

        let mut wm = [1.0f32, 0.5];
        prune_magnitude(&mut wm, 0.5);
        assert_eq!(wm, [1.0, 0.0]);
    }

    #[test]
    fn test_wanda_respects_sparsity_per_row() {
        let input_norms = [1.0f32, 1.0, 1.0, 1.0];
        let mut w: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        prune_wanda(&mut w, 2, 4, &input_norms, 0.5);
        for r in 0..2 {
            let zeros = w[r * 4..(r + 1) * 4]
                .iter()
                .filter(|&&v| v == 0.0)
                .count();
            assert_eq!(zeros, 2, "row {r} should have 2 zeros");
        }
    }

    #[test]
    fn test_wanda_uniform_norms_is_magnitude() {
        let input_norms = [1.0f32; 4];
        let mut w = [4.0f32, 1.0, 3.0, 2.0];
        prune_wanda(&mut w, 1, 4, &input_norms, 0.5);
        assert_eq!(w, [4.0, 0.0, 3.0, 0.0]);
    }
}
