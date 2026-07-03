//! Surface / form inertia — inertial tolerancing of a whole measured surface
//! rather than a single characteristic (Adragna, Pillet, Samper — the *3D and
//! form* extension, arXiv:1002.0251 / thesis tel-00403876).
//!
//! A surface is measured at `m` points on each of `n` parts. Writing `xᵢⱼ` for
//! the deviation of point `j` on part `i` from the nominal surface (target 0),
//! each point carries its own inertia `Iⱼ = √(δⱼ² + σⱼ²)` (off-centering and
//! dispersion of that point across the batch). The **surface inertia** is the
//! quadratic mean of the point inertias:
//!
//! ```text
//! I_S² = (1/m) Σⱼ Iⱼ² = (1/m) Σⱼ (δⱼ² + σⱼ²) = (1/(m·n)) Σᵢⱼ xᵢⱼ² ,
//! ```
//!
//! i.e. the surface inertia is exactly the root-mean-square deviation of every
//! measured point from nominal — the natural generalisation of the scalar
//! `I = √(E[(X−T)²])` to a whole surface.

use crate::inertia::Inertia;
use serde::{Deserialize, Serialize};

/// A batch of surface measurements: `parts[i][j]` is the deviation of point `j`
/// on part `i` from the nominal surface. Every part must expose the same number
/// of points.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FormBatch {
    parts: Vec<Vec<f64>>,
    points: usize,
}

impl FormBatch {
    /// Build from per-part deviation vectors. Returns `None` if the batch is
    /// empty or the parts do not all have the same, non-zero point count.
    pub fn new(parts: Vec<Vec<f64>>) -> Option<Self> {
        let points = parts.first()?.len();
        if points == 0 || parts.iter().any(|p| p.len() != points)
        {
            return None;
        }
        Some(Self { parts, points })
    }

    /// Number of parts `n`.
    pub fn parts(&self) -> usize {
        self.parts.len()
    }

    /// Number of measured points per part `m`.
    pub fn points(&self) -> usize {
        self.points
    }

    /// The raw per-part deviation vectors, e.g. to feed [`crate::modal`].
    pub fn deviations(&self) -> &[Vec<f64>] {
        &self.parts
    }

    /// Per-point inertia `Iⱼ` (off-centering `δⱼ = mean over parts`, dispersion
    /// `σⱼ = population std over parts`), one entry per measured point.
    pub fn point_inertias(&self) -> Vec<Inertia> {
        let n = self.parts.len().max(1) as f64;
        (0..self.points)
            .map(|j| {
                let mean = self.parts.iter().map(|p| p[j]).sum::<f64>() / n;
                let var = self
                    .parts
                    .iter()
                    .map(|p| (p[j] - mean).powi(2))
                    .sum::<f64>()
                    / n;
                // δⱼ = mean − 0 (nominal target is 0).
                Inertia::new(mean, var.sqrt())
            })
            .collect()
    }

    /// The mean form signature: the average deviation at each point across the
    /// batch (`δⱼ`), i.e. the systematic form defect to be modally analysed.
    pub fn mean_form(&self) -> Vec<f64> {
        let n = self.parts.len().max(1) as f64;
        (0..self.points)
            .map(|j| self.parts.iter().map(|p| p[j]).sum::<f64>() / n)
            .collect()
    }

    /// Surface inertia `I_S = √((1/m) Σⱼ Iⱼ²)` — the quadratic mean of the
    /// point inertias, equal to the RMS of every measured deviation from
    /// nominal.
    pub fn surface_inertia(&self) -> f64 {
        if self.points == 0
        {
            return 0.0;
        }
        let i2: f64 = self
            .point_inertias()
            .iter()
            .map(|i| i.mean_squared_deviation())
            .sum::<f64>()
            / self.points as f64;
        i2.sqrt()
    }

    /// The worst point: `(index, inertia)` of the point with the largest
    /// inertia, or `None` for an empty surface.
    pub fn worst_point(&self) -> Option<(usize, Inertia)> {
        self.point_inertias()
            .into_iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.value().total_cmp(&b.value()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn surface_inertia_is_rms_of_all_deviations() {
        // 3 parts × 2 points.
        let batch =
            FormBatch::new(vec![vec![0.10, -0.20], vec![0.00, 0.10], vec![-0.10, 0.10]]).unwrap();
        // Grand mean of squared deviations:
        let all = [0.10, -0.20, 0.0, 0.10, -0.10, 0.10];
        let want = (all.iter().map(|x| x * x).sum::<f64>() / all.len() as f64).sqrt();
        assert_relative_eq!(batch.surface_inertia(), want, epsilon = 1e-12);
    }

    #[test]
    fn surface_inertia_equals_rms_of_point_inertias() {
        let batch = FormBatch::new(vec![
            vec![0.1, 0.2, -0.1],
            vec![-0.1, 0.0, 0.1],
            vec![0.0, 0.1, 0.2],
        ])
        .unwrap();
        let pis = batch.point_inertias();
        let want =
            (pis.iter().map(|i| i.mean_squared_deviation()).sum::<f64>() / pis.len() as f64).sqrt();
        assert_relative_eq!(batch.surface_inertia(), want, epsilon = 1e-12);
    }

    #[test]
    fn mean_form_and_worst_point() {
        let batch = FormBatch::new(vec![vec![0.0, 0.4], vec![0.0, 0.6]]).unwrap();
        assert_relative_eq!(batch.mean_form()[1], 0.5, epsilon = 1e-12);
        let (idx, worst) = batch.worst_point().unwrap();
        assert_eq!(idx, 1); // second point is badly off-nominal
        assert!(worst.value() > 0.5);
    }

    #[test]
    fn rejects_ragged_or_empty_batches() {
        assert!(FormBatch::new(vec![]).is_none());
        assert!(FormBatch::new(vec![vec![]]).is_none());
        assert!(FormBatch::new(vec![vec![0.1, 0.2], vec![0.1]]).is_none());
    }
}
