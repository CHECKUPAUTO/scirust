//! 3D inertial tolerancing by small-displacement torsors (Adragna, Samper,
//! Pillet — arXiv:1002.0253; SDT after Bourdet & Clément).
//!
//! A rigid feature's small deviation from nominal is a **small-displacement
//! torsor** `θ = {T; R}` at a working origin `O`: a translation `T` and a small
//! rotation `R`. The displacement of a point `M` of the feature is
//!
//! ```text
//! d(M) = T + R × OM ,
//! ```
//!
//! and what tolerancing sees is its projection on the local outward normal
//! `n(M)` — the signed gap between actual and nominal surface:
//!
//! ```text
//! e(M) = d(M)·n(M) = T·n(M) + R·(OM × n(M)) = g(M)·θ ,
//! ```
//!
//! using the triple-product identity `(R×OM)·n = R·(OM×n)` and the **influence
//! vector** `g(M) = [n(M); OM(M)×n(M)] ∈ ℝ⁶`. Stacking the `m` sample points
//! gives `e = G·θ` (`G` is `m×6`).
//!
//! Over a batch of parts the torsor has a mean `θ̄` and covariance `Σ_θ`, so the
//! per-point deviation has mean `g·θ̄` and variance `gᵀΣ_θ g`, and the **surface
//! inertia** (the quadratic mean of the point inertias, see [`crate::form`]) is
//!
//! ```text
//! I_S² = θ̄ᵀ H θ̄ + tr(H Σ_θ) ,   H = (1/m) Σ_M g(M) g(M)ᵀ  (6×6),
//! ```
//!
//! the exact **statistical combination of the location (`T`) and orientation
//! (`R`) deviations** through the geometry matrix `H`. Best-fitting a torsor to
//! a measured deviation field (`θ = (GᵀG)⁻¹Gᵀe`) separates the rigid
//! location+orientation part from the **form residual** `e − G·θ`, which the
//! [`crate::modal`] module then decomposes.

// The 6×6 / 6-vector dense linear algebra below reads most clearly with
// explicit index loops; iterator rewrites would obscure the matrix structure.
#![allow(clippy::needless_range_loop)]

use crate::form::FormBatch;
use serde::{Deserialize, Serialize};

/// A 3-vector.
pub type Vec3 = [f64; 3];

fn dot3(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// A small-displacement torsor `{T; R}` at a fixed working origin: a
/// translation `T` and a small rotation `R` (rotation vector, small-angle).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Torsor {
    /// Translation `T` (location deviation).
    pub translation: Vec3,
    /// Small rotation `R` (orientation deviation, rotation-vector form).
    pub rotation: Vec3,
}

impl Torsor {
    /// A torsor from a translation and a small rotation.
    pub fn new(translation: Vec3, rotation: Vec3) -> Self {
        Self {
            translation,
            rotation,
        }
    }

    /// The zero torsor (feature exactly on nominal).
    pub fn zero() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: [0.0; 3],
        }
    }

    /// The six components `(Tx, Ty, Tz, Rx, Ry, Rz)`.
    pub fn to_array(&self) -> [f64; 6] {
        [
            self.translation[0],
            self.translation[1],
            self.translation[2],
            self.rotation[0],
            self.rotation[1],
            self.rotation[2],
        ]
    }

    /// Build a torsor from the six components `(Tx, Ty, Tz, Rx, Ry, Rz)`.
    pub fn from_array(a: [f64; 6]) -> Self {
        Self {
            translation: [a[0], a[1], a[2]],
            rotation: [a[3], a[4], a[5]],
        }
    }

    /// Displacement `d(M) = T + R × OM` of the point at `om = OM` (from the
    /// working origin `O`).
    pub fn displacement(&self, om: Vec3) -> Vec3 {
        let rx = cross3(self.rotation, om);
        [
            self.translation[0] + rx[0],
            self.translation[1] + rx[1],
            self.translation[2] + rx[2],
        ]
    }

    /// Normal deviation `e(M) = d(M)·n = T·n + R·(OM×n)` — the signed gap
    /// between actual and nominal surface at `M` along its outward normal `n`.
    pub fn normal_deviation(&self, om: Vec3, n: Vec3) -> f64 {
        dot3(self.translation, n) + dot3(self.rotation, cross3(om, n))
    }
}

/// The influence vector `g(M) = [n; OM×n] ∈ ℝ⁶` such that `e(M) = g·θ`.
pub fn influence(om: Vec3, n: Vec3) -> [f64; 6] {
    let c = cross3(om, n);
    [n[0], n[1], n[2], c[0], c[1], c[2]]
}

/// A sampled nominal feature: each point contributes its position `OM`
/// (relative to the working origin) and its outward unit normal `n`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Feature {
    points: Vec<(Vec3, Vec3)>,
}

impl Feature {
    /// A feature from `(OM, n)` samples.
    pub fn new(points: Vec<(Vec3, Vec3)>) -> Self {
        Self { points }
    }

    /// Number of sample points `m`.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the feature has no sample points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The `m×6` influence rows `g(M)`.
    pub fn influence_rows(&self) -> Vec<[f64; 6]> {
        self.points
            .iter()
            .map(|&(om, n)| influence(om, n))
            .collect()
    }

    /// The normal-deviation field `e = G·θ` produced by a torsor.
    pub fn deviation_field(&self, torsor: &Torsor) -> Vec<f64> {
        self.points
            .iter()
            .map(|&(om, n)| torsor.normal_deviation(om, n))
            .collect()
    }

    /// The geometry matrix `H = (1/m) Σ_M g(M) g(M)ᵀ` (symmetric `6×6`).
    pub fn geometry_matrix(&self) -> [[f64; 6]; 6] {
        let mut h = [[0.0f64; 6]; 6];
        for &(om, n) in &self.points
        {
            let g = influence(om, n);
            for i in 0..6
            {
                for j in 0..6
                {
                    h[i][j] += g[i] * g[j];
                }
            }
        }
        let m = self.points.len().max(1) as f64;
        for row in &mut h
        {
            for v in row
            {
                *v /= m;
            }
        }
        h
    }

    /// Least-squares best-fit torsor of a measured deviation field:
    /// `θ = (GᵀG)⁻¹Gᵀe`. Returns `None` if the field length is wrong or the
    /// feature is **under-constrained** (`GᵀG` singular — e.g. a single planar
    /// patch cannot observe in-plane translation or the normal rotation).
    pub fn fit_torsor(&self, deviations: &[f64]) -> Option<Torsor> {
        if deviations.len() != self.points.len() || self.points.is_empty()
        {
            return None;
        }
        // Normal equations: (GᵀG) θ = Gᵀe.
        let mut ata = [[0.0f64; 6]; 6];
        let mut atb = [0.0f64; 6];
        for (&(om, n), &e) in self.points.iter().zip(deviations)
        {
            let g = influence(om, n);
            for i in 0..6
            {
                atb[i] += g[i] * e;
                for j in 0..6
                {
                    ata[i][j] += g[i] * g[j];
                }
            }
        }
        // Symmetric Jacobi (diagonal) scaling before the solve. A zero
        // diagonal is an unobservable DOF (that column of G is null); scaling
        // by the diagonal otherwise makes the normal matrix unit-diagonal, so
        // the singularity test is scale-invariant no matter how far the working
        // origin sits from the feature (the rotation block scales as ‖OM‖²).
        let mut d = [0.0f64; 6];
        for i in 0..6
        {
            if ata[i][i] <= 0.0
            {
                return None;
            }
            d[i] = ata[i][i].sqrt();
        }
        let mut sa = [[0.0f64; 6]; 6];
        let mut sb = [0.0f64; 6];
        for i in 0..6
        {
            sb[i] = atb[i] / d[i];
            for j in 0..6
            {
                sa[i][j] = ata[i][j] / (d[i] * d[j]);
            }
        }
        let y = solve6(&sa, &sb)?;
        Some(Torsor::from_array(std::array::from_fn(|i| y[i] / d[i])))
    }

    /// The form residual `e − G·θ̂` after removing the best-fit rigid
    /// location+orientation — the pure form defect (feed it to
    /// [`crate::modal`]). Returns `None` when [`Feature::fit_torsor`] does.
    pub fn form_residual(&self, deviations: &[f64]) -> Option<Vec<f64>> {
        let torsor = self.fit_torsor(deviations)?;
        let fitted = self.deviation_field(&torsor);
        Some(deviations.iter().zip(fitted).map(|(e, f)| e - f).collect())
    }
}

/// Population mean `θ̄` and covariance `Σ_θ` (`6×6`, divisor `n`) of a batch of
/// torsors.
pub fn torsor_moments(torsors: &[Torsor]) -> ([f64; 6], [[f64; 6]; 6]) {
    let n = torsors.len().max(1) as f64;
    let mut mean = [0.0f64; 6];
    for t in torsors
    {
        let a = t.to_array();
        for i in 0..6
        {
            mean[i] += a[i] / n;
        }
    }
    let mut cov = [[0.0f64; 6]; 6];
    for t in torsors
    {
        let a = t.to_array();
        let d: [f64; 6] = std::array::from_fn(|i| a[i] - mean[i]);
        for i in 0..6
        {
            for j in 0..6
            {
                cov[i][j] += d[i] * d[j] / n;
            }
        }
    }
    (mean, cov)
}

/// Surface inertia of a batch of parts, each described by a torsor on the same
/// feature: assembles every part's deviation field and returns the quadratic
/// mean of the point inertias (via [`FormBatch`]). Empty batch ⇒ 0.
pub fn surface_inertia_from_torsors(feature: &Feature, torsors: &[Torsor]) -> f64 {
    if torsors.is_empty() || feature.is_empty()
    {
        return 0.0;
    }
    let fields: Vec<Vec<f64>> = torsors.iter().map(|t| feature.deviation_field(t)).collect();
    FormBatch::new(fields)
        .map(|b| b.surface_inertia())
        .unwrap_or(0.0)
}

/// Surface inertia by the analytical form `I_S² = θ̄ᵀHθ̄ + tr(HΣ_θ)` — the
/// statistical combination of location and orientation through the geometry
/// matrix `H`. Equal to [`surface_inertia_from_torsors`] but computed from the
/// torsor moments, so it also feeds [`inertia_decomposition`].
pub fn surface_inertia_analytical(feature: &Feature, torsors: &[Torsor]) -> f64 {
    let h = feature.geometry_matrix();
    let (mean, cov) = torsor_moments(torsors);
    (quad_form(&h, &mean) + trace_product(&h, &cov))
        .max(0.0)
        .sqrt()
}

/// Decomposition of the squared surface inertia `I_S²` into the contributions
/// of location, orientation, and their coupling — the meaning of a "statistical
/// combination of location and orientation deviations".
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InertiaDecomposition {
    /// Location (translation-block) contribution to `I_S²`.
    pub location: f64,
    /// Orientation (rotation-block) contribution to `I_S²`.
    pub orientation: f64,
    /// Location↔orientation coupling contribution to `I_S²` (may be negative).
    pub coupling: f64,
}

impl InertiaDecomposition {
    /// Total squared surface inertia `location + orientation + coupling`.
    pub fn total(&self) -> f64 {
        self.location + self.orientation + self.coupling
    }
}

/// Split `I_S² = θ̄ᵀHθ̄ + tr(HΣ_θ)` into location (indices 0..3), orientation
/// (indices 3..6), and coupling parts, using both the mean and the covariance.
pub fn inertia_decomposition(feature: &Feature, torsors: &[Torsor]) -> InertiaDecomposition {
    let h = feature.geometry_matrix();
    let (mean, cov) = torsor_moments(torsors);
    // Mean part θ̄ᵀHθ̄ + variance part tr(HΣ), each block-summed over (i,j).
    let mut loc = 0.0;
    let mut ori = 0.0;
    let mut cpl = 0.0;
    for i in 0..6
    {
        for j in 0..6
        {
            let term = mean[i] * h[i][j] * mean[j] + h[i][j] * cov[j][i];
            let (ti, tj) = (i < 3, j < 3);
            if ti && tj
            {
                loc += term;
            }
            else if !ti && !tj
            {
                ori += term;
            }
            else
            {
                cpl += term;
            }
        }
    }
    InertiaDecomposition {
        location: loc,
        orientation: ori,
        coupling: cpl,
    }
}

/// Quadratic form `xᵀ A x`.
fn quad_form(a: &[[f64; 6]; 6], x: &[f64; 6]) -> f64 {
    let mut s = 0.0;
    for i in 0..6
    {
        for j in 0..6
        {
            s += x[i] * a[i][j] * x[j];
        }
    }
    s
}

/// `tr(A B)` for symmetric `A`, `B`.
fn trace_product(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> f64 {
    let mut s = 0.0;
    for i in 0..6
    {
        for j in 0..6
        {
            s += a[i][j] * b[j][i];
        }
    }
    s
}

/// Solve `A x = b` for a `6×6` `A` by Gaussian elimination with partial
/// pivoting. Returns `None` if `A` is (numerically) singular.
fn solve6(a: &[[f64; 6]; 6], b: &[f64; 6]) -> Option<[f64; 6]> {
    // Working copy of the augmented system.
    let mut m = *a;
    let mut y = *b;
    // Scale for a relative singularity test.
    let scale = (0..6)
        .map(|i| (0..6).map(|j| m[i][j].abs()).fold(0.0, f64::max))
        .fold(0.0, f64::max)
        .max(1e-300);
    for col in 0..6
    {
        // Partial pivot.
        let mut piv = col;
        for r in col + 1..6
        {
            if m[r][col].abs() > m[piv][col].abs()
            {
                piv = r;
            }
        }
        if m[piv][col].abs() <= 1e-12 * scale
        {
            return None; // singular / under-constrained
        }
        m.swap(col, piv);
        y.swap(col, piv);
        // Eliminate below.
        for r in col + 1..6
        {
            let f = m[r][col] / m[col][col];
            for c in col..6
            {
                m[r][c] -= f * m[col][c];
            }
            y[r] -= f * y[col];
        }
    }
    // Back-substitution.
    let mut x = [0.0f64; 6];
    for i in (0..6).rev()
    {
        let mut s = y[i];
        for c in i + 1..6
        {
            s -= m[i][c] * x[c];
        }
        x[i] = s / m[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Three mutually-perpendicular faces of a cube (a 3-2-1 datum) — together
    // they observe all six degrees of freedom, so GᵀG is non-singular. (A
    // single sphere/plane/cylinder is rank-deficient, which is physically
    // correct, so it is unsuitable as a full-rank test feature.)
    fn cube_feature() -> Feature {
        let mut pts = Vec::new();
        for &s in &[-0.5, 0.5]
        {
            for &t in &[-0.5, 0.5]
            {
                pts.push(([1.0, s, t], [1.0, 0.0, 0.0])); // +x face
                pts.push(([s, 1.0, t], [0.0, 1.0, 0.0])); // +y face
                pts.push(([s, t, 1.0], [0.0, 0.0, 1.0])); // +z face
            }
        }
        Feature::new(pts)
    }

    #[test]
    fn displacement_and_normal_deviation_match_hand_calc() {
        // Pure z-translation, z-normal ⇒ e = 1.
        let t = Torsor::new([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]);
        assert_relative_eq!(t.normal_deviation([3.0, -2.0, 0.0], [0.0, 0.0, 1.0]), 1.0);
        // Pure x-rotation, point (0,1,0), z-normal: R×OM = (1,0,0)×(0,1,0)=(0,0,1); ·n = 1.
        let r = Torsor::new([0.0; 3], [1.0, 0.0, 0.0]);
        assert_relative_eq!(r.normal_deviation([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), 1.0);
    }

    #[test]
    fn influence_reproduces_normal_deviation() {
        let torsor = Torsor::new([0.2, -0.1, 0.3], [0.05, -0.02, 0.04]);
        let (om, n) = ([1.0, 2.0, -1.0], {
            let v = [0.3, -0.4, 0.5];
            let nrm = dot3(v, v).sqrt();
            [v[0] / nrm, v[1] / nrm, v[2] / nrm]
        });
        let g = influence(om, n);
        let th = torsor.to_array();
        let via_g: f64 = (0..6).map(|i| g[i] * th[i]).sum();
        assert_relative_eq!(via_g, torsor.normal_deviation(om, n), epsilon = 1e-12);
    }

    #[test]
    fn fit_torsor_round_trips_on_a_full_rank_feature() {
        let feat = cube_feature();
        let truth = Torsor::new([0.10, -0.05, 0.20], [0.03, -0.01, 0.02]);
        let e = feat.deviation_field(&truth);
        let fit = feat.fit_torsor(&e).unwrap();
        for (a, b) in fit.to_array().iter().zip(truth.to_array())
        {
            assert_relative_eq!(a, &b, epsilon = 1e-9);
        }
        // A rigid field has no form residual.
        let resid = feat.form_residual(&e).unwrap();
        assert!(resid.iter().all(|r| r.abs() < 1e-9));
    }

    #[test]
    fn form_residual_isolates_the_non_rigid_part() {
        let feat = cube_feature();
        let rigid = Torsor::new([0.05, 0.0, 0.1], [0.0, 0.02, 0.0]);
        let mut e = feat.deviation_field(&rigid);
        // Inject a form bump the rigid torsor cannot represent.
        e[3] += 0.03;
        let resid = feat.form_residual(&e).unwrap();
        // Residual is non-zero and orthogonal to the influence columns
        // (least-squares normal equations Gᵀ·resid = 0).
        let rows = feat.influence_rows();
        for k in 0..6
        {
            let proj: f64 = rows.iter().zip(&resid).map(|(g, r)| g[k] * r).sum();
            assert!(proj.abs() < 1e-9, "residual not orthogonal to column {k}");
        }
        assert!(resid.iter().any(|r| r.abs() > 1e-4));
    }

    #[test]
    fn fit_torsor_handles_a_non_unit_scaled_feature() {
        // A feature offset ~50 units from the origin, so the rotation block of
        // the normal matrix (∝ ‖OM‖²) dwarfs the translation block. The Jacobi
        // (diagonal) scaling keeps the singularity test scale-invariant and the
        // solve well-behaved, so the fit still recovers the torsor accurately.
        // Same 3-face datum as `cube_feature`, but translated by `c` so ‖OM‖≈c;
        // in-plane variation (not along the normal) keeps all 6 DOF observable.
        let c = 50.0;
        let mut pts = Vec::new();
        for &s in &[-0.5, 0.5]
        {
            for &t in &[-0.5, 0.5]
            {
                pts.push(([c + 1.0, c + s, c + t], [1.0, 0.0, 0.0]));
                pts.push(([c + s, c + 1.0, c + t], [0.0, 1.0, 0.0]));
                pts.push(([c + s, c + t, c + 1.0], [0.0, 0.0, 1.0]));
            }
        }
        let shifted = Feature::new(pts);
        let truth = Torsor::new([0.02, -0.01, 0.03], [1e-3, -2e-3, 3e-3]);
        let e = shifted.deviation_field(&truth);
        let fit = shifted
            .fit_torsor(&e)
            .expect("full-rank feature should fit");
        for (a, b) in fit.to_array().iter().zip(truth.to_array())
        {
            assert_relative_eq!(a, &b, epsilon = 1e-7, max_relative = 1e-7);
        }
    }

    #[test]
    fn planar_feature_is_under_constrained() {
        // All points in the z=0 plane with the same z-normal: only Tz, Rx, Ry
        // are observable ⇒ GᵀG singular ⇒ fit returns None.
        let pts: Vec<(Vec3, Vec3)> = (0..9)
            .map(|k| ([(k % 3) as f64, (k / 3) as f64, 0.0], [0.0, 0.0, 1.0]))
            .collect();
        let feat = Feature::new(pts);
        assert!(feat.fit_torsor(&[0.0; 9]).is_none());
    }

    #[test]
    fn analytical_and_empirical_surface_inertia_agree() {
        let feat = cube_feature();
        let batch = [
            Torsor::new([0.02, -0.01, 0.03], [0.01, 0.0, -0.005]),
            Torsor::new([-0.01, 0.02, 0.01], [-0.005, 0.01, 0.002]),
            Torsor::new([0.0, 0.0, -0.02], [0.002, -0.003, 0.004]),
            Torsor::new([0.03, 0.01, 0.0], [0.0, 0.005, -0.001]),
        ];
        let empirical = surface_inertia_from_torsors(&feat, &batch);
        let analytical = surface_inertia_analytical(&feat, &batch);
        assert_relative_eq!(empirical, analytical, epsilon = 1e-12);
    }

    #[test]
    fn decomposition_sums_to_total_squared_inertia() {
        let feat = cube_feature();
        let batch = [
            Torsor::new([0.02, -0.01, 0.03], [0.01, 0.0, -0.005]),
            Torsor::new([-0.01, 0.02, 0.01], [-0.005, 0.01, 0.002]),
            Torsor::new([0.01, 0.0, -0.02], [0.002, -0.003, 0.004]),
        ];
        let d = inertia_decomposition(&feat, &batch);
        let i_s = surface_inertia_analytical(&feat, &batch);
        assert_relative_eq!(d.total(), i_s * i_s, epsilon = 1e-12);
        assert!(d.location >= 0.0 && d.orientation >= 0.0);
    }

    #[test]
    fn solve6_rejects_singular_systems() {
        // Rank-deficient A (row 5 = row 0).
        let mut a = [[0.0; 6]; 6];
        for i in 0..6
        {
            a[i][i] = 1.0;
        }
        a[5] = a[0];
        assert!(solve6(&a, &[1.0; 6]).is_none());
    }
}
