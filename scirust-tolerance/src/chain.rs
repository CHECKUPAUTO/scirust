//! 1D tolerance chains (*chaînes de cotes*): analysis and allocation.
//!
//! A linear assembly relates a functional requirement `Y` to component
//! characteristics `Xᵢ` through influence coefficients `αᵢ` (`±1` for a simple
//! gap stack, a lever ratio otherwise):
//!
//! ```text
//! Y = Σ αᵢ Xᵢ .
//! ```
//!
//! **Analysis** (bottom-up) combines the components' inertias into the
//! assembly's. With independent components,
//!
//! ```text
//! δ_Y = Σ αᵢ δᵢ ,   σ_Y² = Σ αᵢ² σᵢ² ,   I_Y = √(δ_Y² + σ_Y²),
//! ```
//!
//! and, over the space of how each component may split its inertia budget `Iᵢ`
//! between off-centering and dispersion, the two bounds are
//!
//! ```text
//! statistical (typical):   I_Y = √(Σ αᵢ² Iᵢ²)      (off-centerings independent)
//! worst case (guaranteed): I_Y = Σ |αᵢ| Iᵢ         (off-centerings all aligned).
//! ```
//!
//! The worst-case bound is the crux of inertial tolerancing: because `Iᵢ`
//! caps *both* the centering and the spread, `Σ|αᵢ|Iᵢ` is a genuine guarantee
//! on the assembly's RMS-to-target — something a root-sum-square on σ alone
//! cannot give, since it silently assumes every component is centred.
//!
//! **Allocation** (top-down) distributes an assembly inertia budget
//! `I_Y = (tolerance interval)/6` down to the components. Cross-checked against
//! Table 2 of arXiv:1002.0270 (`Y = X₁ − X₂ − X₃ − X₄ − X₅`, `R_Y = 1`):
//!
//! | method                         | `Iᵢ` (uniform, `n` comps)      | value (`n=5`) |
//! |--------------------------------|--------------------------------|---------------|
//! | worst case                     | `R_Y / (6·n)`                  | `0.033`       |
//! | statistical                    | `R_Y / (6·√n)`                 | `0.075`       |
//! | guaranteed `Cpk` (`ICC=1.25`)  | `R_Y / (6·ICC·√n)`             | `0.060`       |

use crate::inertia::Inertia;
use serde::{Deserialize, Serialize};

/// A component of a tolerance chain whose inertia budget `Iᵢ` is known.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contributor {
    /// Human-readable name (`"X1"`, `"shaft"`, …).
    pub name: String,
    /// Influence coefficient `αᵢ` (signed sensitivity of `Y` to this
    /// component; `±1` for a plain additive/subtractive stack).
    pub coeff: f64,
    /// Inertia budget `Iᵢ` of the component.
    pub inertia: f64,
}

impl Contributor {
    /// A contributor with the given name, coefficient and inertia.
    pub fn new(name: impl Into<String>, coeff: f64, inertia: f64) -> Self {
        Self {
            name: name.into(),
            coeff,
            inertia,
        }
    }
}

/// A component whose full state (off-centering `δᵢ` and dispersion `σᵢ`) is
/// known, e.g. from a production sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContributorState {
    /// Human-readable name.
    pub name: String,
    /// Influence coefficient `αᵢ`.
    pub coeff: f64,
    /// Off-centering `δᵢ = μᵢ − Tᵢ`.
    pub off_centering: f64,
    /// Dispersion `σᵢ`.
    pub sigma: f64,
}

impl ContributorState {
    /// A contributor state with the given fields.
    pub fn new(name: impl Into<String>, coeff: f64, off_centering: f64, sigma: f64) -> Self {
        Self {
            name: name.into(),
            coeff,
            off_centering,
            sigma: sigma.abs(),
        }
    }
}

/// Statistical (typical) assembly inertia `I_Y = √(Σ αᵢ² Iᵢ²)`, valid when
/// component off-centerings vary independently from batch to batch.
pub fn assembly_inertia_statistical(contributors: &[Contributor]) -> f64 {
    contributors
        .iter()
        .map(|c| c.coeff * c.coeff * c.inertia * c.inertia)
        .sum::<f64>()
        .sqrt()
}

/// Worst-case (guaranteed) assembly inertia `I_Y = Σ |αᵢ| Iᵢ` — the largest
/// RMS-to-target the assembly can reach for the given component inertia
/// budgets, attained when every off-centering aligns.
pub fn assembly_inertia_worst_case(contributors: &[Contributor]) -> f64 {
    contributors.iter().map(|c| c.coeff.abs() * c.inertia).sum()
}

/// Full assembly state from known component states: propagates the
/// off-centering (`δ_Y = Σ αᵢ δᵢ`) and combines dispersions in quadrature
/// (`σ_Y² = Σ αᵢ² σᵢ²`), returning the resulting [`Inertia`].
pub fn assembly_state(contributors: &[ContributorState]) -> Inertia {
    let delta = contributors
        .iter()
        .map(|c| c.coeff * c.off_centering)
        .sum::<f64>();
    let var = contributors
        .iter()
        .map(|c| c.coeff * c.coeff * c.sigma * c.sigma)
        .sum::<f64>();
    Inertia::new(delta, var.sqrt())
}

/// The inertial coefficient of correction `ICC = √(Cpk² + n/9)`, the factor by
/// which the statistical inertia allocation is tightened so the assembly
/// resultant is guaranteed a capability index of at least `cpk`
/// (arXiv:1002.0270, eq. for `n`-component uniform chains).
pub fn icc(cpk: f64, n: usize) -> f64 {
    (cpk * cpk + n as f64 / 9.0).sqrt()
}

/// Inverse of [`icc`]: the `Cpk` guaranteed on an `n`-component assembly when
/// components are allocated with a given `ICC`, `Cpk = √(ICC² − n/9)`
/// (clamped at 0).
pub fn guaranteed_cpk(icc: f64, n: usize) -> f64 {
    (icc * icc - n as f64 / 9.0).max(0.0).sqrt()
}

/// How an assembly inertia budget is shared out to the components.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Allocation {
    /// `Iᵢ = I_Y / Σ|αⱼ|` — the sum of worst-aligned contributions equals the
    /// budget. Tightest; guarantees the budget even if every part is fully
    /// off-centre.
    WorstCase,
    /// `Iᵢ = I_Y / √(Σ αⱼ²)` — the quadratic (root-sum-square) combination of
    /// contributions equals the budget. The default inertial allocation.
    Statistical,
    /// `Iᵢ = βᵢ · I_Y / √(Σ (αⱼβⱼ)²)` — statistical allocation weighted by
    /// per-component feasibility weights `βᵢ` (larger `βᵢ` ⇒ looser share, for
    /// harder-to-hold components). `βᵢ = 1` for all reduces to
    /// [`Allocation::Statistical`]. The weight slice must match the
    /// contributor count.
    Weighted(Vec<f64>),
    /// `Iᵢ = I_Y / (ICC · √(Σ αⱼ²))` with `ICC = √(cpk² + n/9)` — statistical
    /// allocation tightened so the assembly resultant is guaranteed a
    /// capability index of at least `cpk`.
    GuaranteedCpk(f64),
}

/// Distribute an assembly inertia budget `i_max_assembly` (typically
/// `R_Y / 6`) across `coeffs` (the influence coefficients `αᵢ`) by `method`,
/// returning the per-component inertia budgets `Iᵢ` aligned with `coeffs`.
///
/// Returns an empty vector for empty input, or `Err` when a
/// [`Allocation::Weighted`] weight slice does not match the coefficient count.
pub fn allocate(
    i_max_assembly: f64,
    coeffs: &[f64],
    method: &Allocation,
) -> Result<Vec<f64>, AllocationError> {
    let n = coeffs.len();
    if n == 0
    {
        return Ok(Vec::new());
    }
    match method
    {
        Allocation::WorstCase =>
        {
            let denom: f64 = coeffs.iter().map(|a| a.abs()).sum();
            Ok(vec![i_max_assembly / denom; n])
        },
        Allocation::Statistical =>
        {
            let denom = coeffs.iter().map(|a| a * a).sum::<f64>().sqrt();
            Ok(vec![i_max_assembly / denom; n])
        },
        Allocation::GuaranteedCpk(cpk) =>
        {
            let denom = coeffs.iter().map(|a| a * a).sum::<f64>().sqrt();
            let k = icc(*cpk, n);
            Ok(vec![i_max_assembly / (k * denom); n])
        },
        Allocation::Weighted(betas) =>
        {
            if betas.len() != n
            {
                return Err(AllocationError::WeightCountMismatch {
                    coeffs: n,
                    weights: betas.len(),
                });
            }
            let denom = coeffs
                .iter()
                .zip(betas)
                .map(|(a, b)| (a * b) * (a * b))
                .sum::<f64>()
                .sqrt();
            Ok(betas.iter().map(|b| b * i_max_assembly / denom).collect())
        },
    }
}

/// Error returned by [`allocate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocationError {
    /// A [`Allocation::Weighted`] weight slice length differed from the number
    /// of coefficients.
    WeightCountMismatch {
        /// Number of coefficients supplied.
        coeffs: usize,
        /// Number of weights supplied.
        weights: usize,
    },
}

impl core::fmt::Display for AllocationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self
        {
            AllocationError::WeightCountMismatch { coeffs, weights } => write!(
                f,
                "weighted allocation needs one weight per coefficient (got {weights} weights for {coeffs} coefficients)"
            ),
        }
    }
}

impl std::error::Error for AllocationError {}

/// Traditional tolerance-interval allocation, for side-by-side comparison with
/// the inertial methods. Distributes an assembly tolerance interval `r_y` to
/// component intervals `Rᵢ`.
///
/// - [`TraditionalMethod::WorstCase`]: `Rᵢ = r_y / Σ|αⱼ|`.
/// - [`TraditionalMethod::Statistical`]: `Rᵢ = r_y / √(Σ αⱼ²)`.
/// - [`TraditionalMethod::Inflated`]`(f)`: `Rᵢ = r_y / (f·√(Σ αⱼ²))`, the
///   inflated-statistical rule with safety factor `f`.
pub fn allocate_traditional(r_y: f64, coeffs: &[f64], method: TraditionalMethod) -> Vec<f64> {
    let n = coeffs.len();
    if n == 0
    {
        return Vec::new();
    }
    match method
    {
        TraditionalMethod::WorstCase =>
        {
            let denom: f64 = coeffs.iter().map(|a| a.abs()).sum();
            vec![r_y / denom; n]
        },
        TraditionalMethod::Statistical =>
        {
            let denom = coeffs.iter().map(|a| a * a).sum::<f64>().sqrt();
            vec![r_y / denom; n]
        },
        TraditionalMethod::Inflated(f) =>
        {
            let denom = coeffs.iter().map(|a| a * a).sum::<f64>().sqrt();
            vec![r_y / (f * denom); n]
        },
    }
}

/// Traditional (interval-based) allocation rules, for comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TraditionalMethod {
    /// Arithmetic worst-case sum of intervals.
    WorstCase,
    /// Root-sum-square (statistical) combination of intervals.
    Statistical,
    /// Inflated-statistical with a safety factor `f` (e.g. `1.5`).
    Inflated(f64),
}

/// The maximum centred dispersion `σ_max = R / 6` implied by a tolerance
/// interval `R` (a `Cp = 1` batch). Lets an interval allocation be compared to
/// an inertial one on the same footing (for a centred batch, `σ_max = Iᵢ`).
pub fn max_dispersion(interval: f64) -> f64 {
    interval / 6.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn signs(n: usize) -> Vec<f64> {
        // Y = X1 − X2 − X3 − X4 − X5 : coefficients ±1, magnitude 1.
        let mut v = vec![-1.0; n];
        v[0] = 1.0;
        v
    }

    #[test]
    fn statistical_and_worst_case_bounds_order_correctly() {
        let cs = vec![
            Contributor::new("a", 1.0, 0.03),
            Contributor::new("b", -1.0, 0.04),
            Contributor::new("c", 1.0, 0.12),
        ];
        let stat = assembly_inertia_statistical(&cs);
        let worst = assembly_inertia_worst_case(&cs);
        assert_relative_eq!(
            stat,
            (0.03f64.powi(2) + 0.04f64.powi(2) + 0.12f64.powi(2)).sqrt(),
            epsilon = 1e-12
        );
        assert_relative_eq!(worst, 0.19, epsilon = 1e-12);
        assert!(stat <= worst); // RSS never exceeds the arithmetic sum
    }

    #[test]
    fn assembly_state_propagates_delta_and_variance() {
        let cs = vec![
            ContributorState::new("a", 1.0, 0.1, 0.2),
            ContributorState::new("b", -1.0, 0.1, 0.2),
        ];
        let i = assembly_state(&cs);
        // δ_Y = 0.1 − 0.1 = 0; σ_Y = √(0.04 + 0.04).
        assert_relative_eq!(i.off_centering, 0.0, epsilon = 1e-12);
        assert_relative_eq!(i.sigma, (0.08f64).sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn icc_matches_paper_worked_value() {
        // Cpk = 1, n = 5 ⇒ ICC = √(1 + 5/9) = 1.2472…
        assert_relative_eq!(icc(1.0, 5), 1.247_219, epsilon = 1e-5);
        // Inverse: ICC = 1.5, guarantees Cpk = 1 up to n = 11 (11.25 exact).
        assert_relative_eq!(
            guaranteed_cpk(1.5, 5),
            (2.25f64 - 5.0 / 9.0).sqrt(),
            epsilon = 1e-12
        );
        assert!(guaranteed_cpk(1.5, 11) >= 1.0 && guaranteed_cpk(1.5, 12) < 1.0);
    }

    #[test]
    fn inertial_allocation_reproduces_paper_table2() {
        // R_Y = 1, n = 5, uniform ±1 chain ⇒ I_Y budget = R_Y/6.
        let coeffs = signs(5);
        let i_y = 1.0 / 6.0;

        let worst = allocate(i_y, &coeffs, &Allocation::WorstCase).unwrap();
        assert_relative_eq!(worst[0], 0.0333, epsilon = 1e-3); // R_Y/(6·5)

        let stat = allocate(i_y, &coeffs, &Allocation::Statistical).unwrap();
        assert_relative_eq!(stat[0], 0.0745, epsilon = 1e-3); // R_Y/(6·√5)

        let cpk = allocate(i_y, &coeffs, &Allocation::GuaranteedCpk(1.0)).unwrap();
        assert_relative_eq!(cpk[0], 0.0597, epsilon = 1e-3); // R_Y/(6·1.247·√5)
    }

    #[test]
    fn allocated_statistical_inertias_recombine_to_budget() {
        let coeffs = vec![1.0, -1.0, 2.0, -1.0];
        let i_y = 0.5;
        let alloc = allocate(i_y, &coeffs, &Allocation::Statistical).unwrap();
        let cs: Vec<Contributor> = coeffs
            .iter()
            .zip(&alloc)
            .map(|(a, i)| Contributor::new("x", *a, *i))
            .collect();
        assert_relative_eq!(assembly_inertia_statistical(&cs), i_y, epsilon = 1e-12);
    }

    #[test]
    fn weighted_allocation_recombines_and_respects_weights() {
        let coeffs = vec![1.0, -1.0, 1.0];
        let betas = vec![1.0, 2.0, 1.0];
        let i_y = 0.3;
        let alloc = allocate(i_y, &coeffs, &Allocation::Weighted(betas.clone())).unwrap();
        // Harder component (β=2) gets the loosest inertia.
        assert!(alloc[1] > alloc[0] && alloc[1] > alloc[2]);
        assert_relative_eq!(alloc[1] / alloc[0], 2.0, epsilon = 1e-12);
        let cs: Vec<Contributor> = coeffs
            .iter()
            .zip(&alloc)
            .map(|(a, i)| Contributor::new("x", *a, *i))
            .collect();
        assert_relative_eq!(assembly_inertia_statistical(&cs), i_y, epsilon = 1e-12);
    }

    #[test]
    fn weighted_allocation_rejects_wrong_weight_count() {
        let coeffs = vec![1.0, -1.0, 1.0];
        let err = allocate(0.3, &coeffs, &Allocation::Weighted(vec![1.0, 2.0])).unwrap_err();
        assert_eq!(
            err,
            AllocationError::WeightCountMismatch {
                coeffs: 3,
                weights: 2
            }
        );
    }

    #[test]
    fn traditional_allocation_reproduces_paper_table2() {
        // Worst case R_i = R_Y/n = 0.2 ⇒ σ_max = 0.0333.
        let coeffs = signs(5);
        let wc = allocate_traditional(1.0, &coeffs, TraditionalMethod::WorstCase);
        assert_relative_eq!(wc[0], 0.2, epsilon = 1e-9);
        assert_relative_eq!(max_dispersion(wc[0]), 0.0333, epsilon = 1e-3);
        // Statistical R_i = R_Y/√5 = 0.447 ⇒ σ_max = 0.0745.
        let st = allocate_traditional(1.0, &coeffs, TraditionalMethod::Statistical);
        assert_relative_eq!(st[0], 0.4472, epsilon = 1e-3);
        assert_relative_eq!(max_dispersion(st[0]), 0.0745, epsilon = 1e-3);
        // Inflated f=1.5 ⇒ R_i = 0.298 ⇒ σ_max = 0.0497.
        let inf = allocate_traditional(1.0, &coeffs, TraditionalMethod::Inflated(1.5));
        assert_relative_eq!(inf[0], 0.2981, epsilon = 1e-3);
        assert_relative_eq!(max_dispersion(inf[0]), 0.0497, epsilon = 1e-3);
    }
}
