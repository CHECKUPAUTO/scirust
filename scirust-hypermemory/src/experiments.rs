//! Deterministic falsification experiments F2 and F6.
//!
//! Phase 1 already confirmed **F1** (the sedenion algebra adds nothing to
//! *retrieval* over the same 16 real components). Before investing in later
//! phases, two questions decide whether the *relation* side of the algebra —
//! explicit parenthesized products — is worth anything:
//!
//! * **F2 — zero-divisor / norm-collapse frequency.** How often does a product
//!   `a·b` collapse toward zero (a near-zero divisor), and how far does its norm
//!   fall short of `‖a‖·‖b‖`? If products routinely collapse, relation codes
//!   cannot carry information.
//! * **F6 — structure discrimination.** Do the two parenthesizations of three
//!   atoms, `(a·b)·c` and `a·(b·c)`, evaluate to *distinguishable* sedenion
//!   codes? If they frequently coincide, the code cannot stand in for the
//!   structure.
//!
//! Everything here is **pure and deterministic**: a fixed-seed in-crate LCG (no
//! RNG dependency), `f32` algebra, and fixed index-order (`f64`-accumulated)
//! reductions. A given `(seed, samples, distribution)` always yields the exact
//! same survey on any target where IEEE-754 `f32` is not reassociated — so
//! these results are *reproducible facts*, not timing measurements.
//!
//! The surveys operate directly on [`SedenionSimd`] operands (the algebra
//! layer). That is exactly what relation evaluation composes: an atom resolves
//! to its concept's anchor, and `S16Expr::Product` multiplies. A test in this
//! module cross-checks that the algebra-level associator matches one produced
//! through the [`crate::S16Expr`] machinery.

use scirust_simd::hypercomplex::SedenionSimd;

use crate::diagnostics::ProductDiagnostics;
use crate::representation::norm_sqr_ordered;

/// A fixed-seed linear congruential generator (Knuth MMIX constants) — the same
/// deterministic style used by `scirust-simd`'s own tests. No external RNG.
#[derive(Clone, Debug)]
pub struct Lcg(u64);

impl Lcg {
    /// Seed the generator. A fixed offset decorrelates nearby seeds.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    /// Next raw 64-bit state.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Next `f32` uniform in `[-1, 1)` (top 24 bits of the state).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }

    /// A uniform index in `0..n` (rejection-free modulo; `n` must be > 0).
    #[inline]
    fn index_below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }
}

/// How operands are drawn. The distribution is the whole point of the F2/F6
/// experiments: generic operands and structured/adversarial operands behave
/// very differently, and reporting both is the honest result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandDistribution {
    /// All 16 lanes independent uniform in `[-1, 1)` — generic operands.
    DenseUniform,
    /// Exactly `nonzero_lanes` lanes (at random positions) are non-zero — a
    /// sparser, more cancellation-prone regime. Clamped to `1..=16`.
    Sparse { nonzero_lanes: usize },
    /// Confined to the first `dim` lanes. `dim ∈ {1,2,4}` selects the real /
    /// complex / quaternion **associative** subalgebra (lanes `e₀..e₃` are the
    /// quaternion subalgebra under this crate's Cayley–Dickson convention), so
    /// the associator vanishes by construction — the case that *triggers* F6.
    Subalgebra { dim: usize },
}

impl OperandDistribution {
    /// Draw one operand with a guaranteed non-zero norm.
    fn sample(self, rng: &mut Lcg) -> SedenionSimd {
        loop
        {
            let mut lanes = [0.0f32; 16];
            match self
            {
                Self::DenseUniform =>
                {
                    for lane in &mut lanes
                    {
                        *lane = rng.next_f32();
                    }
                },
                Self::Sparse { nonzero_lanes } =>
                {
                    let k = nonzero_lanes.clamp(1, 16);
                    // Fisher–Yates prefix over 0..16 to pick k distinct lanes.
                    let mut order = [0usize; 16];
                    for (i, slot) in order.iter_mut().enumerate()
                    {
                        *slot = i;
                    }
                    for i in 0..k
                    {
                        let j = i + rng.index_below(16 - i);
                        order.swap(i, j);
                    }
                    for &idx in &order[..k]
                    {
                        lanes[idx] = rng.next_f32();
                    }
                },
                Self::Subalgebra { dim } =>
                {
                    let d = dim.clamp(1, 16);
                    for lane in lanes.iter_mut().take(d)
                    {
                        *lane = rng.next_f32();
                    }
                },
            }
            let s = SedenionSimd::from_array(lanes);
            if norm_sqr_ordered(&s) > 0.0
            {
                return s;
            }
        }
    }
}

#[inline]
fn norm_ordered(s: &SedenionSimd) -> f32 {
    norm_sqr_ordered(s).sqrt()
}

/// Result of the F2 zero-divisor / norm-collapse survey.
///
/// The key quantity per product is the **norm-defect ratio**
/// `r = ‖a·b‖² / (‖a‖²·‖b‖²)`. For a composition algebra `r ≡ 1`; sedenions
/// are not a composition algebra, so `r` falls below 1 and reaches 0 exactly at
/// a zero divisor. All fields are populated in one deterministic pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZeroDivisorSurvey {
    samples: usize,
    near_zero_count: usize,
    exact_zero_count: usize,
    min_defect_ratio: f32,
    max_defect_ratio: f32,
    mean_defect_ratio: f64,
    frac_below_half: f64,
    frac_below_tenth: f64,
    frac_below_hundredth: f64,
}

impl ZeroDivisorSurvey {
    /// Number of products sampled.
    #[must_use]
    pub const fn samples(&self) -> usize {
        self.samples
    }
    /// How many products were flagged near-zero divisors (both operands
    /// non-zero, `‖a·b‖² ≤ threshold`).
    #[must_use]
    pub const fn near_zero_count(&self) -> usize {
        self.near_zero_count
    }
    /// How many products were *exactly* the zero sedenion.
    #[must_use]
    pub const fn exact_zero_count(&self) -> usize {
        self.exact_zero_count
    }
    /// Smallest observed norm-defect ratio `r`.
    #[must_use]
    pub const fn min_defect_ratio(&self) -> f32 {
        self.min_defect_ratio
    }
    /// Largest observed norm-defect ratio `r`.
    #[must_use]
    pub const fn max_defect_ratio(&self) -> f32 {
        self.max_defect_ratio
    }
    /// Mean norm-defect ratio.
    #[must_use]
    pub const fn mean_defect_ratio(&self) -> f64 {
        self.mean_defect_ratio
    }
    /// Fraction of products with `r < 0.5`.
    #[must_use]
    pub const fn frac_below_half(&self) -> f64 {
        self.frac_below_half
    }
    /// Fraction of products with `r < 0.1`.
    #[must_use]
    pub const fn frac_below_tenth(&self) -> f64 {
        self.frac_below_tenth
    }
    /// Fraction of products with `r < 0.01`.
    #[must_use]
    pub const fn frac_below_hundredth(&self) -> f64 {
        self.frac_below_hundredth
    }
    /// Fraction of products flagged as near-zero divisors.
    #[must_use]
    pub fn near_zero_fraction(&self) -> f64 {
        if self.samples == 0
        {
            0.0
        }
        else
        {
            self.near_zero_count as f64 / self.samples as f64
        }
    }

    /// This survey as shared CANR §9 benchmark records
    /// ([`scirust_bench_schema`]). `seed` is the actual LCG seed that
    /// generated the sampled products (the struct does not store it — pass
    /// the seed you gave [`survey_zero_divisors`]); `dataset` names the
    /// operand distribution. Emitting the whole survey as machine-readable
    /// rows lets the F2 falsification result share the workspace's one
    /// benchmark schema instead of a bespoke text table.
    #[must_use]
    pub fn to_bench_records(
        &self,
        seed: u64,
        dataset: &str,
    ) -> Vec<scirust_bench_schema::BenchRecord> {
        let rec = |metric: &str, value: f64| {
            scirust_bench_schema::BenchRecord::new(
                "hypermemory/F2_zero_divisor",
                dataset,
                "sedenion_product",
                seed,
                metric,
                value,
            )
        };
        vec![
            rec("samples", self.samples as f64),
            rec("near_zero_fraction", self.near_zero_fraction()),
            rec("exact_zero_count", self.exact_zero_count as f64),
            rec("min_defect_ratio", f64::from(self.min_defect_ratio)),
            rec("max_defect_ratio", f64::from(self.max_defect_ratio)),
            rec("mean_defect_ratio", self.mean_defect_ratio),
            rec("frac_below_half", self.frac_below_half),
            rec("frac_below_tenth", self.frac_below_tenth),
            rec("frac_below_hundredth", self.frac_below_hundredth),
        ]
    }
}

/// Survey the zero-divisor / norm-collapse behaviour of `samples` products,
/// each `a·b` with `a`, `b` drawn independently from `dist`.
///
/// `threshold` is the near-zero-divisor cutoff on the result's **squared** norm
/// (see [`ProductDiagnostics`]). Deterministic in `(seed, samples, dist,
/// threshold)`.
#[must_use]
pub fn survey_zero_divisors(
    seed: u64,
    samples: usize,
    dist: OperandDistribution,
    threshold: f32,
) -> ZeroDivisorSurvey {
    let mut rng = Lcg::new(seed);
    let mut near_zero_count = 0usize;
    let mut exact_zero_count = 0usize;
    let mut min_ratio = f32::INFINITY;
    let mut max_ratio = 0.0f32;
    let mut sum_ratio = 0.0f64;
    let mut below_half = 0usize;
    let mut below_tenth = 0usize;
    let mut below_hundredth = 0usize;

    for _ in 0..samples
    {
        let a = dist.sample(&mut rng);
        let b = dist.sample(&mut rng);
        let diag = ProductDiagnostics::measure(&a, &b, threshold);
        if diag.near_zero_divisor()
        {
            near_zero_count += 1;
        }
        if diag.result_norm_sqr() == 0.0
        {
            exact_zero_count += 1;
        }
        // Both operands are non-zero by construction, so the denominator is > 0.
        let denom = diag.lhs_norm_sqr() * diag.rhs_norm_sqr();
        let ratio = diag.result_norm_sqr() / denom;
        if ratio < min_ratio
        {
            min_ratio = ratio;
        }
        if ratio > max_ratio
        {
            max_ratio = ratio;
        }
        sum_ratio += ratio as f64;
        if ratio < 0.5
        {
            below_half += 1;
        }
        if ratio < 0.1
        {
            below_tenth += 1;
        }
        if ratio < 0.01
        {
            below_hundredth += 1;
        }
    }

    let n = samples.max(1) as f64;
    ZeroDivisorSurvey {
        samples,
        near_zero_count,
        exact_zero_count,
        min_defect_ratio: if samples == 0 { 0.0 } else { min_ratio },
        max_defect_ratio: max_ratio,
        mean_defect_ratio: sum_ratio / n,
        frac_below_half: below_half as f64 / n,
        frac_below_tenth: below_tenth as f64 / n,
        frac_below_hundredth: below_hundredth as f64 / n,
    }
}

/// Result of the F6 structure-discrimination survey.
///
/// Per triple `(a, b, c)`, with `L = (a·b)·c`, `R = a·(b·c)`, the **relative
/// associator** is `ρ = ‖L − R‖ / (‖L‖ + ‖R‖)` (defined as 0 when both
/// collapse). A triple is *indistinguishable* when `ρ ≤ threshold`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StructureDiscriminationSurvey {
    samples: usize,
    indistinguishable_count: usize,
    min_relative_associator: f32,
    max_relative_associator: f32,
    mean_relative_associator: f64,
    threshold: f32,
}

impl StructureDiscriminationSurvey {
    /// Number of triples sampled.
    #[must_use]
    pub const fn samples(&self) -> usize {
        self.samples
    }
    /// How many triples had `ρ ≤ threshold` (structures indistinguishable by
    /// their codes).
    #[must_use]
    pub const fn indistinguishable_count(&self) -> usize {
        self.indistinguishable_count
    }
    /// Smallest observed relative associator.
    #[must_use]
    pub const fn min_relative_associator(&self) -> f32 {
        self.min_relative_associator
    }
    /// Largest observed relative associator.
    #[must_use]
    pub const fn max_relative_associator(&self) -> f32 {
        self.max_relative_associator
    }
    /// Mean relative associator.
    #[must_use]
    pub const fn mean_relative_associator(&self) -> f64 {
        self.mean_relative_associator
    }
    /// The indistinguishability threshold used.
    #[must_use]
    pub const fn threshold(&self) -> f32 {
        self.threshold
    }
    /// Fraction of triples whose two parenthesizations are *distinguishable*.
    #[must_use]
    pub fn discriminable_fraction(&self) -> f64 {
        if self.samples == 0
        {
            0.0
        }
        else
        {
            (self.samples - self.indistinguishable_count) as f64 / self.samples as f64
        }
    }

    /// This survey as shared CANR §9 benchmark records
    /// ([`scirust_bench_schema`]). `seed` is the actual LCG seed passed to
    /// [`survey_structure_discrimination`]; `dataset` names the operand
    /// distribution. See [`ZeroDivisorSurvey::to_bench_records`].
    #[must_use]
    pub fn to_bench_records(
        &self,
        seed: u64,
        dataset: &str,
    ) -> Vec<scirust_bench_schema::BenchRecord> {
        let rec = |metric: &str, value: f64| {
            scirust_bench_schema::BenchRecord::new(
                "hypermemory/F6_structure_discrimination",
                dataset,
                "sedenion_associator",
                seed,
                metric,
                value,
            )
        };
        vec![
            rec("samples", self.samples as f64),
            rec("discriminable_fraction", self.discriminable_fraction()),
            rec(
                "indistinguishable_count",
                self.indistinguishable_count as f64,
            ),
            rec(
                "min_relative_associator",
                f64::from(self.min_relative_associator),
            ),
            rec(
                "max_relative_associator",
                f64::from(self.max_relative_associator),
            ),
            rec("mean_relative_associator", self.mean_relative_associator),
            rec("threshold", f64::from(self.threshold)),
        ]
    }
}

/// The relative associator `‖(a·b)·c − a·(b·c)‖ / (‖(a·b)·c‖ + ‖a·(b·c)‖)`.
///
/// Returns `0.0` when both parenthesizations collapse to (near-)zero — they are
/// then genuinely indistinguishable by their codes. Exposed for reuse and for
/// the cross-check against the [`crate::S16Expr`] machinery.
#[must_use]
pub fn relative_associator(a: &SedenionSimd, b: &SedenionSimd, c: &SedenionSimd) -> f32 {
    let left = (*a * *b) * *c;
    let right = *a * (*b * *c);
    let assoc = left - right;
    let denom = norm_ordered(&left) + norm_ordered(&right);
    if denom > 0.0
    {
        norm_ordered(&assoc) / denom
    }
    else
    {
        0.0
    }
}

/// Survey structure discrimination over `samples` triples drawn from `dist`.
///
/// `threshold` is the relative-associator cutoff below which the two
/// parenthesizations are counted indistinguishable. Deterministic in
/// `(seed, samples, dist, threshold)`.
#[must_use]
pub fn survey_structure_discrimination(
    seed: u64,
    samples: usize,
    dist: OperandDistribution,
    threshold: f32,
) -> StructureDiscriminationSurvey {
    let mut rng = Lcg::new(seed);
    let mut indistinguishable_count = 0usize;
    let mut min_rho = f32::INFINITY;
    let mut max_rho = 0.0f32;
    let mut sum_rho = 0.0f64;

    for _ in 0..samples
    {
        let a = dist.sample(&mut rng);
        let b = dist.sample(&mut rng);
        let c = dist.sample(&mut rng);
        let rho = relative_associator(&a, &b, &c);
        if rho <= threshold
        {
            indistinguishable_count += 1;
        }
        if rho < min_rho
        {
            min_rho = rho;
        }
        if rho > max_rho
        {
            max_rho = rho;
        }
        sum_rho += rho as f64;
    }

    let n = samples.max(1) as f64;
    StructureDiscriminationSurvey {
        samples,
        indistinguishable_count,
        min_relative_associator: if samples == 0 { 0.0 } else { min_rho },
        max_relative_associator: max_rho,
        mean_relative_associator: sum_rho / n,
        threshold,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::DEFAULT_NEAR_ZERO_THRESHOLD;
    use crate::expr::{ExprLimits, S16Expr};
    use crate::store::{ConceptSpec, S16Store};

    #[test]
    fn surveys_are_deterministic() {
        let a = survey_zero_divisors(7, 500, OperandDistribution::DenseUniform, 1e-12);
        let b = survey_zero_divisors(7, 500, OperandDistribution::DenseUniform, 1e-12);
        assert_eq!(a, b, "F2 survey must be reproducible for a fixed seed");

        let c = survey_structure_discrimination(9, 500, OperandDistribution::DenseUniform, 1e-3);
        let d = survey_structure_discrimination(9, 500, OperandDistribution::DenseUniform, 1e-3);
        assert_eq!(c, d, "F6 survey must be reproducible for a fixed seed");
    }

    #[test]
    fn f2_generic_operands_do_not_collapse() {
        // Generic (dense uniform) operands essentially never hit a zero divisor,
        // and their norm-defect ratio stays bounded well away from 0.
        let s = survey_zero_divisors(
            1,
            5_000,
            OperandDistribution::DenseUniform,
            DEFAULT_NEAR_ZERO_THRESHOLD,
        );
        assert_eq!(
            s.near_zero_count(),
            0,
            "no near-zero divisor expected for generic operands"
        );
        assert_eq!(s.exact_zero_count(), 0);
        assert!(
            s.min_defect_ratio() > 0.0,
            "defect ratio bounded away from 0"
        );
    }

    #[test]
    fn f2_quaternion_subalgebra_is_a_composition_algebra() {
        // In the quaternion subalgebra the Euclidean norm is multiplicative, so
        // the defect ratio r ≈ 1 (up to f32 rounding). This both cross-validates
        // the algebra and proves no norm collapse is possible there.
        let s = survey_zero_divisors(
            5,
            2_000,
            OperandDistribution::Subalgebra { dim: 4 },
            DEFAULT_NEAR_ZERO_THRESHOLD,
        );
        assert!(
            (s.min_defect_ratio() - 1.0).abs() < 1e-3,
            "min r = {}",
            s.min_defect_ratio()
        );
        assert!(
            (s.max_defect_ratio() - 1.0).abs() < 1e-3,
            "max r = {}",
            s.max_defect_ratio()
        );
        assert_eq!(s.near_zero_count(), 0);
    }

    #[test]
    fn f6_generic_operands_are_discriminable() {
        // Generic triples: the two parenthesizations essentially always differ.
        let s = survey_structure_discrimination(2, 5_000, OperandDistribution::DenseUniform, 1e-3);
        assert_eq!(
            s.indistinguishable_count(),
            0,
            "generic parenthesizations must be distinguishable"
        );
        assert!(s.min_relative_associator() > 1e-3);
    }

    #[test]
    fn f6_triggers_inside_an_associative_subalgebra() {
        // Operands confined to the quaternion subalgebra (lanes e0..e3) are
        // associative: (a*b)*c == a*(b*c), so the codes are indistinguishable —
        // F6 fires exactly here, as it must.
        let s = survey_structure_discrimination(
            3,
            2_000,
            OperandDistribution::Subalgebra { dim: 4 },
            1e-3,
        );
        assert_eq!(
            s.discriminable_fraction(),
            0.0,
            "quaternion-subalgebra triples must be indistinguishable"
        );
        assert!(s.max_relative_associator() <= 1e-3);
    }

    #[test]
    fn relative_associator_matches_expression_machinery() {
        // The algebra-level associator equals the one obtained by building the
        // two parenthesizations as S16Expr and evaluating them through the store.
        let mut rng = Lcg::new(123);
        let a = OperandDistribution::DenseUniform.sample(&mut rng);
        let b = OperandDistribution::DenseUniform.sample(&mut rng);
        let c = OperandDistribution::DenseUniform.sample(&mut rng);

        let mut store = S16Store::new();
        let ca = store
            .insert(ConceptSpec::new(b"a".to_vec(), a, 1.0, 0))
            .unwrap();
        let cb = store
            .insert(ConceptSpec::new(b"b".to_vec(), b, 1.0, 0))
            .unwrap();
        let cc = store
            .insert(ConceptSpec::new(b"c".to_vec(), c, 1.0, 0))
            .unwrap();
        let limits = ExprLimits::default();
        let atom = |id| S16Expr::atom(id);
        let left = S16Expr::product(S16Expr::product(atom(ca), atom(cb)), atom(cc));
        let right = S16Expr::product(atom(ca), S16Expr::product(atom(cb), atom(cc)));

        let lv = left.evaluate(&store, &limits).unwrap();
        let rv = right.evaluate(&store, &limits).unwrap();
        // Same values as the direct algebra products.
        assert_eq!(lv.to_array(), ((a * b) * c).to_array());
        assert_eq!(rv.to_array(), (a * (b * c)).to_array());
        // ...and they genuinely differ for a generic triple.
        assert!(relative_associator(&a, &b, &c) > 1e-3);
    }
}
