//! Relational structure discrimination — the "F1 for relations" probe.
//!
//! Phase 1 confirmed **F1** (the sedenion algebra adds nothing to *retrieval*
//! over the same 16 reals), and the F2/F6 surveys showed the relation codes
//! *can* discriminate structure for generic operands. The open question is
//! **usefulness**: does encoding a structured triple through the non-commutative,
//! non-associative sedenion product discriminate its *structure* (operand order
//! and grouping) better than a plain real-vector encoding of the same 16
//! components?
//!
//! This module answers that with a fair, deterministic comparison. A structured
//! triple `(a, b, c)` with a parenthesization ([`TripleShape`]) is encoded to a
//! 16-lane code by one of several [`Encoding`]s:
//!
//! * `Sedenion` — the parenthesized sedenion product (`(a·b)·c` or `a·(b·c)`);
//! * `Real(Sum)` — `a + b + c` (the "16 real numbers, no algebra" bag; blind to
//!   order and grouping);
//! * `Real(Hadamard)` — `a ⊙ b ⊙ c` (elementwise; also order/grouping blind);
//! * `Real(PositionWeighted)` — `1·a + 2·b + 3·c` (order-sensitive by position,
//!   but grouping-blind).
//!
//! Three deterministic metrics quantify discrimination:
//!
//! * **order sensitivity** — mean relative code distance when two operands are
//!   swapped;
//! * **grouping sensitivity** — mean relative code distance between the two
//!   parenthesizations of the same operands;
//! * **structure retrieval** — nearest-neighbour accuracy recovering *which*
//!   structure a noisy query came from, against a codebook of all
//!   order×grouping structures over a fixed atom set.
//!
//! The honest expectation, made executable in the tests: the commutative /
//! associative real baselines discriminate order and/or grouping **by exactly
//! zero** (they are blind to it by construction), the sedenion product
//! discriminates both, and the position-weighted baseline sits in between
//! (order yes, grouping no). Retrieval accuracy then measures whether that
//! capacity survives noise. This establishes *capacity vs the real baselines* —
//! not that the algebra is worth its cost on a real task.

use scirust_simd::hypercomplex::SedenionSimd;

use crate::experiments::{Lcg, OperandDistribution};
use crate::representation::norm_sqr_ordered;

/// The two parenthesizations of a three-atom relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TripleShape {
    /// `(a·b)·c`.
    LeftAssoc,
    /// `a·(b·c)`.
    RightAssoc,
}

impl TripleShape {
    /// Both shapes, in a fixed order.
    pub const ALL: [TripleShape; 2] = [TripleShape::LeftAssoc, TripleShape::RightAssoc];
}

/// A real-vector binding operation over the 16 lanes — the baselines the
/// sedenion product is measured against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RealBinding {
    /// `a + b + c`: commutative and associative → blind to order and grouping.
    Sum,
    /// `a ⊙ b ⊙ c` (elementwise): commutative and associative → also blind.
    Hadamard,
    /// `1·a + 2·b + 3·c`: order-sensitive (position weights), grouping-blind.
    PositionWeighted,
}

/// How a structured triple is encoded to a 16-lane code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// The parenthesized sedenion product.
    Sedenion,
    /// A real-vector baseline over the same 16 components.
    Real(RealBinding),
    /// A **strong** structural baseline: HRR / VSA holographic binding — nested
    /// circular convolution with fixed left/right role vectors. Unlike the
    /// `Real` baselines it *is* both order- and grouping-sensitive (the roles
    /// differ per tree position), so it is the fair opponent for the sedenion
    /// product. See [`circular_convolution`].
    Hrr,
}

impl Encoding {
    /// Every encoding compared by the report, in a fixed order.
    pub const ALL: [Encoding; 5] = [
        Encoding::Sedenion,
        Encoding::Real(RealBinding::Sum),
        Encoding::Real(RealBinding::Hadamard),
        Encoding::Real(RealBinding::PositionWeighted),
        Encoding::Hrr,
    ];

    /// A short human label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self
        {
            Encoding::Sedenion => "Sedenion",
            Encoding::Real(RealBinding::Sum) => "Real(Sum)",
            Encoding::Real(RealBinding::Hadamard) => "Real(Hadamard)",
            Encoding::Real(RealBinding::PositionWeighted) => "Real(PosWeighted)",
            Encoding::Hrr => "HRR(conv+roles)",
        }
    }

    /// Encode the ordered triple `(a, b, c)` with parenthesization `shape`.
    ///
    /// Real bindings are associative, so they ignore `shape` (that is exactly
    /// the point being measured).
    #[must_use]
    pub fn encode(
        self,
        a: &SedenionSimd,
        b: &SedenionSimd,
        c: &SedenionSimd,
        shape: TripleShape,
    ) -> [f32; 16] {
        match self
        {
            Encoding::Sedenion =>
            {
                let code = match shape
                {
                    TripleShape::LeftAssoc => (*a * *b) * *c,
                    TripleShape::RightAssoc => *a * (*b * *c),
                };
                code.to_array()
            },
            Encoding::Real(binding) =>
            {
                let (x, y, z) = (a.to_array(), b.to_array(), c.to_array());
                let mut out = [0.0f32; 16];
                for i in 0..16
                {
                    out[i] = match binding
                    {
                        RealBinding::Sum => x[i] + y[i] + z[i],
                        RealBinding::Hadamard => x[i] * y[i] * z[i],
                        RealBinding::PositionWeighted => x[i] + 2.0 * y[i] + 3.0 * z[i],
                    };
                }
                out
            },
            Encoding::Hrr =>
            {
                // HRR tree encoding with fixed role vectors L (left child) and
                // R (right child). A leaf is the atom; a node `(l·r)` is
                // `L ⊛ enc(l) + R ⊛ enc(r)`. The two shapes bind the three atoms
                // to *different* nested role products, so order and grouping are
                // both preserved — the fair strong baseline for the sedenion.
                let (l, r) = hrr_roles();
                let (x, y, z) = (a.to_array(), b.to_array(), c.to_array());
                match shape
                {
                    // (a·b)·c = L⊛(L⊛a + R⊛b) + R⊛c
                    TripleShape::LeftAssoc =>
                    {
                        let inner =
                            add16(&circular_convolution(&l, &x), &circular_convolution(&r, &y));
                        add16(
                            &circular_convolution(&l, &inner),
                            &circular_convolution(&r, &z),
                        )
                    },
                    // a·(b·c) = L⊛a + R⊛(L⊛b + R⊛c)
                    TripleShape::RightAssoc =>
                    {
                        let inner =
                            add16(&circular_convolution(&l, &y), &circular_convolution(&r, &z));
                        add16(
                            &circular_convolution(&l, &x),
                            &circular_convolution(&r, &inner),
                        )
                    },
                }
            },
        }
    }
}

/// Fixed seed for the HRR role vectors (structural constants, shared by every
/// HRR encode call so a codebook and its queries use the same roles).
const HRR_ROLE_SEED: u64 = 0x4110_C0DE;

/// The fixed left/right role vectors for the HRR tree encoding, drawn
/// deterministically from [`HRR_ROLE_SEED`].
fn hrr_roles() -> ([f32; 16], [f32; 16]) {
    let mut rng = Lcg::new(HRR_ROLE_SEED);
    let mut l = [0.0f32; 16];
    let mut r = [0.0f32; 16];
    for v in &mut l
    {
        *v = rng.next_f32();
    }
    for v in &mut r
    {
        *v = rng.next_f32();
    }
    (l, r)
}

/// Elementwise sum of two 16-lane vectors, in fixed index order.
fn add16(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..16
    {
        out[i] = a[i] + b[i];
    }
    out
}

/// Circular convolution `(x ⊛ y)[k] = Σⱼ x[j]·y[(k − j) mod 16]`, the HRR/VSA
/// binding operator. Computed directly in fixed index order (deterministic).
#[must_use]
pub fn circular_convolution(x: &[f32; 16], y: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for k in 0..16
    {
        let mut acc = 0.0f32;
        for j in 0..16
        {
            acc += x[j] * y[(k + 16 - j) % 16];
        }
        out[k] = acc;
    }
    out
}

/// Cosine similarity of two 16-lane codes in fixed index order; `0.0` if either
/// is the zero vector (never `NaN`).
#[must_use]
pub fn cosine16(u: &[f32; 16], v: &[f32; 16]) -> f32 {
    let mut dot = 0.0f32;
    let mut nu = 0.0f32;
    let mut nv = 0.0f32;
    for i in 0..16
    {
        dot += u[i] * v[i];
        nu += u[i] * u[i];
        nv += v[i] * v[i];
    }
    if nu <= 0.0 || nv <= 0.0
    {
        0.0
    }
    else
    {
        dot / (nu.sqrt() * nv.sqrt())
    }
}

/// Relative distance `‖u − v‖ / (‖u‖ + ‖v‖)` in fixed index order; `0.0` when
/// both are (near-)zero.
#[must_use]
pub fn relative_distance16(u: &[f32; 16], v: &[f32; 16]) -> f32 {
    let mut diff = 0.0f32;
    let mut nu = 0.0f32;
    let mut nv = 0.0f32;
    for i in 0..16
    {
        let d = u[i] - v[i];
        diff += d * d;
        nu += u[i] * u[i];
        nv += v[i] * v[i];
    }
    let denom = nu.sqrt() + nv.sqrt();
    if denom > 0.0
    {
        diff.sqrt() / denom
    }
    else
    {
        0.0
    }
}

/// Mean relative code distance when the first two operands are swapped —
/// `enc(a,b,c)` vs `enc(b,a,c)` — over `samples` random triples. Measures
/// **order sensitivity**. Commutative encodings return exactly `0`.
#[must_use]
pub fn order_sensitivity(enc: Encoding, seed: u64, samples: usize) -> f64 {
    let mut rng = Lcg::new(seed);
    let dist = OperandDistribution::DenseUniform;
    let mut acc = 0.0f64;
    for _ in 0..samples
    {
        let a = sample(&mut rng, dist);
        let b = sample(&mut rng, dist);
        let c = sample(&mut rng, dist);
        let straight = enc.encode(&a, &b, &c, TripleShape::LeftAssoc);
        let swapped = enc.encode(&b, &a, &c, TripleShape::LeftAssoc);
        acc += relative_distance16(&straight, &swapped) as f64;
    }
    acc / samples.max(1) as f64
}

/// Mean relative code distance between the two parenthesizations of the same
/// operands — `enc(a,b,c,Left)` vs `enc(a,b,c,Right)` — over `samples` random
/// triples. Measures **grouping sensitivity**. Associative encodings return
/// exactly `0`.
#[must_use]
pub fn grouping_sensitivity(enc: Encoding, seed: u64, samples: usize) -> f64 {
    let mut rng = Lcg::new(seed);
    let dist = OperandDistribution::DenseUniform;
    let mut acc = 0.0f64;
    for _ in 0..samples
    {
        let a = sample(&mut rng, dist);
        let b = sample(&mut rng, dist);
        let c = sample(&mut rng, dist);
        let left = enc.encode(&a, &b, &c, TripleShape::LeftAssoc);
        let right = enc.encode(&a, &b, &c, TripleShape::RightAssoc);
        acc += relative_distance16(&left, &right) as f64;
    }
    acc / samples.max(1) as f64
}

/// The six orderings of three positions.
const PERMS3: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

/// Result of the noisy structure-retrieval experiment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetrievalAccuracy {
    encoding: Encoding,
    trials: usize,
    correct: usize,
    structures: usize,
    noise: f32,
}

impl RetrievalAccuracy {
    /// The encoding measured.
    #[must_use]
    pub const fn encoding(&self) -> Encoding {
        self.encoding
    }
    /// Total retrieval trials.
    #[must_use]
    pub const fn trials(&self) -> usize {
        self.trials
    }
    /// Correct retrievals.
    #[must_use]
    pub const fn correct(&self) -> usize {
        self.correct
    }
    /// Codebook size (distinct structures = orderings × parenthesizations).
    #[must_use]
    pub const fn structures(&self) -> usize {
        self.structures
    }
    /// The per-lane noise amplitude used.
    #[must_use]
    pub const fn noise(&self) -> f32 {
        self.noise
    }
    /// Retrieval accuracy in `[0, 1]`.
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        if self.trials == 0
        {
            0.0
        }
        else
        {
            self.correct as f64 / self.trials as f64
        }
    }
    /// Chance accuracy `1 / structures`.
    #[must_use]
    pub fn chance(&self) -> f64 {
        if self.structures == 0
        {
            0.0
        }
        else
        {
            1.0 / self.structures as f64
        }
    }
}

/// Nearest-neighbour structure retrieval under noise.
///
/// For each of `atom_sets` fixed random atom triples, a codebook of all
/// `6 × 2 = 12` structures (orderings × parenthesizations) is built from the
/// clean atoms. Then `trials_per_set` times a target structure is chosen, its
/// three atoms are perturbed by uniform per-lane noise of amplitude `noise`, the
/// noisy triple is re-encoded, and the codebook entry with the highest
/// [`cosine16`] is retrieved. A trial is correct iff the retrieved structure is
/// the target. Deterministic in all arguments.
#[must_use]
pub fn structure_retrieval(
    enc: Encoding,
    seed: u64,
    atom_sets: usize,
    trials_per_set: usize,
    noise: f32,
) -> RetrievalAccuracy {
    // The 12 labelled structures: (ordering, shape).
    let structures: Vec<([usize; 3], TripleShape)> = PERMS3
        .iter()
        .flat_map(|&p| TripleShape::ALL.iter().map(move |&s| (p, s)))
        .collect();
    let n_struct = structures.len();

    let mut rng = Lcg::new(seed);
    let mut correct = 0usize;
    let mut trials = 0usize;

    for _ in 0..atom_sets
    {
        // Three fixed atoms for this codebook.
        let atoms = [
            sample(&mut rng, OperandDistribution::DenseUniform),
            sample(&mut rng, OperandDistribution::DenseUniform),
            sample(&mut rng, OperandDistribution::DenseUniform),
        ];
        let codebook: Vec<[f32; 16]> = structures
            .iter()
            .map(|&(p, s)| enc.encode(&atoms[p[0]], &atoms[p[1]], &atoms[p[2]], s))
            .collect();

        for _ in 0..trials_per_set
        {
            let target = rng.next_u64() as usize % n_struct;
            let (p, s) = structures[target];
            // Perturb the three atoms.
            let noisy = [
                perturb(&atoms[0], &mut rng, noise),
                perturb(&atoms[1], &mut rng, noise),
                perturb(&atoms[2], &mut rng, noise),
            ];
            let query = enc.encode(&noisy[p[0]], &noisy[p[1]], &noisy[p[2]], s);

            // Argmax cosine over the codebook (first wins ties → deterministic).
            let mut best = 0usize;
            let mut best_sim = f32::NEG_INFINITY;
            for (i, code) in codebook.iter().enumerate()
            {
                let sim = cosine16(&query, code);
                if sim > best_sim
                {
                    best_sim = sim;
                    best = i;
                }
            }
            if best == target
            {
                correct += 1;
            }
            trials += 1;
        }
    }

    RetrievalAccuracy {
        encoding: enc,
        trials,
        correct,
        structures: n_struct,
        noise,
    }
}

/// Draw one non-zero operand from `dist`.
fn sample(rng: &mut Lcg, dist: OperandDistribution) -> SedenionSimd {
    // Reuse the experiments generator's distributions via a tiny reimpl of the
    // dense path (the only one used here) to keep this module self-contained.
    match dist
    {
        OperandDistribution::DenseUniform =>
        {
            loop
            {
                let mut lanes = [0.0f32; 16];
                for lane in &mut lanes
                {
                    *lane = rng.next_f32();
                }
                let s = SedenionSimd::from_array(lanes);
                if norm_sqr_ordered(&s) > 0.0
                {
                    return s;
                }
            }
        },
        // Only DenseUniform is used by this module; fall back to it otherwise.
        _ => sample(rng, OperandDistribution::DenseUniform),
    }
}

/// Add uniform per-lane noise of amplitude `noise` to `base`.
fn perturb(base: &SedenionSimd, rng: &mut Lcg, noise: f32) -> SedenionSimd {
    let b = base.to_array();
    let mut out = [0.0f32; 16];
    for i in 0..16
    {
        out[i] = b[i] + noise * rng.next_f32();
    }
    SedenionSimd::from_array(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUM: Encoding = Encoding::Real(RealBinding::Sum);
    const HAD: Encoding = Encoding::Real(RealBinding::Hadamard);
    const POS: Encoding = Encoding::Real(RealBinding::PositionWeighted);
    const SED: Encoding = Encoding::Sedenion;
    const HRR: Encoding = Encoding::Hrr;

    #[test]
    fn commutative_baselines_are_order_blind() {
        // a+b+c and a⊙b⊙c are commutative → swapping operands changes nothing,
        // exactly (IEEE + and * are commutative).
        assert_eq!(order_sensitivity(SUM, 1, 1_000), 0.0);
        assert_eq!(order_sensitivity(HAD, 1, 1_000), 0.0);
    }

    #[test]
    fn all_real_baselines_are_grouping_blind() {
        // Real bindings are associative → parenthesization is ignored, exactly.
        assert_eq!(grouping_sensitivity(SUM, 2, 1_000), 0.0);
        assert_eq!(grouping_sensitivity(HAD, 2, 1_000), 0.0);
        assert_eq!(grouping_sensitivity(POS, 2, 1_000), 0.0);
    }

    #[test]
    fn position_weighted_sees_order_but_not_grouping() {
        assert!(order_sensitivity(POS, 3, 1_000) > 0.0);
        assert_eq!(grouping_sensitivity(POS, 3, 1_000), 0.0);
    }

    #[test]
    fn sedenion_sees_both_order_and_grouping() {
        assert!(order_sensitivity(SED, 4, 1_000) > 0.0);
        assert!(grouping_sensitivity(SED, 4, 1_000) > 0.0);
    }

    #[test]
    fn retrieval_is_deterministic() {
        let a = structure_retrieval(SED, 7, 20, 50, 0.1);
        let b = structure_retrieval(SED, 7, 20, 50, 0.1);
        assert_eq!(a, b);
    }

    #[test]
    fn sedenion_retrieval_beats_the_order_blind_baseline() {
        // Sum collides all 12 structures over the same atoms → near chance.
        // Sedenion distinguishes order AND grouping → far above chance and far
        // above Sum, at a modest noise level.
        let noise = 0.1;
        let sed = structure_retrieval(SED, 11, 40, 100, noise);
        let sum = structure_retrieval(SUM, 11, 40, 100, noise);
        let pos = structure_retrieval(POS, 11, 40, 100, noise);

        assert!(
            sum.accuracy() <= sum.chance() + 0.05,
            "Sum should be ~chance ({:.3}), got {:.3}",
            sum.chance(),
            sum.accuracy()
        );
        assert!(
            sed.accuracy() > 0.8,
            "Sedenion retrieval should be high, got {:.3}",
            sed.accuracy()
        );
        assert!(
            sed.accuracy() > pos.accuracy(),
            "Sedenion ({:.3}) should beat position-weighted ({:.3}) — grouping matters",
            sed.accuracy(),
            pos.accuracy()
        );
    }

    #[test]
    fn circular_convolution_is_commutative() {
        // HRR's binding operator is commutative (multiplication in the DFT
        // domain); the tree encoding's order/grouping sensitivity comes from the
        // distinct role vectors, not from the operator.
        let mut rng = Lcg::new(321);
        let x = sample(&mut rng, OperandDistribution::DenseUniform).to_array();
        let y = sample(&mut rng, OperandDistribution::DenseUniform).to_array();
        // Commutative mathematically; the two summation orders differ only by
        // f32 rounding, so compare within tolerance rather than bit-exactly.
        let xy = circular_convolution(&x, &y);
        let yx = circular_convolution(&y, &x);
        assert!(relative_distance16(&xy, &yx) < 1e-5);
    }

    #[test]
    fn hrr_is_order_and_grouping_sensitive() {
        // The strong baseline sees both, unlike the simple real bindings.
        assert!(order_sensitivity(HRR, 5, 1_000) > 0.0);
        assert!(grouping_sensitivity(HRR, 5, 1_000) > 0.0);
    }

    #[test]
    fn hrr_is_a_strong_structural_baseline() {
        // HRR must clear structure retrieval well above chance and above the
        // grouping-blind position-weighted baseline — otherwise it would not be
        // a fair strong opponent.
        let noise = 0.1;
        let hrr = structure_retrieval(HRR, 11, 40, 100, noise);
        let pos = structure_retrieval(POS, 11, 40, 100, noise);
        assert!(
            hrr.accuracy() > 0.8,
            "HRR retrieval should be high, got {:.3}",
            hrr.accuracy()
        );
        assert!(
            hrr.accuracy() > pos.accuracy(),
            "HRR ({:.3}) should beat position-weighted ({:.3})",
            hrr.accuracy(),
            pos.accuracy()
        );
    }
}
