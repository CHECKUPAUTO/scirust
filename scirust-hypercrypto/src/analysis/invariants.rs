//! Experiment 5 — norm and conjugation invariants (spec §Experiment 5).
//!
//! The modular norm `N(x) = Σ x_i²` is multiplicative over the octonion product
//! (Degen's eight-square identity, an integer polynomial identity — spec §7.4),
//! so it survives every `⊗`, `conj`, and `PERM_π` layer. The gating question is
//! whether any norm-derived quantity survives the *full* round through `ROT_λ`
//! and `XORC`. A full-round surviving invariant is a KILL criterion.

use crate::algebra::Oct;
use crate::algebra::word::Word;
use crate::analysis::util::{Coverage, test_points};

/// Result of testing one candidate invariant relation.
#[derive(Clone, Debug)]
pub struct InvariantResult {
    /// Name of the relation.
    pub name: String,
    /// `true` iff the relation held on every tested point.
    pub holds: bool,
    /// Fraction of violations, in parts-per-million.
    pub violation_ppm: u64,
    /// Coverage of the test.
    pub coverage: Coverage,
}

/// Test `N(a ⊗ x) == N(a)·N(x)` for a fixed `a` (spec: multiplicative norm).
pub fn norm_multiplicative_left<W: Word>(a: Oct<W>, seed: u64, sample: usize) -> InvariantResult {
    let (points, coverage) = test_points::<W>(seed, sample);
    let na = a.norm();
    let n = points.len().max(1) as u64;
    let mut bad = 0u64;
    for x in &points
    {
        if a.mul(*x).norm() != na.wmul(x.norm())
        {
            bad += 1;
        }
    }
    InvariantResult {
        name: "N(a⊗x) == N(a)·N(x)".to_string(),
        holds: bad == 0 && !points.is_empty(),
        violation_ppm: bad.saturating_mul(1_000_000) / n,
        coverage,
    }
}

/// Test whether a layer map `g` preserves the norm: `N(g(x)) == N(x)`.
pub fn layer_preserves_norm<W: Word>(
    name: &str,
    g: impl Fn(Oct<W>) -> Oct<W>,
    seed: u64,
    sample: usize,
) -> InvariantResult {
    let (points, coverage) = test_points::<W>(seed, sample);
    let n = points.len().max(1) as u64;
    let mut bad = 0u64;
    for x in &points
    {
        if g(*x).norm() != x.norm()
        {
            bad += 1;
        }
    }
    InvariantResult {
        name: format!("N({name}(x)) == N(x)"),
        holds: bad == 0 && !points.is_empty(),
        violation_ppm: bad.saturating_mul(1_000_000) / n,
        coverage,
    }
}

/// Result of the full-round "is `N(f(x))` a function of `N(x)`?" probe.
#[derive(Clone, Debug)]
pub struct FunctionalNormResult {
    /// `true` iff no two tested inputs shared `N(x)` but differed in `N(f(x))`.
    /// `true` here would mean a norm-determined invariant survived — a KILL.
    pub norm_determines_output_norm: bool,
    /// A witness pair `(N(x), N(f(x1)), N(f(x2)))` breaking functionality, if any.
    pub witness: Option<(u64, u64, u64)>,
    /// Coverage of the probe.
    pub coverage: Coverage,
}

/// Probe whether `N(f(x))` is determined by `N(x)` (spec: does a norm invariant
/// survive the full round?). Finding two inputs with equal `N(x)` and unequal
/// `N(f(x))` refutes any norm-only invariant.
pub fn norm_functional_through<W: Word>(
    f: impl Fn(Oct<W>) -> Oct<W>,
    seed: u64,
    sample: usize,
) -> FunctionalNormResult {
    use std::collections::HashMap;
    let (points, coverage) = test_points::<W>(seed, sample);
    let mut seen: HashMap<u64, u64> = HashMap::new();
    let mut witness = None;
    let mut collisions = 0u64; // how many inputs shared an already-seen N(x)
    for x in &points
    {
        let nx = x.norm().to_u64();
        let nf = f(*x).norm().to_u64();
        match seen.get(&nx)
        {
            Some(&prev) =>
            {
                collisions += 1;
                if prev != nf
                {
                    witness = Some((nx, prev, nf));
                    break;
                }
            },
            None =>
            {
                seen.insert(nx, nf);
            },
        }
    }
    // Only claim a surviving norm invariant if we actually observed many
    // same-N(x) collisions and none broke functionality. Without collisions the
    // probe is inconclusive (reported as "not determined", never a false kill).
    FunctionalNormResult {
        norm_determines_output_norm: witness.is_none() && collisions >= 64,
        witness,
        coverage,
    }
}

/// Result of the associator-evenness structural probe.
#[derive(Clone, Debug)]
pub struct AssociatorResult {
    /// `true` iff every associator `(x⊗y)⊗z ⊟ x⊗(y⊗z)` had all-even coefficients.
    pub all_even: bool,
    /// An example associator (as `u64` coefficients) for the report.
    pub example: [u64; 8],
    /// Coverage of the probe.
    pub coverage: Coverage,
}

/// Probe whether the octonion associator always lies in the even sublattice
/// `2·R^8` (spec §7.4: "associator outputs carrying a factor of 2"). Documented
/// as attack surface, not a kill.
pub fn associator_evenness<W: Word>(seed: u64, sample: usize) -> AssociatorResult {
    let xs = crate::analysis::util::sample_octs::<W>(seed, sample);
    let ys = crate::analysis::util::sample_octs::<W>(seed ^ 0x1111, sample);
    let zs = crate::analysis::util::sample_octs::<W>(seed ^ 0x2222, sample);
    let mut all_even = true;
    let mut example = [0u64; 8];
    let mut first = true;
    for ((x, y), z) in xs.iter().zip(ys.iter()).zip(zs.iter())
    {
        let assoc = x.mul(*y).mul(*z).sub(x.mul(y.mul(*z)));
        if first
        {
            example = assoc.to_u64s();
            first = false;
        }
        if assoc.c.iter().any(|w| w.to_u64() & 1 == 1)
        {
            all_even = false;
            example = assoc.to_u64s();
            break;
        }
    }
    AssociatorResult {
        all_even,
        example,
        coverage: Coverage::Sampled {
            count: sample,
            seed,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::OctLayers;
    use crate::algebra::word::{W2, W8};
    use crate::fixtures::{Fixture, FixtureId};
    use crate::permutation::round::f_round;

    #[test]
    fn norm_is_multiplicative_over_product() {
        // Degen identity: should hold exactly for every a (exhaustive over W2).
        for code in [0u64, 1, 7, 40000]
        {
            let a = Oct::<W2>::from_u64s(std::array::from_fn(|i| (code >> (2 * i)) & 3));
            let r = norm_multiplicative_left(a, 1, 0);
            assert!(r.holds, "norm not multiplicative for a={code}");
        }
    }

    #[test]
    fn conj_and_perm_preserve_norm_rotation_does_not() {
        let r_conj = layer_preserves_norm::<W2>("conj", |x| x.conj(), 1, 0);
        assert!(r_conj.holds);
        let r_perm = layer_preserves_norm::<W2>("PERM", |x| x.perm_pi(), 1, 0);
        assert!(r_perm.holds);
        let r_rot = layer_preserves_norm::<W2>("ROT", |x| x.rot_lambda(), 1, 0);
        assert!(!r_rot.holds, "bit rotation must not preserve the norm");
    }

    #[test]
    fn full_round_breaks_norm_functionality() {
        // The full round must NOT let N(x) determine N(F(x)) (else it's a kill).
        let m = Fixture::new(FixtureId::PseudoRandom(3)).round_material::<W8>(0);
        let r = norm_functional_through::<W8>(|x| f_round(x, &m), 5, 50000);
        assert!(
            !r.norm_determines_output_norm,
            "found norm-determined output norm -> would be a KILL"
        );
    }

    #[test]
    fn associator_is_even() {
        let r = associator_evenness::<W8>(1, 5000);
        assert!(
            r.all_even,
            "octonion associator should be in the even sublattice"
        );
    }
}
