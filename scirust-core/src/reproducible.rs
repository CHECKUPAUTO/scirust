//! Order-independent, reproducible floating-point reductions.
//!
//! Floating-point addition is **not associative**, so a plain sum depends on the
//! order the terms are added — and therefore on the number of threads, the
//! chunking, or the iteration order. That breaks bit-exact reproducibility, one
//! of scirust's fundamentals (cf. [`GROWTH_PLAN`](../docs/GROWTH_PLAN.md)).
//!
//! These reductions are a **pure function of the multiset** of inputs: any
//! permutation yields a **bit-identical** result. The technique (after Demmel &
//! Nguyen, *Reproducible Floating-Point Summation*, and Shewchuk's exact
//! expansions) is:
//!
//! 1. **canonical ordering** — sort by [`f32::total_cmp`]/[`f64::total_cmp`], so
//!    the same values are always summed in the same order;
//! 2. **exact expansion** — accumulate into a non-overlapping expansion (the
//!    `fsum` algorithm), giving the correctly-rounded sum.
//!
//! The result is both reproducible *and* far more accurate than a naive sum
//! (it survives catastrophic cancellation — see the tests).

/// Correctly-rounded sum of `vals` in a **canonical order** (Shewchuk's exact
/// expansion over a sorted input). Order-independent by construction.
fn fsum_canonical(mut vals: Vec<f64>) -> f64 {
    vals.sort_by(f64::total_cmp);
    // Non-overlapping expansion: `partials` always sum (exactly) to the running
    // total, with no two members overlapping in significance.
    let mut partials: Vec<f64> = Vec::new();
    for xi in vals
    {
        let mut x = xi;
        let mut i = 0;
        for k in 0..partials.len()
        {
            let mut y = partials[k];
            if x.abs() < y.abs()
            {
                core::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x); // exact rounding error of x + y (|x| ≥ |y|)
            if lo != 0.0
            {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        partials.push(x);
    }
    partials.iter().sum()
}

/// Reproducible sum of `f32` values: bit-identical regardless of order or how
/// the slice was chunked across threads.
pub fn reproducible_sum(xs: &[f32]) -> f32 {
    fsum_canonical(xs.iter().map(|&x| x as f64).collect()) as f32
}

/// Reproducible mean (0.0 for an empty slice).
pub fn reproducible_mean(xs: &[f32]) -> f32 {
    if xs.is_empty()
    {
        return 0.0;
    }
    (fsum_canonical(xs.iter().map(|&x| x as f64).collect()) / xs.len() as f64) as f32
}

/// Reproducible dot product `Σ aᵢ·bᵢ`: products are formed in `f64` and summed
/// in a canonical order, so the result is order-independent and near-exact.
pub fn reproducible_dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "reproducible_dot: length mismatch");
    fsum_canonical(
        a.iter()
            .zip(b)
            .map(|(&x, &y)| x as f64 * y as f64)
            .collect(),
    ) as f32
}

/// `exp(x)` par **promotion en `f64`** : l'argument `f32` est promu, l'exponentielle
/// est évaluée en double précision, puis arrondie (au plus près) vers `f32`.
///
/// Classe de garantie, énoncée précisément :
/// - **précision** : tant que l'`exp` `f64` de la plate-forme est fidèle
///   (erreur < 1 ulp `f64`, le cas de toutes les libm courantes), le résultat
///   est le `f32` **correctement arrondi**, sauf si la valeur exacte tombe à
///   ≈ 2⁻⁵² (relatif) d'une frontière d'arrondi `f32` (dilemme du fabricant de
///   tables) — cas non prouvés impossibles, mais de mesure quasi nulle ;
/// - **déterminisme** : bit-stable sur un binaire/une libm donnés. L'identité
///   bit-à-bit *inter-plates-formes* est très probable (il faudrait qu'une
///   erreur libm `f64` traverse une frontière d'arrondi `f32`) mais **pas
///   prouvée**, contrairement aux réductions ci-dessus qui sont exactes.
///
/// C'est la même classe de technique que les transcendantales de RepDL
/// (Microsoft, arXiv:2510.09180) ; les transcendantales correctement arrondies
/// *prouvées* en Rust pur restent le travail futur acté dans
/// `paper/RELATED_WORK.md`.
pub fn exp_via_f64(x: f32) -> f32 {
    (x as f64).exp() as f32
}

/// `ln(x)` par promotion en `f64` — mêmes garanties que [`exp_via_f64`].
pub fn ln_via_f64(x: f32) -> f32 {
    (x as f64).ln() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::PcgEngine;

    fn shuffled(xs: &[f32], rng: &mut PcgEngine) -> Vec<f32> {
        let mut v = xs.to_vec();
        for i in (1..v.len()).rev()
        {
            let j = ((rng.float() * (i as f32 + 1.0)) as usize).min(i);
            v.swap(i, j);
        }
        v
    }

    /// The headline property: every permutation of the same values yields a
    /// **bit-identical** sum (what a plain `fold` cannot promise).
    #[test]
    fn reproducible_sum_is_order_independent() {
        let xs: Vec<f32> = (0..1000)
            .map(|i| ((i as f32 * 0.137).sin()) * 10f32.powi((i % 12) - 6))
            .collect();
        let reference = reproducible_sum(&xs);
        let mut rng = PcgEngine::new(99);
        for _ in 0..50
        {
            let s = reproducible_sum(&shuffled(&xs, &mut rng));
            assert_eq!(
                s.to_bits(),
                reference.to_bits(),
                "sum changed under permutation"
            );
        }
    }

    /// It survives catastrophic cancellation where a naive `f32` fold collapses
    /// to the wrong answer.
    #[test]
    fn reproducible_sum_beats_naive() {
        let xs = [1e8f32, 1.0, -1e8];
        let naive = xs.iter().fold(0.0f32, |a, &b| a + b);
        assert_eq!(naive, 0.0, "naive f32 sum loses the 1.0");
        assert_eq!(reproducible_sum(&xs), 1.0, "reproducible sum recovers it");
    }

    /// Accuracy: matches an `f64` reference to within f32 rounding.
    #[test]
    fn reproducible_sum_matches_f64_reference() {
        let xs: Vec<f32> = (0..500).map(|i| (i as f32 * 0.01 - 2.5).tanh()).collect();
        let f64_ref: f64 = xs.iter().map(|&x| x as f64).sum();
        assert!((reproducible_sum(&xs) as f64 - f64_ref).abs() < 1e-3);
    }

    /// Dot product is order-independent and correct.
    #[test]
    fn reproducible_dot_works() {
        let a: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();
        let b: Vec<f32> = (0..256).map(|i| (i as f32 * 0.03).cos()).collect();
        let reference = reproducible_dot(&a, &b);

        // Permuting both operands in lock-step keeps the multiset of products.
        let mut rng = PcgEngine::new(7);
        let mut idx: Vec<usize> = (0..a.len()).collect();
        for i in (1..idx.len()).rev()
        {
            let j = ((rng.float() * (i as f32 + 1.0)) as usize).min(i);
            idx.swap(i, j);
        }
        let ap: Vec<f32> = idx.iter().map(|&i| a[i]).collect();
        let bp: Vec<f32> = idx.iter().map(|&i| b[i]).collect();
        assert_eq!(reproducible_dot(&ap, &bp).to_bits(), reference.to_bits());

        let f64_ref: f64 = a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum();
        assert!((reference as f64 - f64_ref).abs() < 1e-4);
    }

    /// exp/ln promus : à ≤ 0,5 ulp `f32` (+ marge) de la référence `f64`,
    /// c'est-à-dire fidèlement arrondis sur tout l'échantillon.
    #[test]
    fn promoted_exp_ln_are_faithful() {
        for i in 0..4000
        {
            let x = -40.0 + i as f32 * 0.02; // [-40, 40)
            let r = exp_via_f64(x) as f64;
            let t = (x as f64).exp();
            assert!(
                (r - t).abs() <= t * 1.0001 * 2f64.powi(-24),
                "exp_via_f64({x}) = {r}, référence f64 = {t}"
            );
        }
        for i in 1..4000
        {
            let x = i as f32 * 0.25; // (0, 1000)
            let r = ln_via_f64(x) as f64;
            let t = (x as f64).ln();
            assert!(
                (r - t).abs() <= t.abs().max(f64::MIN_POSITIVE) * 1.0001 * 2f64.powi(-24) + 1e-12,
                "ln_via_f64({x}) = {r}, référence f64 = {t}"
            );
        }
        assert_eq!(exp_via_f64(0.0), 1.0);
        assert_eq!(ln_via_f64(1.0), 0.0);
        assert!((ln_via_f64(exp_via_f64(1.0)) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn reproducible_edge_cases() {
        assert_eq!(reproducible_sum(&[]), 0.0);
        assert_eq!(reproducible_sum(&[3.5]), 3.5);
        assert_eq!(reproducible_mean(&[]), 0.0);
        assert_eq!(reproducible_mean(&[2.0, 4.0]), 3.0);
    }
}
