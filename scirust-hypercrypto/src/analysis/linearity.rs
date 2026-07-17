//! Experiments 2–3 — affinity detection over `Z/2^k` and over `GF(2)`
//! (spec §Experiment 2, §Experiment 3).
//!
//! Three independent detectors operate on an arbitrary map `f: Oct<W> → Oct<W>`:
//!
//! * [`gf2_affine_test`] — does `f(x⊕y) ⊕ f(0) = f(x) ⊕ f(y)` hold? (GF(2)).
//! * [`ring_affine_recover`] — recover `A·x ⊞ b` over `Z/2^k` and verify it.
//! * [`bit_affine_recover`] — recover a Boolean affine map `A·x ⊕ c` over
//!   `GF(2)` from basis evaluations and verify it (small widths only).
//!
//! A *failed* detection excludes only that simple model; it is NOT a claim of
//! cryptographic nonlinearity (spec §Experiment 3).

use crate::algebra::ModMatrix;
use crate::algebra::Oct;
use crate::algebra::word::Word;
use crate::analysis::util::{Coverage, oct_domain_exhaustible, sample_octs, test_points};

/// Outcome of an affinity detector.
#[derive(Clone, Debug)]
pub struct AffinityResult {
    /// `true` iff the tested model represented `f` exactly on all tested points.
    pub holds: bool,
    /// Fraction of tested points where the model disagreed (`0.0` if it holds).
    /// Reported as parts-per-million to stay integer-friendly in JSON.
    pub disagreement_ppm: u64,
    /// Coverage of the check.
    pub coverage: Coverage,
    /// Human-readable note (e.g. recovered-matrix invertibility).
    pub note: String,
}

/// GF(2)-affinity: test `f(x⊕y) ⊕ f(0) == f(x) ⊕ f(y)` on pairs.
pub fn gf2_affine_test<W: Word>(
    f: impl Fn(Oct<W>) -> Oct<W>,
    seed: u64,
    sample_pairs: usize,
) -> AffinityResult {
    let f0 = f(Oct::<W>::zero());
    // draw two independent deterministic streams of points
    let xs = sample_octs::<W>(seed, sample_pairs);
    let ys = sample_octs::<W>(seed ^ 0xA5A5_A5A5_A5A5_A5A5, sample_pairs);
    let mut bad = 0u64;
    let n = xs.len().max(1) as u64;
    for (x, y) in xs.iter().zip(ys.iter())
    {
        let lhs = f(x.xor(*y)).xor(f0);
        let rhs = f(*x).xor(f(*y));
        if lhs != rhs
        {
            bad += 1;
        }
    }
    AffinityResult {
        // never claim "affine" without having tested at least one pair
        holds: bad == 0 && !xs.is_empty(),
        disagreement_ppm: bad.saturating_mul(1_000_000) / n,
        coverage: Coverage::Sampled {
            count: xs.len(),
            seed,
        },
        note: "GF(2) additive-homomorphism test on random pairs".to_string(),
    }
}

/// Ring-affinity over `Z/2^k`: recover `A` (columns `f(e_j) ⊟ f(0)`) and `b =
/// f(0)`, then verify `f(x) == A·x ⊞ b` exhaustively (`k=2`) or on a sample.
pub fn ring_affine_recover<W: Word>(
    f: impl Fn(Oct<W>) -> Oct<W>,
    seed: u64,
    sample: usize,
) -> (AffinityResult, ModMatrix<W>, Oct<W>) {
    let b = f(Oct::<W>::zero());
    let mut a = ModMatrix::<W>::zeros(8, 8);
    for j in 0..8
    {
        let col = f(Oct::<W>::e(j)).sub(b); // A e_j = f(e_j) - f(0)
        a.set_col(j, &col.c);
    }
    let (points, coverage) = test_points::<W>(seed, sample);
    let mut bad = 0u64;
    let n = points.len().max(1) as u64;
    for x in &points
    {
        let mv: [W; 8] = a.matvec(&x.c).try_into().expect("matvec length 8");
        let model = Oct::from_coeffs(mv).add(b);
        if model != f(*x)
        {
            bad += 1;
        }
    }
    let holds = bad == 0 && !points.is_empty();
    let note = if holds
    {
        format!("recovered A over Z/2^k; A invertible={}", a.is_unit())
    }
    else
    {
        "no exact A·x ⊞ b representation over Z/2^k".to_string()
    };
    (
        AffinityResult {
            holds,
            disagreement_ppm: bad.saturating_mul(1_000_000) / n,
            coverage,
            note,
        },
        a,
        b,
    )
}

// --- Boolean (GF(2)) affine recovery for small total bit-widths -------------

fn oct_to_bits<W: Word>(o: Oct<W>) -> u64 {
    let bits = W::BITS;
    let mut acc = 0u64;
    for i in 0..8
    {
        acc |= o.c[i].to_u64() << (i as u32 * bits);
    }
    acc
}

fn bits_to_oct<W: Word>(v: u64) -> Oct<W> {
    let bits = W::BITS;
    let mask = if bits >= 64
    {
        u64::MAX
    }
    else
    {
        (1u64 << bits) - 1
    };
    let mut c = [W::ZERO; 8];
    for (i, slot) in c.iter_mut().enumerate()
    {
        *slot = W::from_u64((v >> (i as u32 * bits)) & mask);
    }
    Oct::from_coeffs(c)
}

/// Boolean affine recovery over `GF(2)`. Applicable only when the total bit
/// width `8k ≤ 64`. Returns `None` when not applicable (e.g. `k = 16`).
pub fn bit_affine_recover<W: Word>(
    f: impl Fn(Oct<W>) -> Oct<W>,
    seed: u64,
    sample: usize,
) -> Option<AffinityResult> {
    let n = 8 * W::BITS; // total input/output bits
    if n > 64
    {
        return None;
    }
    let c = oct_to_bits::<W>(f(Oct::<W>::zero()));
    // column j = f(unit bit j) ^ c ; store as per-output-bit masks
    let mut cols = [0u64; 64];
    for j in 0..n as usize
    {
        let ej = bits_to_oct::<W>(1u64 << j);
        cols[j] = oct_to_bits::<W>(f(ej)) ^ c;
    }
    // eval model at x: out bit o = parity over j of (x_j & col_j has bit o) ^ c_o
    let eval = |x: u64| -> u64 {
        let mut out = c;
        for j in 0..n as usize
        {
            if (x >> j) & 1 == 1
            {
                out ^= cols[j];
            }
        }
        out
    };
    // verify exhaustively if n small enough, else sample
    let exhaustive = n <= 18;
    let (bad, tested, coverage) = if exhaustive
    {
        let total = 1u64 << n;
        let mut bad = 0u64;
        for x in 0..total
        {
            if eval(x) != oct_to_bits::<W>(f(bits_to_oct::<W>(x)))
            {
                bad += 1;
            }
        }
        (bad, total, Coverage::Exhaustive)
    }
    else
    {
        let pts = sample_octs::<W>(seed, sample);
        let mut bad = 0u64;
        for p in &pts
        {
            let x = oct_to_bits::<W>(*p);
            if eval(x) != oct_to_bits::<W>(f(*p))
            {
                bad += 1;
            }
        }
        (
            bad,
            pts.len() as u64,
            Coverage::Sampled {
                count: sample,
                seed,
            },
        )
    };
    Some(AffinityResult {
        holds: bad == 0 && tested > 0,
        disagreement_ppm: bad.saturating_mul(1_000_000) / tested.max(1),
        coverage,
        note: format!("GF(2) affine recovery over {n} input bits"),
    })
}

/// Convenience: is the single-octonion domain exhaustively verifiable here?
pub fn exhaustive_here<W: Word>() -> bool {
    oct_domain_exhaustible::<W>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::word::{W2, W8};
    use crate::fixtures::{Fixture, FixtureId};
    use crate::permutation::Variant;
    use crate::permutation::round::{f_round, g_pre_rotation};

    #[test]
    fn control_b_ring_linear_is_recovered() {
        // (K1 ⊗ R) ⊗ K2 must be recovered as exact ring-linear (b=0).
        let m = Fixture::new(FixtureId::OddNorm).round_material::<W2>(0);
        let f = |x: Oct<W2>| Variant::ControlB.round(x, &m);
        let (res, _a, b) = ring_affine_recover::<W2>(f, 1, 0);
        assert!(res.holds, "Control B must be ring-affine");
        assert_eq!(res.coverage, Coverage::Exhaustive);
        assert_eq!(b, Oct::<W2>::zero(), "pure linear => zero offset");
    }

    #[test]
    fn pre_rotation_g_is_ring_affine() {
        // G(x) = (K1 ⊗ (x ⊞ K0)) ⊗ K2 must be affine over Z/2^k.
        let m = Fixture::new(FixtureId::PseudoRandom(3)).round_material::<W2>(0);
        let f = |x: Oct<W2>| g_pre_rotation(x, &m);
        let (res, _a, _b) = ring_affine_recover::<W2>(f, 1, 0);
        assert!(res.holds, "G must be ring-affine over Z/2^k");
    }

    #[test]
    fn full_round_f_is_not_ring_affine() {
        // ROT + XORC break Z/2^k affinity of the full round.
        let m = Fixture::new(FixtureId::PseudoRandom(5)).round_material::<W2>(0);
        let f = |x: Oct<W2>| f_round(x, &m);
        let (res, _a, _b) = ring_affine_recover::<W2>(f, 1, 0);
        assert!(!res.holds, "full F should NOT be ring-affine");
    }

    #[test]
    fn control_a_is_ring_affine_but_not_gf2_affine() {
        let m = Fixture::new(FixtureId::PseudoRandom(2)).round_material::<W8>(0);
        let fa = |x: Oct<W8>| Variant::ControlA.round(x, &m);
        // additive affine holds (perm + wrapping add of K0)
        let (ring, _a, _b) = ring_affine_recover::<W8>(fa, 9, 20000);
        assert!(ring.holds, "Control A must be ring-affine");
        // but wrapping add has carries -> not GF(2)-affine in general
        let gf2 = gf2_affine_test::<W8>(fa, 9, 20000);
        assert!(
            !gf2.holds,
            "Control A add-with-carry should not be GF(2)-affine"
        );
    }
}
