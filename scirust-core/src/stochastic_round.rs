//! **Deterministic stochastic rounding** — the second Phase-A slice of the CANR
//! study (`docs/research/CANR_CERTIFIED_ADAPTIVE_REPRESENTATIONS_2026-07-16.md`,
//! §3.6 / §6 determinism ladder / experiment Y4).
//!
//! Stochastic rounding (SR) maps a real `x` lying between two representable
//! neighbours `lo ≤ x ≤ hi` to `hi` with probability `(x − lo)/(hi − lo)` and to
//! `lo` otherwise. Two consequences make it valuable where round-to-nearest
//! (RN) fails (Croci, Fasi, Higham, Mary & Mikaitis, *Stochastic rounding:
//! implementation, error analysis and applications*, R. Soc. Open Sci. 9:211631,
//! 2022):
//!
//! * **Unbiased**: `E[SR(x)] = lo·(1 − p) + hi·p = x` exactly. A long
//!   accumulation therefore has no systematic drift.
//! * **Stagnation-immune**: RN loses every update smaller than half a ULP of the
//!   running sum (`fl(s + δ) = s`), so a stream of tiny increments to a large
//!   accumulator is silently dropped. SR keeps a proportional chance of bumping
//!   the accumulator up, so the sum still grows (CANR Y4: RN stuck at 0.25/20,
//!   SR 20.16 ± 0.56).
//!
//! ## Why *deterministic* SR
//!
//! Naive SR draws from a stateful PRNG, so the result depends on draw order and
//! is not reproducible across threads or reruns — it sits at rung **D3** of the
//! CANR determinism ladder (statistical only). Keying the randomness to the
//! element **index** through the counter-based [`Philox4x32`](crate::philox)
//! generator instead makes each rounding decision a pure function of
//! `(seed, stream, index)`. The result is then **bitwise reproducible given the
//! seed and the accumulation order** — rung **D2** — while remaining unbiased
//! and stagnation-immune. (Order still matters: SR accumulation is sequential,
//! so this is D2, not the order-independent D1 of the exact accumulator in
//! [`crate::exact_acc`]. Choose the exact accumulator when you need D1; choose SR
//! when you need an unbiased *low-precision* accumulator that does not stagnate.)
//!
//! ## Scope and relation to [`crate::lowprec`]
//!
//! [`crate::lowprec`] already provides index-keyed deterministic SR for the
//! narrow storage *formats* (`f32 → bf16`, `f32 → fp8`). This module is
//! complementary on two axes it does not cover:
//!
//! * the `f64 → f32` rounding step ([`stochastic_round_to_f32`]), the common
//!   "compute wide, store narrow" case; and
//! * a stochastic-rounding **accumulator** ([`StochasticSum`]) — the unbiased,
//!   stagnation-immune low-precision reduction of CANR Y4, which the per-value
//!   format converters in `lowprec` do not offer.
//!
//! The primitive [`stochastic_round_to_f32`] takes the uniform variate
//! explicitly so it composes with any randomness source; [`StochasticSum`] and
//! [`stochastic_sum_f32`] wire in Philox for the deterministic, index-keyed
//! variant.

use crate::philox::Philox4x32;

/// Largest `f32` less than or equal to `x` (the `f32`-grid floor of a finite
/// `f64`). `x` must be finite.
#[inline]
fn f32_floor_of(x: f64) -> f32 {
    // `as f32` is round-to-nearest, so it lands on one of the two bracketing
    // grid points; step down by one ULP when it rounded up past `x`.
    let r = x as f32;
    if (r as f64) <= x { r } else { r.next_down() }
}

/// Stochastically round a finite `f64` to `f32` using a uniform variate
/// `u ∈ [0, 1)`.
///
/// Rounds up to the next `f32` with probability equal to the fractional
/// position of `x` between its two `f32` neighbours, so `E[·] = x` exactly for
/// any `x` in `f32` range. Values already representable in `f32` are returned
/// unchanged (probability 0 of moving). Non-finite `x` passes through as
/// `x as f32`.
///
/// A finite `x` outside the `f32` range saturates to the nearest finite bound
/// (`f32::MAX` / `f32::MIN`): a finite input never rounds to an infinity.
#[inline]
pub fn stochastic_round_to_f32(x: f64, u: f32) -> f32 {
    if !x.is_finite()
    {
        return x as f32;
    }
    // Finite `x` outside the f32 range saturates to the nearest finite bound:
    // a finite input must never stochastically round to an infinity. (This also
    // keeps the bracket below well-defined — the f32-floor of a sub-MIN value is
    // −∞, which has no finite upper neighbour.)
    if x >= f32::MAX as f64
    {
        return f32::MAX;
    }
    if x <= f32::MIN as f64
    {
        return f32::MIN;
    }
    let lo = f32_floor_of(x);
    if (lo as f64) == x
    {
        return lo;
    }
    // `lo < f32::MAX` after the saturation guards, so `hi` is finite and `span`
    // is a positive normal/​subnormal — no infinities enter the ratio.
    let hi = lo.next_up();
    let span = hi as f64 - lo as f64;
    let p_up = (x - lo as f64) / span;
    if (u as f64) < p_up { hi } else { lo }
}

/// A deterministic stochastic-rounding accumulator that stores its running sum
/// in `f32` while performing each addition in exact `f64` before the SR step.
///
/// Randomness is keyed to the running element index through [`Philox4x32`], so
/// the final value is **bitwise reproducible** for a given `(seed, stream)` and
/// a given order of [`StochasticSum::add`] calls (CANR ladder rung D2). Unbiased
/// and stagnation-immune (see the module docs).
#[derive(Debug, Clone)]
pub struct StochasticSum {
    rng: Philox4x32,
    stream: u32,
    acc: f32,
    index: u64,
}

impl StochasticSum {
    /// New accumulator seeded with `seed`, drawing from Philox `stream`.
    pub fn new(seed: u64, stream: u32) -> Self {
        Self {
            rng: Philox4x32::new(seed),
            stream,
            acc: 0.0,
            index: 0,
        }
    }

    /// Add one term, rounding the exact `f64` partial sum back to `f32` with a
    /// stochastic step keyed to this term's index.
    #[inline]
    pub fn add(&mut self, x: f32) {
        // f32 → f64 is exact, and the sum of two f64s formed from f32s is far
        // inside f64's range, so `t` is the exact real partial sum.
        let t = self.acc as f64 + x as f64;
        let u = self.rng.f32_at(self.stream, self.index);
        self.acc = stochastic_round_to_f32(t, u);
        self.index += 1;
    }

    /// The current `f32` running sum.
    #[inline]
    pub fn value(&self) -> f32 {
        self.acc
    }

    /// Number of terms added so far (also the next Philox index).
    #[inline]
    pub fn count(&self) -> u64 {
        self.index
    }
}

/// Deterministic stochastic-rounding sum of `xs` (f32 storage, exact f64
/// partial sums, index-keyed Philox randomness). Bitwise reproducible for a
/// given `(seed, stream)` and input order.
pub fn stochastic_sum_f32(xs: &[f32], seed: u64, stream: u32) -> f32 {
    let mut s = StochasticSum::new(seed, stream);
    for &x in xs
    {
        s.add(x);
    }
    s.value()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values already on the f32 grid never move, whatever the variate.
    #[test]
    fn representable_values_are_fixed_points() {
        for &x in &[0.0f32, 1.0, -2.5, 1024.0, f32::MIN_POSITIVE, f32::MAX]
        {
            for &u in &[0.0f32, 0.3, 0.5, 0.999_999]
            {
                assert_eq!(stochastic_round_to_f32(x as f64, u), x, "x={x} u={u}");
            }
        }
    }

    /// SR returns exactly one of the two bracketing f32 neighbours, and the
    /// variate selects the branch monotonically (u below p ⇒ up, else down).
    #[test]
    fn rounds_to_a_bracketing_neighbour() {
        // 0.1 is not representable in f32; bracket it explicitly.
        let x = 0.1f64;
        let lo = f32_floor_of(x);
        let hi = lo.next_up();
        assert!((lo as f64) < x && x < (hi as f64));
        assert_eq!(stochastic_round_to_f32(x, 0.0), hi, "u=0 must round up");
        assert_eq!(
            stochastic_round_to_f32(x, 0.999_999),
            lo,
            "u≈1 must round down"
        );
    }

    /// Unbiasedness: the empirical mean of many SR draws of a fixed `x` matches
    /// `x` to well within the sampling error, using Philox variates.
    #[test]
    fn empirical_mean_is_unbiased() {
        let rng = Philox4x32::new(2026_0716);
        let x = 0.1f64;
        let n = 200_000u64;
        let mut sum = 0.0f64;
        for i in 0..n
        {
            sum += stochastic_round_to_f32(x, rng.f32_at(0, i)) as f64;
        }
        let mean = sum / n as f64;
        // f32 ULP near 0.1 is ~7.5e-9; the SE of the mean over 2e5 draws is a
        // few times 1e-11, so 1e-9 is a comfortable, non-flaky bound.
        assert!((mean - x).abs() < 1e-9, "mean={mean} x={x}");
    }

    /// The headline property (CANR Y4): RN stagnates on tiny increments to a
    /// large accumulator; deterministic SR keeps growing and stays unbiased.
    #[test]
    fn cures_stagnation_where_round_to_nearest_stalls() {
        let start = 100.0f32;
        let inc = 1e-6f32; // far below half a ULP of 100.0 in f32 (~3.8e-6)
        let steps = 400_000u64;
        let true_sum = start as f64 + inc as f64 * steps as f64;

        // Round-to-nearest f32 accumulation: every add rounds straight back.
        let mut rn = start;
        for _ in 0..steps
        {
            rn += inc;
        }
        assert_eq!(rn, start, "RN must stagnate for this increment");

        // Deterministic SR over several streams: unbiased, no stagnation.
        let mut worst_rel = 0.0f64;
        for stream in 0..5u32
        {
            let mut s = StochasticSum::new(7, stream);
            s.acc = start; // start the accumulator at `start`
            for _ in 0..steps
            {
                s.add(inc);
            }
            let rel = ((s.value() as f64 - true_sum) / true_sum).abs();
            worst_rel = worst_rel.max(rel);
            assert!(s.value() > start, "SR must move off the stagnation point");
        }
        // Each SR step adds O(ULP) noise; over 4e5 steps the relative error
        // stays well under 1e-3 (true sum ~= 100.4).
        assert!(worst_rel < 1e-3, "worst SR relative error {worst_rel}");
    }

    /// Determinism: same seed, stream and order ⇒ bit-identical result;
    /// changing the stream changes it.
    #[test]
    fn deterministic_sum_is_reproducible_and_stream_dependent() {
        let xs: Vec<f32> = (0..10_000).map(|i| (i as f32).sin() * 1e-4).collect();
        let a = stochastic_sum_f32(&xs, 42, 0);
        let b = stochastic_sum_f32(&xs, 42, 0);
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "same (seed, stream, order) must match bitwise"
        );
        let c = stochastic_sum_f32(&xs, 42, 1);
        assert_ne!(a.to_bits(), c.to_bits(), "different stream should differ");
    }

    /// Overflow rounds to the finite max, never spuriously to infinity.
    #[test]
    fn overflow_saturates_to_max_not_infinity() {
        let big = (f32::MAX as f64) * 1.001;
        for &u in &[0.0f32, 0.5, 0.999]
        {
            assert_eq!(stochastic_round_to_f32(big, u), f32::MAX);
            assert_eq!(stochastic_round_to_f32(-big, u), f32::MIN);
        }
    }

    /// Non-finite inputs pass through.
    #[test]
    fn non_finite_passes_through() {
        assert!(stochastic_round_to_f32(f64::NAN, 0.3).is_nan());
        assert_eq!(stochastic_round_to_f32(f64::INFINITY, 0.3), f32::INFINITY);
        assert_eq!(
            stochastic_round_to_f32(f64::NEG_INFINITY, 0.3),
            f32::NEG_INFINITY
        );
    }
}
