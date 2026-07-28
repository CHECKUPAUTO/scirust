//! Regression tests for `radar::vi_cfar`'s reference-window overflow
//! validation, driven entirely through the crate's *public* API.
//!
//! The bug these pin: `CfarConfig::validate` checked that the half-window
//! span `reference_cells + guard_cells` and the pooled estimator size
//! `2 * reference_cells` each fit a `usize`, but never checked their
//! combination `2 * (reference_cells + guard_cells)` — which is exactly what
//! `CfarDetector::evaluate` computes, twice, for its output capacity and its
//! `n > 2 * half` loop bound. A config landing in the narrow band where the
//! half-window fits but its double does not therefore passed validation and
//! reached that arithmetic, where it panicked with `overflow-checks` on
//! (debug) and wrapped silently with them off (release) — the latter then
//! entering the loop with `half > n` and panicking on a slice index instead.
//! Either way the documented contract ("only ever returns `Ok` or a
//! structured `CfarError`") was broken.
//!
//! The counterexample was found by `tests/vi_cfar_proptest.rs`'s
//! `never_panics_on_any_config_and_power`, whose `any_small_count` strategy
//! draws `usize::MAX / 2` for `guard_cells`. It is pinned here as explicit
//! deterministic cases rather than as a `.proptest-regressions` seed: the
//! band is a vanishingly small fraction of that strategy's space (it did not
//! recur in 30 local runs of the full 256-case suite), so a saved seed is a
//! far weaker guard than naming the inputs outright.
//!
//! Chosen behaviour: **reject at validation**, with the already-existing
//! `CfarError::ReferenceWindowTooLarge` — the variant whose own documentation
//! says such configs are "caught here, deterministically, rather than left to
//! panic on overflow". Saturating the arithmetic instead would have silently
//! turned an unrepresentable window into a different, wrong one. Because
//! `CfarDetector::new`, `CfarStreamDetector::new` and `evaluate_slice` all
//! validate before touching window arithmetic, this closes every entry point,
//! and does so identically in debug and release.

use scirust_signal::radar::vi_cfar::{
    CfarConfig, CfarDetector, CfarError, CfarStreamDetector, DetectorPolicy, EdgePolicy,
    InputValidationPolicy, RobustNoiseEstimator, evaluate_slice,
};

/// An otherwise entirely ordinary, valid configuration; each test perturbs
/// only the window fields under test.
fn base_config() -> CfarConfig {
    CfarConfig {
        reference_cells: 4,
        guard_cells: 2,
        pfa: 0.01,
        edge_policy: EdgePolicy::Exclude,
        input_validation: InputValidationPolicy::RejectNegative,
        detector: DetectorPolicy::Ca,
        robust_estimator: RobustNoiseEstimator::TrimmedMean {
            trim_low: 1,
            trim_high: 1,
        },
    }
}

/// The exact minimal counterexample: `reference_cells = 4`, `guard_cells =
/// usize::MAX / 2` (== `i64::MAX` on a 64-bit target). `half = 2^63 + 3`
/// fits a `usize`; `2 * half = 2^64 + 6` does not.
#[test]
fn evaluate_slice_rejects_the_proptest_counterexample_without_panicking() {
    let mut config = base_config();
    config.guard_cells = usize::MAX / 2;

    // The half-window itself is representable — this is the band the old
    // validation let through, not a config any prior check would have caught.
    assert!(
        config
            .reference_cells
            .checked_add(config.guard_cells)
            .is_some()
    );
    assert!(
        config
            .reference_cells
            .checked_add(config.guard_cells)
            .and_then(|half| half.checked_mul(2))
            .is_none()
    );

    let power = vec![1.0_f64; 64];
    assert_eq!(
        evaluate_slice(&power, &config),
        Err(CfarError::ReferenceWindowTooLarge {
            reference_cells: 4,
            guard_cells: usize::MAX / 2,
        })
    );

    // The same config must be rejected at every entry point that owns window
    // arithmetic, not just the one the counterexample happened to reach.
    assert!(matches!(
        CfarDetector::new(config),
        Err(CfarError::ReferenceWindowTooLarge { .. })
    ));
    assert!(matches!(
        CfarStreamDetector::new(config),
        Err(CfarError::ReferenceWindowTooLarge { .. })
    ));
}

/// An empty slice is the degenerate "insufficient input" case, and must go
/// through the same structured rejection rather than short-circuiting past
/// validation.
#[test]
fn empty_input_still_rejects_an_unrepresentable_window() {
    let mut config = base_config();
    config.guard_cells = usize::MAX / 2;
    assert!(matches!(
        evaluate_slice(&[], &config),
        Err(CfarError::ReferenceWindowTooLarge { .. })
    ));
}

/// The accept/reject boundary must be sharp: the check is a genuine bound on
/// `2 * (reference_cells + guard_cells)`, not a blanket rejection of large
/// `guard_cells`. With `reference_cells = 4` the largest usable guard is
/// `usize::MAX / 2 - 4` (half = `2^63 - 1`, double = `2^64 - 2`).
#[test]
fn the_largest_representable_window_is_accepted_and_runs() {
    let mut config = base_config();
    config.guard_cells = usize::MAX / 2 - 4;
    assert!(config.validate().is_ok());

    // And it must *run*, not merely validate: an input far shorter than the
    // window yields no decision (`EdgePolicy::Exclude`), with no panic.
    let power = vec![1.0_f64; 64];
    let decisions = evaluate_slice(&power, &config).expect("representable window must evaluate");
    assert!(decisions.is_empty());

    config.guard_cells = usize::MAX / 2 - 3; // half = 2^63, double = 2^64
    assert_eq!(
        evaluate_slice(&power, &config),
        Err(CfarError::ReferenceWindowTooLarge {
            reference_cells: 4,
            guard_cells: usize::MAX / 2 - 3,
        })
    );
}

/// The overflowing sum `reference_cells + guard_cells` — the case that was
/// already covered — must stay covered: the widened check subsumes it rather
/// than replacing it.
#[test]
fn the_half_window_sum_overflow_remains_rejected() {
    let mut config = base_config();
    config.guard_cells = usize::MAX;
    assert!(
        config
            .reference_cells
            .checked_add(config.guard_cells)
            .is_none()
    );
    assert_eq!(
        evaluate_slice(&[1.0; 64], &config),
        Err(CfarError::ReferenceWindowTooLarge {
            reference_cells: 4,
            guard_cells: usize::MAX,
        })
    );
}

/// No accepted config can make `evaluate` reserve an oversized allocation.
///
/// `evaluate` sizes its output `Vec` with `n.saturating_sub(2 * half)`. Had
/// `2 * half` wrapped (release, `overflow-checks` off) it would have become a
/// small number, and the saturating subtraction — designed to clamp to zero
/// for a too-short input — would instead have reserved close to `n`. That is
/// bounded by the input length rather than unbounded, but it is still the
/// wrong capacity for a window that does not fit the input at all. With the
/// window validated up front, the reserved capacity is never more than the
/// number of cells that can actually be decided.
#[test]
fn accepted_windows_never_over_reserve_output_capacity() {
    let power = vec![1.0_f64; 64];
    for guard in [0_usize, 2, 8, usize::MAX / 2 - 4]
    {
        let mut config = base_config();
        config.guard_cells = guard;
        let decisions = evaluate_slice(&power, &config).expect("window must be representable");

        let half = config.reference_cells + config.guard_cells;
        let expected = power.len().saturating_sub(2 * half);
        assert_eq!(decisions.len(), expected, "guard_cells = {guard}");
        assert!(decisions.capacity() <= power.len(), "guard_cells = {guard}");
    }
}
