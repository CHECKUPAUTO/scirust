//! Supplementary regression tests for `radar::vi_cfar`'s reference-window
//! overflow validation, driven entirely through the crate's *public* API.
//!
//! The defect itself — `CfarConfig::validate` checking the half-window span
//! `reference_cells + guard_cells` but never its double
//! `2 * (reference_cells + guard_cells)`, which `CfarDetector::evaluate`
//! computes for both its output capacity and its `n > 2 * half` loop bound —
//! is fixed, and the module's own unit tests pin the rejecting side of that
//! fix along with the committed proptest seed that found it.
//!
//! What is pinned *here* is what those leave open, both of which matter if
//! the check is ever touched again:
//!
//! * **The boundary is exact.** The condition is
//!   `2 * (reference_cells + guard_cells) <= usize::MAX`, so a large
//!   `guard_cells` is not by itself disqualifying. Replacing the check with
//!   something conservative-but-wrong — a fixed ceiling, a saturating
//!   substitution — still rejects the counterexample, and would pass a suite
//!   that only tests rejection.
//! * **Nothing over-reserves.** `evaluate` sizes its output with
//!   `n.saturating_sub(2 * half)`. When `2 * half` wrapped (release,
//!   `overflow-checks` off) it became a small number, and that saturating
//!   subtraction — meant to clamp to zero for a too-short input — reserved
//!   close to `n` instead. Bounded by the input length rather than unbounded,
//!   but still the wrong capacity for a window that does not fit the input at
//!   all.

use scirust_signal::radar::vi_cfar::{
    CfarConfig, CfarError, DetectorPolicy, EdgePolicy, InputValidationPolicy, RobustNoiseEstimator,
    evaluate_slice,
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

/// The accept/reject boundary must be sharp in both directions. With
/// `reference_cells = 4` the largest usable guard is `usize::MAX / 2 - 4`
/// (half = `2^63 - 1`, double = `2^64 - 2`); the next value up overflows.
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

/// No accepted config can make `evaluate` reserve an oversized allocation:
/// with the window validated up front, the reserved capacity is never more
/// than the number of cells that can actually be decided.
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
