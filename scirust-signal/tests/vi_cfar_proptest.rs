//! Property-based (randomized) invariants for `radar::vi_cfar`, mirroring
//! `tests/denoise_proptest.rs`'s convention in this crate: the example-based
//! tests elsewhere pin *specific* behaviors (a target is detected, a mode is
//! selected, a design `P_fa` holds); these check invariants that must hold
//! for **every** input, catching the class of bug no hand-picked example
//! happens to try — an adversarial combination of `reference_cells`,
//! `guard_cells`, `pfa`, trim counts, and switching thresholds, or a power
//! slice laced with `NaN`/`±∞`/negative values.
//!
//! Invariants covered:
//! * [`CfarConfig::validate`]/[`evaluate_slice`] never panics on *any*
//!   combination of field values this proptest can construct, valid or not
//!   — only ever returns `Ok` or a structured [`CfarError`];
//! * a config with any single invalid field (too few reference cells, `pfa`
//!   outside `(0, 1)`, trim counts leaving no retained cell, or invalid
//!   switching thresholds under [`DetectorPolicy::ClassicalViCfar`]) is
//!   always rejected with the matching [`CfarError`] variant, never
//!   silently accepted;
//! * a genuinely valid config always rejects non-finite or (under
//!   [`InputValidationPolicy::RejectNegative`]) negative power samples with
//!   a structured error, never a panic and never a silently-produced
//!   decision;
//! * a genuinely valid config over a genuinely valid power slice always
//!   succeeds, and every decision's `threshold`/`noise_estimate` is finite
//!   and non-negative, with `cut_index` in bounds.

use proptest::prelude::*;
use scirust_signal::radar::vi_cfar::{
    CfarConfig, CfarError, DetectorPolicy, EdgePolicy, InputValidationPolicy, RobustNoiseEstimator,
    SwitchingThresholds, evaluate_slice,
};

const MAX_REFERENCE_CELLS: usize = 40;

/// Any `usize` a `reference_cells`/`guard_cells`/trim field might hold,
/// including the small edge values (`0`, `1`) that are invalid for
/// `reference_cells` specifically, and — the whole point being to exercise
/// both sides of every boundary, not only the "reasonable" range — the huge
/// values (`usize::MAX` and near it) that must be rejected with a structured
/// `ReferenceWindowTooLarge`/`InvalidTrimCounts` error rather than panicking
/// on overflow in the `reference_cells + guard_cells`/`2 * reference_cells`/
/// `trim_low + trim_high` arithmetic.
fn any_small_count() -> impl Strategy<Value = usize> {
    prop_oneof![
        6 => 0..=MAX_REFERENCE_CELLS,
        1 => Just(0usize),
        1 => Just(1usize),
        1 => Just(usize::MAX),
        1 => Just(usize::MAX - 1),
        1 => Just(usize::MAX / 2),
    ]
}

/// Any `f64` a `pfa`/`k_vi`/`k_mr` field might hold: ordinary finite values
/// in and out of the valid range, plus the non-finite values validation
/// must reject rather than propagate.
fn any_config_scalar() -> impl Strategy<Value = f64> {
    prop_oneof![
        3 => -2.0f64..2.0,
        1 => Just(f64::NAN),
        1 => Just(f64::INFINITY),
        1 => Just(f64::NEG_INFINITY),
        1 => Just(0.0f64),
        1 => Just(1.0f64),
    ]
}

fn any_robust_estimator() -> impl Strategy<Value = RobustNoiseEstimator> {
    (any::<bool>(), any_small_count(), any_small_count()).prop_map(
        |(censored, trim_low, trim_high)| {
            if censored
            {
                RobustNoiseEstimator::CensoredMean {
                    trim_low,
                    trim_high,
                }
            }
            else
            {
                RobustNoiseEstimator::TrimmedMean {
                    trim_low,
                    trim_high,
                }
            }
        },
    )
}

fn any_detector_policy() -> impl Strategy<Value = DetectorPolicy> {
    prop_oneof![
        Just(DetectorPolicy::Ca),
        Just(DetectorPolicy::Go),
        Just(DetectorPolicy::So),
        Just(DetectorPolicy::AlwaysRobust),
        (any_config_scalar(), any_config_scalar()).prop_map(|(k_vi, k_mr)| {
            DetectorPolicy::ClassicalViCfar(SwitchingThresholds { k_vi, k_mr })
        }),
    ]
}

/// A `CfarConfig` with every field independently drawn from its full
/// possible range, including values [`CfarConfig::validate`] must reject —
/// deliberately not filtered down to "valid configs only". `guard_cells`
/// shares [`any_small_count`]'s distribution (including near-`usize::MAX`
/// draws) since `reference_cells + guard_cells` is exactly the other half of
/// the overflow-prone window-size arithmetic.
fn any_config() -> impl Strategy<Value = CfarConfig> {
    (
        any_small_count(),
        any_small_count(),
        any_config_scalar(),
        any_detector_policy(),
        any_robust_estimator(),
    )
        .prop_map(
            |(reference_cells, guard_cells, pfa, detector, robust_estimator)| CfarConfig {
                reference_cells,
                guard_cells,
                pfa,
                edge_policy: EdgePolicy::Exclude,
                input_validation: InputValidationPolicy::RejectNegative,
                detector,
                robust_estimator,
            },
        )
}

/// A [`CfarConfig`] guaranteed to pass [`CfarConfig::validate`]: every field
/// drawn from within its documented valid range instead of
/// [`any_config`]'s full (valid-or-not) range.
fn valid_config() -> impl Strategy<Value = CfarConfig> {
    (2usize..=MAX_REFERENCE_CELLS, 0usize..=4).prop_flat_map(|(reference_cells, guard_cells)| {
        let n_ref = 2 * reference_cells;
        (
            Just(reference_cells),
            Just(guard_cells),
            0.001f64..0.5,
            0..(n_ref / 2).max(1),
            0..(n_ref / 2).max(1),
            any::<bool>(),
            any::<bool>(), // ClassicalViCfar vs. a forced single mode
        )
            .prop_map(
                |(reference_cells, guard_cells, pfa, trim_low, trim_high, censored, classical)| {
                    let robust_estimator = if censored
                    {
                        RobustNoiseEstimator::CensoredMean {
                            trim_low,
                            trim_high,
                        }
                    }
                    else
                    {
                        RobustNoiseEstimator::TrimmedMean {
                            trim_low,
                            trim_high,
                        }
                    };
                    let detector = if classical
                    {
                        DetectorPolicy::ClassicalViCfar(SwitchingThresholds {
                            k_vi: 6.0,
                            k_mr: 2.0,
                        })
                    }
                    else
                    {
                        DetectorPolicy::Ca
                    };
                    CfarConfig {
                        reference_cells,
                        guard_cells,
                        pfa,
                        edge_policy: EdgePolicy::Exclude,
                        input_validation: InputValidationPolicy::RejectNegative,
                        detector,
                        robust_estimator,
                    }
                },
            )
    })
}

/// A power slice of finite, non-negative values (valid under
/// [`InputValidationPolicy::RejectNegative`], the only policy [`valid_config`]
/// uses), long enough to give any `reference_cells` up to
/// [`MAX_REFERENCE_CELLS`] at least a few evaluable CUTs.
fn valid_power() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(
        0.0f64..1.0e3,
        4 * MAX_REFERENCE_CELLS..(6 * MAX_REFERENCE_CELLS),
    )
}

/// A power slice that may contain `NaN`/`±∞`/negative values, of arbitrary
/// (including very short or empty) length.
fn wild_power() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(
        prop_oneof![
            Just(f64::NAN),
            Just(f64::INFINITY),
            Just(f64::NEG_INFINITY),
            Just(0.0f64),
            -1.0e6f64..1.0e6,
        ],
        0..64,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// However adversarial the config and the power slice, `evaluate_slice`
    /// never panics — it only ever returns `Ok` or a structured `CfarError`.
    #[test]
    fn never_panics_on_any_config_and_power(config in any_config(), power in wild_power()) {
        let _ = evaluate_slice(&power, &config);
    }

    /// `validate` rejects `reference_cells < 2` with `TooFewReferenceCells`,
    /// regardless of every other field.
    #[test]
    fn too_few_reference_cells_is_always_rejected(
        reference_cells in 0usize..2,
        guard_cells in 0usize..8,
        detector in any_detector_policy(),
        robust_estimator in any_robust_estimator(),
    ) {
        let config = CfarConfig {
            reference_cells,
            guard_cells,
            pfa: 0.05,
            edge_policy: EdgePolicy::Exclude,
            input_validation: InputValidationPolicy::RejectNegative,
            detector,
            robust_estimator,
        };
        prop_assert!(matches!(
            config.validate(),
            Err(CfarError::TooFewReferenceCells(_))
        ));
    }

    /// `validate` rejects a `pfa` outside `(0, 1)` (including non-finite)
    /// with `InvalidPfa`, given otherwise-valid fields.
    #[test]
    fn invalid_pfa_is_always_rejected(
        pfa in prop_oneof![
            Just(0.0f64), Just(1.0f64), Just(-0.1f64), Just(1.1f64),
            Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY),
        ],
        reference_cells in 2usize..=MAX_REFERENCE_CELLS,
    ) {
        let config = CfarConfig {
            reference_cells,
            guard_cells: 2,
            pfa,
            edge_policy: EdgePolicy::Exclude,
            input_validation: InputValidationPolicy::RejectNegative,
            detector: DetectorPolicy::Ca,
            robust_estimator: RobustNoiseEstimator::TrimmedMean {
                trim_low: 1,
                trim_high: 1,
            },
        };
        prop_assert!(matches!(config.validate(), Err(CfarError::InvalidPfa(_))));
    }

    /// `validate` rejects trim counts that leave no retained cell in the
    /// pooled `2 * reference_cells`-cell window, with `InvalidTrimCounts`.
    #[test]
    fn trim_counts_covering_the_whole_pooled_window_are_rejected(
        reference_cells in 2usize..=20,
        extra in 0usize..10,
    ) {
        let n_ref = 2 * reference_cells;
        // trim_low + trim_high == n_ref + extra >= n_ref: strictly no
        // retained cell can remain.
        let trim_low = n_ref / 2 + extra;
        let trim_high = n_ref - n_ref / 2;
        let config = CfarConfig {
            reference_cells,
            guard_cells: 2,
            pfa: 0.05,
            edge_policy: EdgePolicy::Exclude,
            input_validation: InputValidationPolicy::RejectNegative,
            detector: DetectorPolicy::Ca,
            robust_estimator: RobustNoiseEstimator::TrimmedMean {
                trim_low,
                trim_high,
            },
        };
        let is_invalid_trim_counts = matches!(config.validate(), Err(CfarError::InvalidTrimCounts { .. }));
        prop_assert!(is_invalid_trim_counts);
    }

    /// `validate` rejects non-finite, non-positive, or `k_mr < 1.0` switching
    /// thresholds under `ClassicalViCfar`, with `InvalidSwitchingThresholds`.
    #[test]
    fn invalid_switching_thresholds_are_always_rejected(
        k_vi in prop_oneof![
            Just(0.0f64), Just(-1.0f64), Just(f64::NAN), Just(f64::INFINITY),
        ],
        k_mr in 0.5f64..0.99,
    ) {
        let config = CfarConfig {
            reference_cells: 16,
            guard_cells: 2,
            pfa: 0.05,
            edge_policy: EdgePolicy::Exclude,
            input_validation: InputValidationPolicy::RejectNegative,
            detector: DetectorPolicy::ClassicalViCfar(SwitchingThresholds { k_vi, k_mr }),
            robust_estimator: RobustNoiseEstimator::TrimmedMean {
                trim_low: 1,
                trim_high: 1,
            },
        };
        let is_invalid_thresholds = matches!(
            config.validate(),
            Err(CfarError::InvalidSwitchingThresholds { .. })
        );
        prop_assert!(is_invalid_thresholds);
    }

    /// A genuinely valid config rejects non-finite or negative power samples
    /// with a structured error (`NonFiniteSample`/`NegativeSample`), never a
    /// panic and never a silently-produced decision.
    #[test]
    fn valid_config_rejects_bad_samples_cleanly(config in valid_config(), power in wild_power()) {
        match evaluate_slice(&power, &config) {
            Ok(_) => {
                // Every sample happened to be finite and non-negative --
                // legitimate, not a bug (`wild_power` can draw an
                // all-clean stream by chance).
                prop_assert!(power.iter().all(|&x| x.is_finite() && x >= 0.0));
            },
            Err(CfarError::NonFiniteSample { .. } | CfarError::NegativeSample { .. }) => {},
            Err(other) => prop_assert!(false, "unexpected error variant: {other:?}"),
        }
    }

    /// A genuinely valid config over a genuinely valid power slice always
    /// succeeds, with every decision's `threshold`/`noise_estimate` finite
    /// and non-negative and `cut_index` in bounds.
    #[test]
    fn valid_config_and_power_always_succeeds_with_sane_decisions(
        config in valid_config(),
        power in valid_power(),
    ) {
        let decisions = evaluate_slice(&power, &config)
            .expect("a valid config and a valid power slice must not error");
        for d in &decisions {
            prop_assert!(d.cut_index < power.len());
            prop_assert!(d.threshold.is_finite() && d.threshold >= 0.0, "{:?}", d);
            prop_assert!(d.noise_estimate.is_finite() && d.noise_estimate >= 0.0, "{:?}", d);
        }
    }
}
