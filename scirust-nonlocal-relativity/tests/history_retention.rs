//! Phase 3 regression tests for the step-doubling controller's persistent
//! history-retention strategies ([`HistoryRetention::EndpointOnly`] vs
//! [`HistoryRetention::RefinedAcceptedHistory`]).
//!
//! The measured convergence comparison against a fine independent reference,
//! and the resulting decision to keep `EndpointOnly` as the default, live in
//! `experiments/nonlocal-relativity-v2` and the v2 paper. These tests pin the
//! structural invariants the two strategies must satisfy.

use std::cell::RefCell;
use std::rc::Rc;

use scirust_nonlocal_relativity::{
    AdaptiveNonlocalConfig, AdaptiveStepperPolicy, CompleteUniformHistory, HistoryBackend,
    HistoryRetention, HistoryTransport, IdentityHistoryTransport, MemoryLaw, NonlocalConfig,
    NonlocalResult, NonuniformCaputoCoordinateMemory, WorldlineState,
    simulate_nonlocal_worldline_adaptive_with_stepper_policy_retention,
};
use scirust_relativity::Schwarzschild;
use std::f64::consts::FRAC_PI_2;

fn circular_schwarzschild_state(mass: f64, radius: f64) -> WorldlineState<4> {
    let denominator = (1.0 - 3.0 * mass / radius).sqrt();
    let u_t = 1.0 / denominator;
    let u_phi = (mass / (radius * radius * radius)).sqrt() / denominator;
    WorldlineState::new([0.0, radius, FRAC_PI_2, 0.0], [u_t, 0.0, 0.0, u_phi])
}

fn l2(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    (0..4).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt()
}

/// A memory law that records the retained affine-parameter list at every
/// evaluation and delegates the actual memory to
/// [`NonuniformCaputoCoordinateMemory`], so dynamics are realistic.
#[derive(Clone)]
struct RecordingMemory {
    snapshots: Rc<RefCell<Vec<Vec<f64>>>>,
}

impl<const D: usize> MemoryLaw<D> for RecordingMemory {
    fn memory_vector<H, T>(
        &self,
        history: &H,
        transport: &T,
        current_state: &WorldlineState<D>,
        step_index: usize,
        config: NonlocalConfig,
    ) -> NonlocalResult<[f64; D]>
    where
        H: HistoryBackend<D>,
        T: HistoryTransport<D>,
    {
        let retained = history.retained_samples();
        let params: Vec<f64> = (0..retained)
            .map(|i| history.entry(i).expect("entry").parameter)
            .collect();
        self.snapshots.borrow_mut().push(params);
        NonuniformCaputoCoordinateMemory.memory_vector(
            history,
            transport,
            current_state,
            step_index,
            config,
        )
    }
}

fn generous_config(tolerance: f64) -> AdaptiveNonlocalConfig {
    // A generous initial step relative to the tolerance forces early
    // rejections and shrinkage, so these tests also exercise the "rejected
    // trials do not leak into persistent history" path.
    AdaptiveNonlocalConfig::new(
        0.55, 0.02, 0.15, 0.00005, 0.2, tolerance, 1.0e-8, 0.8, 50_000, 60,
    )
    .unwrap()
}

fn run_recording(retention: HistoryRetention, tolerance: f64) -> (usize, Vec<Vec<f64>>, Vec<f64>) {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let mut initial = circular_schwarzschild_state(1.0, 10.0);
    initial.velocity[1] = -0.01;
    let snapshots = Rc::new(RefCell::new(Vec::new()));
    let memory = RecordingMemory {
        snapshots: Rc::clone(&snapshots),
    };
    let trajectory = simulate_nonlocal_worldline_adaptive_with_stepper_policy_retention(
        &background,
        initial,
        generous_config(tolerance),
        AdaptiveStepperPolicy::new(
            CompleteUniformHistory::<4>::new(),
            memory,
            IdentityHistoryTransport,
        ),
        retention,
    )
    .unwrap();
    let accepted_steps = trajectory.len() - 1;
    let endpoint_parameters: Vec<f64> = trajectory
        .diagnostics()
        .iter()
        .map(|d| d.affine_parameter)
        .collect();
    let snaps = Rc::try_unwrap(snapshots).unwrap().into_inner();
    (accepted_steps, snaps, endpoint_parameters)
}

#[test]
fn every_evaluation_history_is_strictly_ordered_with_no_duplicates() {
    for retention in [
        HistoryRetention::EndpointOnly,
        HistoryRetention::RefinedAcceptedHistory,
    ]
    {
        let (_steps, snapshots, _endpoints) = run_recording(retention, 1.0e-8);
        assert!(!snapshots.is_empty());
        for snapshot in &snapshots
        {
            for window in snapshot.windows(2)
            {
                assert!(
                    window[1] > window[0],
                    "{retention:?}: parameters not strictly increasing: {window:?}"
                );
            }
        }
    }
}

#[test]
fn retained_counts_match_strategy_and_reject_nothing_from_failed_trials() {
    // EndpointOnly retains exactly one sample per accepted step (plus the
    // initial); RefinedAcceptedHistory retains two. The exact counts prove no
    // rejected trial leaked a sample into persistent history.
    let (endpoint_steps, endpoint_snaps, _e) =
        run_recording(HistoryRetention::EndpointOnly, 1.0e-8);
    let (refined_steps, refined_snaps, _r) =
        run_recording(HistoryRetention::RefinedAcceptedHistory, 1.0e-8);

    let endpoint_final = endpoint_snaps.last().unwrap().len();
    let refined_final = refined_snaps.last().unwrap().len();

    assert_eq!(endpoint_final, endpoint_steps + 1);
    assert_eq!(refined_final, 2 * refined_steps + 1);
}

#[test]
fn refined_history_records_true_midpoint_parameters() {
    let (steps, snapshots, endpoints) =
        run_recording(HistoryRetention::RefinedAcceptedHistory, 1.0e-8);
    let final_history = snapshots.last().unwrap();
    assert_eq!(final_history.len(), 2 * steps + 1);
    assert_eq!(endpoints.len(), steps + 1);

    for k in 0..=steps
    {
        // Even positions are the accepted endpoints, recorded identically to
        // the trajectory diagnostics.
        assert_eq!(
            final_history[2 * k].to_bits(),
            endpoints[k].to_bits(),
            "endpoint parameter mismatch at accepted step {k}"
        );
    }
    for k in 1..=steps
    {
        // Odd positions are the true midpoints: strictly between the bracketing
        // endpoints and equal to their average to within rounding.
        let midpoint = final_history[2 * k - 1];
        let previous = endpoints[k - 1];
        let next = endpoints[k];
        assert!(
            midpoint > previous && midpoint < next,
            "midpoint {midpoint} not strictly between {previous} and {next}"
        );
        let average = 0.5 * (previous + next);
        assert!(
            (midpoint - average).abs() <= 1.0e-12 * next.abs().max(1.0),
            "midpoint {midpoint} is not the endpoint average {average}"
        );
    }
}

#[test]
fn endpoint_only_history_matches_the_accepted_endpoints_exactly() {
    let (steps, snapshots, endpoints) = run_recording(HistoryRetention::EndpointOnly, 1.0e-8);
    let final_history = snapshots.last().unwrap();
    assert_eq!(final_history.len(), steps + 1);
    for (recorded, endpoint) in final_history.iter().zip(&endpoints)
    {
        assert_eq!(recorded.to_bits(), endpoint.to_bits());
    }
}

#[test]
fn refined_history_does_not_change_the_accepted_step_count_or_endpoints() {
    // Empirically, retaining midpoints leaves the accepted-step decisions and
    // the endpoint essentially unchanged here (the memory force is a small
    // perturbation): the same number of steps, and endpoints agreeing far
    // more tightly than the tolerance. This is the evidence behind keeping
    // EndpointOnly as the default.
    let background = Schwarzschild::try_new(1.0).unwrap();
    let mut initial = circular_schwarzschild_state(1.0, 10.0);
    initial.velocity[1] = -0.01;
    let config = generous_config(1.0e-8);

    let run = |retention| {
        simulate_nonlocal_worldline_adaptive_with_stepper_policy_retention(
            &background,
            initial,
            config,
            AdaptiveStepperPolicy::new(
                CompleteUniformHistory::<4>::new(),
                NonuniformCaputoCoordinateMemory,
                IdentityHistoryTransport,
            ),
            retention,
        )
        .unwrap()
    };

    let endpoint = run(HistoryRetention::EndpointOnly);
    let refined = run(HistoryRetention::RefinedAcceptedHistory);

    assert_eq!(endpoint.len(), refined.len(), "accepted-step count changed");

    let endpoint_final = endpoint.final_state().unwrap();
    let refined_final = refined.final_state().unwrap();
    let coordinate_gap = l2(&endpoint_final.coordinates, &refined_final.coordinates);
    let velocity_gap = l2(&endpoint_final.velocity, &refined_final.velocity);
    assert!(
        coordinate_gap < 1.0e-9 && velocity_gap < 1.0e-9,
        "refined and endpoint endpoints diverged: coord={coordinate_gap:e}, vel={velocity_gap:e}"
    );
}

#[test]
fn endpoint_error_decreases_under_tolerance_refinement_for_both_strategies() {
    use scirust_nonlocal_relativity::{
        NonlocalSimulationPolicy, SemiImplicitEulerStepper, simulate_nonlocal_worldline_with_policy,
    };
    let background = Schwarzschild::try_new(1.0).unwrap();
    let mut initial = circular_schwarzschild_state(1.0, 10.0);
    initial.velocity[1] = -0.01;
    let target: f64 = 0.8;

    // Independent fine fixed-step reference (same model and memory law).
    let fine_step: f64 = 0.0005;
    let fine_steps = (target / fine_step).round() as usize;
    let reference = simulate_nonlocal_worldline_with_policy(
        &background,
        initial,
        NonlocalConfig::new(0.55, 0.02, fine_step, fine_steps, 1.0e-8).unwrap(),
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(fine_steps + 1),
            NonuniformCaputoCoordinateMemory,
            IdentityHistoryTransport,
            SemiImplicitEulerStepper,
        ),
    )
    .unwrap();
    let reference_final = reference.final_state().unwrap();

    let endpoint_error = |tolerance: f64, retention: HistoryRetention| -> f64 {
        let config = AdaptiveNonlocalConfig::new(
            0.55, 0.02, 0.02, 0.00002, 0.05, tolerance, 1.0e-8, target, 50_000, 60,
        )
        .unwrap();
        let trajectory = simulate_nonlocal_worldline_adaptive_with_stepper_policy_retention(
            &background,
            initial,
            config,
            AdaptiveStepperPolicy::new(
                CompleteUniformHistory::<4>::new(),
                NonuniformCaputoCoordinateMemory,
                IdentityHistoryTransport,
            ),
            retention,
        )
        .unwrap();
        l2(
            &trajectory.final_state().unwrap().coordinates,
            &reference_final.coordinates,
        )
    };

    for retention in [
        HistoryRetention::EndpointOnly,
        HistoryRetention::RefinedAcceptedHistory,
    ]
    {
        let loose = endpoint_error(1.0e-6, retention);
        let tight = endpoint_error(1.0e-8, retention);
        assert!(
            tight < loose,
            "{retention:?}: endpoint error did not decrease under refinement: \
             loose={loose:e}, tight={tight:e}"
        );
    }
}
