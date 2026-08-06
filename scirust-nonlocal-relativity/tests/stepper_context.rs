//! Phase 5 regression tests: [`HeunPeceStepper`] must derive the provisional
//! (post-step) affine parameter from `StepperContext::current_parameter`, not
//! by reconstructing it as `step_index * config.step` (valid only under
//! uniform spacing).

use std::cell::RefCell;
use std::rc::Rc;

use scirust_nonlocal_relativity::{
    CompleteUniformHistory, HeunPeceStepper, HistoryBackend, HistoryEntry, HistoryTransport,
    IdentityHistoryTransport, MemoryLaw, NonlocalConfig, NonlocalResult,
    NonuniformCaputoCoordinateMemory, StepperContext, WorldlineState, WorldlineStepper,
};
use scirust_relativity::Minkowski;

/// A memory law that records the affine parameter of the last retained
/// history entry each time it is evaluated, and otherwise returns zero memory.
/// This lets a test observe exactly which parameter the stepper recorded for
/// its provisional predictor point.
#[derive(Clone)]
struct ParameterCapturingMemory {
    seen_last_parameters: Rc<RefCell<Vec<f64>>>,
}

impl<const D: usize> MemoryLaw<D> for ParameterCapturingMemory {
    fn memory_vector<H, T>(
        &self,
        history: &H,
        _transport: &T,
        _current_state: &WorldlineState<D>,
        _step_index: usize,
        _config: NonlocalConfig,
    ) -> NonlocalResult<[f64; D]>
    where
        H: HistoryBackend<D>,
        T: HistoryTransport<D>,
    {
        let retained = history.retained_samples();
        if retained > 0
        {
            let entry = history.entry(retained - 1).expect("entry available");
            self.seen_last_parameters.borrow_mut().push(entry.parameter);
        }
        Ok([0.0; D])
    }
}

fn history_with_parameters(parameters: &[f64]) -> CompleteUniformHistory<4> {
    let mut history = CompleteUniformHistory::<4>::new();
    for (index, &parameter) in parameters.iter().enumerate()
    {
        let velocity = [1.2 + 0.01 * index as f64, 0.05 * index as f64, 0.0, 0.0];
        history
            .push_entry(
                &Minkowski,
                &IdentityHistoryTransport,
                HistoryEntry::new([0.0, 0.0, 0.0, 0.0], velocity, parameter),
            )
            .unwrap();
    }
    history
}

#[test]
fn heun_pece_records_provisional_parameter_from_current_parameter() {
    let history = history_with_parameters(&[0.0, 0.1, 0.2]);
    let state = WorldlineState::new([0.0, 0.0, 0.0, 0.0], [1.2, 0.1, 0.0, 0.0]);
    let accepted_acceleration = [0.0, 0.0, 0.0, 0.0];
    let config = NonlocalConfig::new(0.5, 0.02, 0.1, 1, 1.0e-8).unwrap();

    let run = |current_parameter: f64, step_index: usize| -> f64 {
        let capture = ParameterCapturingMemory {
            seen_last_parameters: Rc::new(RefCell::new(Vec::new())),
        };
        HeunPeceStepper
            .advance(StepperContext {
                background: &Minkowski,
                state: &state,
                accepted_acceleration: &accepted_acceleration,
                history: &history,
                memory_law: &capture,
                transport: &IdentityHistoryTransport,
                initial_metric_norm: -1.43,
                current_parameter,
                step_index,
                config,
            })
            .unwrap();
        let seen = capture.seen_last_parameters.borrow();
        assert_eq!(seen.len(), 1, "predictor evaluates memory exactly once");
        seen[0]
    };

    // current_parameter = 0.2, step = 0.1 -> provisional point at
    // current_parameter + step, even though step_index * step = 5 * 0.1 = 0.5
    // is a completely different value.
    let step = config.step();
    assert_eq!(run(0.2, 5).to_bits(), (0.2_f64 + step).to_bits());
    // The recorded parameter tracks current_parameter, not step_index.
    assert_eq!(run(0.9, 5).to_bits(), (0.9_f64 + step).to_bits());
    // Changing only step_index leaves the recorded provisional parameter
    // unchanged: the old `step_index * step` reconstruction is gone.
    assert_eq!(run(0.2, 987).to_bits(), (0.2_f64 + step).to_bits());
}

#[test]
fn heun_pece_result_depends_on_current_parameter_not_step_index() {
    // A genuine non-uniform memory law: the corrector depends on the
    // provisional point's parameter through `caputo_l1_nonuniform`.
    let history = history_with_parameters(&[0.0, 0.1, 0.2]);
    let state = WorldlineState::new([0.0, 0.0, 0.0, 0.0], [1.2, 0.1, 0.0, 0.0]);
    let accepted_acceleration = [0.01, 0.02, 0.0, 0.0];
    let config = NonlocalConfig::new(0.5, 0.05, 0.1, 1, 1.0e-8).unwrap();

    let run = |current_parameter: f64, step_index: usize| -> WorldlineState<4> {
        HeunPeceStepper
            .advance(StepperContext {
                background: &Minkowski,
                state: &state,
                accepted_acceleration: &accepted_acceleration,
                history: &history,
                memory_law: &NonuniformCaputoCoordinateMemory,
                transport: &IdentityHistoryTransport,
                initial_metric_norm: -1.43,
                current_parameter,
                step_index,
                config,
            })
            .unwrap()
    };

    let baseline = run(0.2, 5);
    let different_step_index = run(0.2, 987);
    let different_parameter = run(0.5, 5);

    // step_index does not affect the numerical result at all.
    for component in 0..4
    {
        assert_eq!(
            baseline.coordinates[component].to_bits(),
            different_step_index.coordinates[component].to_bits(),
            "coordinate {component} changed with step_index"
        );
        assert_eq!(
            baseline.velocity[component].to_bits(),
            different_step_index.velocity[component].to_bits(),
            "velocity {component} changed with step_index"
        );
    }

    // current_parameter does affect the result (the provisional point's
    // parameter feeds the non-uniform Caputo memory).
    let differs = (0..4).any(|component| {
        baseline.velocity[component].to_bits() != different_parameter.velocity[component].to_bits()
    });
    assert!(
        differs,
        "result did not depend on current_parameter as expected"
    );
}
