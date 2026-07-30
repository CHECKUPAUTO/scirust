//! The canonical tensor facade, reached through the public `scirust` entry
//! point only, executed for real on the deterministic CPU adapter.
//!
//! Two things are under test here that nothing else covers:
//!
//! 1. **The public surface.** Every import below is `scirust::…`. If a type the
//!    facade needs were missing from `scirust::tensor_canonical`, this file
//!    would fail to compile — which is the whole point of the re-export list.
//! 2. **The values.** `cargo test` builds examples but never runs them, so the
//!    assertions that matter live here.
//!
//! The file is gated on `tensor-canonical-cpu`; without it the crate compiles
//! to nothing at all, so a default `cargo test -p scirust` is unaffected.

#![cfg(feature = "tensor-canonical-cpu")]

use scirust::tensor_canonical::{
    CanonicalExecutionError, CanonicalInputs, CanonicalOutputs, CanonicalProgram, CanonicalSession,
    CpuComputeAdapter, ReferencePlanRuntime, TensorND,
};

type Session = CanonicalSession<CpuComputeAdapter>;

fn tensor(values: &[f32], shape: &[usize]) -> TensorND {
    TensorND::try_new(values.to_vec(), shape.to_vec()).expect("consistent tensor")
}

/// `x -> add(ones) -> relu -> output`, prepared on the CPU adapter.
fn biased_relu_session() -> (Session, scirust::tensor_canonical::CanonicalInput) {
    let mut program = CanonicalProgram::new();

    let x = program.input("x", &[2, 2]).expect("input");
    let bias = program
        .constant(tensor(&[1.0, 1.0, 1.0, 1.0], &[2, 2]))
        .expect("constant");

    let biased = program.add(x, bias).expect("add");
    let activated = program.relu(biased).expect("relu");
    program.set_outputs([activated]).expect("outputs");

    let session = program
        .prepare(ReferencePlanRuntime::new(CpuComputeAdapter::new()))
        .expect("preparable program");

    (session, x)
}

fn run(
    session: &Session,
    input: scirust::tensor_canonical::CanonicalInput,
    values: &TensorND,
) -> CanonicalOutputs {
    let mut inputs = CanonicalInputs::new();
    inputs.bind(input, values);
    session.execute(&inputs).expect("successful run")
}

#[test]
fn the_facade_runs_a_real_computation_on_the_cpu_backend() {
    let (session, x) = biased_relu_session();

    // The constant is the session's own; only `x` is required.
    assert_eq!(session.inputs().len(), 1);
    assert_eq!(session.inputs()[0].name, "x");
    assert_eq!(session.inputs()[0].shape, vec![2, 2]);
    assert_eq!(session.outputs().len(), 1);
    assert_eq!(session.outputs()[0].shape, vec![2, 2]);

    let values = tensor(&[-2.0, 0.0, 1.0, 3.0], &[2, 2]);
    let outputs = run(&session, x, &values);

    assert_eq!(outputs.len(), 1);
    // [-2, 0, 1, 3] + 1 = [-1, 1, 2, 4]; relu clamps the negative to zero.
    assert_eq!(outputs.values()[0].data, vec![0.0, 1.0, 2.0, 4.0]);
    assert_eq!(outputs.values()[0].shape, vec![2, 2]);
}

#[test]
fn one_prepared_session_serves_several_runs() {
    let (session, x) = biased_relu_session();

    let first = tensor(&[-2.0, 0.0, 1.0, 3.0], &[2, 2]);
    let second = tensor(&[10.0, -10.0, 0.5, -0.5], &[2, 2]);

    assert_eq!(
        run(&session, x, &first).values()[0].data,
        vec![0.0, 1.0, 2.0, 4.0]
    );
    assert_eq!(
        run(&session, x, &second).values()[0].data,
        vec![11.0, 0.0, 1.5, 0.5]
    );
    // The constant is re-injected identically, so the first result reproduces.
    assert_eq!(
        run(&session, x, &first).values()[0].data,
        vec![0.0, 1.0, 2.0, 4.0]
    );
}

#[test]
fn a_rejected_input_leaves_the_session_usable() {
    let (session, x) = biased_relu_session();

    let wrong = tensor(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let mut inputs = CanonicalInputs::new();
    inputs.bind(x, &wrong);

    let error = session.execute(&inputs).expect_err("shape mismatch");
    assert!(
        matches!(error, CanonicalExecutionError::InputShapeMismatch { .. }),
        "expected a shape mismatch, got {error:?}"
    );

    let values = tensor(&[-2.0, 0.0, 1.0, 3.0], &[2, 2]);
    assert_eq!(
        run(&session, x, &values).values()[0].data,
        vec![0.0, 1.0, 2.0, 4.0]
    );
}

#[test]
fn errors_are_reachable_and_typed_through_the_public_path() {
    use scirust::tensor_canonical::{CanonicalBuildError, CanonicalPreparationError};

    let mut program = CanonicalProgram::new();
    let a = program.input("a", &[2, 3]).expect("input");
    let b = program.input("b", &[6]).expect("input");

    // No broadcasting: the shapes must match exactly.
    assert!(matches!(
        program.add(a, b),
        Err(CanonicalBuildError::ShapeMismatch { .. })
    ));

    // A program without an output cannot be prepared.
    let error = program
        .prepare(ReferencePlanRuntime::new(CpuComputeAdapter::new()))
        .expect_err("no outputs");
    assert!(matches!(error, CanonicalPreparationError::NoOutputs));
}
