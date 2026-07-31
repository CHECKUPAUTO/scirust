//! The canonical tensor facade executed on a CUDA device, through the public
//! `scirust` entry point only.
//!
//! ```text
//! CanonicalProgram -> CanonicalSession -> CudaReferenceAdapter
//!                  -> generated CUDA C -> NVRTC -> PTX -> GPU -> TensorND
//! ```
//!
//! Every import below is `scirust::…`: if a type the CUDA path needs were
//! missing from `scirust::tensor_canonical`, this file would not compile.
//!
//! # Running these
//!
//! Without the CUDA driver, without NVRTC or without a device there is no
//! adapter and each test skips. Set `SCIRUST_REQUIRE_CUDA=1` to turn that skip
//! into a failure — the self-hosted Jetson job does, so a silent no-op there
//! cannot masquerade as a pass.

#![cfg(feature = "tensor-canonical-cuda")]

use scirust::tensor_canonical::{
    CanonicalInput, CanonicalInputs, CanonicalOutputs, CanonicalProgram, CanonicalSession,
    CpuComputeAdapter, CudaReferenceAdapter, ReferencePlanRuntime, TensorND,
};

/// The device these tests run on. Explicit on purpose: this adapter has no
/// default ordinal and never falls back to another device.
const DEVICE_ORDINAL: usize = 0;

fn adapter_or_skip() -> Option<CudaReferenceAdapter> {
    match CudaReferenceAdapter::new(DEVICE_ORDINAL)
    {
        Ok(adapter) =>
        {
            let info = adapter.device_info();
            eprintln!(
                "cuda device {}: {} sm_{}{}",
                info.ordinal, info.name, info.compute_capability.0, info.compute_capability.1
            );
            Some(adapter)
        },
        Err(error) =>
        {
            assert!(
                std::env::var_os("SCIRUST_REQUIRE_CUDA").is_none(),
                "SCIRUST_REQUIRE_CUDA is set, so a real CUDA device is mandatory, but none \
                 could be acquired: {error}"
            );
            eprintln!("skipping: no CUDA device available ({error})");
            None
        },
    }
}

fn tensor(values: &[f32], shape: &[usize]) -> TensorND {
    TensorND::try_new(values.to_vec(), shape.to_vec()).expect("consistent tensor")
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// `x -> add(ones) -> relu -> permute -> reshape -> output`, on any backend.
fn build_program() -> (CanonicalProgram, CanonicalInput) {
    let mut program = CanonicalProgram::new();

    let x = program.input("x", &[2, 3]).expect("input");
    let bias = program
        .constant(tensor(&[1.0; 6], &[2, 3]))
        .expect("constant");

    let biased = program.add(x, bias).expect("add");
    let activated = program.relu(biased).expect("relu");
    let transposed = program.permute(activated, &[1, 0]).expect("permute");
    let flattened = program.reshape(transposed, &[6]).expect("reshape");
    program.set_outputs([flattened]).expect("outputs");

    (program, x)
}

fn run<B: scirust::tensor_canonical::ComputeBackend>(
    session: &CanonicalSession<B>,
    input: CanonicalInput,
    values: &TensorND,
) -> CanonicalOutputs {
    let mut inputs = CanonicalInputs::new();
    inputs.bind(input, values);
    session.execute(&inputs).expect("successful run")
}

#[test]
fn the_facade_runs_on_a_cuda_device() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let (program, x) = build_program();
    let session = program
        .prepare(ReferencePlanRuntime::new(adapter))
        .expect("preparable program");

    assert_eq!(
        session.inputs().len(),
        1,
        "the constant is the session's own"
    );
    assert_eq!(session.outputs().len(), 1);
    assert_eq!(session.outputs()[0].shape, vec![6]);

    let values = tensor(&[-3.0, -1.0, 0.0, 1.0, 4.0, 9.0], &[2, 3]);
    let outputs = run(&session, x, &values);

    // +1 gives [-2, 0, 1, 2, 5, 10]; relu gives [0, 0, 1, 2, 5, 10]; the
    // transpose of [[0,0,1],[2,5,10]] flattens to [0, 2, 0, 5, 1, 10].
    assert_eq!(
        outputs.values()[0].data,
        vec![0.0, 2.0, 0.0, 5.0, 1.0, 10.0]
    );
    assert_eq!(outputs.values()[0].shape, vec![6]);
}

#[test]
fn the_cuda_and_cpu_backends_agree_on_a_multi_operation_program() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let values = tensor(&[-3.5, -1.25, 0.0, 1.5, 4.75, 9.125], &[2, 3]);

    let (program, x) = build_program();
    let gpu_session = program
        .prepare(ReferencePlanRuntime::new(adapter))
        .expect("preparable on cuda");
    let gpu = run(&gpu_session, x, &values).into_values();

    let (program, x) = build_program();
    let cpu_session = program
        .prepare(ReferencePlanRuntime::new(CpuComputeAdapter::new()))
        .expect("preparable on cpu");
    let cpu = run(&cpu_session, x, &values).into_values();

    assert_eq!(gpu[0].shape, cpu[0].shape);
    assert_eq!(gpu[0].data.len(), 6);

    // The program ends in permute + reshape, which move raw words, and the
    // arithmetic before them is exact on these values — so this comparison is
    // legitimately bitwise. See the module docs for where that stops holding.
    assert_eq!(
        bits(&gpu[0].data),
        bits(&cpu[0].data),
        "cuda and cpu disagree: {:?} vs {:?}",
        gpu[0].data,
        cpu[0].data
    );
}

#[test]
fn one_prepared_session_serves_several_cuda_executions() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let (program, x) = build_program();
    let session = program
        .prepare(ReferencePlanRuntime::new(adapter))
        .expect("preparable program");

    let first = tensor(&[-3.0, -1.0, 0.0, 1.0, 4.0, 9.0], &[2, 3]);
    let second = tensor(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 3]);

    assert_eq!(
        run(&session, x, &first).values()[0].data,
        vec![0.0, 2.0, 0.0, 5.0, 1.0, 10.0]
    );
    // All zeros plus the constant one, relu'd, then transposed and flattened.
    assert_eq!(
        run(&session, x, &second).values()[0].data,
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    );
    // The first result reproduces: nothing was recompiled and no state carried.
    assert_eq!(
        run(&session, x, &first).values()[0].data,
        vec![0.0, 2.0, 0.0, 5.0, 1.0, 10.0]
    );
}

#[test]
fn a_rejected_input_leaves_the_cuda_session_usable() {
    use scirust::tensor_canonical::CanonicalExecutionError;

    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let (program, x) = build_program();
    let session = program
        .prepare(ReferencePlanRuntime::new(adapter))
        .expect("preparable program");

    let wrong = tensor(&[1.0; 6], &[6]);
    let mut inputs = CanonicalInputs::new();
    inputs.bind(x, &wrong);

    let error = session.execute(&inputs).expect_err("shape mismatch");
    assert!(
        matches!(error, CanonicalExecutionError::InputShapeMismatch { .. }),
        "expected a shape mismatch, got {error:?}"
    );

    let values = tensor(&[-3.0, -1.0, 0.0, 1.0, 4.0, 9.0], &[2, 3]);
    assert_eq!(
        run(&session, x, &values).values()[0].data,
        vec![0.0, 2.0, 0.0, 5.0, 1.0, 10.0]
    );
}

#[test]
fn an_absent_cuda_device_is_reported_and_never_silently_replaced() {
    // Whatever this machine has, asking for an implausible ordinal must fail —
    // it must not quietly hand back device zero, and it must not compute the
    // program on the host.
    let error = CudaReferenceAdapter::new(4096).expect_err("no such CUDA device");
    eprintln!("cuda ordinal 4096: {error}");
}
