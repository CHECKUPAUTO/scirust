//! The canonical tensor facade, end to end, on the deterministic CPU adapter.
//!
//! ```text
//! TensorND -> CanonicalProgram -> add(constant) -> relu
//!          -> CanonicalSession -> CpuComputeAdapter -> CanonicalOutputs
//! ```
//!
//! Run it with:
//!
//! ```text
//! cargo run --features tensor-canonical-cpu --example canonical_tensor_cpu
//! ```
//!
//! Everything below is imported from `scirust` alone — no `scirust_tensor_*`
//! and no `scirust_gpu` in sight, which is the point of the facade.

use scirust::tensor_canonical::{
    CanonicalInputs, CanonicalProgram, CpuComputeAdapter, ReferencePlanRuntime, TensorND,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Describe the computation. Handles are opaque: no NodeId, no graph.
    let mut program = CanonicalProgram::new();

    let x = program.input("x", &[2, 2])?;
    let bias = program.constant(TensorND::try_new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2])?)?;

    let biased = program.add(x, bias)?;
    let activated = program.relu(biased)?;
    program.set_outputs([activated])?;

    // 2. Prepare once. This compiles, lowers and prepares every kernel; the
    //    program is consumed, so it cannot be prepared twice by accident.
    let session = program.prepare(ReferencePlanRuntime::new(CpuComputeAdapter::new()))?;

    println!("inputs required : {:?}", session.inputs().len());
    println!("outputs produced: {:?}", session.outputs().len());

    // The constant is never asked of the caller — the session owns it.
    assert_eq!(session.inputs().len(), 1);
    assert_eq!(session.inputs()[0].name, "x");
    assert_eq!(session.inputs()[0].shape, vec![2, 2]);

    // 3. Execute. The same prepared session serves any number of runs.
    let first = TensorND::try_new(vec![-2.0, 0.0, 1.0, 3.0], vec![2, 2])?;
    let mut inputs = CanonicalInputs::new();
    inputs.bind(x, &first);

    let outputs = session.execute(&inputs)?;
    let result = &outputs.values()[0];

    println!(
        "relu(x + 1)      = {:?}  shape {:?}",
        result.data, result.shape
    );

    // [-2, 0, 1, 3] + 1 = [-1, 1, 2, 4], then relu clamps the negative to zero.
    assert_eq!(result.data, vec![0.0, 1.0, 2.0, 4.0]);
    assert_eq!(result.shape, vec![2, 2]);

    // 4. Run it again with different values. Nothing is recompiled, and the
    //    constant is injected automatically, identically, every time.
    let second = TensorND::try_new(vec![10.0, -10.0, 0.5, -0.5], vec![2, 2])?;
    let mut again = CanonicalInputs::new();
    again.bind(x, &second);

    let outputs = session.execute(&again)?;
    let result = &outputs.values()[0];

    println!(
        "relu(x + 1)      = {:?}  shape {:?}",
        result.data, result.shape
    );
    assert_eq!(result.data, vec![11.0, 0.0, 1.5, 0.5]);

    Ok(())
}
