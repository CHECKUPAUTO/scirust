//! Tensor runtimes for SciRust.
//!
//! This crate holds four layers. The first three are the canonical pipeline,
//! each stacked on the one below; the fourth is the historical runtime,
//! unrelated to any of them. Nothing migrates implicitly between them.
//!
//! ```text
//! CanonicalProgram / CanonicalSession   TensorND in, TensorND out
//!   ReferenceGraphSession               canonical Graph -> prepared session
//!     ReferencePlanRuntime              prepared LoweredPlan -> executions
//!
//! TensorRuntime                         the historical named-register machine
//! ```
//!
//! Start at the top. The two layers under it exist for callers who already hold
//! a `scirust_tensor_ir::Graph` or a `scirust_tensor_compile::LoweredPlan`.
//!
//! # [`CanonicalProgram`] — the user-facing façade
//!
//! Builds a computation from opaque handles, prepares it once, and runs it as
//! often as wanted, speaking only [`scirust_tensor_core::TensorND`]:
//!
//! ```text
//! CanonicalProgram          input / constant / add / relu / … / set_outputs
//!   -> prepare(runtime)     compile, lower, prepare — once
//!   -> CanonicalSession
//!   -> CanonicalInputs      TensorND keyed by handle
//!   -> execute
//!   -> CanonicalOutputs     owned TensorND, in the declared output order
//! ```
//!
//! A caller of this layer never names a `Graph`, a `NodeId`, a `TensorType`, a
//! `DType`, a `ConstantId`, a `LogicalBindingId`, a [`GraphInputs`], a
//! [`GraphConstants`] or a [`PlanOutputs`], and never sees a backend buffer,
//! stream or event.
//!
//! Points worth knowing before using it:
//!
//! * **Tensor type.** [`scirust_tensor_core::TensorND`] — dense row-major `f32`,
//!   no device, no dtype field. The eager 2-D
//!   `scirust_core::autodiff::reverse::Tensor` is untouched and unrelated; it
//!   could not express a scalar, a rank-3 value or a zero dimension anyway.
//! * **Conversions are free.** An input lends `&tensor.data[..]`; an output
//!   moves its `Vec<f32>` into a new tensor. No copy, no byte reinterpretation.
//! * **Eight operations.** `add`, `sub`, `mul`, `div`, `relu`, `scale`,
//!   `reshape`, `permute`. `exp`, `log` and `matmul` are deliberately absent —
//!   the layers below reject them, so exposing them would advertise a capability
//!   that does not exist.
//! * **No broadcasting.** Binary operations require identical shapes, because
//!   the canonical IR compares whole tensor types.
//! * **Constants are owned by the program** and injected on every execution; a
//!   caller supplies them once, at construction, and never again.
//! * **Preparation consumes the program**, which makes single preparation
//!   structural rather than merely documented.
//!
//! # [`ReferenceGraphSession`] — the canonical graph façade
//!
//! Compiles, lowers and prepares a `scirust_tensor_ir::Graph` **once**, then
//! runs it as often as wanted:
//!
//! ```text
//! Graph + GraphConstants
//!   -> ReferenceGraphSession::prepare
//!   -> GraphInputs keyed by NodeId
//!   -> ReferenceGraphSession::execute
//!   -> PlanOutputs, in the graph's output order
//! ```
//!
//! What it changes for a caller:
//!
//! * inputs are addressed by `scirust_tensor_ir::NodeId`, never by the internal
//!   `LogicalBindingId`, which stays entirely inside the session;
//! * constants are never asked for per execution — see below;
//! * outputs come back as [`PlanOutputs`], reused as is, in the graph's declared
//!   order, duplicated outputs included;
//! * nothing in [`ReferenceGraphSession::execute`] recompiles, re-lowers,
//!   regenerates an artefact or rebuilds binding metadata;
//! * the session holds no execution state at all, so two runs are independent
//!   and a failed one leaves it immediately reusable.
//!
//! ## Constant payloads live outside the graph
//!
//! `scirust_tensor_ir` stores no tensor payload: an `Operation::Constant`
//! carries a `ConstantId` and a `TensorType`, and that is the IR's deliberate
//! design. Values therefore come from the caller through [`GraphConstants`],
//! **once, at preparation**. The session clones only the constants that survive
//! dead-code elimination, owns them for its whole life, and injects them itself
//! on every execution. Bits are preserved verbatim — signed zeros, NaN payloads,
//! infinities and subnormals included.
//!
//! Likewise, an input eliminated as dead is not a required input of the session:
//! the prepared plan, not the source graph, decides what a caller must supply.
//!
//! Same limits as the plan runtime below: `F32` only, `Exp` and `Log` rejected.
//!
//! # [`ReferencePlanRuntime`] — the canonical plan runtime
//!
//! Executes a `scirust_tensor_compile::LoweredPlan` end to end on any
//! [`scirust_compute::ComputeBackend`] able to run Reference kernels:
//!
//! ```text
//! Graph -> CanonicalCompiler -> ExecutionPlan -> MemoryPlan
//!       -> KernelLowerer -> LoweredPlan
//!       -> ReferencePlanRuntime::prepare   (generate, validate, compile once)
//!       -> ReferencePlanRuntime::execute   (allocate, import, dispatch, read back)
//! ```
//!
//! Preparation and execution are separate, so one prepared plan serves many
//! executions with different values.
//!
//! ## Backend requirements
//!
//! The runtime is generic over [`scirust_compute::ComputeBackend`], but that
//! does not mean every backend can run a Reference plan. `DeviceCapabilities`
//! exposes neither an alignment nor a memory space, so the runtime cannot
//! negotiate them: it allocates with `alignment = 1` and
//! [`scirust_compute::MemorySpace::Host`], the placement the CPU Reference
//! adapter requires. A backend handed to [`ReferencePlanRuntime`] must accept
//! [`scirust_compute::KernelFormat::Reference`] modules, host memory at
//! alignment `1`, [`scirust_compute::DType::F32`], and the canonical
//! `[1, 1, 1]` grid and block with no shared memory. What the contract *does*
//! expose — `F32` support, the workgroup limit and `max_buffer_bytes` — is
//! checked at preparation, and no advertised limit is ever bypassed.
//!
//! ## Determinism
//!
//! The plan runtime contains no `HashMap`. Identifiers are dense, so every
//! table is a `Vec` indexed by identifier and every traversal follows a
//! canonical order: kernels are generated and compiled in `LogicalKernelId`
//! order, buffer slots are allocated in `BufferSlot` order, external values are
//! imported in `LogicalBindingId` order, dispatches run in the exact order of
//! `LoweredPlan::instructions`, and outputs are read back in the exact order of
//! `LoweredPlan::outputs`. The order in which a caller supplies external values
//! has no effect — they are resolved by identifier. There is no clock, no
//! random identifier and no concurrency. Numeric behaviour is entirely the
//! backend's: this runtime only moves bytes and issues dispatches.
//!
//! ## Supported operations
//!
//! `Relu`, `Scale`, `Add`, `Sub`, `Mul`, `Div`, `ShapeCopy` and `Permute`.
//! `Exp` and `Log` are rejected by [`ReferencePlanRuntime::prepare`] itself,
//! before any kernel is compiled: the executable opcode set is a property of
//! this runtime, not of whichever backend happens to be supplied.
//!
//! # [`TensorRuntime`] — the historical register machine
//!
//! A named-register machine that executes compiled element-wise kernels and
//! multi-operand contractions over [`TensorND`] values, driving
//! `scirust_tensor_compile`'s original `TensorGraph`/`FusedKernel` API. It
//! predates the canonical pipeline, is unrelated to it, and is preserved
//! unchanged: same behaviour, same `String` errors, same
//! `HashMap<String, TensorND>` registers, and the same unimplemented
//! `OptimizedContraction` path in [`TensorRuntime::run_graph`]. Nothing in this
//! phase migrates it.

mod canonical;
mod error;
mod graph_session;
mod reference_plan;

pub use canonical::{
    CanonicalInput, CanonicalInputSpec, CanonicalInputs, CanonicalOutputSpec, CanonicalOutputs,
    CanonicalProgram, CanonicalSession, CanonicalValue,
};
pub use error::{
    CanonicalBuildError, CanonicalExecutionError, CanonicalPreparationError,
    GraphSessionExecutionError, GraphSessionPreparationError, PlanExecutionError,
    PlanPreparationError,
};
pub use graph_session::{
    GraphConstantSpec, GraphConstants, GraphInputSpec, GraphInputs, GraphOutputSpec,
    ReferenceGraphSession,
};
pub use reference_plan::{
    BufferSlotSpecification, ExternalValueSpec, PlanExternalValues, PlanOutputSpec,
    PlanOutputValue, PlanOutputs, PreparedReferencePlan, ReferencePlanRuntime,
};

use scirust_tensor_compile::{ContractionPlan, FusedKernel, FusedOp, TensorGraph, TensorOp};
use scirust_tensor_core::TensorND;
use std::collections::HashMap;

/// The historical named-register runtime.
///
/// Unrelated to [`ReferencePlanRuntime`]; see the crate documentation.
#[derive(Default)]
pub struct TensorRuntime {
    regs: HashMap<String, TensorND>,
}

impl TensorRuntime {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a tensor to a register name.
    pub fn set(&mut self, name: &str, tensor: TensorND) {
        self.regs.insert(name.to_string(), tensor);
    }

    pub fn get(&self, name: &str) -> Option<&TensorND> {
        self.regs.get(name)
    }

    /// Apply a fused element-wise kernel from `input` into `output`.
    pub fn run_fused(
        &mut self,
        input: &str,
        kernel: &FusedKernel,
        output: &str,
    ) -> Result<(), String> {
        let t = self
            .regs
            .get(input)
            .ok_or_else(|| format!("unknown register '{input}'"))?;
        let r = kernel.apply(t);
        self.regs.insert(output.to_string(), r);
        Ok(())
    }

    /// Execute a contraction plan over the named input registers into `output`.
    pub fn run_contraction(
        &mut self,
        plan: &ContractionPlan,
        inputs: &[&str],
        output: &str,
    ) -> Result<(), String> {
        let mut tensors = Vec::with_capacity(inputs.len());
        for name in inputs
        {
            tensors.push(
                self.regs
                    .get(*name)
                    .ok_or_else(|| format!("unknown register '{name}'"))?,
            );
        }
        let r = plan.execute(&tensors)?;
        self.regs.insert(output.to_string(), r);
        Ok(())
    }

    /// Execute a full `TensorGraph`. This implementation currently uses the graph's
    /// internal buffer indices as temporary register slots.
    pub fn run_graph(&mut self, graph: &TensorGraph) -> Result<TensorND, String> {
        let mut buffers = graph.buffers.clone();

        for op in &graph.ops
        {
            match op
            {
                TensorOp::MatMul(a, b) =>
                {
                    let res =
                        scirust_tensor_einsum::einsum("ij,jk->ik", &[&buffers[*a], &buffers[*b]])?;
                    buffers.push(res);
                },
                TensorOp::Add(a, b) =>
                {
                    if buffers[*a].shape != buffers[*b].shape
                    {
                        return Err("Add: shape mismatch".to_string());
                    }
                    let data = buffers[*a]
                        .data
                        .iter()
                        .zip(&buffers[*b].data)
                        .map(|(x, y)| x + y)
                        .collect();
                    buffers.push(TensorND::new(data, buffers[*a].shape.clone()));
                },
                TensorOp::ReLU(a) =>
                {
                    let data = buffers[*a].data.iter().map(|x| x.max(0.0)).collect();
                    buffers.push(TensorND::new(data, buffers[*a].shape.clone()));
                },
                TensorOp::Fused(fused) =>
                {
                    match fused
                    {
                        FusedOp::Linear {
                            input_idx,
                            weight_idx,
                            bias_idx,
                            activation,
                        } =>
                        {
                            let mut res = scirust_tensor_einsum::einsum(
                                "ij,jk->ik",
                                &[&buffers[*input_idx], &buffers[*weight_idx]],
                            )?;
                            if let Some(b_idx) = bias_idx
                            {
                                // Simple bias addition (assumes bias is a row vector)
                                if buffers[*b_idx].data.len() != res.shape[1]
                                {
                                    return Err("Bias dimension mismatch".to_string());
                                }
                                for r in 0..res.shape[0]
                                {
                                    for c in 0..res.shape[1]
                                    {
                                        res.data[r * res.shape[1] + c] += buffers[*b_idx].data[c];
                                    }
                                }
                            }
                            if let Some(kernel) = activation
                            {
                                res = kernel.apply(&res);
                            }
                            buffers.push(res);
                        },
                        FusedOp::OptimizedContraction(_plan) =>
                        {
                            // This is a simplification: OptimizedContraction in a graph
                            // would need to know which buffers to use as inputs.
                            return Err(
                                "OptimizedContraction in TensorGraph not yet fully implemented"
                                    .to_string(),
                            );
                        },
                    }
                },
            }
        }

        buffers.pop().ok_or_else(|| "Graph empty".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_tensor_compile::{ElementwiseOp, GraphCompiler};

    #[test]
    fn executes_contraction_then_fused_activation() {
        let mut rt = TensorRuntime::new();
        rt.set("a", TensorND::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]));
        rt.set("b", TensorND::new(vec![1., 0., 0., 1., 1., 0.], vec![3, 2]));

        let plan = ContractionPlan::new("ij,jk->ik").unwrap();
        rt.run_contraction(&plan, &["a", "b"], "c").unwrap();

        // Then subtract 10 and ReLU, fused in one pass.
        let kernel = GraphCompiler::new()
            .op(ElementwiseOp::AddScalar(-10.0))
            .op(ElementwiseOp::Relu)
            .compile();
        rt.run_fused("c", &kernel, "out").unwrap();

        let out = rt.get("out").unwrap();
        assert_eq!(out.shape, vec![2, 2]);
        // c = [[1+? ...]] -> verify all non-negative after relu
        assert!(out.data.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn errors_on_missing_register() {
        let mut rt = TensorRuntime::new();
        let plan = ContractionPlan::new("ij->ij").unwrap();
        assert!(rt.run_contraction(&plan, &["nope"], "x").is_err());
    }
}
