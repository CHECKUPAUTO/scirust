//! Turns a canonical `Graph` into a prepared, reusable execution session.
//!
//! # What the session removes
//!
//! Running a graph used to mean building `CanonicalCompiler`, `ExternalBindings`
//! and `KernelLowerer` by hand, then addressing every external value by the
//! `LogicalBindingId` the lowerer happened to assign. A session does all of that
//! once and lets the caller speak the graph's own vocabulary:
//!
//! ```text
//! Graph + GraphConstants
//!   -> ReferenceGraphSession::prepare     (compile, lower, prepare — once)
//!   -> GraphInputs keyed by NodeId
//!   -> ReferenceGraphSession::execute     (validate, inject constants, run)
//!   -> PlanOutputs, in the graph's output order
//! ```
//!
//! `LogicalBindingId` never appears in the session's API.
//!
//! # Constant payloads are external to the graph
//!
//! `scirust_tensor_ir::Graph` deliberately stores no tensor payload: an
//! `Operation::Constant` carries a `ConstantId` and a `TensorType`, nothing
//! more. The values therefore come from the caller, through [`GraphConstants`],
//! **once, at preparation**. The session clones only the constants that survive
//! dead-code elimination, owns them for its whole life, and re-injects them on
//! every execution. A caller never supplies a constant again.
//!
//! # Preparation once, execution many times
//!
//! [`ReferenceGraphSession::prepare`] compiles, lowers and prepares. Nothing in
//! [`ReferenceGraphSession::execute`] recompiles the graph, re-lowers it,
//! regenerates a Reference artefact, recompiles a backend kernel or rebuilds
//! binding metadata. The session holds no execution state — no buffer, no
//! stream, no event, no previous input, no previous output, no counter and no
//! cache — so two executions are entirely independent and a failed one leaves
//! the session immediately reusable.
//!
//! # Scope
//!
//! `F32` only. `Exp` and `Log` are rejected by the plan runtime, before any
//! kernel is compiled. One device, one stream, no session cache, no
//! serialisation.

use scirust_compute::{ComputeBackend, DType};
use scirust_tensor_compile::{
    CanonicalCompiler, ExternalBindings, ExternalValueKind, KernelLowerer, LogicalBindingId,
};
use scirust_tensor_ir::{ConstantId, Graph, NodeId, Operation, TensorType};

use crate::error::{GraphSessionExecutionError, GraphSessionPreparationError};
use crate::reference_plan::{
    PlanExternalValues, PlanOutputs, PreparedReferencePlan, ReferencePlanRuntime,
};

// ---------------------------------------------------------------------------
// Public metadata
// ---------------------------------------------------------------------------

/// One graph input a session needs before it can run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphInputSpec {
    pub node: NodeId,
    pub tensor_type: TensorType,
}

/// One graph constant a session injects on its own.
///
/// Metadata only: the payload stays private to the session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphConstantSpec {
    pub node: NodeId,
    pub constant: ConstantId,
    pub tensor_type: TensorType,
}

/// One value a session produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphOutputSpec {
    pub node: NodeId,
    pub tensor_type: TensorType,
}

/// The input values one execution needs, addressed by graph `NodeId`.
///
/// Insertion order has no functional effect: everything is resolved by
/// identifier. A duplicate node is accepted while building and rejected by
/// [`ReferenceGraphSession::execute`], before any backend call.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GraphInputs<'a> {
    entries: Vec<(NodeId, &'a [f32])>,
}

impl<'a> GraphInputs<'a> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Supplies one input value. Chainable.
    pub fn bind(&mut self, node: NodeId, values: &'a [f32]) -> &mut Self {
        self.entries.push((node, values));
        self
    }

    pub fn entries(&self) -> &[(NodeId, &'a [f32])] {
        &self.entries
    }
}

/// The constant payloads a graph refers to, addressed by `ConstantId`.
///
/// Supplied once, to [`ReferenceGraphSession::prepare`]. A payload for a
/// constant dead-code elimination removed is accepted and ignored, so one store
/// covering a whole graph works whatever the elimination result; a payload for a
/// `ConstantId` the graph never references is rejected, so a mistyped identifier
/// cannot pass unnoticed.
///
/// Insertion order has no functional effect.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GraphConstants<'a> {
    entries: Vec<(ConstantId, &'a [f32])>,
}

impl<'a> GraphConstants<'a> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Supplies one constant payload. Chainable.
    pub fn bind(&mut self, constant: ConstantId, values: &'a [f32]) -> &mut Self {
        self.entries.push((constant, values));
        self
    }

    pub fn entries(&self) -> &[(ConstantId, &'a [f32])] {
        &self.entries
    }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// One constant payload, owned by the session.
///
/// Stored once per `ConstantId`, not once per node: several constant nodes may
/// share an identifier, and all of them then read the same values.
#[derive(Debug, Clone, PartialEq)]
struct PreparedConstantPayload {
    constant: ConstantId,
    values: Vec<f32>,
}

/// One constant binding of the prepared plan, pointing at its payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PreparedConstantBinding {
    binding: LogicalBindingId,
    payload_index: usize,
}

/// A compiled, prepared graph ready to run any number of times.
///
/// Immutable: `execute` takes `&self` and the session contains no `Cell`,
/// `RefCell`, `Mutex`, counter or cache. `Send` and `Sync` are never imposed;
/// they follow from `B` and `B::Kernel`.
#[derive(Debug)]
pub struct ReferenceGraphSession<B: ComputeBackend> {
    runtime: ReferencePlanRuntime<B>,
    prepared: PreparedReferencePlan<B::Kernel>,
    /// Surviving inputs, in ascending `LogicalBindingId` order.
    inputs: Vec<GraphInputSpec>,
    /// Binding of `inputs[i]`.
    input_bindings: Vec<LogicalBindingId>,
    /// Element count of `inputs[i]`, computed once.
    input_lengths: Vec<usize>,
    /// Surviving constants, in ascending `LogicalBindingId` order.
    constants: Vec<GraphConstantSpec>,
    /// Binding and payload of `constants[i]`.
    constant_bindings: Vec<PreparedConstantBinding>,
    /// Deduplicated constant payloads, in first-use order.
    payloads: Vec<PreparedConstantPayload>,
    /// Outputs, in the graph's declared order.
    outputs: Vec<GraphOutputSpec>,
}

impl<B: ComputeBackend> ReferenceGraphSession<B> {
    /// Compiles, lowers and prepares `graph`, exactly once.
    ///
    /// The graph, its execution plan and its lowered plan are all dropped before
    /// this returns: the session keeps only what an execution needs.
    ///
    /// Constant payloads are validated in full *before* the plan is prepared, so
    /// a missing, duplicated, unknown, mistyped or wrongly sized constant costs
    /// no backend compilation at all.
    pub fn prepare(
        runtime: ReferencePlanRuntime<B>,
        graph: &Graph,
        constants: &GraphConstants<'_>,
    ) -> Result<Self, GraphSessionPreparationError> {
        // 1. Canonical compilation. Validates the graph and eliminates every
        //    node unreachable from the declared outputs.
        let execution_plan = CanonicalCompiler::new()
            .compile(graph)
            .map_err(|source| GraphSessionPreparationError::GraphCompilation { source })?;

        // 2. External binding table, in the canonical order of the retained
        //    instructions. The caller never chooses this order.
        let bindings = ExternalBindings::derive(&execution_plan);

        // 3. Lowering.
        let lowered = KernelLowerer::new()
            .lower(&execution_plan, &bindings)
            .map_err(|source| GraphSessionPreparationError::KernelLowering { source })?;

        // 4-6. Walk the plan's own binding table — the post-elimination source
        //      of truth — and correlate every entry with the graph.
        let resolved = resolve_bindings(graph, &lowered)?;

        // 7-8. Validate the whole constant store, then clone only what survives.
        let (payloads, constant_bindings) = resolve_constants(graph, &resolved, constants)?;

        // 9. Physical preparation. The graph-supplied types complete the
        //    metadata a lowered plan cannot carry.
        let prepared = runtime
            .prepare_with_external_types(&lowered, &resolved.external_types)
            .map_err(|source| GraphSessionPreparationError::PlanPreparation { source })?;

        // 10. The prepared plan must describe exactly the bindings we resolved.
        check_prepared_externals(&prepared, &resolved)?;

        // 11. The prepared outputs must be the graph's outputs, in order.
        let outputs = resolve_outputs(graph, &prepared)?;

        // 12. Immutable session.
        Ok(Self {
            runtime,
            prepared,
            inputs: resolved.inputs,
            input_bindings: resolved.input_bindings,
            input_lengths: resolved.input_lengths,
            constants: resolved.constants,
            constant_bindings,
            payloads,
            outputs,
        })
    }

    pub const fn runtime(&self) -> &ReferencePlanRuntime<B> {
        &self.runtime
    }

    pub const fn prepared_plan(&self) -> &PreparedReferencePlan<B::Kernel> {
        &self.prepared
    }

    /// Inputs this session requires, in ascending `LogicalBindingId` order.
    ///
    /// An input dead-code elimination removed is absent: it is not required, and
    /// supplying it is an error.
    pub fn inputs(&self) -> &[GraphInputSpec] {
        &self.inputs
    }

    /// Constants this session injects on its own, in ascending
    /// `LogicalBindingId` order. Payloads are not exposed.
    pub fn constants(&self) -> &[GraphConstantSpec] {
        &self.constants
    }

    /// Outputs this session produces, in the graph's declared order.
    pub fn outputs(&self) -> &[GraphOutputSpec] {
        &self.outputs
    }

    /// Runs the prepared session over `inputs`.
    ///
    /// Every input is validated — known, required, unique, complete, exactly
    /// sized — before a single backend call is issued, so a rejected set of
    /// inputs allocates nothing, writes nothing and launches nothing. Constants
    /// are added afterwards, from the session's own storage.
    ///
    /// The order the caller bound values in has no effect: values are placed at
    /// their binding's position either way.
    pub fn execute(
        &self,
        inputs: &GraphInputs<'_>,
    ) -> Result<PlanOutputs, GraphSessionExecutionError> {
        let supplied = self.resolve_inputs(inputs)?;

        let mut values = PlanExternalValues::new();

        // Inputs, in ascending binding order.
        for ((&binding, &value), _) in self
            .input_bindings
            .iter()
            .zip(&supplied)
            .zip(&self.input_lengths)
        {
            values.bind(binding, value);
        }

        // Constants, in ascending binding order.
        for entry in &self.constant_bindings
        {
            let payload = self.payloads.get(entry.payload_index).ok_or(
                GraphSessionExecutionError::UnresolvedConstantPayload {
                    binding: entry.binding,
                },
            )?;
            values.bind(entry.binding, &payload.values);
        }

        self.runtime
            .execute(&self.prepared, &values)
            .map_err(|source| GraphSessionExecutionError::PlanExecution { source })
    }

    /// Validates the caller's inputs and orders them by binding.
    ///
    /// Runs to completion before anything else, and touches no backend state.
    fn resolve_inputs<'a>(
        &self,
        inputs: &GraphInputs<'a>,
    ) -> Result<Vec<&'a [f32]>, GraphSessionExecutionError> {
        let mut slots: Vec<Option<&'a [f32]>> = vec![None; self.inputs.len()];

        for &(node, value) in inputs.entries()
        {
            let position = self
                .inputs
                .iter()
                .position(|spec| spec.node == node)
                .ok_or(GraphSessionExecutionError::UnexpectedInput { node })?;

            match slots.get_mut(position)
            {
                Some(entry) =>
                {
                    if entry.is_some()
                    {
                        return Err(GraphSessionExecutionError::DuplicateInput { node });
                    }
                    *entry = Some(value);
                },
                // Defensive: `position` indexes the very vector `slots` was
                // sized from.
                None => return Err(GraphSessionExecutionError::UnexpectedInput { node }),
            }
        }

        let mut resolved = Vec::with_capacity(self.inputs.len());
        for ((spec, &expected), entry) in self.inputs.iter().zip(&self.input_lengths).zip(slots)
        {
            let value =
                entry.ok_or(GraphSessionExecutionError::MissingInput { node: spec.node })?;

            if value.len() != expected
            {
                return Err(GraphSessionExecutionError::InputLengthMismatch {
                    node: spec.node,
                    expected,
                    actual: value.len(),
                });
            }

            resolved.push(value);
        }

        Ok(resolved)
    }
}

// ---------------------------------------------------------------------------
// Preparation helpers
// ---------------------------------------------------------------------------

/// One constant binding, before its payload is known.
struct ConstantBinding {
    binding: LogicalBindingId,
    node: NodeId,
    constant: ConstantId,
    elements: usize,
}

/// Everything the binding table and the graph agree on.
struct ResolvedBindings {
    /// Graph-supplied tensor type of every binding, in binding order.
    external_types: Vec<(NodeId, TensorType)>,
    /// Kind of every binding, in binding order.
    kinds: Vec<ExternalValueKind>,
    inputs: Vec<GraphInputSpec>,
    input_bindings: Vec<LogicalBindingId>,
    input_lengths: Vec<usize>,
    constants: Vec<GraphConstantSpec>,
    constant_bindings: Vec<ConstantBinding>,
}

/// Correlates the plan's binding table with the graph's own metadata.
///
/// The binding table is the source of truth for *which* values are needed — it
/// already reflects dead-code elimination — and the graph is the source of truth
/// for *what* each of them is.
fn resolve_bindings(
    graph: &Graph,
    lowered: &scirust_tensor_compile::LoweredPlan,
) -> Result<ResolvedBindings, GraphSessionPreparationError> {
    let entries = lowered.bindings();

    let mut resolved = ResolvedBindings {
        external_types: Vec::with_capacity(entries.len()),
        kinds: Vec::with_capacity(entries.len()),
        inputs: Vec::new(),
        input_bindings: Vec::new(),
        input_lengths: Vec::new(),
        constants: Vec::new(),
        constant_bindings: Vec::new(),
    };

    for (index, entry) in entries.iter().enumerate()
    {
        let node = entry.node;
        let binding = LogicalBindingId::new(binding_index(index));

        let graph_node = graph
            .nodes()
            .get(node.get() as usize)
            .ok_or(GraphSessionPreparationError::UnknownExternalNode { node })?;

        // `ExternalValueKind` is `#[non_exhaustive]`: a future variant is
        // rejected here rather than guessed at.
        match entry.kind
        {
            ExternalValueKind::Input | ExternalValueKind::Constant =>
            {},
            _ => return Err(GraphSessionPreparationError::UnsupportedExternalKind { node }),
        }

        let tensor_type = graph_node.output.clone();
        resolved.external_types.push((node, tensor_type.clone()));
        resolved.kinds.push(entry.kind);

        match &graph_node.operation
        {
            Operation::Input { .. } =>
            {
                if entry.kind != ExternalValueKind::Input
                {
                    return Err(GraphSessionPreparationError::ExternalKindMismatch {
                        node,
                        expected: ExternalValueKind::Input,
                        actual: entry.kind,
                    });
                }

                if tensor_type.dtype != DType::F32
                {
                    return Err(GraphSessionPreparationError::UnsupportedInputDType {
                        node,
                        dtype: tensor_type.dtype,
                    });
                }

                let elements = tensor_type
                    .shape
                    .checked_num_elements()
                    .map_err(|_| GraphSessionPreparationError::InputSizeOverflow { node })?;

                resolved.inputs.push(GraphInputSpec { node, tensor_type });
                resolved.input_bindings.push(binding);
                resolved.input_lengths.push(elements);
            },
            Operation::Constant { id } =>
            {
                if entry.kind != ExternalValueKind::Constant
                {
                    return Err(GraphSessionPreparationError::ExternalKindMismatch {
                        node,
                        expected: ExternalValueKind::Constant,
                        actual: entry.kind,
                    });
                }

                if tensor_type.dtype != DType::F32
                {
                    return Err(GraphSessionPreparationError::UnsupportedConstantDType {
                        node,
                        dtype: tensor_type.dtype,
                    });
                }

                let elements = tensor_type
                    .shape
                    .checked_num_elements()
                    .map_err(|_| GraphSessionPreparationError::ConstantSizeOverflow { node })?;

                resolved.constants.push(GraphConstantSpec {
                    node,
                    constant: *id,
                    tensor_type: tensor_type.clone(),
                });
                resolved.constant_bindings.push(ConstantBinding {
                    binding,
                    node,
                    constant: *id,
                    elements,
                });
            },
            // Defensive: the memory planner classifies a value as external
            // exactly when its operation is `Input` or `Constant`.
            _ => return Err(GraphSessionPreparationError::UnknownExternalNode { node }),
        }
    }

    Ok(resolved)
}

/// Every `ConstantId` the graph references, in canonical node order, once each.
///
/// Built from the *whole* graph, before dead-code elimination, so a payload for
/// an eliminated constant is recognised as legitimate rather than mistyped.
fn referenced_constants(graph: &Graph) -> Vec<ConstantId> {
    let mut referenced = Vec::new();

    for node in graph.nodes()
    {
        if let Operation::Constant { id } = &node.operation
        {
            if !referenced.contains(id)
            {
                referenced.push(*id);
            }
        }
    }

    referenced
}

/// Validates the constant store, then clones the payloads that survive.
///
/// Diagnostics do not depend on the order the store was built in: an unknown
/// identifier is reported by its smallest value, a duplicate by the graph's own
/// constant order, and everything else in ascending binding order.
fn resolve_constants(
    graph: &Graph,
    resolved: &ResolvedBindings,
    constants: &GraphConstants<'_>,
) -> Result<
    (Vec<PreparedConstantPayload>, Vec<PreparedConstantBinding>),
    GraphSessionPreparationError,
> {
    let referenced = referenced_constants(graph);

    // A payload for a constant no node of the graph declares is a mistake worth
    // naming; one for a constant the graph declares but elimination removed is
    // simply unused.
    let mut unexpected: Option<ConstantId> = None;
    for &(constant, _) in constants.entries()
    {
        if !referenced.contains(&constant)
        {
            unexpected = Some(match unexpected
            {
                Some(smallest) if smallest <= constant => smallest,
                _ => constant,
            });
        }
    }
    if let Some(constant) = unexpected
    {
        return Err(GraphSessionPreparationError::UnexpectedConstantPayload { constant });
    }

    for &constant in &referenced
    {
        let supplied = constants
            .entries()
            .iter()
            .filter(|(id, _)| *id == constant)
            .count();

        if supplied > 1
        {
            return Err(GraphSessionPreparationError::DuplicateConstantPayload { constant });
        }
    }

    let mut payloads: Vec<PreparedConstantPayload> = Vec::new();
    let mut prepared = Vec::with_capacity(resolved.constant_bindings.len());

    for entry in &resolved.constant_bindings
    {
        let values = constants
            .entries()
            .iter()
            .find(|(id, _)| *id == entry.constant)
            .map(|&(_, values)| values)
            .ok_or(GraphSessionPreparationError::MissingConstantPayload {
                node: entry.node,
                constant: entry.constant,
            })?;

        // One payload serves every surviving node sharing this identifier, so it
        // has to satisfy each of them. Shapes may differ; element counts and
        // dtypes may not.
        if values.len() != entry.elements
        {
            return Err(GraphSessionPreparationError::ConstantLengthMismatch {
                node: entry.node,
                constant: entry.constant,
                expected: entry.elements,
                actual: values.len(),
            });
        }

        // One clone per identifier, never one per node. `to_vec` copies the
        // `f32` values verbatim: signed zeros, NaN payloads, infinities and
        // subnormals all survive untouched.
        let payload_index = match payloads
            .iter()
            .position(|payload| payload.constant == entry.constant)
        {
            Some(index) => index,
            None =>
            {
                payloads.push(PreparedConstantPayload {
                    constant: entry.constant,
                    values: values.to_vec(),
                });
                payloads.len().saturating_sub(1)
            },
        };

        prepared.push(PreparedConstantBinding {
            binding: entry.binding,
            payload_index,
        });
    }

    Ok((payloads, prepared))
}

/// Confronts the prepared plan's external metadata with the resolved bindings.
///
/// Defensive throughout: preparation was handed the plan's own binding table and
/// a type per entry, and hints can neither add, remove nor reorder a binding.
fn check_prepared_externals<K>(
    prepared: &PreparedReferencePlan<K>,
    resolved: &ResolvedBindings,
) -> Result<(), GraphSessionPreparationError> {
    let specifications = prepared.external_values();

    if specifications.len() != resolved.external_types.len()
    {
        return Err(
            GraphSessionPreparationError::PreparedExternalCountMismatch {
                expected: resolved.external_types.len(),
                actual: specifications.len(),
            },
        );
    }

    for (index, ((specification, (node, tensor_type)), kind)) in specifications
        .iter()
        .zip(&resolved.external_types)
        .zip(&resolved.kinds)
        .enumerate()
    {
        let binding = LogicalBindingId::new(binding_index(index));

        if specification.binding != binding
            || specification.node != *node
            || specification.kind != *kind
            || specification.tensor_type != *tensor_type
        {
            return Err(GraphSessionPreparationError::PreparedExternalMismatch {
                binding,
                node: *node,
            });
        }
    }

    Ok(())
}

/// Checks that the prepared outputs are the graph's outputs, in order.
///
/// Duplicated graph outputs are preserved as they are: the same node declared
/// twice yields two output values.
fn resolve_outputs<K>(
    graph: &Graph,
    prepared: &PreparedReferencePlan<K>,
) -> Result<Vec<GraphOutputSpec>, GraphSessionPreparationError> {
    let declared = graph.outputs();
    let produced = prepared.outputs();

    if declared.len() != produced.len()
    {
        return Err(GraphSessionPreparationError::OutputCountMismatch {
            graph: declared.len(),
            plan: produced.len(),
        });
    }

    let mut outputs = Vec::with_capacity(declared.len());
    for (index, (&node, specification)) in declared.iter().zip(&produced).enumerate()
    {
        if specification.node != node
        {
            return Err(GraphSessionPreparationError::OutputNodeMismatch {
                index,
                graph: node,
                plan: specification.node,
            });
        }

        let tensor_type = graph
            .nodes()
            .get(node.get() as usize)
            .map(|graph_node| graph_node.output.clone())
            .ok_or(GraphSessionPreparationError::UnknownOutputNode { node })?;

        if tensor_type != specification.tensor_type
        {
            return Err(GraphSessionPreparationError::OutputTypeMismatch {
                index,
                node,
                graph: tensor_type,
                plan: specification.tensor_type.clone(),
            });
        }

        outputs.push(GraphOutputSpec { node, tensor_type });
    }

    Ok(outputs)
}

/// Binding positions are bounded by the lowerer, which already caps the binding
/// table below `u32::MAX`, so the saturating conversion never loses information.
fn binding_index(index: usize) -> u32 {
    u32::try_from(index).unwrap_or(u32::MAX)
}

#[cfg(test)]
impl<B: ComputeBackend> ReferenceGraphSession<B> {
    /// Number of distinct payloads the session owns — one per surviving
    /// `ConstantId`, not one per constant node.
    fn payload_count(&self) -> usize {
        self.payloads.len()
    }

    /// Values stored for the payload at `index`.
    fn payload_values(&self, index: usize) -> Option<&[f32]> {
        self.payloads
            .get(index)
            .map(|payload| payload.values.as_slice())
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of the session, driven by the recording backend.
    //!
    //! The real CPU path is covered end to end in
    //! `tests/reference_graph_session.rs`. What a real backend cannot show is
    //! *when* the session touches it: that a rejected constant costs no
    //! compilation, that a rejected input costs no allocation, and that repeated
    //! executions compile nothing new.

    use super::*;

    use core::error::Error as _;
    use std::rc::Rc;

    use scirust_compute::{
        BufferBinding, ComputeResult, DeviceCapabilities, LaunchConfig, MemorySpace, Shape,
    };
    use scirust_tensor_ir::Scalar;

    use crate::error::PlanExecutionError;
    use crate::reference_plan::test_backend::{Call, RecordingBackend, RecordingBuffer};

    // -----------------------------------------------------------------------
    // A recording backend the test keeps a handle on
    // -----------------------------------------------------------------------

    /// `ReferenceGraphSession::prepare` takes the runtime by value, so a failed
    /// preparation drops the backend with it. Sharing the recorder keeps its
    /// call log observable after such a failure — which is exactly what "no
    /// backend call happened" tests need.
    #[derive(Debug, Clone)]
    struct SharedBackend(Rc<RecordingBackend>);

    impl SharedBackend {
        fn new() -> Self {
            Self(Rc::new(RecordingBackend::new()))
        }

        fn failing_at(dispatch_index: usize) -> Self {
            Self(Rc::new(
                RecordingBackend::new().fail_launch_at(dispatch_index),
            ))
        }

        fn clear_launch_failure(&self) {
            self.0.clear_launch_failure();
        }

        fn calls(&self) -> Vec<Call> {
            self.0.calls()
        }

        fn count<F: Fn(&Call) -> bool>(&self, predicate: F) -> usize {
            self.0.count(predicate)
        }
    }

    impl ComputeBackend for SharedBackend {
        type Buffer = RecordingBuffer;
        type Kernel = String;
        type Stream = ();
        type Event = ();

        fn capabilities(&self) -> &DeviceCapabilities {
            self.0.capabilities()
        }

        fn allocate(
            &self,
            bytes: usize,
            alignment: usize,
            memory_space: MemorySpace,
        ) -> ComputeResult<Self::Buffer> {
            self.0.allocate(bytes, alignment, memory_space)
        }

        fn write(
            &self,
            destination: &Self::Buffer,
            offset_bytes: usize,
            data: &[u8],
        ) -> ComputeResult<()> {
            self.0.write(destination, offset_bytes, data)
        }

        fn read(
            &self,
            source: &Self::Buffer,
            offset_bytes: usize,
            destination: &mut [u8],
        ) -> ComputeResult<()> {
            self.0.read(source, offset_bytes, destination)
        }

        fn compile(&self, module: &scirust_compute::KernelModule) -> ComputeResult<Self::Kernel> {
            self.0.compile(module)
        }

        fn create_stream(&self) -> ComputeResult<Self::Stream> {
            self.0.create_stream()
        }

        fn launch(
            &self,
            kernel: &Self::Kernel,
            stream: &Self::Stream,
            config: LaunchConfig,
            bindings: &[BufferBinding<'_, Self::Buffer>],
        ) -> ComputeResult<Self::Event> {
            self.0.launch(kernel, stream, config, bindings)
        }

        fn wait(&self, event: &Self::Event) -> ComputeResult<()> {
            self.0.wait(event)
        }

        fn synchronize(&self, stream: &Self::Stream) -> ComputeResult<()> {
            self.0.synchronize(stream)
        }
    }

    // -----------------------------------------------------------------------
    // Fixtures
    // -----------------------------------------------------------------------

    fn f32_type(dims: Vec<usize>) -> TensorType {
        TensorType::new(DType::F32, Shape::new(dims))
    }

    fn prepare(
        backend: &SharedBackend,
        graph: &Graph,
        constants: &GraphConstants<'_>,
    ) -> Result<ReferenceGraphSession<SharedBackend>, GraphSessionPreparationError> {
        ReferenceGraphSession::prepare(ReferencePlanRuntime::new(backend.clone()), graph, constants)
    }

    /// `x -> relu -> relu -> output`: two dispatches sharing one kernel.
    fn shared_kernel_graph() -> Graph {
        let ty = f32_type(vec![2]);
        let mut graph = Graph::new();
        let input = graph.add_input("x", ty.clone()).expect("input");
        let first = graph
            .add_node(Operation::Relu, vec![input], ty.clone())
            .expect("relu");
        let second = graph
            .add_node(Operation::Relu, vec![first], ty)
            .expect("relu");
        graph.set_outputs(vec![second]).expect("outputs");
        graph
    }

    /// `c -> relu -> output`, with `c` a constant of `elements` values.
    fn constant_graph(constant: ConstantId, dims: Vec<usize>) -> Graph {
        let ty = f32_type(dims);
        let mut graph = Graph::new();
        let node = graph.add_constant(constant, ty.clone()).expect("constant");
        let relu = graph
            .add_node(Operation::Relu, vec![node], ty)
            .expect("relu");
        graph.set_outputs(vec![relu]).expect("outputs");
        graph
    }

    // -----------------------------------------------------------------------
    // Preparation happens once
    // -----------------------------------------------------------------------

    #[test]
    fn preparation_compiles_each_logical_kernel_once_and_nothing_else() {
        let backend = SharedBackend::new();
        let session =
            prepare(&backend, &shared_kernel_graph(), &GraphConstants::new()).expect("preparable");

        assert_eq!(session.prepared_plan().dispatch_count(), 2);
        assert_eq!(session.prepared_plan().kernel_count(), 1);
        assert_eq!(
            backend.calls(),
            vec![Call::Compile {
                entry_point: "scirust_reference_kernel_0".to_string(),
            }],
            "preparation compiles, and does nothing else"
        );
    }

    #[test]
    fn repeated_executions_compile_nothing_new_and_keep_the_metadata_stable() {
        let backend = SharedBackend::new();
        let session =
            prepare(&backend, &shared_kernel_graph(), &GraphConstants::new()).expect("preparable");

        let node = session.inputs()[0].node;
        let before_inputs = session.inputs().to_vec();
        let before_outputs = session.outputs().to_vec();

        for values in [[-1.0f32, 2.0], [3.0, -4.0], [0.0, 0.0]]
        {
            let mut inputs = GraphInputs::new();
            inputs.bind(node, &values);
            session.execute(&inputs).expect("runs");
        }

        assert_eq!(
            backend.count(|call| matches!(call, Call::Compile { .. })),
            1,
            "three executions must not recompile anything"
        );
        assert_eq!(session.inputs(), before_inputs.as_slice());
        assert_eq!(session.outputs(), before_outputs.as_slice());
    }

    // -----------------------------------------------------------------------
    // Constants are validated before any compilation
    // -----------------------------------------------------------------------

    #[test]
    fn a_missing_constant_payload_is_rejected_before_any_compilation() {
        let backend = SharedBackend::new();
        let graph = constant_graph(ConstantId::new(7), vec![3]);

        assert_eq!(
            prepare(&backend, &graph, &GraphConstants::new()).err(),
            Some(GraphSessionPreparationError::MissingConstantPayload {
                node: NodeId::new(0),
                constant: ConstantId::new(7),
            })
        );
        assert!(
            backend.calls().is_empty(),
            "an invalid constant must cost no backend call"
        );
    }

    #[test]
    fn a_duplicated_constant_payload_is_rejected_before_any_compilation() {
        let backend = SharedBackend::new();
        let graph = constant_graph(ConstantId::new(7), vec![3]);

        let first = [1.0f32, 2.0, 3.0];
        let second = [4.0f32, 5.0, 6.0];
        let mut constants = GraphConstants::new();
        constants
            .bind(ConstantId::new(7), &first)
            .bind(ConstantId::new(7), &second);

        assert_eq!(
            prepare(&backend, &graph, &constants).err(),
            Some(GraphSessionPreparationError::DuplicateConstantPayload {
                constant: ConstantId::new(7),
            })
        );
        assert!(backend.calls().is_empty());
    }

    #[test]
    fn a_payload_for_an_unknown_constant_is_rejected_before_any_compilation() {
        let backend = SharedBackend::new();
        let graph = constant_graph(ConstantId::new(7), vec![3]);

        let values = [1.0f32, 2.0, 3.0];
        let stranger = [0.0f32];
        let mut constants = GraphConstants::new();
        constants
            .bind(ConstantId::new(7), &values)
            .bind(ConstantId::new(99), &stranger);

        assert_eq!(
            prepare(&backend, &graph, &constants).err(),
            Some(GraphSessionPreparationError::UnexpectedConstantPayload {
                constant: ConstantId::new(99),
            })
        );
        assert!(backend.calls().is_empty());
    }

    #[test]
    fn an_unknown_constant_is_reported_independently_of_insertion_order() {
        let backend = SharedBackend::new();
        let graph = constant_graph(ConstantId::new(7), vec![1]);
        let values = [1.0f32];

        let mut ascending = GraphConstants::new();
        ascending
            .bind(ConstantId::new(7), &values)
            .bind(ConstantId::new(40), &values)
            .bind(ConstantId::new(90), &values);

        let mut descending = GraphConstants::new();
        descending
            .bind(ConstantId::new(90), &values)
            .bind(ConstantId::new(40), &values)
            .bind(ConstantId::new(7), &values);

        let expected = Some(GraphSessionPreparationError::UnexpectedConstantPayload {
            constant: ConstantId::new(40),
        });
        assert_eq!(prepare(&backend, &graph, &ascending).err(), expected);
        assert_eq!(prepare(&backend, &graph, &descending).err(), expected);
    }

    #[test]
    fn a_constant_eliminated_as_dead_is_accepted_and_never_stored() {
        let ty = f32_type(vec![2]);
        let mut graph = Graph::new();
        let input = graph.add_input("x", ty.clone()).expect("input");
        let live = graph
            .add_node(Operation::Relu, vec![input], ty.clone())
            .expect("relu");
        let dead = graph
            .add_constant(ConstantId::new(5), ty.clone())
            .expect("constant");
        graph
            .add_node(Operation::Relu, vec![dead], ty)
            .expect("dead relu");
        graph.set_outputs(vec![live]).expect("outputs");

        let payload = [1.0f32, 2.0];
        let mut constants = GraphConstants::new();
        constants.bind(ConstantId::new(5), &payload);

        let backend = SharedBackend::new();
        let session = prepare(&backend, &graph, &constants).expect("dead payloads are ignored");

        assert!(
            session.constants().is_empty(),
            "an eliminated constant is not a constant of the session"
        );
        assert_eq!(
            session.payload_count(),
            0,
            "an eliminated constant must not be cloned"
        );
        assert_eq!(session.prepared_plan().external_values().len(), 1);
    }

    // -----------------------------------------------------------------------
    // One payload, several nodes
    // -----------------------------------------------------------------------

    /// Two constant nodes sharing one `ConstantId`, with different shapes but
    /// the same element count.
    fn shared_constant_graph(second_dims: Vec<usize>) -> Graph {
        let mut graph = Graph::new();
        let flat = f32_type(vec![4]);
        let other = f32_type(second_dims);

        let first = graph
            .add_constant(ConstantId::new(3), flat.clone())
            .expect("constant");
        let second = graph
            .add_constant(ConstantId::new(3), other.clone())
            .expect("constant");
        let left = graph
            .add_node(Operation::Relu, vec![first], flat)
            .expect("relu");
        let right = graph
            .add_node(Operation::Relu, vec![second], other)
            .expect("relu");
        graph.set_outputs(vec![left, right]).expect("outputs");
        graph
    }

    #[test]
    fn one_payload_serves_every_node_sharing_its_identifier() {
        let graph = shared_constant_graph(vec![2, 2]);
        let payload = [1.0f32, -2.0, 3.0, -4.0];
        let mut constants = GraphConstants::new();
        constants.bind(ConstantId::new(3), &payload);

        let backend = SharedBackend::new();
        let session = prepare(&backend, &graph, &constants).expect("preparable");

        assert_eq!(session.constants().len(), 2, "two constant nodes");
        assert_eq!(
            session.payload_count(),
            1,
            "one identifier, one stored payload"
        );
        assert_eq!(session.payload_values(0), Some(payload.as_slice()));
        assert_eq!(
            session.constants()[0].tensor_type.shape.dims(),
            &[4],
            "shapes may differ between nodes"
        );
        assert_eq!(session.constants()[1].tensor_type.shape.dims(), &[2, 2]);
    }

    #[test]
    fn a_shared_payload_must_satisfy_every_node_that_uses_it() {
        // The second node wants three elements; the payload holds four.
        let graph = shared_constant_graph(vec![3]);
        let payload = [1.0f32, 2.0, 3.0, 4.0];
        let mut constants = GraphConstants::new();
        constants.bind(ConstantId::new(3), &payload);

        let backend = SharedBackend::new();
        assert_eq!(
            prepare(&backend, &graph, &constants).err(),
            Some(GraphSessionPreparationError::ConstantLengthMismatch {
                node: NodeId::new(1),
                constant: ConstantId::new(3),
                expected: 3,
                actual: 4,
            })
        );
        assert!(backend.calls().is_empty());
    }

    // -----------------------------------------------------------------------
    // Input validation happens before the backend is touched
    // -----------------------------------------------------------------------

    #[test]
    fn an_invalid_input_costs_no_backend_call() {
        let backend = SharedBackend::new();
        let graph = shared_kernel_graph();
        let session = prepare(&backend, &graph, &GraphConstants::new()).expect("preparable");
        let node = session.inputs()[0].node;

        let before = backend.calls();
        let values = [1.0f32, 2.0];
        let short = [1.0f32];
        let stranger = NodeId::new(2);

        let mut missing = GraphInputs::new();
        assert_eq!(
            session.execute(&missing).err(),
            Some(GraphSessionExecutionError::MissingInput { node })
        );

        let mut extra = GraphInputs::new();
        extra.bind(node, &values).bind(stranger, &values);
        assert_eq!(
            session.execute(&extra).err(),
            Some(GraphSessionExecutionError::UnexpectedInput { node: stranger })
        );

        let mut duplicate = GraphInputs::new();
        duplicate.bind(node, &values).bind(node, &values);
        assert_eq!(
            session.execute(&duplicate).err(),
            Some(GraphSessionExecutionError::DuplicateInput { node })
        );

        let mut wrong_length = GraphInputs::new();
        wrong_length.bind(node, &short);
        assert_eq!(
            session.execute(&wrong_length).err(),
            Some(GraphSessionExecutionError::InputLengthMismatch {
                node,
                expected: 2,
                actual: 1,
            })
        );

        // Four rejections later, the session still runs.
        missing.bind(node, &values);
        session.execute(&missing).expect("valid run");

        assert_eq!(
            backend.calls()[..before.len()],
            before[..],
            "rejected inputs must leave the earlier call log untouched"
        );
        assert_eq!(
            backend.count(|call| matches!(call, Call::Allocate { .. })),
            3,
            "only the one valid run allocates"
        );
    }

    // -----------------------------------------------------------------------
    // Constants are injected automatically, in binding order
    // -----------------------------------------------------------------------

    #[test]
    fn constants_are_injected_automatically_and_identically_every_run() {
        let ty = f32_type(vec![2]);
        let mut graph = Graph::new();
        let input = graph.add_input("x", ty.clone()).expect("input");
        let constant = graph
            .add_constant(ConstantId::new(1), ty.clone())
            .expect("constant");
        let sum = graph
            .add_node(Operation::Add, vec![input, constant], ty.clone())
            .expect("add");
        let scaled = graph
            .add_node(
                Operation::Scale {
                    factor: Scalar::f32(2.0),
                },
                vec![sum],
                ty,
            )
            .expect("scale");
        graph.set_outputs(vec![scaled]).expect("outputs");

        let payload = [10.0f32, 20.0];
        let mut constants = GraphConstants::new();
        constants.bind(ConstantId::new(1), &payload);

        let backend = SharedBackend::new();
        let session = prepare(&backend, &graph, &constants).expect("preparable");

        assert_eq!(session.inputs().len(), 1, "only the input is required");
        assert_eq!(session.constants().len(), 1);
        assert_eq!(session.constants()[0].constant, ConstantId::new(1));

        let first = [1.0f32, 2.0];
        let second = [100.0f32, 200.0];
        for values in [first, second]
        {
            let mut inputs = GraphInputs::new();
            inputs.bind(input, &values);
            session.execute(&inputs).expect("runs");
        }

        // Two runs, two external buffers each, and the constant bytes are the
        // same both times.
        assert_eq!(
            backend.count(|call| matches!(call, Call::Write { bytes: 8 })),
            4,
            "each run imports the input and the constant"
        );
        assert_eq!(session.payload_values(0), Some(payload.as_slice()));
    }

    #[test]
    fn the_order_inputs_are_bound_in_has_no_effect() {
        let ty = f32_type(vec![2]);
        let mut graph = Graph::new();
        let lhs = graph.add_input("lhs", ty.clone()).expect("input");
        let rhs = graph.add_input("rhs", ty.clone()).expect("input");
        let difference = graph
            .add_node(Operation::Sub, vec![lhs, rhs], ty)
            .expect("sub");
        graph.set_outputs(vec![difference]).expect("outputs");

        let backend = SharedBackend::new();
        let session = prepare(&backend, &graph, &GraphConstants::new()).expect("preparable");

        let left = [5.0f32, 6.0];
        let right = [1.0f32, 2.0];

        let mut forward = GraphInputs::new();
        forward.bind(lhs, &left).bind(rhs, &right);
        let mut backward = GraphInputs::new();
        backward.bind(rhs, &right).bind(lhs, &left);

        assert_eq!(
            session.execute(&forward).expect("runs"),
            session.execute(&backward).expect("runs")
        );
    }

    // -----------------------------------------------------------------------
    // Failure leaves the session intact
    // -----------------------------------------------------------------------

    #[test]
    fn a_session_survives_a_backend_failure() {
        let backend = SharedBackend::failing_at(0);
        let graph = shared_kernel_graph();
        let session = prepare(&backend, &graph, &GraphConstants::new()).expect("preparable");

        let node = session.inputs()[0].node;
        let before_inputs = session.inputs().to_vec();
        let before_constants = session.constants().to_vec();
        let before_outputs = session.outputs().to_vec();

        let values = [1.0f32, 2.0];
        let mut inputs = GraphInputs::new();
        inputs.bind(node, &values);

        let error = session.execute(&inputs).expect_err("injected failure");
        assert!(
            matches!(
                error,
                GraphSessionExecutionError::PlanExecution {
                    source: PlanExecutionError::KernelLaunch { .. }
                }
            ),
            "the plan error must be kept whole, got {error:?}"
        );
        assert!(error.source().is_some(), "the source must stay reachable");

        assert_eq!(session.inputs(), before_inputs.as_slice());
        assert_eq!(session.constants(), before_constants.as_slice());
        assert_eq!(session.outputs(), before_outputs.as_slice());
        assert_eq!(values, [1.0, 2.0], "caller values are never mutated");

        // The recording backend computes nothing, so only the shape of the run
        // is asserted here; numeric behaviour belongs to the real CPU tests.
        backend.clear_launch_failure();
        let outputs = session
            .execute(&inputs)
            .expect("the session is still usable");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs.values()[0].node, session.outputs()[0].node);
        assert_eq!(outputs.values()[0].values.len(), 2);
    }
}
