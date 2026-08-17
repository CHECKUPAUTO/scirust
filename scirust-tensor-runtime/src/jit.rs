use std::fmt;

use scirust_compute::ComputeBackend;
use scirust_tensor_ir::{
    Graph, NodeId, Operation, OptimizationConfig, OptimizationError, OptimizedGraph, TensorType,
    optimize_graph,
};

use crate::{
    GraphConstants, GraphInputs, GraphSessionExecutionError, GraphSessionPreparationError,
    PlanOutputs, ReferenceGraphSession, ReferencePlanRuntime,
};

#[derive(Debug)]
pub enum JitPreparationError {
    Optimization(OptimizationError),
    Compilation(GraphSessionPreparationError),
    InputMappingLost { optimized: NodeId },
}

impl fmt::Display for JitPreparationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Optimization(error) => write!(f, "JIT optimization failed: {error}"),
            Self::Compilation(error) => write!(f, "JIT compilation failed: {error}"),
            Self::InputMappingLost { optimized } => write!(
                f,
                "JIT could not map optimized input node {} back to the source graph",
                optimized.get()
            ),
        }
    }
}

impl std::error::Error for JitPreparationError {}

#[derive(Debug)]
pub enum JitExecutionError {
    UnexpectedInput { node: NodeId },
    DuplicateInput { node: NodeId },
    MissingInput { node: NodeId },
    Execution(GraphSessionExecutionError),
}

impl fmt::Display for JitExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::UnexpectedInput { node } => write!(f, "unexpected JIT input node {}", node.get()),
            Self::DuplicateInput { node } => write!(f, "duplicate JIT input node {}", node.get()),
            Self::MissingInput { node } => write!(f, "missing JIT input node {}", node.get()),
            Self::Execution(error) => write!(f, "JIT execution failed: {error}"),
        }
    }
}

impl std::error::Error for JitExecutionError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JitInputSpec {
    pub source_node: NodeId,
    pub optimized_node: NodeId,
    pub tensor_type: TensorType,
}

/// One optimized and backend-compiled canonical graph.
///
/// `prepare` is the compilation boundary. `execute` performs no graph
/// optimization, lowering, kernel generation, or backend compilation.
#[derive(Debug)]
pub struct ReferenceJitSession<B: ComputeBackend> {
    session: ReferenceGraphSession<B>,
    inputs: Vec<JitInputSpec>,
    optimization: OptimizedGraph,
}

impl<B: ComputeBackend> ReferenceJitSession<B> {
    pub fn prepare(
        runtime: ReferencePlanRuntime<B>,
        graph: &Graph,
        constants: &GraphConstants<'_>,
    ) -> Result<Self, JitPreparationError> {
        Self::prepare_with_config(runtime, graph, constants, OptimizationConfig::default())
    }

    pub fn prepare_with_config(
        runtime: ReferencePlanRuntime<B>,
        graph: &Graph,
        constants: &GraphConstants<'_>,
        config: OptimizationConfig,
    ) -> Result<Self, JitPreparationError> {
        let optimization =
            optimize_graph(graph, config).map_err(JitPreparationError::Optimization)?;
        let session = ReferenceGraphSession::prepare(runtime, &optimization.graph, constants)
            .map_err(JitPreparationError::Compilation)?;

        let mut inputs = Vec::with_capacity(session.inputs().len());
        for optimized_spec in session.inputs()
        {
            let source_node = source_input_for_optimized(graph, &optimization, optimized_spec.node)
                .ok_or(JitPreparationError::InputMappingLost {
                    optimized: optimized_spec.node,
                })?;
            inputs.push(JitInputSpec {
                source_node,
                optimized_node: optimized_spec.node,
                tensor_type: optimized_spec.tensor_type.clone(),
            });
        }

        Ok(Self {
            session,
            inputs,
            optimization,
        })
    }

    pub fn inputs(&self) -> &[JitInputSpec] {
        &self.inputs
    }

    pub fn optimization(&self) -> &OptimizedGraph {
        &self.optimization
    }

    pub fn compiled_kernel_count(&self) -> usize {
        self.session.prepared_plan().kernel_count()
    }

    pub fn dispatch_count(&self) -> usize {
        self.session.prepared_plan().dispatch_count()
    }

    pub fn execute(&self, inputs: &GraphInputs<'_>) -> Result<PlanOutputs, JitExecutionError> {
        let mut ordered: Vec<Option<&[f32]>> = vec![None; self.inputs.len()];

        for &(source_node, values) in inputs.entries()
        {
            let Some(position) = self
                .inputs
                .iter()
                .position(|spec| spec.source_node == source_node)
            else
            {
                return Err(JitExecutionError::UnexpectedInput { node: source_node });
            };
            if ordered[position].is_some()
            {
                return Err(JitExecutionError::DuplicateInput { node: source_node });
            }
            ordered[position] = Some(values);
        }

        let mut remapped = GraphInputs::new();
        for (spec, values) in self.inputs.iter().zip(ordered)
        {
            let values = values.ok_or(JitExecutionError::MissingInput {
                node: spec.source_node,
            })?;
            remapped.bind(spec.optimized_node, values);
        }

        self.session
            .execute(&remapped)
            .map_err(JitExecutionError::Execution)
    }
}

fn source_input_for_optimized(
    source: &Graph,
    optimization: &OptimizedGraph,
    optimized: NodeId,
) -> Option<NodeId> {
    source
        .nodes()
        .iter()
        .enumerate()
        .filter(|(_, node)| matches!(&node.operation, Operation::Input { .. }))
        .find_map(|(index, _)| {
            let source_id = NodeId::new(index as u32);
            match optimization.old_to_new.get(index).copied().flatten()
            {
                Some(mapped) if mapped == optimized => Some(source_id),
                _ => None,
            }
        })
}
