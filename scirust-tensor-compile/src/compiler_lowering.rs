use core::fmt;

use crate::{
    CompilerIr, CompilerIrError, ExecutionPlan, ExternalBindings, KernelLowerer, LoweredPlan,
    LoweringError, MemoryPlan,
};

/// Failures of the transitional compiler-IR lowering path.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompilerIrLoweringError {
    InvalidCompilerIr(CompilerIrError),
    IncompatibleMemoryPlan,
    Lowering(LoweringError),
}

impl fmt::Display for CompilerIrLoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidCompilerIr(error) => write!(formatter, "invalid compiler IR: {error}"),
            Self::IncompatibleMemoryPlan => formatter.write_str(
                "memory plan does not match the deterministic plan derived from compiler IR",
            ),
            Self::Lowering(error) => write!(formatter, "compiler-IR lowering failed: {error}"),
        }
    }
}

impl std::error::Error for CompilerIrLoweringError {}

impl From<CompilerIrError> for CompilerIrLoweringError {
    fn from(error: CompilerIrError) -> Self {
        Self::InvalidCompilerIr(error)
    }
}

impl From<LoweringError> for CompilerIrLoweringError {
    fn from(error: LoweringError) -> Self {
        Self::Lowering(error)
    }
}

/// Transitional lowering adapter for transformed [`CompilerIr`].
///
/// The existing logical-kernel lowerer is deliberately reused rather than
/// duplicated. The adapter first proves that `memory` is exactly the
/// deterministic plan for the supplied compiler IR, then projects the current
/// SSA operations and edges into the legacy execution-plan shape consumed by
/// [`KernelLowerer`]. Because the projection reads operations from `CompilerIr`,
/// compiler rewrites are visible to lowering instead of being lost by rereading
/// stale canonical instructions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompilerIrLowerer;

impl CompilerIrLowerer {
    pub const fn new() -> Self {
        Self
    }

    /// Derive the default external binding order from compiler IR and its
    /// validated post-pass memory plan.
    pub fn derive_bindings(
        self,
        ir: &CompilerIr,
        memory: &MemoryPlan,
    ) -> Result<ExternalBindings, CompilerIrLoweringError> {
        let plan = projected_plan(ir, memory)?;
        Ok(ExternalBindings::derive(&plan))
    }

    /// Lower transformed compiler IR with a caller-supplied external binding
    /// order.
    pub fn lower(
        self,
        ir: &CompilerIr,
        memory: &MemoryPlan,
        bindings: &ExternalBindings,
    ) -> Result<LoweredPlan, CompilerIrLoweringError> {
        let plan = projected_plan(ir, memory)?;
        Ok(KernelLowerer::new().lower(&plan, bindings)?)
    }

    /// Lower transformed compiler IR using the deterministic default external
    /// binding order.
    pub fn lower_with_derived_bindings(
        self,
        ir: &CompilerIr,
        memory: &MemoryPlan,
    ) -> Result<LoweredPlan, CompilerIrLoweringError> {
        let plan = projected_plan(ir, memory)?;
        let bindings = ExternalBindings::derive(&plan);
        Ok(KernelLowerer::new().lower(&plan, &bindings)?)
    }
}

fn projected_plan(
    ir: &CompilerIr,
    memory: &MemoryPlan,
) -> Result<ExecutionPlan, CompilerIrLoweringError> {
    let expected = MemoryPlan::from_compiler_ir(ir)?;
    if expected != *memory
    {
        return Err(CompilerIrLoweringError::IncompatibleMemoryPlan);
    }

    Ok(ExecutionPlan::project_compiler_ir(ir, memory.clone())?)
}

#[cfg(test)]
mod tests {
    use scirust_tensor_ir::{DType, Graph, Operation, Shape, TensorType};

    use crate::{
        CanonicalCompiler, ExternalBindings, KernelFamily, OperationRewrite, Rewriter, UnaryKernel,
    };

    use super::*;

    fn ty() -> TensorType {
        TensorType::new(DType::F32, Shape::new(vec![4]))
    }

    fn sample_graph() -> Graph {
        let mut graph = Graph::new();
        let input = graph.add_input("input", ty()).unwrap();
        let relu = graph.add_node(Operation::Relu, vec![input], ty()).unwrap();
        let output = graph.add_node(Operation::Exp, vec![relu], ty()).unwrap();
        graph.set_outputs(vec![output]).unwrap();
        graph
    }

    #[test]
    fn compiler_ir_lowering_matches_legacy_without_rewrites() {
        let execution = CanonicalCompiler::new().compile(&sample_graph()).unwrap();
        let ir = CompilerIr::from_execution_plan(&execution).unwrap();
        let memory = MemoryPlan::from_compiler_ir(&ir).unwrap();

        let legacy_bindings = ExternalBindings::derive(&execution);
        let legacy = KernelLowerer::new()
            .lower(&execution, &legacy_bindings)
            .unwrap();
        let compiler = CompilerIrLowerer::new()
            .lower_with_derived_bindings(&ir, &memory)
            .unwrap();

        assert_eq!(compiler, legacy);
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct ReluToLogRewrite;

    impl OperationRewrite for ReluToLogRewrite {
        fn name(&self) -> &'static str {
            "relu-to-log-test"
        }

        fn rewrite(&mut self, operation: &crate::IrOperation) -> Option<Operation> {
            matches!(operation.operation(), Operation::Relu).then_some(Operation::Log)
        }
    }

    #[test]
    fn compiler_ir_lowering_observes_post_canonical_rewrite() {
        let execution = CanonicalCompiler::new().compile(&sample_graph()).unwrap();
        let mut ir = CompilerIr::from_execution_plan(&execution).unwrap();

        let mut rewrite = ReluToLogRewrite;
        let stats = Rewriter::new(&mut ir).unwrap().apply(&mut rewrite).unwrap();
        assert_eq!(stats.operations_rewritten, 1);

        let memory = MemoryPlan::from_compiler_ir(&ir).unwrap();
        let lowered = CompilerIrLowerer::new()
            .lower_with_derived_bindings(&ir, &memory)
            .unwrap();

        assert!(matches!(
            lowered.kernel(lowered.instructions()[0].kernel).unwrap().family(),
            KernelFamily::ElementwiseUnary(UnaryKernel::Log)
        ));

        let legacy_bindings = ExternalBindings::derive(&execution);
        let legacy = KernelLowerer::new()
            .lower(&execution, &legacy_bindings)
            .unwrap();
        assert!(matches!(
            legacy.kernel(legacy.instructions()[0].kernel).unwrap().family(),
            KernelFamily::ElementwiseUnary(UnaryKernel::Relu)
        ));
    }

    #[test]
    fn compiler_ir_lowering_rejects_memory_from_another_program() {
        let execution = CanonicalCompiler::new().compile(&sample_graph()).unwrap();
        let ir = CompilerIr::from_execution_plan(&execution).unwrap();

        let mut other_graph = Graph::new();
        let input = other_graph.add_input("input", ty()).unwrap();
        let first = other_graph
            .add_node(Operation::Relu, vec![input], ty())
            .unwrap();
        let second = other_graph
            .add_node(Operation::Exp, vec![input], ty())
            .unwrap();
        other_graph.set_outputs(vec![first, second]).unwrap();
        let other_execution = CanonicalCompiler::new().compile(&other_graph).unwrap();
        let other_ir = CompilerIr::from_execution_plan(&other_execution).unwrap();
        let other_memory = MemoryPlan::from_compiler_ir(&other_ir).unwrap();

        assert_eq!(
            CompilerIrLowerer::new().derive_bindings(&ir, &other_memory),
            Err(CompilerIrLoweringError::IncompatibleMemoryPlan)
        );
    }
}
