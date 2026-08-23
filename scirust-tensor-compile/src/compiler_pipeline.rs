use core::fmt;

use crate::{
    CompilerIr, CompilerIrError, CompilerIrLowerer, CompilerIrLoweringError, CompilerPass,
    ExecutionPlan, LoweredPlan, MemoryPlan, PassManager, PassManagerStats,
};

/// Failure of the compiler-IR pipeline after canonical scheduling.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompilerPipelineError {
    CompilerIr(CompilerIrError),
    Lowering(CompilerIrLoweringError),
}

impl fmt::Display for CompilerPipelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::CompilerIr(error) => write!(formatter, "compiler IR pipeline failed: {error}"),
            Self::Lowering(error) => write!(formatter, "compiler IR lowering failed: {error}"),
        }
    }
}

impl std::error::Error for CompilerPipelineError {}

impl From<CompilerIrError> for CompilerPipelineError {
    fn from(error: CompilerIrError) -> Self {
        Self::CompilerIr(error)
    }
}

impl From<CompilerIrLoweringError> for CompilerPipelineError {
    fn from(error: CompilerIrLoweringError) -> Self {
        Self::Lowering(error)
    }
}

/// Complete backend-neutral result of compiler-IR optimization and lowering.
///
/// Keeping all four products together makes their provenance explicit: the
/// memory and lowered plans were derived from this exact transformed SSA, after
/// the recorded pass-manager run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledTensorProgram {
    compiler_ir: CompilerIr,
    pass_stats: PassManagerStats,
    memory: MemoryPlan,
    lowered: LoweredPlan,
}

impl CompiledTensorProgram {
    pub const fn compiler_ir(&self) -> &CompilerIr {
        &self.compiler_ir
    }

    pub const fn pass_stats(&self) -> PassManagerStats {
        self.pass_stats
    }

    pub const fn memory_plan(&self) -> &MemoryPlan {
        &self.memory
    }

    pub const fn lowered_plan(&self) -> &LoweredPlan {
        &self.lowered
    }

    pub fn into_parts(self) -> (CompilerIr, PassManagerStats, MemoryPlan, LoweredPlan) {
        (self.compiler_ir, self.pass_stats, self.memory, self.lowered)
    }
}

/// Ordered compiler pipeline between canonical scheduling and backend codegen.
///
/// The pipeline owns its [`PassManager`] and establishes the migration target:
///
/// `ExecutionPlan -> CompilerIr -> passes -> liveness -> MemoryPlan -> LoweredPlan`.
///
/// No backend state or physical allocation is introduced here. The canonical
/// [`ExecutionPlan`] remains an input boundary while runtime consumers migrate
/// incrementally to [`CompiledTensorProgram`].
#[derive(Default)]
pub struct CompilerPipeline {
    passes: PassManager,
}

impl CompilerPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn passes(&self) -> &PassManager {
        &self.passes
    }

    pub fn passes_mut(&mut self) -> &mut PassManager {
        &mut self.passes
    }

    pub fn push_pass<P>(&mut self, pass: P)
    where
        P: CompilerPass + 'static,
    {
        self.passes.push(pass);
    }

    /// Compile one canonical execution plan through transformed compiler SSA.
    pub fn compile_execution_plan(
        &mut self,
        execution: &ExecutionPlan,
    ) -> Result<CompiledTensorProgram, CompilerPipelineError> {
        let mut compiler_ir = CompilerIr::from_execution_plan(execution)?;
        let (pass_stats, memory) = compiler_ir.run_passes_and_plan_memory(&mut self.passes)?;
        let lowered =
            CompilerIrLowerer::new().lower_with_derived_bindings(&compiler_ir, &memory)?;

        Ok(CompiledTensorProgram {
            compiler_ir,
            pass_stats,
            memory,
            lowered,
        })
    }
}

#[cfg(test)]
mod tests {
    use scirust_tensor_ir::{DType, Graph, Operation, Shape, TensorType};

    use crate::{
        CanonicalCompiler, CompilerPass, ExternalBindings, IrOperation, KernelFamily,
        KernelLowerer, OperationRewrite, PassResult, Rewriter, UnaryKernel,
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
    fn empty_pipeline_matches_legacy_memory_and_lowering() {
        let execution = CanonicalCompiler::new().compile(&sample_graph()).unwrap();
        let legacy_bindings = ExternalBindings::derive(&execution);
        let legacy_lowered = KernelLowerer::new()
            .lower(&execution, &legacy_bindings)
            .unwrap();

        let compiled = CompilerPipeline::new()
            .compile_execution_plan(&execution)
            .unwrap();

        assert_eq!(compiled.pass_stats(), PassManagerStats::default());
        assert_eq!(compiled.memory_plan(), execution.memory_plan());
        assert_eq!(compiled.lowered_plan(), &legacy_lowered);
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct ReluToLogRewrite;

    impl OperationRewrite for ReluToLogRewrite {
        fn name(&self) -> &'static str {
            "relu-to-log-e2e-test"
        }

        fn rewrite(&mut self, operation: &IrOperation) -> Option<Operation> {
            matches!(operation.operation(), Operation::Relu).then_some(Operation::Log)
        }
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct ReluToLogPass;

    impl CompilerPass for ReluToLogPass {
        fn name(&self) -> &'static str {
            "relu-to-log-e2e-test"
        }

        fn run(&mut self, ir: &mut CompilerIr) -> Result<PassResult, CompilerIrError> {
            let mut rewrite = ReluToLogRewrite;
            let stats = Rewriter::new(ir)?.apply(&mut rewrite)?;
            Ok(PassResult {
                changed: stats.changed(),
            })
        }
    }

    #[test]
    fn pipeline_lowers_the_post_pass_operation() {
        let execution = CanonicalCompiler::new().compile(&sample_graph()).unwrap();
        let mut pipeline = CompilerPipeline::new();
        pipeline.push_pass(ReluToLogPass);

        let compiled = pipeline.compile_execution_plan(&execution).unwrap();

        assert_eq!(
            compiled.pass_stats(),
            PassManagerStats {
                passes_run: 1,
                passes_changed: 1,
            }
        );
        assert!(matches!(
            compiled.compiler_ir().operations()[1].operation(),
            Operation::Log
        ));
        assert!(matches!(
            compiled
                .lowered_plan()
                .kernel(compiled.lowered_plan().instructions()[0].kernel)
                .unwrap()
                .family(),
            KernelFamily::ElementwiseUnary(UnaryKernel::Log)
        ));
    }
}
