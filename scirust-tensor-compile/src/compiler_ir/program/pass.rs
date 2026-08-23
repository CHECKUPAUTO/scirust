use scirust_tensor_ir::Operation;

use super::super::{operation::IrOperation, verify::verify_compiler_ir};
use super::rewrite::{OperationRewrite, Rewriter};
use super::{CompilerIr, CompilerIrError};

/// Result of one compiler pass invocation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PassResult {
    pub changed: bool,
}

/// Deterministic transformation over [`CompilerIr`].
///
/// A pass may mutate compiler IR but must leave it structurally valid.
/// [`PassManager`] verifies the IR before the pipeline and after every pass.
pub trait CompilerPass {
    fn name(&self) -> &'static str;

    fn run(&mut self, ir: &mut CompilerIr) -> Result<PassResult, CompilerIrError>;
}

/// Aggregate statistics for one pass-manager run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PassManagerStats {
    pub passes_run: usize,
    pub passes_changed: usize,
}

/// Ordered deterministic compiler-pass pipeline.
#[derive(Default)]
pub struct PassManager {
    passes: Vec<Box<dyn CompilerPass>>,
}

impl PassManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push<P>(&mut self, pass: P)
    where
        P: CompilerPass + 'static,
    {
        self.passes.push(Box::new(pass));
    }

    pub fn len(&self) -> usize {
        self.passes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    pub fn run(&mut self, ir: &mut CompilerIr) -> Result<PassManagerStats, CompilerIrError> {
        verify_compiler_ir(ir)?;

        let mut stats = PassManagerStats::default();

        for pass in &mut self.passes
        {
            let result = pass.run(ir)?;
            stats.passes_run += 1;

            if result.changed
            {
                stats.passes_changed += 1;
            }

            verify_compiler_ir(ir)?;
        }

        Ok(stats)
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ScaleZeroRewrite;

impl OperationRewrite for ScaleZeroRewrite {
    fn name(&self) -> &'static str {
        "scale-zero"
    }

    fn rewrite(&mut self, operation: &IrOperation) -> Option<Operation> {
        matches!(
            operation.operation(),
            Operation::Scale { factor } if factor.as_f32() == Some(0.0)
        )
        .then_some(Operation::ZerosLike)
    }
}

/// First compiler-IR canonicalization pass.
///
/// `Scale(0)` and `ZerosLike` have the same unary structural contract. Rewriting
/// between them therefore changes neither SSA topology nor tensor identity.
///
/// The canonical tensor optimizer performs the same algebraic simplification
/// when it is explicitly run. This compiler pass is still useful because an
/// [`crate::ExecutionPlan`] may be built directly from a valid canonical graph
/// without first calling that optimizer.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScaleZeroCanonicalizationPass;

impl CompilerPass for ScaleZeroCanonicalizationPass {
    fn name(&self) -> &'static str {
        "scale-zero-canonicalization"
    }

    fn run(&mut self, ir: &mut CompilerIr) -> Result<PassResult, CompilerIrError> {
        let mut rewrite = ScaleZeroRewrite;
        let stats = Rewriter::new(ir)?.apply(&mut rewrite)?;
        Ok(PassResult {
            changed: stats.changed(),
        })
    }
}

#[cfg(test)]
mod tests {
    use scirust_tensor_ir::{DType, Graph, Operation, Scalar, Shape, TensorType};

    use crate::CanonicalCompiler;

    use super::*;

    fn ty() -> TensorType {
        TensorType::new(DType::F32, Shape::new(vec![4]))
    }

    #[test]
    fn pass_manager_runs_scale_zero_canonicalization_without_changing_ssa() {
        let mut graph = Graph::new();

        let input = graph.add_input("input", ty()).unwrap();
        let zero = graph
            .add_node(
                Operation::Scale {
                    factor: Scalar::f32(0.0),
                },
                vec![input],
                ty(),
            )
            .unwrap();
        let output = graph.add_node(Operation::Relu, vec![zero], ty()).unwrap();

        graph.set_outputs(vec![output]).unwrap();

        let plan = CanonicalCompiler::new().compile(&graph).unwrap();
        let mut ir = CompilerIr::from_execution_plan(&plan).unwrap();

        let before_operands = ir.operations()[1].operands().to_vec();
        let before_result = ir.operations()[1].result();
        let before_node = ir.operations()[1].canonical_node();

        assert!(matches!(
            ir.operations()[1].operation(),
            Operation::Scale { .. }
        ));

        let mut passes = PassManager::new();
        passes.push(ScaleZeroCanonicalizationPass);

        let stats = passes.run(&mut ir).unwrap();

        assert_eq!(
            stats,
            PassManagerStats {
                passes_run: 1,
                passes_changed: 1,
            }
        );

        assert!(matches!(
            ir.operations()[1].operation(),
            Operation::ZerosLike
        ));

        assert_eq!(ir.operations()[1].operands(), before_operands);
        assert_eq!(ir.operations()[1].result(), before_result);
        assert_eq!(ir.operations()[1].canonical_node(), before_node);
        assert_eq!(verify_compiler_ir(&ir), Ok(()));
    }

    #[test]
    fn scale_zero_canonicalization_is_idempotent() {
        let mut graph = Graph::new();

        let input = graph.add_input("input", ty()).unwrap();
        let output = graph
            .add_node(
                Operation::Scale {
                    factor: Scalar::f32(0.0),
                },
                vec![input],
                ty(),
            )
            .unwrap();

        graph.set_outputs(vec![output]).unwrap();

        let plan = CanonicalCompiler::new().compile(&graph).unwrap();
        let mut ir = CompilerIr::from_execution_plan(&plan).unwrap();

        let mut passes = PassManager::new();
        passes.push(ScaleZeroCanonicalizationPass);

        assert_eq!(passes.run(&mut ir).unwrap().passes_changed, 1);
        assert_eq!(passes.run(&mut ir).unwrap().passes_changed, 0);
        assert_eq!(verify_compiler_ir(&ir), Ok(()));
    }
}
