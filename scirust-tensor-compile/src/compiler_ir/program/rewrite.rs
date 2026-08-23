use scirust_tensor_ir::Operation;

use super::super::{operation::IrOperation, verify::verify_compiler_ir};
use super::{CompilerIr, CompilerIrError};

/// One local operation rewrite over immutable compiler-IR structure.
pub trait OperationRewrite {
    fn name(&self) -> &'static str;

    /// Return a replacement tensor operation, or `None` when the operation does
    /// not match this rewrite.
    ///
    /// Rewrites are intentionally local in this first phase: operands, result
    /// value, compiler identifiers and canonical provenance remain unchanged.
    fn rewrite(&mut self, operation: &IrOperation) -> Option<Operation>;
}

/// Aggregate result of one [`Rewriter::apply`] traversal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RewriteStats {
    pub operations_visited: usize,
    pub rewrites_applied: usize,
}

impl RewriteStats {
    pub const fn changed(self) -> bool {
        self.rewrites_applied != 0
    }
}

/// Controlled mutation boundary for local compiler-IR operation rewrites.
///
/// The rewriter first verifies the incoming IR, computes every replacement
/// without mutating the program, validates replacement arity, and only then
/// applies the complete rewrite set. This keeps arity failures atomic.
pub struct Rewriter<'a> {
    ir: &'a mut CompilerIr,
}

impl<'a> Rewriter<'a> {
    pub fn new(ir: &'a mut CompilerIr) -> Result<Self, CompilerIrError> {
        verify_compiler_ir(ir)?;
        Ok(Self { ir })
    }

    pub fn apply<R>(&mut self, rewrite: &mut R) -> Result<RewriteStats, CompilerIrError>
    where
        R: OperationRewrite + ?Sized,
    {
        let mut stats = RewriteStats::default();
        let mut replacements = Vec::<(usize, Operation)>::new();

        for (index, operation) in self.ir.operations.iter().enumerate()
        {
            stats.operations_visited += 1;

            let Some(replacement) = rewrite.rewrite(operation)
            else
            {
                continue;
            };

            if replacement == *operation.operation()
            {
                continue;
            }

            let expected = replacement.expected_arity();
            let actual = operation.operands().len();
            if expected != actual
            {
                return Err(CompilerIrError::OperationArityMismatch {
                    operation: operation.id(),
                    expected,
                    actual,
                });
            }

            replacements.push((index, replacement));
        }

        stats.rewrites_applied = replacements.len();

        for (index, replacement) in replacements
        {
            let rewritten = {
                let operation = &self.ir.operations[index];
                IrOperation::new(
                    operation.id(),
                    operation.canonical_node(),
                    replacement,
                    operation.operands().to_vec(),
                    operation.result(),
                )
            };

            self.ir.operations[index] = rewritten;
        }

        verify_compiler_ir(self.ir)?;
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use scirust_tensor_ir::{DType, Graph, Operation, Shape, TensorType};

    use crate::CanonicalCompiler;

    use super::*;

    fn ty() -> TensorType {
        TensorType::new(DType::F32, Shape::new(vec![4]))
    }

    #[derive(Default)]
    struct ReluToAdd;

    impl OperationRewrite for ReluToAdd {
        fn name(&self) -> &'static str {
            "relu-to-add"
        }

        fn rewrite(&mut self, operation: &IrOperation) -> Option<Operation> {
            matches!(operation.operation(), Operation::Relu).then_some(Operation::Add)
        }
    }

    #[test]
    fn rewriter_rejects_arity_change_atomically() {
        let mut graph = Graph::new();
        let input = graph.add_input("input", ty()).unwrap();
        let output = graph.add_node(Operation::Relu, vec![input], ty()).unwrap();
        graph.set_outputs(vec![output]).unwrap();

        let plan = CanonicalCompiler::new().compile(&graph).unwrap();
        let mut ir = CompilerIr::from_execution_plan(&plan).unwrap();
        let before = ir.clone();

        let mut rewrite = ReluToAdd;
        let error = Rewriter::new(&mut ir)
            .unwrap()
            .apply(&mut rewrite)
            .unwrap_err();

        assert_eq!(
            error,
            CompilerIrError::OperationArityMismatch {
                operation: ir.operations()[1].id(),
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(ir, before);
    }

    #[derive(Default)]
    struct IdentityRelu;

    impl OperationRewrite for IdentityRelu {
        fn name(&self) -> &'static str {
            "identity-relu"
        }

        fn rewrite(&mut self, operation: &IrOperation) -> Option<Operation> {
            matches!(operation.operation(), Operation::Relu).then_some(Operation::Relu)
        }
    }

    #[test]
    fn identical_replacement_is_not_reported_as_a_change() {
        let mut graph = Graph::new();
        let input = graph.add_input("input", ty()).unwrap();
        let output = graph.add_node(Operation::Relu, vec![input], ty()).unwrap();
        graph.set_outputs(vec![output]).unwrap();

        let plan = CanonicalCompiler::new().compile(&graph).unwrap();
        let mut ir = CompilerIr::from_execution_plan(&plan).unwrap();

        let mut rewrite = IdentityRelu;
        let stats = Rewriter::new(&mut ir).unwrap().apply(&mut rewrite).unwrap();

        assert_eq!(
            stats,
            RewriteStats {
                operations_visited: 2,
                rewrites_applied: 0,
            }
        );
        assert!(!stats.changed());
    }
}
