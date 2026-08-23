use core::fmt;
use std::collections::BTreeMap;

use scirust_tensor_ir::NodeId;

use crate::ExecutionPlan;

use super::{
    block::IrBlock,
    ids::{IrBlockId, IrOperationId, IrRegionId, IrValueId},
    operation::IrOperation,
    region::IrRegion,
    value::IrValue,
    verify::verify_compiler_ir,
};

mod analysis;
mod pass;
mod rewrite;
pub use analysis::{
    AnalysisManager, AnalysisManagerError, CompilerAnalysis, LinearLiveRange, LinearLiveness,
    LinearLivenessAnalysis, UseCountAnalysis, UseCounts,
};
pub use pass::{
    CompilerPass, PassManager, PassManagerStats, PassResult, ScaleZeroCanonicalizationPass,
};
pub use rewrite::{OperationRewrite, RewriteStats, Rewriter};

/// Identifier namespace whose deterministic `u32` space overflowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompilerIrIdentifierSpace {
    Value,
    Operation,
    Block,
    Region,
}

impl fmt::Display for CompilerIrIdentifierSpace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Value => formatter.write_str("value"),
            Self::Operation => formatter.write_str("operation"),
            Self::Block => formatter.write_str("block"),
            Self::Region => formatter.write_str("region"),
        }
    }
}

/// Typed structural errors in the compiler IR.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompilerIrError {
    IdentifierOverflow {
        space: CompilerIrIdentifierSpace,
    },
    MissingOperand {
        node: NodeId,
        operand: NodeId,
    },
    MissingOutput {
        node: NodeId,
    },
    InvalidEntryRegion {
        region: IrRegionId,
    },
    InvalidBlock {
        region: IrRegionId,
        block: IrBlockId,
    },
    InvalidOperation {
        block: IrBlockId,
        operation: IrOperationId,
    },
    InvalidValue {
        operation: IrOperationId,
        value: IrValueId,
    },
    InvalidOutputValue {
        value: IrValueId,
    },
    IdentifierMismatch {
        space: CompilerIrIdentifierSpace,
        expected: u32,
        actual: u32,
    },
    OperationArityMismatch {
        operation: IrOperationId,
        expected: usize,
        actual: usize,
    },
    NonSsaOperand {
        operation: IrOperationId,
        operand: IrValueId,
        defining_operation: IrOperationId,
    },
    ResultDefinitionMismatch {
        operation: IrOperationId,
        result: IrValueId,
        defining_operation: IrOperationId,
    },
    CanonicalNodeMismatch {
        operation: IrOperationId,
        value: IrValueId,
        operation_node: NodeId,
        value_node: NodeId,
    },
}

impl fmt::Display for CompilerIrError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::IdentifierOverflow { space } =>
            {
                write!(
                    formatter,
                    "compiler IR {space} identifier space overflowed u32"
                )
            },
            Self::MissingOperand { node, operand } => write!(
                formatter,
                "canonical node {} references operand {} before it exists in compiler IR",
                node.get(),
                operand.get()
            ),
            Self::MissingOutput { node } => write!(
                formatter,
                "canonical output node {} has no compiler-IR SSA value",
                node.get()
            ),
            Self::InvalidEntryRegion { region } => write!(
                formatter,
                "compiler IR entry region {} does not exist",
                region.get()
            ),
            Self::InvalidBlock { region, block } => write!(
                formatter,
                "compiler IR region {} references missing block {}",
                region.get(),
                block.get()
            ),
            Self::InvalidOperation { block, operation } => write!(
                formatter,
                "compiler IR block {} references missing operation {}",
                block.get(),
                operation.get()
            ),
            Self::InvalidValue { operation, value } => write!(
                formatter,
                "compiler IR operation {} references missing value {}",
                operation.get(),
                value.get()
            ),
            Self::InvalidOutputValue { value } => write!(
                formatter,
                "compiler IR output references missing value {}",
                value.get()
            ),
            Self::IdentifierMismatch {
                space,
                expected,
                actual,
            } => write!(
                formatter,
                "compiler IR {space} identifier mismatch: expected {expected}, got {actual}"
            ),
            Self::OperationArityMismatch {
                operation,
                expected,
                actual,
            } => write!(
                formatter,
                "compiler IR operation {} arity mismatch: expected {expected}, got {actual}",
                operation.get()
            ),
            Self::NonSsaOperand {
                operation,
                operand,
                defining_operation,
            } => write!(
                formatter,
                "operation {} uses value {} defined by non-preceding operation {}",
                operation.get(),
                operand.get(),
                defining_operation.get()
            ),
            Self::ResultDefinitionMismatch {
                operation,
                result,
                defining_operation,
            } => write!(
                formatter,
                "operation {} result value {} claims defining operation {}",
                operation.get(),
                result.get(),
                defining_operation.get()
            ),
            Self::CanonicalNodeMismatch {
                operation,
                value,
                operation_node,
                value_node,
            } => write!(
                formatter,
                "operation {} and result value {} disagree on canonical node: {} versus {}",
                operation.get(),
                value.get(),
                operation_node.get(),
                value_node.get()
            ),
        }
    }
}

impl std::error::Error for CompilerIrError {}

/// Structural, backend-neutral SSA compiler representation.
///
/// This first implementation is a lossless structural bridge from
/// [`ExecutionPlan`].  It does **not** yet replace the execution-plan or kernel
/// lowering pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilerIr {
    values: Vec<IrValue>,
    operations: Vec<IrOperation>,
    blocks: Vec<IrBlock>,
    regions: Vec<IrRegion>,
    entry_region: IrRegionId,
    outputs: Vec<IrValueId>,
}

impl CompilerIr {
    /// Build the initial single-region, single-block SSA compiler IR.
    pub fn from_execution_plan(plan: &ExecutionPlan) -> Result<Self, CompilerIrError> {
        let entry_region = IrRegionId::new(0);
        let entry_block = IrBlockId::new(0);

        let mut values = Vec::with_capacity(plan.instructions().len());
        let mut operations = Vec::with_capacity(plan.instructions().len());
        let mut operation_ids = Vec::with_capacity(plan.instructions().len());
        let mut canonical_values = BTreeMap::<NodeId, IrValueId>::new();

        for instruction in plan.instructions()
        {
            let operation_id = IrOperationId::new(checked_id(
                operations.len(),
                CompilerIrIdentifierSpace::Operation,
            )?);
            let result_id =
                IrValueId::new(checked_id(values.len(), CompilerIrIdentifierSpace::Value)?);

            let operands = instruction
                .inputs
                .iter()
                .map(|operand| {
                    canonical_values
                        .get(operand)
                        .copied()
                        .ok_or(CompilerIrError::MissingOperand {
                            node: instruction.id,
                            operand: *operand,
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;

            operations.push(IrOperation::new(
                operation_id,
                instruction.id,
                instruction.operation.clone(),
                operands,
                result_id,
            ));
            values.push(IrValue::new(
                result_id,
                instruction.id,
                instruction.output.clone(),
                operation_id,
            ));
            operation_ids.push(operation_id);
            canonical_values.insert(instruction.id, result_id);
        }

        let outputs = plan
            .outputs()
            .iter()
            .map(|node| {
                canonical_values
                    .get(node)
                    .copied()
                    .ok_or(CompilerIrError::MissingOutput { node: *node })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let blocks = vec![IrBlock::new(entry_block, operation_ids)];
        let regions = vec![IrRegion::new(entry_region, vec![entry_block])];

        let ir = Self {
            values,
            operations,
            blocks,
            regions,
            entry_region,
            outputs,
        };

        verify_compiler_ir(&ir)?;
        Ok(ir)
    }

    pub fn values(&self) -> &[IrValue] {
        &self.values
    }

    pub fn operations(&self) -> &[IrOperation] {
        &self.operations
    }

    pub fn blocks(&self) -> &[IrBlock] {
        &self.blocks
    }

    pub fn regions(&self) -> &[IrRegion] {
        &self.regions
    }

    pub const fn entry_region(&self) -> IrRegionId {
        self.entry_region
    }

    pub fn outputs(&self) -> &[IrValueId] {
        &self.outputs
    }

    pub fn value(&self, id: IrValueId) -> Option<&IrValue> {
        usize::try_from(id.get())
            .ok()
            .and_then(|index| self.values.get(index))
    }

    pub fn operation(&self, id: IrOperationId) -> Option<&IrOperation> {
        usize::try_from(id.get())
            .ok()
            .and_then(|index| self.operations.get(index))
    }

    pub fn block(&self, id: IrBlockId) -> Option<&IrBlock> {
        usize::try_from(id.get())
            .ok()
            .and_then(|index| self.blocks.get(index))
    }

    pub fn region(&self, id: IrRegionId) -> Option<&IrRegion> {
        usize::try_from(id.get())
            .ok()
            .and_then(|index| self.regions.get(index))
    }
}

fn checked_id(len: usize, space: CompilerIrIdentifierSpace) -> Result<u32, CompilerIrError> {
    u32::try_from(len).map_err(|_| CompilerIrError::IdentifierOverflow { space })
}

#[cfg(test)]
mod tests {
    use scirust_tensor_ir::{DType, Graph, Operation, Shape, TensorType};

    use crate::CanonicalCompiler;

    use super::*;

    fn ty() -> TensorType {
        TensorType::new(DType::F32, Shape::new(vec![2, 2]))
    }

    #[test]
    fn execution_plan_becomes_deterministic_single_block_ssa() {
        let mut graph = Graph::new();
        let lhs = graph.add_input("lhs", ty()).unwrap();
        let rhs = graph.add_input("rhs", ty()).unwrap();
        let sum = graph
            .add_node(Operation::Add, vec![lhs, rhs], ty())
            .unwrap();
        let output = graph.add_node(Operation::Relu, vec![sum], ty()).unwrap();
        graph.set_outputs(vec![output]).unwrap();

        let plan = CanonicalCompiler::new().compile(&graph).unwrap();
        let ir = CompilerIr::from_execution_plan(&plan).unwrap();

        assert_eq!(ir.regions().len(), 1);
        assert_eq!(ir.blocks().len(), 1);
        assert_eq!(ir.operations().len(), 4);
        assert_eq!(ir.values().len(), 4);
        assert_eq!(ir.outputs().len(), 1);

        assert_eq!(ir.entry_region(), IrRegionId::new(0));
        assert_eq!(
            ir.region(IrRegionId::new(0)).unwrap().blocks(),
            &[IrBlockId::new(0)]
        );

        let add = ir.operation(IrOperationId::new(2)).unwrap();
        assert_eq!(add.canonical_node(), sum);
        assert_eq!(add.operands(), &[IrValueId::new(0), IrValueId::new(1)]);
        assert_eq!(add.result(), IrValueId::new(2));

        let relu = ir.operation(IrOperationId::new(3)).unwrap();
        assert_eq!(relu.operands(), &[IrValueId::new(2)]);
        assert_eq!(ir.outputs(), &[IrValueId::new(3)]);

        assert_eq!(verify_compiler_ir(&ir), Ok(()));
    }

    #[test]
    fn verifier_rejects_non_positional_block_identifier() {
        let mut graph = Graph::new();
        let input = graph.add_input("input", ty()).unwrap();
        graph.set_outputs(vec![input]).unwrap();

        let plan = CanonicalCompiler::new().compile(&graph).unwrap();
        let mut ir = CompilerIr::from_execution_plan(&plan).unwrap();

        ir.blocks[0] = IrBlock::new(IrBlockId::new(7), ir.blocks[0].operations().to_vec());

        assert_eq!(
            verify_compiler_ir(&ir),
            Err(CompilerIrError::IdentifierMismatch {
                space: CompilerIrIdentifierSpace::Block,
                expected: 0,
                actual: 7,
            })
        );
    }

    #[test]
    fn compiler_ir_preserves_sparse_canonical_node_ids_after_dce() {
        let mut graph = Graph::new();
        let input = graph.add_input("input", ty()).unwrap();
        let dead = graph.add_node(Operation::Exp, vec![input], ty()).unwrap();
        let live = graph.add_node(Operation::Relu, vec![input], ty()).unwrap();

        assert_eq!(dead.get(), 1);
        assert_eq!(live.get(), 2);

        graph.set_outputs(vec![live]).unwrap();

        let plan = CanonicalCompiler::new().compile(&graph).unwrap();
        let ir = CompilerIr::from_execution_plan(&plan).unwrap();

        assert_eq!(ir.operations().len(), 2);
        assert_eq!(
            ir.operation(IrOperationId::new(1))
                .unwrap()
                .canonical_node(),
            live
        );
        assert_eq!(ir.outputs(), &[IrValueId::new(1)]);
    }
}
