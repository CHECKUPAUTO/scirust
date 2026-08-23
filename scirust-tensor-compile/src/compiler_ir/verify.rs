use super::{
    ids::{IrBlockId, IrOperationId},
    program::{CompilerIr, CompilerIrError, CompilerIrIdentifierSpace},
};

/// Verify structural and SSA invariants of a compiler IR.
///
/// Verification is deliberately independent from tensor semantic validation:
/// the canonical graph already owns tensor semantics.  This verifier owns
/// compiler structure — identifier stability, containment, operation arity,
/// SSA use-before-def rules and operation/result consistency.
pub fn verify_compiler_ir(ir: &CompilerIr) -> Result<(), CompilerIrError> {
    let entry_index = usize::try_from(ir.entry_region().get()).ok();
    if entry_index
        .and_then(|index| ir.regions().get(index))
        .is_none()
    {
        return Err(CompilerIrError::InvalidEntryRegion {
            region: ir.entry_region(),
        });
    }

    for (index, block) in ir.blocks().iter().enumerate()
    {
        let expected = checked_index(index, CompilerIrIdentifierSpace::Block)?;
        if block.id().get() != expected
        {
            return Err(CompilerIrError::IdentifierMismatch {
                space: CompilerIrIdentifierSpace::Block,
                expected,
                actual: block.id().get(),
            });
        }
    }

    for (index, region) in ir.regions().iter().enumerate()
    {
        let expected = checked_index(index, CompilerIrIdentifierSpace::Region)?;
        if region.id().get() != expected
        {
            return Err(CompilerIrError::IdentifierMismatch {
                space: CompilerIrIdentifierSpace::Region,
                expected,
                actual: region.id().get(),
            });
        }

        for &block_id in region.blocks()
        {
            let Some(block) = ir.block(block_id)
            else
            {
                return Err(CompilerIrError::InvalidBlock {
                    region: region.id(),
                    block: block_id,
                });
            };

            verify_block(ir, block.id(), block.operations())?;
        }
    }

    for (index, operation) in ir.operations().iter().enumerate()
    {
        let expected = checked_index(index, CompilerIrIdentifierSpace::Operation)?;
        if operation.id().get() != expected
        {
            return Err(CompilerIrError::IdentifierMismatch {
                space: CompilerIrIdentifierSpace::Operation,
                expected,
                actual: operation.id().get(),
            });
        }

        let expected_arity = operation.operation().expected_arity();
        let actual_arity = operation.operands().len();
        if expected_arity != actual_arity
        {
            return Err(CompilerIrError::OperationArityMismatch {
                operation: operation.id(),
                expected: expected_arity,
                actual: actual_arity,
            });
        }

        let Some(result) = ir.value(operation.result())
        else
        {
            return Err(CompilerIrError::InvalidValue {
                operation: operation.id(),
                value: operation.result(),
            });
        };

        if result.defining_operation() != operation.id()
        {
            return Err(CompilerIrError::ResultDefinitionMismatch {
                operation: operation.id(),
                result: result.id(),
                defining_operation: result.defining_operation(),
            });
        }

        if result.canonical_node() != operation.canonical_node()
        {
            return Err(CompilerIrError::CanonicalNodeMismatch {
                operation: operation.id(),
                value: result.id(),
                operation_node: operation.canonical_node(),
                value_node: result.canonical_node(),
            });
        }

        for &operand in operation.operands()
        {
            let Some(value) = ir.value(operand)
            else
            {
                return Err(CompilerIrError::InvalidValue {
                    operation: operation.id(),
                    value: operand,
                });
            };

            if value.defining_operation().get() >= operation.id().get()
            {
                return Err(CompilerIrError::NonSsaOperand {
                    operation: operation.id(),
                    operand,
                    defining_operation: value.defining_operation(),
                });
            }
        }
    }

    for (index, value) in ir.values().iter().enumerate()
    {
        let expected = checked_index(index, CompilerIrIdentifierSpace::Value)?;
        if value.id().get() != expected
        {
            return Err(CompilerIrError::IdentifierMismatch {
                space: CompilerIrIdentifierSpace::Value,
                expected,
                actual: value.id().get(),
            });
        }
    }

    for &output in ir.outputs()
    {
        if ir.value(output).is_none()
        {
            return Err(CompilerIrError::InvalidOutputValue { value: output });
        }
    }

    Ok(())
}

fn verify_block(
    ir: &CompilerIr,
    block_id: IrBlockId,
    operations: &[IrOperationId],
) -> Result<(), CompilerIrError> {
    for &operation in operations
    {
        if ir.operation(operation).is_none()
        {
            return Err(CompilerIrError::InvalidOperation {
                block: block_id,
                operation,
            });
        }
    }

    Ok(())
}

fn checked_index(index: usize, space: CompilerIrIdentifierSpace) -> Result<u32, CompilerIrError> {
    u32::try_from(index).map_err(|_| CompilerIrError::IdentifierOverflow { space })
}
