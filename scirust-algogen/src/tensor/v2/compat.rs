//! Compatibility adapter: lift a frozen V1 [`TensorProgram`](crate::tensor::ir::TensorProgram)
//! into V2.
//!
//! V1 remains byte-stable; this adapter is the one-way migration path. The
//! mapping is exact:
//!
//! | V1 | V2 |
//! |---|---|
//! | `Input { i }` | reference to input `i` |
//! | `Add { lhs, rhs }` | `Op::Add` |
//! | `MatMul { lhs, rhs }` | `Op::MatMul` |
//! | `Transpose2d { src }` | `Op::Transpose(perm = [1, 0])` |
//! | `Relu { src }` | `Op::Max(src, Const(0.0))` (identical native semantics) |
//! | `Scale { src, factor }` | `Op::Mul(src, Const(factor))` |
//!
//! Execution of the lifted program is bit-identical to V1 for finite inputs
//! (tested in `tests.rs`).

use super::ir::{Bin, Op, Ref, ResearchProgram, Section, ValueId};
use super::types::{DType, ScalarValue, ValueType};
use crate::tensor::ir::TensorInstruction;

/// Lift a straight-line V1 program into an equivalent V2 expression program.
///
/// `input_shapes` must describe every V1 input; all values are `f32`.
pub fn from_v1(
    program: &crate::tensor::ir::TensorProgram,
    input_shapes: &[Vec<usize>],
) -> ResearchProgram {
    // Alias map: V1 register -> V2 value id (finalize-section index) or a
    // direct reference for inputs.
    enum Binding {
        Input(usize),
        Value(ValueId),
    }

    let mut bindings: Vec<Binding> = Vec::with_capacity(program.len());
    let mut ops: Vec<Op> = Vec::new();

    let alias = |bindings: &[Binding], register: usize| -> Ref {
        match &bindings[register]
        {
            Binding::Input(index) => Ref::Input(*index),
            Binding::Value(id) => Ref::Local(*id),
        }
    };

    for instruction in &program.instructions
    {
        let op = match *instruction
        {
            TensorInstruction::Input { input } =>
            {
                bindings.push(Binding::Input(input));
                continue;
            },
            TensorInstruction::Add { lhs, rhs } =>
            {
                Op::Add(Bin::new(alias(&bindings, lhs), alias(&bindings, rhs)))
            },
            TensorInstruction::MatMul { lhs, rhs } =>
            {
                Op::MatMul(Bin::new(alias(&bindings, lhs), alias(&bindings, rhs)))
            },
            TensorInstruction::Transpose2d { src } => Op::Transpose(super::ir::Permute {
                src: alias(&bindings, src),
                perm: vec![1, 0],
            }),
            TensorInstruction::Relu { src } =>
            {
                // relu(x) == max(x, 0.0); identical native f32 semantics.
                let zero = Op::Const(ScalarValue::F32(0.0));
                ops.push(zero);
                let zero_id = ops.len() - 1;
                Op::Max(Bin::new(alias(&bindings, src), Ref::Local(zero_id)))
            },
            TensorInstruction::Scale { src, factor } =>
            {
                let constant = Op::Const(ScalarValue::F32(factor));
                ops.push(constant);
                let constant_id = ops.len() - 1;
                Op::Mul(Bin::new(alias(&bindings, src), Ref::Local(constant_id)))
            },
        };
        ops.push(op);
        bindings.push(Binding::Value(ops.len() - 1));
    }

    // Resolve the output binding; an output that aliases an input directly
    // becomes an explicit identity via a copy-free trick: outputs must name
    // finalize values, so materialise a broadcast-to-same-shape passthrough
    // when needed.
    let output_binding = &bindings[program.output];
    let outputs = match *output_binding
    {
        Binding::Value(id) => vec![id],
        Binding::Input(index) =>
        {
            // Identity of an input: BroadcastTo its own shape is a no-op view
            // semantically, but V2 shape ops require a defined value; use an
            // explicit reshape to the same shape (pure row-major copy).
            let shape = input_shapes[index].clone();
            ops.push(Op::BroadcastTo(super::ir::ShapeTo {
                src: Ref::Input(index),
                shape,
            }));
            vec![ops.len() - 1]
        },
    };

    let inputs = input_shapes
        .iter()
        .map(|shape| ValueType::new(DType::F32, shape.clone()))
        .collect();

    ResearchProgram::expression(inputs, Section::new(ops), outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::ir::TensorProgram;

    #[test]
    fn lifts_a_composed_v1_program() {
        let v1 = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Scale {
                    src: 0,
                    factor: 2.0,
                },
                TensorInstruction::Relu { src: 1 },
            ],
            2,
        );
        let v2 = from_v1(&v1, &[vec![2, 2]]);
        assert_eq!(v2.steps, 0);
        assert_eq!(v2.inputs, vec![ValueType::new(DType::F32, vec![2, 2])]);
        assert_eq!(v2.outputs.len(), 1);
        assert!(v2.finalize.len() >= 4);
    }

    #[test]
    fn lifts_identity_output_aliasing_an_input() {
        let v1 = TensorProgram::new(vec![TensorInstruction::Input { input: 0 }], 0);
        let v2 = from_v1(&v1, &[vec![3]]);
        assert_eq!(v2.outputs, vec![v2.finalize.len() - 1]);
    }

    #[test]
    fn lifted_programs_verify() {
        let v1 = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Input { input: 1 },
                TensorInstruction::MatMul { lhs: 0, rhs: 1 },
                TensorInstruction::Add { lhs: 2, rhs: 2 },
            ],
            3,
        );
        let shapes = [vec![2, 3], vec![3, 2]];
        let v2 = from_v1(&v1, &shapes);
        super::super::verify::verify_program(
            &v2,
            super::super::verify::VerificationLimits::default(),
        )
        .expect("lifted program must verify");
    }
}
