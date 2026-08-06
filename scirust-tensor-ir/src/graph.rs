use alloc::{string::String, vec::Vec};

use scirust_compute::{DType, Shape};

use crate::{ConstantId, GraphError, NodeId, Operation};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorType {
    pub dtype: DType,
    pub shape: Shape,
}

impl TensorType {
    pub const fn new(dtype: DType, shape: Shape) -> Self {
        Self { dtype, shape }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub operation: Operation,
    pub inputs: Vec<NodeId>,
    pub output: TensorType,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Graph {
    nodes: Vec<Node>,
    outputs: Vec<NodeId>,
}

impl Graph {
    pub const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    pub fn add_input(
        &mut self,
        name: impl Into<String>,
        output: TensorType,
    ) -> Result<NodeId, GraphError> {
        self.add_node(Operation::Input { name: name.into() }, Vec::new(), output)
    }

    pub fn add_constant(
        &mut self,
        id: ConstantId,
        output: TensorType,
    ) -> Result<NodeId, GraphError> {
        self.add_node(Operation::Constant { id }, Vec::new(), output)
    }

    pub fn add_node(
        &mut self,
        operation: Operation,
        inputs: Vec<NodeId>,
        output: TensorType,
    ) -> Result<NodeId, GraphError> {
        let expected = operation.expected_arity();
        if inputs.len() != expected
        {
            return Err(GraphError::ArityMismatch {
                expected,
                actual: inputs.len(),
            });
        }

        let node_index = self.nodes.len();
        for &input in &inputs
        {
            if input.get() as usize >= node_index
            {
                return Err(GraphError::InvalidInput { node_index, input });
            }
        }

        let raw_id = u32::try_from(node_index).map_err(|_| GraphError::TooManyNodes)?;
        let id = NodeId::new(raw_id);

        self.nodes.push(Node {
            operation,
            inputs,
            output,
        });

        Ok(id)
    }

    pub fn set_outputs(&mut self, outputs: Vec<NodeId>) -> Result<(), GraphError> {
        for &output in &outputs
        {
            if output.get() as usize >= self.nodes.len()
            {
                return Err(GraphError::InvalidOutput { output });
            }
        }

        self.outputs = outputs;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), GraphError> {
        for (node_index, node) in self.nodes.iter().enumerate()
        {
            let expected = node.operation.expected_arity();
            if node.inputs.len() != expected
            {
                return Err(GraphError::ArityMismatch {
                    expected,
                    actual: node.inputs.len(),
                });
            }

            for &input in &node.inputs
            {
                if input.get() as usize >= node_index
                {
                    return Err(GraphError::InvalidInput { node_index, input });
                }
            }
        }

        for &output in &self.outputs
        {
            if output.get() as usize >= self.nodes.len()
            {
                return Err(GraphError::InvalidOutput { output });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConstantId, Operation, Scalar};

    fn matrix_type() -> TensorType {
        TensorType::new(DType::F32, Shape::new(vec![2, 2]))
    }

    #[test]
    fn graph_uses_stable_topological_identifiers() {
        let mut graph = Graph::new();

        let lhs = graph.add_input("lhs", matrix_type()).unwrap();
        let rhs = graph.add_input("rhs", matrix_type()).unwrap();
        let sum = graph
            .add_node(Operation::Add, vec![lhs, rhs], matrix_type())
            .unwrap();

        graph.set_outputs(vec![sum]).unwrap();

        assert_eq!(lhs.get(), 0);
        assert_eq!(rhs.get(), 1);
        assert_eq!(sum.get(), 2);
        assert_eq!(graph.validate(), Ok(()));
    }

    #[test]
    fn graph_rejects_forward_references_and_wrong_arity() {
        let mut graph = Graph::new();

        assert_eq!(
            graph.add_node(Operation::Relu, vec![NodeId::new(1)], matrix_type(),),
            Err(GraphError::InvalidInput {
                node_index: 0,
                input: NodeId::new(1),
            })
        );

        assert_eq!(
            graph.add_node(Operation::Add, Vec::new(), matrix_type()),
            Err(GraphError::ArityMismatch {
                expected: 2,
                actual: 0,
            })
        );
    }

    #[test]
    fn constants_are_external_references_not_tensor_payloads() {
        let mut graph = Graph::new();
        let constant = graph
            .add_constant(ConstantId::new(42), matrix_type())
            .unwrap();

        assert_eq!(
            graph.nodes()[constant.get() as usize].operation,
            Operation::Constant {
                id: ConstantId::new(42),
            }
        );
    }

    #[test]
    fn scalar_attributes_are_bit_exact() {
        let minus_zero = Scalar::f32(-0.0);
        let plus_zero = Scalar::f32(0.0);

        assert_ne!(minus_zero, plus_zero);
        assert_eq!(minus_zero.as_f32(), Some(-0.0));
    }
}
