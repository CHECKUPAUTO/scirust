use std::fmt;

use scirust_tensor_core::{Tensor, TensorDevice, TensorError};
use scirust_tensor_ir::{AutodiffError, DType, Graph, NodeId, TensorType, grad, jvp, vjp};

use crate::{Core2Constants, Core2Inputs, Core2ReferenceSession, Core2RuntimeError};

#[derive(Debug)]
pub enum DifferentialError {
    Autodiff(AutodiffError),
    Runtime(Core2RuntimeError),
    Tensor(TensorError),
    InvalidNode(NodeId),
    UnsupportedDType { node: NodeId, dtype: DType },
    ScalarOutputRequired { node: NodeId, elements: usize },
    MissingTransformedOutput(NodeId),
}

impl fmt::Display for DifferentialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Autodiff(error) => write!(f, "differential graph construction failed: {error}"),
            Self::Runtime(error) => write!(f, "differential reference execution failed: {error}"),
            Self::Tensor(error) => write!(f, "differential tensor construction failed: {error}"),
            Self::InvalidNode(node) => write!(f, "unknown differential node {}", node.get()),
            Self::UnsupportedDType { node, dtype } => write!(
                f,
                "dense differential materialization currently requires F32 at node {}, got {dtype:?}",
                node.get()
            ),
            Self::ScalarOutputRequired { node, elements } => write!(
                f,
                "Hessian requires a scalar output; node {} has {elements} elements",
                node.get()
            ),
            Self::MissingTransformedOutput(node) => write!(
                f,
                "differential transform did not produce expected node {}",
                node.get()
            ),
        }
    }
}

impl std::error::Error for DifferentialError {}

impl From<AutodiffError> for DifferentialError {
    fn from(value: AutodiffError) -> Self {
        Self::Autodiff(value)
    }
}

impl From<Core2RuntimeError> for DifferentialError {
    fn from(value: Core2RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

impl From<TensorError> for DifferentialError {
    fn from(value: TensorError) -> Self {
        Self::Tensor(value)
    }
}

#[derive(Debug, Clone)]
pub struct DifferentialBinding {
    pub node: NodeId,
    pub value: Tensor,
}

impl DifferentialBinding {
    pub fn new(node: NodeId, value: Tensor) -> Self {
        Self { node, value }
    }
}

/// Materialize a dense Jacobian with forward-mode JVPs.
/// The result shape is `output.shape ++ wrt.shape`.
pub fn jacfwd_reference(
    graph: &Graph,
    output: NodeId,
    wrt: NodeId,
    bindings: &[DifferentialBinding],
    constants: &Core2Constants,
) -> Result<Tensor, DifferentialError> {
    let input_type = tensor_type(graph, wrt)?.clone();
    let output_type = tensor_type(graph, output)?.clone();
    require_f32(wrt, &input_type)?;
    require_f32(output, &output_type)?;

    let transformed = jvp(graph, output, &[wrt])?;
    let tangent_input = transformed.tangent_inputs[0];
    let tangent_output = transformed.tangent_output;
    let session = Core2ReferenceSession::prepare(transformed.graph, constants.clone())?;
    let input_elements = checked_elements(&input_type)?;
    let output_elements = checked_elements(&output_type)?;
    let matrix_elements = input_elements
        .checked_mul(output_elements)
        .ok_or(DifferentialError::Tensor(TensorError::ShapeOverflow))?;
    let mut jacobian = vec![0.0f32; matrix_elements];

    for input_index in 0..input_elements {
        let mut tangent = vec![0.0f32; input_elements];
        tangent[input_index] = 1.0;
        let mut execution_inputs = copy_bindings(bindings)?;
        execution_inputs.insert(
            tangent_input,
            Tensor::from_f32(tangent, input_type.shape.dims().to_vec())?,
        )?;
        let outputs = session.execute(&execution_inputs)?;
        let tangent_value = outputs
            .get(tangent_output)
            .ok_or(DifferentialError::MissingTransformedOutput(tangent_output))?
            .to_f32_vec()?;
        for (output_index, derivative) in tangent_value.into_iter().enumerate() {
            jacobian[output_index * input_elements + input_index] = derivative;
        }
    }

    Ok(Tensor::from_f32(
        jacobian,
        jacobian_shape(&output_type, &input_type),
    )?)
}

/// Materialize a dense Jacobian with reverse-mode VJPs.
pub fn jacrev_reference(
    graph: &Graph,
    output: NodeId,
    wrt: NodeId,
    bindings: &[DifferentialBinding],
    constants: &Core2Constants,
) -> Result<Tensor, DifferentialError> {
    let input_type = tensor_type(graph, wrt)?.clone();
    let output_type = tensor_type(graph, output)?.clone();
    require_f32(wrt, &input_type)?;
    require_f32(output, &output_type)?;

    let transformed = vjp(graph, output, &[wrt])?;
    let cotangent_input = transformed.cotangent_input;
    let gradient_output = transformed.gradients[0];
    let session = Core2ReferenceSession::prepare(transformed.graph, constants.clone())?;
    let input_elements = checked_elements(&input_type)?;
    let output_elements = checked_elements(&output_type)?;
    let matrix_elements = input_elements
        .checked_mul(output_elements)
        .ok_or(DifferentialError::Tensor(TensorError::ShapeOverflow))?;
    let mut jacobian = vec![0.0f32; matrix_elements];

    for output_index in 0..output_elements {
        let mut cotangent = vec![0.0f32; output_elements];
        cotangent[output_index] = 1.0;
        let mut execution_inputs = copy_bindings(bindings)?;
        execution_inputs.insert(
            cotangent_input,
            Tensor::from_f32(cotangent, output_type.shape.dims().to_vec())?,
        )?;
        let outputs = session.execute(&execution_inputs)?;
        let gradient = outputs
            .get(gradient_output)
            .ok_or(DifferentialError::MissingTransformedOutput(gradient_output))?
            .to_f32_vec()?;
        let row_start = output_index * input_elements;
        jacobian[row_start..row_start + input_elements].copy_from_slice(&gradient);
    }

    Ok(Tensor::from_f32(
        jacobian,
        jacobian_shape(&output_type, &input_type),
    )?)
}

/// Choose the dense Jacobian mode requiring the smaller canonical basis.
pub fn jacobian_reference(
    graph: &Graph,
    output: NodeId,
    wrt: NodeId,
    bindings: &[DifferentialBinding],
    constants: &Core2Constants,
) -> Result<Tensor, DifferentialError> {
    let input_elements = checked_elements(tensor_type(graph, wrt)?)?;
    let output_elements = checked_elements(tensor_type(graph, output)?)?;
    if input_elements <= output_elements {
        jacfwd_reference(graph, output, wrt, bindings, constants)
    } else {
        jacrev_reference(graph, output, wrt, bindings, constants)
    }
}

/// Dense Hessian of a scalar F32 output with respect to one tensor node.
pub fn hessian_reference(
    graph: &Graph,
    output: NodeId,
    wrt: NodeId,
    bindings: &[DifferentialBinding],
    constants: &Core2Constants,
) -> Result<Tensor, DifferentialError> {
    let output_elements = checked_elements(tensor_type(graph, output)?)?;
    if output_elements != 1 {
        return Err(DifferentialError::ScalarOutputRequired {
            node: output,
            elements: output_elements,
        });
    }
    let gradient_graph = grad(graph, output, &[wrt])?;
    jacfwd_reference(
        &gradient_graph.graph,
        gradient_graph.gradients[0],
        wrt,
        bindings,
        constants,
    )
}

fn tensor_type(graph: &Graph, node: NodeId) -> Result<&TensorType, DifferentialError> {
    graph
        .nodes()
        .get(node.get() as usize)
        .map(|node| &node.output)
        .ok_or(DifferentialError::InvalidNode(node))
}

fn checked_elements(tensor_type: &TensorType) -> Result<usize, DifferentialError> {
    tensor_type
        .shape
        .checked_num_elements()
        .map_err(|_| DifferentialError::Tensor(TensorError::ShapeOverflow))
}

fn require_f32(node: NodeId, tensor_type: &TensorType) -> Result<(), DifferentialError> {
    if tensor_type.dtype == DType::F32 {
        Ok(())
    } else {
        Err(DifferentialError::UnsupportedDType {
            node,
            dtype: tensor_type.dtype,
        })
    }
}

fn jacobian_shape(output: &TensorType, input: &TensorType) -> Vec<usize> {
    let mut shape = Vec::with_capacity(output.shape.rank() + input.shape.rank());
    shape.extend_from_slice(output.shape.dims());
    shape.extend_from_slice(input.shape.dims());
    shape
}

fn copy_bindings(bindings: &[DifferentialBinding]) -> Result<Core2Inputs, DifferentialError> {
    let mut inputs = Core2Inputs::new();
    for binding in bindings {
        if binding.value.device() != &TensorDevice::Host {
            return Err(DifferentialError::Runtime(
                Core2RuntimeError::HostReferenceRequired { node: binding.node },
            ));
        }
        inputs.insert(binding.node, binding.value.clone())?;
    }
    Ok(inputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_tensor_ir::{Operation, Shape};

    fn ty(shape: &[usize]) -> TensorType {
        TensorType::new(DType::F32, Shape::new(shape.to_vec()))
    }

    #[test]
    fn forward_and_reverse_jacobians_match_for_elementwise_square() {
        let mut graph = Graph::new();
        let x = graph.add_input("x", ty(&[3])).unwrap();
        let y = graph
            .add_node(Operation::Mul, vec![x, x], ty(&[3]))
            .unwrap();
        graph.set_outputs(vec![y]).unwrap();
        let bindings = [DifferentialBinding::new(
            x,
            Tensor::from_f32(vec![2., 3., 4.], vec![3]).unwrap(),
        )];

        let fwd = jacfwd_reference(&graph, y, x, &bindings, &Core2Constants::new()).unwrap();
        let rev = jacrev_reference(&graph, y, x, &bindings, &Core2Constants::new()).unwrap();
        assert_eq!(fwd.to_f32_vec().unwrap(), rev.to_f32_vec().unwrap());
        assert_eq!(
            fwd.to_f32_vec().unwrap(),
            vec![4., 0., 0., 0., 6., 0., 0., 0., 8.]
        );
        assert_eq!(fwd.shape().dims(), &[3, 3]);
    }

    #[test]
    fn hessian_of_scalar_square_is_two() {
        let mut graph = Graph::new();
        let x = graph.add_input("x", ty(&[1])).unwrap();
        let square = graph
            .add_node(Operation::Mul, vec![x, x], ty(&[1]))
            .unwrap();
        graph.set_outputs(vec![square]).unwrap();
        let bindings = [DifferentialBinding::new(
            x,
            Tensor::from_f32(vec![3.], vec![1]).unwrap(),
        )];

        let hessian = hessian_reference(
            &graph,
            square,
            x,
            &bindings,
            &Core2Constants::new(),
        )
        .unwrap();
        assert_eq!(hessian.shape().dims(), &[1, 1]);
        assert_eq!(hessian.to_f32_vec().unwrap(), vec![2.]);
    }
}
