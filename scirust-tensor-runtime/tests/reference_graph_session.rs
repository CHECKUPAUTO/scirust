//! End-to-end tests of the reference graph session on the real CPU backend.
//!
//! Every test drives the genuine pipeline — no mock, no hand-built plan:
//!
//! ```text
//! Graph + GraphConstants
//!   -> ReferenceGraphSession::prepare
//!   -> GraphInputs keyed by NodeId
//!   -> CpuComputeAdapter -> ReferenceInterpreter
//!   -> PlanOutputs
//! ```
//!
//! Bit-level assertions use `to_bits()` wherever the semantics demand it:
//! signed zeros and NaN payloads are indistinguishable under `==` (or compare
//! unequal, for NaN), so value equality would silently pass where the contract
//! requires an exact bit pattern.

use scirust_compute::{DType, Shape};
use scirust_gpu::CpuComputeAdapter;
use scirust_tensor_compile::{CompileError, LoweringError};
use scirust_tensor_ir::{ConstantId, Graph, NodeId, Operation, Scalar, TensorType};
use scirust_tensor_runtime::{
    GraphConstants, GraphInputs, GraphSessionExecutionError, GraphSessionPreparationError,
    PlanPreparationError, ReferenceGraphSession, ReferencePlanRuntime,
};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

type Session = ReferenceGraphSession<CpuComputeAdapter>;

fn f32_type(dims: Vec<usize>) -> TensorType {
    TensorType::new(DType::F32, Shape::new(dims))
}

fn runtime() -> ReferencePlanRuntime<CpuComputeAdapter> {
    ReferencePlanRuntime::new(CpuComputeAdapter::new())
}

fn try_session(
    graph: &Graph,
    constants: &GraphConstants<'_>,
) -> Result<Session, GraphSessionPreparationError> {
    ReferenceGraphSession::prepare(runtime(), graph, constants)
}

fn session(graph: &Graph) -> Session {
    try_session(graph, &GraphConstants::new()).expect("preparable graph")
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// `x -> op -> output`.
fn unary_graph(op: Operation, dims: Vec<usize>) -> (Graph, NodeId) {
    let ty = f32_type(dims);
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).unwrap();
    let result = graph.add_node(op, vec![input], ty).unwrap();
    graph.set_outputs(vec![result]).unwrap();
    (graph, input)
}

/// `a, b -> op -> output`.
fn binary_graph(op: Operation, dims: Vec<usize>) -> (Graph, NodeId, NodeId) {
    let ty = f32_type(dims);
    let mut graph = Graph::new();
    let lhs = graph.add_input("lhs", ty.clone()).unwrap();
    let rhs = graph.add_input("rhs", ty.clone()).unwrap();
    let result = graph.add_node(op, vec![lhs, rhs], ty).unwrap();
    graph.set_outputs(vec![result]).unwrap();
    (graph, lhs, rhs)
}

fn run_unary(op: Operation, dims: Vec<usize>, values: &[f32]) -> Vec<f32> {
    let (graph, input) = unary_graph(op, dims);
    let session = session(&graph);

    let mut inputs = GraphInputs::new();
    inputs.bind(input, values);

    let outputs = session.execute(&inputs).expect("successful run");
    assert_eq!(outputs.len(), 1);
    outputs.into_values()[0].values.clone()
}

fn run_binary(op: Operation, dims: Vec<usize>, left: &[f32], right: &[f32]) -> Vec<f32> {
    let (graph, lhs, rhs) = binary_graph(op, dims);
    let session = session(&graph);

    let mut inputs = GraphInputs::new();
    inputs.bind(lhs, left).bind(rhs, right);

    let outputs = session.execute(&inputs).expect("successful run");
    assert_eq!(outputs.len(), 1);
    outputs.into_values()[0].values.clone()
}

// ---------------------------------------------------------------------------
// The eight supported opcodes
// ---------------------------------------------------------------------------

#[test]
fn executes_relu() {
    assert_eq!(
        run_unary(Operation::Relu, vec![4], &[-2.0, -0.5, 0.0, 3.0]),
        vec![0.0, 0.0, 0.0, 3.0]
    );
}

#[test]
fn executes_scale() {
    assert_eq!(
        run_unary(
            Operation::Scale {
                factor: Scalar::f32(-2.0),
            },
            vec![3],
            &[1.0, -2.0, 0.5],
        ),
        vec![-2.0, 4.0, -1.0]
    );
}

#[test]
fn executes_the_four_binary_operations() {
    let left = [6.0f32, -4.0];
    let right = [2.0f32, 8.0];

    assert_eq!(
        run_binary(Operation::Add, vec![2], &left, &right),
        vec![8.0, 4.0]
    );
    assert_eq!(
        run_binary(Operation::Sub, vec![2], &left, &right),
        vec![4.0, -12.0]
    );
    assert_eq!(
        run_binary(Operation::Mul, vec![2], &left, &right),
        vec![12.0, -32.0]
    );
    assert_eq!(
        run_binary(Operation::Div, vec![2], &left, &right),
        vec![3.0, -0.5]
    );
}

#[test]
fn executes_shape_copy() {
    let mut graph = Graph::new();
    let input = graph.add_input("x", f32_type(vec![2, 3])).unwrap();
    let reshaped = graph
        .add_node(
            Operation::Reshape {
                shape: Shape::new(vec![6]),
            },
            vec![input],
            f32_type(vec![6]),
        )
        .unwrap();
    graph.set_outputs(vec![reshaped]).unwrap();

    let session = session(&graph);
    let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut inputs = GraphInputs::new();
    inputs.bind(input, &values);

    assert_eq!(
        session.execute(&inputs).expect("runs").into_values()[0].values,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn executes_permute() {
    let mut graph = Graph::new();
    let input = graph.add_input("x", f32_type(vec![2, 3])).unwrap();
    let transposed = graph
        .add_node(
            Operation::Transpose {
                permutation: vec![1, 0],
            },
            vec![input],
            f32_type(vec![3, 2]),
        )
        .unwrap();
    graph.set_outputs(vec![transposed]).unwrap();

    let session = session(&graph);
    let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut inputs = GraphInputs::new();
    inputs.bind(input, &values);

    assert_eq!(
        session.execute(&inputs).expect("runs").into_values()[0].values,
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

// ---------------------------------------------------------------------------
// Multi-node plans
// ---------------------------------------------------------------------------

#[test]
fn executes_scale_then_relu() {
    let ty = f32_type(vec![4]);
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).unwrap();
    let scaled = graph
        .add_node(
            Operation::Scale {
                factor: Scalar::f32(-1.0),
            },
            vec![input],
            ty.clone(),
        )
        .unwrap();
    let activated = graph.add_node(Operation::Relu, vec![scaled], ty).unwrap();
    graph.set_outputs(vec![activated]).unwrap();

    let session = session(&graph);
    let values = [1.0f32, -2.0, 3.0, -4.0];
    let mut inputs = GraphInputs::new();
    inputs.bind(input, &values);

    assert_eq!(
        session.execute(&inputs).expect("runs").into_values()[0].values,
        vec![0.0, 2.0, 0.0, 4.0]
    );
}

#[test]
fn executes_add_then_scale() {
    let ty = f32_type(vec![3]);
    let mut graph = Graph::new();
    let lhs = graph.add_input("a", ty.clone()).unwrap();
    let rhs = graph.add_input("b", ty.clone()).unwrap();
    let sum = graph
        .add_node(Operation::Add, vec![lhs, rhs], ty.clone())
        .unwrap();
    let scaled = graph
        .add_node(
            Operation::Scale {
                factor: Scalar::f32(0.5),
            },
            vec![sum],
            ty,
        )
        .unwrap();
    graph.set_outputs(vec![scaled]).unwrap();

    let session = session(&graph);
    let left = [1.0f32, 3.0, 5.0];
    let right = [1.0f32, 1.0, 1.0];
    let mut inputs = GraphInputs::new();
    inputs.bind(lhs, &left).bind(rhs, &right);

    assert_eq!(
        session.execute(&inputs).expect("runs").into_values()[0].values,
        vec![1.0, 2.0, 3.0]
    );
}

#[test]
fn executes_permute_then_shape_copy() {
    let mut graph = Graph::new();
    let input = graph.add_input("x", f32_type(vec![2, 3])).unwrap();
    let transposed = graph
        .add_node(
            Operation::Transpose {
                permutation: vec![1, 0],
            },
            vec![input],
            f32_type(vec![3, 2]),
        )
        .unwrap();
    let flattened = graph
        .add_node(
            Operation::Reshape {
                shape: Shape::new(vec![6]),
            },
            vec![transposed],
            f32_type(vec![6]),
        )
        .unwrap();
    graph.set_outputs(vec![flattened]).unwrap();

    let session = session(&graph);
    let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut inputs = GraphInputs::new();
    inputs.bind(input, &values);

    assert_eq!(
        session.execute(&inputs).expect("runs").into_values()[0].values,
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// `c -> op -> output`, with `c` a constant node.
fn unary_constant_graph(op: Operation, dims: Vec<usize>, constant: ConstantId) -> Graph {
    let ty = f32_type(dims);
    let mut graph = Graph::new();
    let node = graph.add_constant(constant, ty.clone()).unwrap();
    let result = graph.add_node(op, vec![node], ty).unwrap();
    graph.set_outputs(vec![result]).unwrap();
    graph
}

#[test]
fn a_constant_can_be_the_operand_of_a_unary_operation() {
    let graph = unary_constant_graph(
        Operation::Scale {
            factor: Scalar::f32(3.0),
        },
        vec![3],
        ConstantId::new(1),
    );

    let payload = [1.0f32, 2.0, -1.0];
    let mut constants = GraphConstants::new();
    constants.bind(ConstantId::new(1), &payload);
    let session = try_session(&graph, &constants).expect("preparable");

    assert!(
        session.inputs().is_empty(),
        "nothing is asked of the caller"
    );
    assert_eq!(session.constants().len(), 1);

    let outputs = session.execute(&GraphInputs::new()).expect("runs");
    assert_eq!(outputs.into_values()[0].values, vec![3.0, 6.0, -3.0]);
}

#[test]
fn a_constant_can_be_an_operand_of_a_binary_operation() {
    let ty = f32_type(vec![3]);
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).unwrap();
    let weights = graph.add_constant(ConstantId::new(4), ty.clone()).unwrap();
    let product = graph
        .add_node(Operation::Mul, vec![input, weights], ty)
        .unwrap();
    graph.set_outputs(vec![product]).unwrap();

    let payload = [2.0f32, 4.0, 8.0];
    let mut constants = GraphConstants::new();
    constants.bind(ConstantId::new(4), &payload);
    let session = try_session(&graph, &constants).expect("preparable");

    assert_eq!(session.inputs().len(), 1, "only the input is required");
    assert_eq!(session.inputs()[0].node, input);
    assert_eq!(session.constants()[0].constant, ConstantId::new(4));

    let values = [1.0f32, 1.0, 1.0];
    let mut inputs = GraphInputs::new();
    inputs.bind(input, &values);

    assert_eq!(
        session.execute(&inputs).expect("runs").into_values()[0].values,
        vec![2.0, 4.0, 8.0]
    );
}

#[test]
fn a_constant_is_never_asked_of_the_caller() {
    let graph = unary_constant_graph(Operation::Relu, vec![2], ConstantId::new(2));
    let payload = [-1.0f32, 5.0];
    let mut constants = GraphConstants::new();
    constants.bind(ConstantId::new(2), &payload);
    let session = try_session(&graph, &constants).expect("preparable");

    // The constant node is not an input of the session, so binding it fails.
    let constant_node = session.constants()[0].node;
    let mut inputs = GraphInputs::new();
    inputs.bind(constant_node, &payload);

    assert_eq!(
        session.execute(&inputs).err(),
        Some(GraphSessionExecutionError::UnexpectedInput {
            node: constant_node,
        })
    );

    // Supplying nothing at all is the correct call.
    assert_eq!(
        session
            .execute(&GraphInputs::new())
            .expect("runs")
            .into_values()[0]
            .values,
        vec![0.0, 5.0]
    );
}

#[test]
fn a_constant_eliminated_by_dead_code_is_neither_required_nor_stored() {
    let ty = f32_type(vec![2]);
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).unwrap();
    let live = graph
        .add_node(Operation::Relu, vec![input], ty.clone())
        .unwrap();
    let dead = graph.add_constant(ConstantId::new(8), ty.clone()).unwrap();
    graph.add_node(Operation::Relu, vec![dead], ty).unwrap();
    graph.set_outputs(vec![live]).unwrap();

    // The store covers the whole graph, elimination or not.
    let payload = [7.0f32, 9.0];
    let mut constants = GraphConstants::new();
    constants.bind(ConstantId::new(8), &payload);
    let session = try_session(&graph, &constants).expect("dead payloads are ignored");

    assert!(session.constants().is_empty());
    assert_eq!(session.inputs().len(), 1);

    let values = [-3.0f32, 4.0];
    let mut inputs = GraphInputs::new();
    inputs.bind(input, &values);
    assert_eq!(
        session.execute(&inputs).expect("runs").into_values()[0].values,
        vec![0.0, 4.0]
    );
}

#[test]
fn a_constant_keeps_its_exact_bits_through_a_structural_copy() {
    let mut graph = Graph::new();
    let constant = graph
        .add_constant(ConstantId::new(11), f32_type(vec![2, 2]))
        .unwrap();
    let flattened = graph
        .add_node(
            Operation::Reshape {
                shape: Shape::new(vec![4]),
            },
            vec![constant],
            f32_type(vec![4]),
        )
        .unwrap();
    graph.set_outputs(vec![flattened]).unwrap();

    let nan = f32::from_bits(0x7fc0_dead);
    let payload = [-0.0f32, nan, f32::INFINITY, f32::NEG_INFINITY];
    let mut constants = GraphConstants::new();
    constants.bind(ConstantId::new(11), &payload);
    let session = try_session(&graph, &constants).expect("preparable");

    let outputs = session.execute(&GraphInputs::new()).expect("runs");
    assert_eq!(bits(&outputs.values()[0].values), bits(&payload));
}

#[test]
fn a_constant_keeps_its_exact_bits_through_a_permutation() {
    let mut graph = Graph::new();
    let constant = graph
        .add_constant(ConstantId::new(12), f32_type(vec![2, 2]))
        .unwrap();
    let transposed = graph
        .add_node(
            Operation::Transpose {
                permutation: vec![1, 0],
            },
            vec![constant],
            f32_type(vec![2, 2]),
        )
        .unwrap();
    graph.set_outputs(vec![transposed]).unwrap();

    let nan = f32::from_bits(0xffc0_0001);
    let payload = [-0.0f32, nan, 1.0, 2.0];
    let mut constants = GraphConstants::new();
    constants.bind(ConstantId::new(12), &payload);
    let session = try_session(&graph, &constants).expect("preparable");

    let outputs = session.execute(&GraphInputs::new()).expect("runs");
    assert_eq!(
        bits(&outputs.values()[0].values),
        bits(&[-0.0, 1.0, nan, 2.0])
    );
}

#[test]
fn a_non_f32_constant_is_rejected() {
    let ty = TensorType::new(DType::F64, Shape::new(vec![2]));
    let mut graph = Graph::new();
    let constant = graph.add_constant(ConstantId::new(13), ty.clone()).unwrap();
    let result = graph.add_node(Operation::Relu, vec![constant], ty).unwrap();
    graph.set_outputs(vec![result]).unwrap();

    let payload = [1.0f32, 2.0];
    let mut constants = GraphConstants::new();
    constants.bind(ConstantId::new(13), &payload);

    assert_eq!(
        try_session(&graph, &constants).err(),
        Some(GraphSessionPreparationError::UnsupportedConstantDType {
            node: NodeId::new(0),
            dtype: DType::F64,
        })
    );
}

#[test]
fn a_missing_constant_payload_is_rejected() {
    let graph = unary_constant_graph(Operation::Relu, vec![2], ConstantId::new(14));

    assert_eq!(
        try_session(&graph, &GraphConstants::new()).err(),
        Some(GraphSessionPreparationError::MissingConstantPayload {
            node: NodeId::new(0),
            constant: ConstantId::new(14),
        })
    );
}

#[test]
fn a_duplicated_constant_payload_is_rejected() {
    let graph = unary_constant_graph(Operation::Relu, vec![2], ConstantId::new(15));
    let first = [1.0f32, 2.0];
    let second = [3.0f32, 4.0];
    let mut constants = GraphConstants::new();
    constants
        .bind(ConstantId::new(15), &first)
        .bind(ConstantId::new(15), &second);

    assert_eq!(
        try_session(&graph, &constants).err(),
        Some(GraphSessionPreparationError::DuplicateConstantPayload {
            constant: ConstantId::new(15),
        })
    );
}

#[test]
fn a_payload_for_an_unknown_constant_is_rejected() {
    let graph = unary_constant_graph(Operation::Relu, vec![2], ConstantId::new(16));
    let payload = [1.0f32, 2.0];
    let mut constants = GraphConstants::new();
    constants
        .bind(ConstantId::new(16), &payload)
        .bind(ConstantId::new(17), &payload);

    assert_eq!(
        try_session(&graph, &constants).err(),
        Some(GraphSessionPreparationError::UnexpectedConstantPayload {
            constant: ConstantId::new(17),
        })
    );
}

#[test]
fn a_constant_of_the_wrong_length_is_rejected() {
    let graph = unary_constant_graph(Operation::Relu, vec![3], ConstantId::new(18));
    let payload = [1.0f32, 2.0];
    let mut constants = GraphConstants::new();
    constants.bind(ConstantId::new(18), &payload);

    assert_eq!(
        try_session(&graph, &constants).err(),
        Some(GraphSessionPreparationError::ConstantLengthMismatch {
            node: NodeId::new(0),
            constant: ConstantId::new(18),
            expected: 3,
            actual: 2,
        })
    );
}

#[test]
fn one_payload_serves_several_nodes_sharing_a_constant_identifier() {
    let mut graph = Graph::new();
    let flat = f32_type(vec![4]);
    let square = f32_type(vec![2, 2]);

    let first = graph
        .add_constant(ConstantId::new(19), flat.clone())
        .unwrap();
    let second = graph
        .add_constant(ConstantId::new(19), square.clone())
        .unwrap();
    let left = graph.add_node(Operation::Relu, vec![first], flat).unwrap();
    let right = graph
        .add_node(
            Operation::Scale {
                factor: Scalar::f32(2.0),
            },
            vec![second],
            square,
        )
        .unwrap();
    graph.set_outputs(vec![left, right]).unwrap();

    let payload = [-1.0f32, 2.0, -3.0, 4.0];
    let mut constants = GraphConstants::new();
    constants.bind(ConstantId::new(19), &payload);
    let session = try_session(&graph, &constants).expect("preparable");

    assert_eq!(session.constants().len(), 2, "two nodes, one identifier");

    let outputs = session.execute(&GraphInputs::new()).expect("runs");
    let values = outputs.into_values();
    assert_eq!(values[0].values, vec![0.0, 2.0, 0.0, 4.0]);
    assert_eq!(values[1].values, vec![-2.0, 4.0, -6.0, 8.0]);
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

#[test]
fn the_order_inputs_are_bound_in_has_no_effect() {
    let (graph, lhs, rhs) = binary_graph(Operation::Sub, vec![3]);
    let session = session(&graph);

    let left = [10.0f32, 20.0, 30.0];
    let right = [1.0f32, 2.0, 3.0];

    let mut forward = GraphInputs::new();
    forward.bind(lhs, &left).bind(rhs, &right);
    let mut backward = GraphInputs::new();
    backward.bind(rhs, &right).bind(lhs, &left);

    let expected = vec![9.0f32, 18.0, 27.0];
    assert_eq!(
        session.execute(&forward).expect("runs").into_values()[0].values,
        expected
    );
    assert_eq!(
        session.execute(&backward).expect("runs").into_values()[0].values,
        expected
    );
}

#[test]
fn a_missing_input_is_rejected() {
    let (graph, lhs, rhs) = binary_graph(Operation::Add, vec![2]);
    let session = session(&graph);

    let left = [1.0f32, 2.0];
    let mut inputs = GraphInputs::new();
    inputs.bind(lhs, &left);

    assert_eq!(
        session.execute(&inputs).err(),
        Some(GraphSessionExecutionError::MissingInput { node: rhs })
    );
}

#[test]
fn an_unexpected_input_is_rejected() {
    let (graph, input) = unary_graph(Operation::Relu, vec![2]);
    let session = session(&graph);

    let values = [1.0f32, 2.0];
    let stranger = NodeId::new(1);
    let mut inputs = GraphInputs::new();
    inputs.bind(input, &values).bind(stranger, &values);

    assert_eq!(
        session.execute(&inputs).err(),
        Some(GraphSessionExecutionError::UnexpectedInput { node: stranger })
    );
}

#[test]
fn a_duplicated_input_is_rejected() {
    let (graph, input) = unary_graph(Operation::Relu, vec![2]);
    let session = session(&graph);

    let values = [1.0f32, 2.0];
    let mut inputs = GraphInputs::new();
    inputs.bind(input, &values).bind(input, &values);

    assert_eq!(
        session.execute(&inputs).err(),
        Some(GraphSessionExecutionError::DuplicateInput { node: input })
    );
}

#[test]
fn an_input_of_the_wrong_length_is_rejected() {
    let (graph, input) = unary_graph(Operation::Relu, vec![3]);
    let session = session(&graph);

    let short = [1.0f32, 2.0];
    let long = [1.0f32, 2.0, 3.0, 4.0];

    let mut too_short = GraphInputs::new();
    too_short.bind(input, &short);
    assert_eq!(
        session.execute(&too_short).err(),
        Some(GraphSessionExecutionError::InputLengthMismatch {
            node: input,
            expected: 3,
            actual: 2,
        })
    );

    let mut too_long = GraphInputs::new();
    too_long.bind(input, &long);
    assert_eq!(
        session.execute(&too_long).err(),
        Some(GraphSessionExecutionError::InputLengthMismatch {
            node: input,
            expected: 3,
            actual: 4,
        })
    );
}

#[test]
fn an_input_eliminated_by_dead_code_is_neither_required_nor_accepted() {
    let ty = f32_type(vec![2]);
    let mut graph = Graph::new();
    let live = graph.add_input("live", ty.clone()).unwrap();
    let dead = graph.add_input("dead", ty.clone()).unwrap();
    let result = graph
        .add_node(Operation::Relu, vec![live], ty.clone())
        .unwrap();
    graph.add_node(Operation::Relu, vec![dead], ty).unwrap();
    graph.set_outputs(vec![result]).unwrap();

    let session = session(&graph);
    assert_eq!(session.inputs().len(), 1, "the dead input is not required");
    assert_eq!(session.inputs()[0].node, live);

    let values = [-1.0f32, 2.0];

    // Supplying it is an error: the prepared plan, not the source graph, decides
    // what a caller must provide.
    let mut with_dead = GraphInputs::new();
    with_dead.bind(live, &values).bind(dead, &values);
    assert_eq!(
        session.execute(&with_dead).err(),
        Some(GraphSessionExecutionError::UnexpectedInput { node: dead })
    );

    let mut without_dead = GraphInputs::new();
    without_dead.bind(live, &values);
    assert_eq!(
        session.execute(&without_dead).expect("runs").into_values()[0].values,
        vec![0.0, 2.0]
    );
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

#[test]
fn several_outputs_are_returned_in_the_graphs_order() {
    let ty = f32_type(vec![2]);
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).unwrap();
    let relu = graph
        .add_node(Operation::Relu, vec![input], ty.clone())
        .unwrap();
    let scaled = graph
        .add_node(
            Operation::Scale {
                factor: Scalar::f32(10.0),
            },
            vec![input],
            ty,
        )
        .unwrap();
    // Deliberately not the node order.
    graph.set_outputs(vec![scaled, relu]).unwrap();

    let session = session(&graph);
    assert_eq!(
        session
            .outputs()
            .iter()
            .map(|output| output.node)
            .collect::<Vec<_>>(),
        vec![scaled, relu]
    );

    let values = [-1.0f32, 2.0];
    let mut inputs = GraphInputs::new();
    inputs.bind(input, &values);

    let produced = session.execute(&inputs).expect("runs").into_values();
    assert_eq!(produced[0].node, scaled);
    assert_eq!(produced[0].values, vec![-10.0, 20.0]);
    assert_eq!(produced[1].node, relu);
    assert_eq!(produced[1].values, vec![0.0, 2.0]);
}

#[test]
fn a_duplicated_graph_output_is_preserved_as_is() {
    let (graph_base, input) = unary_graph(Operation::Relu, vec![2]);
    let mut graph = graph_base;
    let result = NodeId::new(1);
    graph.set_outputs(vec![result, result]).unwrap();

    let session = session(&graph);
    assert_eq!(session.outputs().len(), 2);
    assert_eq!(session.outputs()[0].node, result);
    assert_eq!(session.outputs()[1].node, result);

    let values = [-4.0f32, 5.0];
    let mut inputs = GraphInputs::new();
    inputs.bind(input, &values);

    let produced = session.execute(&inputs).expect("runs").into_values();
    assert_eq!(produced.len(), 2);
    assert_eq!(produced[0].values, vec![0.0, 5.0]);
    assert_eq!(produced[1].values, vec![0.0, 5.0]);
}

#[test]
fn an_output_can_come_straight_from_an_input() {
    let ty = f32_type(vec![4]);
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).unwrap();
    graph.set_outputs(vec![input]).unwrap();

    let session = session(&graph);
    assert_eq!(session.inputs().len(), 1);
    assert_eq!(session.outputs()[0].tensor_type, ty);
    assert_eq!(
        session.prepared_plan().kernel_count(),
        0,
        "a graph without an instruction compiles nothing"
    );
    assert_eq!(session.prepared_plan().dispatch_count(), 0);

    let nan = f32::from_bits(0x7fc0_1234);
    let values = [-0.0f32, nan, f32::INFINITY, 1.5];
    let mut inputs = GraphInputs::new();
    inputs.bind(input, &values);

    let produced = session.execute(&inputs).expect("runs").into_values();
    assert_eq!(bits(&produced[0].values), bits(&values));
}

#[test]
fn an_output_can_come_straight_from_a_constant() {
    let ty = f32_type(vec![3]);
    let mut graph = Graph::new();
    let constant = graph.add_constant(ConstantId::new(20), ty.clone()).unwrap();
    graph.set_outputs(vec![constant]).unwrap();

    let nan = f32::from_bits(0xffc0_5678);
    let payload = [-0.0f32, nan, f32::NEG_INFINITY];
    let mut constants = GraphConstants::new();
    constants.bind(ConstantId::new(20), &payload);
    let session = try_session(&graph, &constants).expect("preparable");

    assert!(session.inputs().is_empty());
    assert_eq!(session.prepared_plan().kernel_count(), 0);
    assert_eq!(session.prepared_plan().dispatch_count(), 0);

    let produced = session
        .execute(&GraphInputs::new())
        .expect("runs")
        .into_values();
    assert_eq!(bits(&produced[0].values), bits(&payload));
}

#[test]
fn a_graph_without_an_output_is_rejected_by_the_compiler() {
    let mut graph = Graph::new();
    graph.add_input("x", f32_type(vec![2])).unwrap();

    assert_eq!(
        try_session(&graph, &GraphConstants::new()).err(),
        Some(GraphSessionPreparationError::GraphCompilation {
            source: CompileError::NoOutputs,
        })
    );
}

#[test]
fn executes_a_scalar_tensor() {
    assert_eq!(run_unary(Operation::Relu, vec![], &[-4.0]), vec![0.0]);
}

#[test]
fn executes_a_zero_element_tensor() {
    assert!(run_unary(Operation::Relu, vec![0, 3], &[]).is_empty());
}

// ---------------------------------------------------------------------------
// Reuse
// ---------------------------------------------------------------------------

#[test]
fn one_session_serves_several_executions_with_different_values() {
    let (graph, input) = unary_graph(Operation::Relu, vec![3]);
    let session = session(&graph);

    for (values, expected) in [
        ([-1.0f32, 2.0, -3.0], vec![0.0f32, 2.0, 0.0]),
        ([4.0, -5.0, 6.0], vec![4.0, 0.0, 6.0]),
        ([0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]),
    ]
    {
        let mut inputs = GraphInputs::new();
        inputs.bind(input, &values);
        assert_eq!(
            session.execute(&inputs).expect("runs").into_values()[0].values,
            expected
        );
    }
}

#[test]
fn identical_inputs_reproduce_identical_bits() {
    let (graph, input) = unary_graph(
        Operation::Scale {
            factor: Scalar::f32(0.1),
        },
        vec![4],
    );
    let session = session(&graph);

    let values = [1.0f32, -0.0, 3.3, -7.7];
    let mut inputs = GraphInputs::new();
    inputs.bind(input, &values);

    let first = session.execute(&inputs).expect("runs").into_values();
    let second = session.execute(&inputs).expect("runs").into_values();

    assert_eq!(bits(&first[0].values), bits(&second[0].values));
}

#[test]
fn a_session_stays_usable_after_a_rejected_input() {
    let (graph, input) = unary_graph(Operation::Relu, vec![2]);
    let session = session(&graph);

    let short = [1.0f32];
    let mut wrong = GraphInputs::new();
    wrong.bind(input, &short);
    assert!(session.execute(&wrong).is_err());

    let values = [-1.0f32, 2.0];
    let mut right = GraphInputs::new();
    right.bind(input, &values);
    assert_eq!(
        session.execute(&right).expect("runs").into_values()[0].values,
        vec![0.0, 2.0]
    );
}

#[test]
fn constants_stay_identical_across_executions() {
    let ty = f32_type(vec![2]);
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).unwrap();
    let bias = graph.add_constant(ConstantId::new(21), ty.clone()).unwrap();
    let sum = graph
        .add_node(Operation::Add, vec![input, bias], ty)
        .unwrap();
    graph.set_outputs(vec![sum]).unwrap();

    let payload = [100.0f32, 200.0];
    let mut constants = GraphConstants::new();
    constants.bind(ConstantId::new(21), &payload);
    let session = try_session(&graph, &constants).expect("preparable");

    let first = [1.0f32, 2.0];
    let second = [3.0f32, 4.0];

    let mut inputs = GraphInputs::new();
    inputs.bind(input, &first);
    assert_eq!(
        session.execute(&inputs).expect("runs").into_values()[0].values,
        vec![101.0, 202.0]
    );

    let mut again = GraphInputs::new();
    again.bind(input, &second);
    assert_eq!(
        session.execute(&again).expect("runs").into_values()[0].values,
        vec![103.0, 204.0]
    );

    assert_eq!(payload, [100.0, 200.0], "the store is never mutated");
}

// ---------------------------------------------------------------------------
// Rejections
// ---------------------------------------------------------------------------

#[test]
fn exp_and_log_are_rejected() {
    for operation in [Operation::Exp, Operation::Log]
    {
        let label = format!("{operation:?}");
        let (graph, _) = unary_graph(operation, vec![4]);

        let error = try_session(&graph, &GraphConstants::new())
            .err()
            .unwrap_or_else(|| panic!("{label} must be rejected"));

        assert!(
            matches!(
                error,
                GraphSessionPreparationError::PlanPreparation {
                    source: PlanPreparationError::DeterministicMathUnavailable { .. }
                }
            ),
            "{label} must be rejected as non-reproducible, got {error:?}"
        );
    }
}

#[test]
fn a_non_f32_input_is_rejected() {
    let ty = TensorType::new(DType::F64, Shape::new(vec![2]));
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).unwrap();
    let result = graph.add_node(Operation::Relu, vec![input], ty).unwrap();
    graph.set_outputs(vec![result]).unwrap();

    assert_eq!(
        try_session(&graph, &GraphConstants::new()).err(),
        Some(GraphSessionPreparationError::UnsupportedInputDType {
            node: input,
            dtype: DType::F64,
        })
    );
}

#[test]
fn matmul_is_rejected_by_the_lowerer() {
    let ty = f32_type(vec![2, 2]);
    let mut graph = Graph::new();
    let lhs = graph.add_input("a", ty.clone()).unwrap();
    let rhs = graph.add_input("b", ty.clone()).unwrap();
    let product = graph
        .add_node(Operation::MatMul, vec![lhs, rhs], ty)
        .unwrap();
    graph.set_outputs(vec![product]).unwrap();

    assert_eq!(
        try_session(&graph, &GraphConstants::new()).err(),
        Some(GraphSessionPreparationError::KernelLowering {
            source: LoweringError::UnsupportedOperation {
                node: product,
                operation: Operation::MatMul,
            },
        })
    );
}

#[test]
fn an_inconsistent_reshape_is_rejected_by_the_lowerer() {
    let mut graph = Graph::new();
    let input = graph.add_input("x", f32_type(vec![2, 3])).unwrap();
    // Six elements cannot become four.
    let reshaped = graph
        .add_node(
            Operation::Reshape {
                shape: Shape::new(vec![4]),
            },
            vec![input],
            f32_type(vec![4]),
        )
        .unwrap();
    graph.set_outputs(vec![reshaped]).unwrap();

    assert_eq!(
        try_session(&graph, &GraphConstants::new()).err(),
        Some(GraphSessionPreparationError::KernelLowering {
            source: LoweringError::InvalidReshape { node: reshaped },
        })
    );
}

/// `Graph::validate` only checks arity and reference direction, and both are
/// already enforced by `Graph::add_node`, so an invalid *structure* cannot be
/// built through the public API. What a caller can still build is a
/// semantically inconsistent graph, and the lowerer rejects it.
#[test]
fn a_semantically_inconsistent_graph_is_rejected_by_the_lowerer() {
    let ty = f32_type(vec![2]);
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).unwrap();
    // A binary operation whose operands disagree with its result type.
    let wrong = graph
        .add_node(Operation::Add, vec![input, input], f32_type(vec![3]))
        .unwrap();
    graph.set_outputs(vec![wrong]).unwrap();

    assert!(
        matches!(
            try_session(&graph, &GraphConstants::new()).err(),
            Some(GraphSessionPreparationError::KernelLowering {
                source: LoweringError::OperandTypeMismatch { .. },
            })
        ),
        "a semantically inconsistent graph must be rejected"
    );
}

// ---------------------------------------------------------------------------
// Session metadata
// ---------------------------------------------------------------------------

#[test]
fn session_metadata_describes_the_graph_after_elimination() {
    let ty = f32_type(vec![2]);
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).unwrap();
    let bias = graph.add_constant(ConstantId::new(22), ty.clone()).unwrap();
    let dead_input = graph.add_input("unused", ty.clone()).unwrap();
    let sum = graph
        .add_node(Operation::Add, vec![input, bias], ty.clone())
        .unwrap();
    graph
        .add_node(Operation::Relu, vec![dead_input], ty)
        .unwrap();
    graph.set_outputs(vec![sum]).unwrap();

    let payload = [1.0f32, 1.0];
    let mut constants = GraphConstants::new();
    constants.bind(ConstantId::new(22), &payload);
    let session = try_session(&graph, &constants).expect("preparable");

    assert_eq!(session.inputs().len(), 1);
    assert_eq!(session.inputs()[0].node, input);
    assert_eq!(session.inputs()[0].tensor_type, f32_type(vec![2]));

    assert_eq!(session.constants().len(), 1);
    assert_eq!(session.constants()[0].node, bias);
    assert_eq!(session.constants()[0].constant, ConstantId::new(22));

    assert_eq!(session.outputs().len(), 1);
    assert_eq!(session.outputs()[0].node, sum);
    assert_eq!(session.outputs()[0].tensor_type, f32_type(vec![2]));

    // The prepared plan is reachable, and it is the session's own.
    assert_eq!(session.prepared_plan().external_values().len(), 2);
    assert_eq!(session.prepared_plan().dispatch_count(), 1);
    assert!(
        session
            .runtime()
            .backend()
            .capabilities()
            .supports_dtype(DType::F32)
    );
}
