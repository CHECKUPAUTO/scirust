//! Public-API integration tests for representation planning.
//!
//! These exercise exactly what downstream crates can reach: no crate-private
//! constructors, no plan internals. Any break of the published surface fails
//! here first.

use scirust_compute::{DType, Shape};
use scirust_tensor_ir::{
    Graph, NodeId, PrimitiveRepresentation, RepresentationError, RepresentationId,
    RepresentationPlan, StorageBits, TensorType,
};

fn tensor_type(dtype: DType, dims: &[usize]) -> TensorType {
    TensorType::new(dtype, Shape::new(dims.to_vec()))
}

#[test]
fn downstream_crates_plan_composite_representations_end_to_end() {
    let mut graph = Graph::new();
    let weight = graph
        .add_input("weight", tensor_type(DType::F32, &[8, 8]))
        .unwrap();
    graph.set_outputs(vec![weight]).unwrap();

    // Dense seeding over the canonical graph.
    let mut plan = RepresentationPlan::dense(&graph).unwrap();

    // Declaration kernel interns building blocks deterministically.
    let dense_u8 = plan
        .declare(PrimitiveRepresentation::dense(DType::U8))
        .unwrap();
    let dense_f16 = plan
        .declare(PrimitiveRepresentation::dense(DType::F16))
        .unwrap();
    assert!(dense_u8.get() < dense_f16.get());
    assert_eq!(
        plan.declare(PrimitiveRepresentation::dense(DType::U8)),
        Ok(dense_u8)
    );

    let codes = plan
        .component(tensor_type(DType::U8, &[8, 8]), dense_u8)
        .unwrap();
    let scales = plan
        .component(tensor_type(DType::F16, &[8]), dense_f16)
        .unwrap();
    let quantized = plan
        .declare(PrimitiveRepresentation::quantized(codes, scales))
        .unwrap();

    // Binding and exact accounting through the published surface.
    plan.assign(&graph, weight, quantized).unwrap();

    assert_eq!(plan.assignment(weight), Some(quantized));
    assert_eq!(
        plan.node_storage_bits(&graph, weight),
        Ok(StorageBits::new(8 * 8 * 8 + 8 * 2 * 8))
    );
    assert_eq!(
        plan.total_storage_bits(&graph),
        Ok(StorageBits::new(8 * 8 * 8 + 8 * 2 * 8))
    );

    // The assignment table mirrors canonical node order.
    assert_eq!(plan.assignments(), &[quantized]);

    // Planning leaves the canonical graph untouched.
    assert_eq!(
        graph.nodes()[weight.get() as usize].output.dtype,
        DType::F32
    );
}

#[test]
fn foreign_values_cannot_bypass_the_public_declaration_kernel() {
    let mut source_graph = Graph::new();
    let u8_node = source_graph
        .add_input("u8", tensor_type(DType::U8, &[4]))
        .unwrap();
    let f16_node = source_graph
        .add_input("f16", tensor_type(DType::F16, &[4]))
        .unwrap();
    source_graph.set_outputs(vec![u8_node, f16_node]).unwrap();

    let source = RepresentationPlan::dense(&source_graph).unwrap();
    let codes = source
        .component(tensor_type(DType::U8, &[4]), RepresentationId::new(0))
        .unwrap();
    let scales = source
        .component(tensor_type(DType::F16, &[4]), RepresentationId::new(1))
        .unwrap();
    let foreign = PrimitiveRepresentation::quantized(codes, scales);

    // The target plan only knows identifier 0, so depending on identifier 1
    // is a forward reference there and must be rejected.
    let mut target_graph = Graph::new();
    let only = target_graph
        .add_input("u8", tensor_type(DType::U8, &[4]))
        .unwrap();
    target_graph.set_outputs(vec![only]).unwrap();
    let mut target = RepresentationPlan::dense(&target_graph).unwrap();

    assert_eq!(
        target.declare(foreign.clone()),
        Err(RepresentationError::InvalidRepresentationId {
            id: RepresentationId::new(1)
        })
    );

    // Plans are append-only: once the target declares its own identifier 1,
    // the identical value becomes a legitimate backward reference.
    target
        .declare(PrimitiveRepresentation::dense(DType::F16))
        .unwrap();
    assert!(target.declare(foreign).is_ok());
}

#[test]
fn incompatible_bindings_rejected_atomically_through_the_public_surface() {
    let mut graph = Graph::new();
    let weight = graph
        .add_input("weight", tensor_type(DType::F32, &[4, 2]))
        .unwrap();
    graph.set_outputs(vec![weight]).unwrap();

    let mut plan = RepresentationPlan::dense(&graph).unwrap();
    let dense_default = plan.assignment(weight).unwrap();

    // Factors contracting to [2, 5] cannot represent the node's [4, 2].
    let dense_f16 = plan
        .declare(PrimitiveRepresentation::dense(DType::F16))
        .unwrap();
    let left = plan
        .component(tensor_type(DType::F16, &[2, 3]), dense_f16)
        .unwrap();
    let right = plan
        .component(tensor_type(DType::F16, &[3, 5]), dense_f16)
        .unwrap();
    let factored = plan
        .declare(PrimitiveRepresentation::factorized(left, right))
        .unwrap();

    assert_eq!(
        plan.assign(&graph, weight, factored),
        Err(RepresentationError::FactorizedIncompatibleShapes {
            left: Shape::new(vec![2, 3]),
            right: Shape::new(vec![3, 5]),
            logical: Shape::new(vec![4, 2]),
        })
    );
    assert_eq!(plan.assignment(weight), Some(dense_default));
}

#[test]
fn plans_reject_graphs_they_were_not_seeded_from() {
    let mut graph = Graph::new();
    let weight = graph
        .add_input(
            "weight",
            TensorType::new(DType::F32, Shape::new(vec![8, 8])),
        )
        .unwrap();
    graph.set_outputs(vec![weight]).unwrap();

    let mut plan = RepresentationPlan::dense(&graph).unwrap();

    // A rewritten graph that retypes the value must be rejected instead of
    // being silently planned against stale assumptions.
    let mut retyped = Graph::new();
    let f16_weight = retyped
        .add_input(
            "weight",
            TensorType::new(DType::F16, Shape::new(vec![8, 8])),
        )
        .unwrap();
    retyped.set_outputs(vec![f16_weight]).unwrap();

    assert!(plan.ensure_compatible_with(&retyped).is_err());
    assert!(plan.node_storage_bits(&retyped, NodeId::new(0)).is_err());
    assert!(plan.total_storage_bits(&retyped).is_err());

    assert_eq!(
        plan.total_storage_bits(&retyped),
        Err(RepresentationError::GraphNodeTypeMismatch {
            node: NodeId::new(0),
            expected: TensorType::new(DType::F32, Shape::new(vec![8, 8])),
            actual: TensorType::new(DType::F16, Shape::new(vec![8, 8])),
        })
    );

    // An identically rebuilt graph remains fully usable.
    let mut rebuilt = Graph::new();
    let same = rebuilt
        .add_input(
            "weight",
            TensorType::new(DType::F32, Shape::new(vec![8, 8])),
        )
        .unwrap();
    rebuilt.set_outputs(vec![same]).unwrap();

    let dense_f32 = plan.assignment(NodeId::new(0)).unwrap();
    plan.assign(&rebuilt, NodeId::new(0), dense_f32).unwrap();
    assert_eq!(plan.assignment(NodeId::new(0)), Some(dense_f32));
}
