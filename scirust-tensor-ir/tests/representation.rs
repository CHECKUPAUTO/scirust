//! Public-API integration tests for representation planning.
//!
//! These exercise exactly what downstream crates can reach: no crate-private
//! constructors, no plan internals. Any break of the published surface fails
//! here first.

use scirust_compute::{DType, Shape};
use scirust_tensor_ir::{
    Graph, NodeId, Rebinding, RepresentationError, RepresentationId, RepresentationPlan,
    StorageBits, TensorType,
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

    let mut plan = RepresentationPlan::dense(&graph).unwrap();

    let dense_f16 = plan.declare_dense(DType::F16).unwrap();
    assert_eq!(plan.declare_dense(DType::F16), Ok(dense_f16));

    let factorized = plan
        .declare_factorized(
            tensor_type(DType::F16, &[8, 2]),
            dense_f16,
            tensor_type(DType::F16, &[2, 8]),
            dense_f16,
        )
        .unwrap();

    // Equal declarations remain deterministically interned.
    assert_eq!(
        plan.declare_factorized(
            tensor_type(DType::F16, &[8, 2]),
            dense_f16,
            tensor_type(DType::F16, &[2, 8]),
            dense_f16,
        ),
        Ok(factorized)
    );

    plan.assign(&graph, weight, factorized).unwrap();

    assert_eq!(plan.assignment(weight), Some(factorized));

    // 8*2 + 2*8 F16 elements = 32 half words = 512 physical bits.
    assert_eq!(
        plan.node_storage_bits(&graph, weight),
        Ok(StorageBits::new(512))
    );
    assert_eq!(plan.total_storage_bits(&graph), Ok(StorageBits::new(512)));

    assert_eq!(plan.assignments(), &[factorized]);

    // Representation planning remains a side table.
    assert_eq!(
        graph.nodes()[weight.get() as usize].output,
        tensor_type(DType::F32, &[8, 8])
    );
}

#[test]
fn public_declarations_resolve_dependencies_in_the_target_plan() {
    let mut graph = Graph::new();
    let value = graph
        .add_input("value", tensor_type(DType::U8, &[2, 2]))
        .unwrap();
    graph.set_outputs(vec![value]).unwrap();

    let mut plan = RepresentationPlan::dense(&graph).unwrap();

    // The seeded U8 representation is identifier 0. Identifier 1 is not yet
    // part of this plan and therefore cannot be used as a component.
    let dense_u8 = plan.assignment(value).unwrap();
    assert_eq!(dense_u8, RepresentationId::new(0));

    let missing = RepresentationId::new(1);
    assert_eq!(
        plan.declare_factorized(
            tensor_type(DType::U8, &[2, 2]),
            dense_u8,
            tensor_type(DType::U16, &[2, 2]),
            missing,
        ),
        Err(RepresentationError::InvalidRepresentationId { id: missing })
    );

    // Once this plan itself declares identifier 1, the same numeric identifier
    // resolves to this plan's own declaration. No foreign component object is
    // accepted by the public API.
    let dense_u16 = plan.declare_dense(DType::U16).unwrap();
    assert_eq!(dense_u16, missing);

    assert!(
        plan.declare_factorized(
            tensor_type(DType::U8, &[2, 2]),
            dense_u8,
            tensor_type(DType::U16, &[2, 2]),
            dense_u16,
        )
        .is_ok()
    );
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

    let dense_f16 = plan.declare_dense(DType::F16).unwrap();
    let factored = plan
        .declare_factorized(
            tensor_type(DType::F16, &[2, 3]),
            dense_f16,
            tensor_type(DType::F16, &[3, 5]),
            dense_f16,
        )
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

#[test]
fn replan_batches_apply_atomically_from_the_public_surface() {
    let mut graph = Graph::new();
    let weight = graph
        .add_input(
            "weight",
            TensorType::new(DType::F32, Shape::new(vec![4, 2])),
        )
        .unwrap();
    let bias = graph
        .add_input("bias", TensorType::new(DType::F32, Shape::new(vec![4])))
        .unwrap();
    graph.set_outputs(vec![weight, bias]).unwrap();

    let mut plan = RepresentationPlan::dense(&graph).unwrap();
    let before_total = plan.total_storage_bits(&graph).unwrap();
    let dense_f32 = plan.assignment(bias).unwrap();

    let dense_f16 = plan.declare_dense(DType::F16).unwrap();
    let factored = plan
        .declare_factorized(
            TensorType::new(DType::F16, Shape::new(vec![4, 1])),
            dense_f16,
            TensorType::new(DType::F16, Shape::new(vec![1, 2])),
            dense_f16,
        )
        .unwrap();

    // One invalid decision rejects the complete batch.
    plan.replan(
        &graph,
        &[
            Rebinding {
                node: weight,
                representation: factored,
            },
            Rebinding {
                node: bias,
                representation: factored,
            },
        ],
    )
    .expect_err("factorized cannot represent a vector bias");

    assert_ne!(plan.assignment(weight), Some(factored));
    assert_eq!(plan.assignment(bias), Some(dense_f32));

    // A corrected batch applies atomically.
    plan.replan(
        &graph,
        &[
            Rebinding {
                node: weight,
                representation: factored,
            },
            Rebinding {
                node: bias,
                representation: dense_f32,
            },
        ],
    )
    .unwrap();

    assert_eq!(plan.assignments(), &[factored, dense_f32]);

    let after_total = plan.total_storage_bits(&graph).unwrap();
    assert!(after_total.get() < before_total.get());
}
