//! End-to-end tests: real `Graph -> ExecutionPlan -> LoweredPlan` fixtures,
//! driven through `ReferenceKernelGenerator`, plus decoder tests against
//! deliberately malformed byte streams.
//!
//! # What is, and is not, reachable through the public API
//!
//! `scirust_tensor_compile::KernelLowerer::lower` already enforces, before a
//! `LoweredPlan` can exist: uniform operand/result dtype, exact operand arity,
//! valid reshape/permutation semantics, sequential `LogicalKernelId`
//! assignment, and rejection of `MatMul`. Consequently several of
//! [`crate::ReferenceGenerationError`]'s variants — `UnsupportedKernelFamily`,
//! `MixedDTypes`, `ScaleFactorTypeMismatch`, `ScaleFactorBitsOverflow`,
//! `AxisIndexOverflow`, generator-side `UnexpectedOperandCount`,
// `TooManyKernels`, `DuplicateLogicalKernel`, `OutOfOrderLogicalKernel` and
//! `MissingLogicalKernel` — cannot be produced by feeding this crate a real
//! `LoweredPlan`. They are not exercised here, per the same "don't fabricate
//! API surface just to force coverage" discipline `scirust-tensor-compile`
//! itself followed for its own defensive lowering errors. What *is*
//! reachable — `UnsupportedDType` and `RankTooLarge` at generation time, and
//! every [`crate::ReferenceDecodeError`] variant, reachable by feeding the
//! decoder deliberately malformed bytes — is tested below.

use scirust_compute::{DType, KernelFormat, Shape};
use scirust_tensor_compile::{CanonicalCompiler, ExternalBindings, KernelLowerer, LoweredPlan};
use scirust_tensor_ir::{Graph, Operation, Scalar, TensorType};

use crate::{
    REFERENCE_FORMAT_VERSION, REFERENCE_MAGIC, REFERENCE_MAX_RANK, ReferenceAttributes,
    ReferenceDecodeError, ReferenceGenerationError, ReferenceKernelArtifact,
    ReferenceKernelGenerator, ReferenceOpcode,
};

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

fn f32_type(dims: Vec<usize>) -> TensorType {
    TensorType::new(DType::F32, Shape::new(dims))
}

fn lower(graph: &Graph) -> LoweredPlan {
    let plan = CanonicalCompiler::new()
        .compile(graph)
        .expect("valid graph");
    let bindings = ExternalBindings::derive(&plan);
    KernelLowerer::new()
        .lower(&plan, &bindings)
        .expect("valid lowering")
}

fn single_unary_plan(op: Operation, ty: TensorType) -> LoweredPlan {
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).unwrap();
    let result = graph.add_node(op, vec![input], ty).unwrap();
    graph.set_outputs(vec![result]).unwrap();
    lower(&graph)
}

fn single_binary_plan(op: Operation, ty: TensorType) -> LoweredPlan {
    let mut graph = Graph::new();
    let lhs = graph.add_input("lhs", ty.clone()).unwrap();
    let rhs = graph.add_input("rhs", ty.clone()).unwrap();
    let result = graph.add_node(op, vec![lhs, rhs], ty).unwrap();
    graph.set_outputs(vec![result]).unwrap();
    lower(&graph)
}

fn reshape_plan(input_ty: TensorType, output_ty: TensorType, target_shape: Shape) -> LoweredPlan {
    let mut graph = Graph::new();
    let input = graph.add_input("x", input_ty).unwrap();
    let result = graph
        .add_node(
            Operation::Reshape {
                shape: target_shape,
            },
            vec![input],
            output_ty,
        )
        .unwrap();
    graph.set_outputs(vec![result]).unwrap();
    lower(&graph)
}

fn transpose_plan(
    input_ty: TensorType,
    output_ty: TensorType,
    permutation: Vec<usize>,
) -> LoweredPlan {
    let mut graph = Graph::new();
    let input = graph.add_input("x", input_ty).unwrap();
    let result = graph
        .add_node(Operation::Transpose { permutation }, vec![input], output_ty)
        .unwrap();
    graph.set_outputs(vec![result]).unwrap();
    lower(&graph)
}

fn scale_plan(ty: TensorType, factor: Scalar) -> LoweredPlan {
    single_unary_plan(Operation::Scale { factor }, ty)
}

fn only_artifact(plan: &LoweredPlan) -> ReferenceKernelArtifact {
    let set = ReferenceKernelGenerator::new().generate(plan).unwrap();
    assert_eq!(set.len(), 1);
    set.artifacts()[0].clone()
}

/// Byte length of the fixed header, before the first operand layout.
fn header_len() -> usize {
    REFERENCE_MAGIC.len() // magic
        + 2 + 2 // version major, minor
        + 4 // kernel id
        + 2 // opcode
        + 2 // dtype tag
        + 2 + 2 // operand_count, result_count
}

/// Byte length of one `LAYOUT` (rank, dims, elements) for a given rank.
fn layout_total_len(rank: usize) -> usize {
    2 + rank * 8 + 8
}

// ---------------------------------------------------------------------------
// One test per supported operation
// ---------------------------------------------------------------------------

#[test]
fn relu_generates_the_relu_opcode() {
    let plan = single_unary_plan(Operation::Relu, f32_type(vec![2, 2]));
    let artifact = only_artifact(&plan);

    assert_eq!(artifact.opcode(), ReferenceOpcode::Relu);
    assert_eq!(artifact.dtype(), DType::F32);
    assert_eq!(artifact.operands().len(), 1);
    assert_eq!(artifact.operands()[0].dims(), &[2, 2]);
    assert_eq!(artifact.result().dims(), &[2, 2]);
    assert_eq!(artifact.attributes(), &ReferenceAttributes::None);
}

#[test]
fn exp_generates_the_exp_opcode() {
    let plan = single_unary_plan(Operation::Exp, f32_type(vec![3]));
    let artifact = only_artifact(&plan);

    assert_eq!(artifact.opcode(), ReferenceOpcode::Exp);
    assert_eq!(artifact.attributes(), &ReferenceAttributes::None);
}

#[test]
fn log_generates_the_log_opcode() {
    let plan = single_unary_plan(Operation::Log, f32_type(vec![5]));
    let artifact = only_artifact(&plan);

    assert_eq!(artifact.opcode(), ReferenceOpcode::Log);
    assert_eq!(artifact.attributes(), &ReferenceAttributes::None);
}

#[test]
fn add_generates_the_add_opcode_with_two_operands() {
    let plan = single_binary_plan(Operation::Add, f32_type(vec![2, 2]));
    let artifact = only_artifact(&plan);

    assert_eq!(artifact.opcode(), ReferenceOpcode::Add);
    assert_eq!(artifact.operands().len(), 2);
}

#[test]
fn sub_generates_the_sub_opcode() {
    let plan = single_binary_plan(Operation::Sub, f32_type(vec![2, 2]));
    assert_eq!(only_artifact(&plan).opcode(), ReferenceOpcode::Sub);
}

#[test]
fn mul_generates_the_mul_opcode() {
    let plan = single_binary_plan(Operation::Mul, f32_type(vec![2, 2]));
    assert_eq!(only_artifact(&plan).opcode(), ReferenceOpcode::Mul);
}

#[test]
fn div_generates_the_div_opcode() {
    let plan = single_binary_plan(Operation::Div, f32_type(vec![2, 2]));
    assert_eq!(only_artifact(&plan).opcode(), ReferenceOpcode::Div);
}

#[test]
fn shape_copy_preserves_source_and_destination_shapes_and_element_count() {
    let plan = reshape_plan(f32_type(vec![2, 3]), f32_type(vec![6]), Shape::new(vec![6]));
    let artifact = only_artifact(&plan);

    assert_eq!(artifact.opcode(), ReferenceOpcode::ShapeCopy);
    assert_eq!(artifact.operands()[0].dims(), &[2, 3]);
    assert_eq!(artifact.operands()[0].elements(), 6);
    assert_eq!(artifact.result().dims(), &[6]);
    assert_eq!(artifact.result().elements(), 6);
    assert_eq!(artifact.attributes(), &ReferenceAttributes::None);
}

#[test]
fn permute_preserves_source_shape_result_shape_and_axes() {
    let plan = transpose_plan(f32_type(vec![2, 3]), f32_type(vec![3, 2]), vec![1, 0]);
    let artifact = only_artifact(&plan);

    assert_eq!(artifact.opcode(), ReferenceOpcode::Permute);
    assert_eq!(artifact.operands()[0].dims(), &[2, 3]);
    assert_eq!(artifact.result().dims(), &[3, 2]);
    assert_eq!(
        artifact.attributes(),
        &ReferenceAttributes::Permute { axes: vec![1, 0] }
    );
}

#[test]
fn scale_encodes_the_factor_bit_exact_and_distinct_special_values_stay_distinct() {
    let cases: [u32; 6] = [
        0x0000_0000, // +0.0
        0x8000_0000, // -0.0
        0x7f80_0000, // +inf
        0xff80_0000, // -inf
        0x7fc0_0001, // NaN, payload 1
        0x7fc0_0002, // NaN, payload 2
    ];

    for &bits in &cases
    {
        let factor = Scalar::f32(f32::from_bits(bits));
        let plan = scale_plan(f32_type(vec![4]), factor);
        let artifact = only_artifact(&plan);

        let ReferenceAttributes::Scale { factor_bits } = artifact.attributes()
        else
        {
            panic!("expected a Scale attribute");
        };
        assert_eq!(*factor_bits, bits);

        let decoded = ReferenceKernelArtifact::decode(&artifact.encode()).unwrap();
        assert_eq!(decoded, artifact);
    }

    // Every case above is pairwise distinct after the full generate/encode/
    // decode round trip — none collapsed onto another.
    let mut seen = cases.to_vec();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), cases.len());
}

#[test]
fn scale_attribute_payload_is_exactly_four_bytes() {
    let plan = scale_plan(f32_type(vec![4]), Scalar::f32(1.5));
    let artifact = only_artifact(&plan);
    let bytes = artifact.encode();

    // header + one rank-1 operand layout + one rank-1 result layout +
    // attribute_tag(2) + factor_bits(4): no room exists for anything wider.
    let expected_len = header_len() + layout_total_len(1) * 2 + 2 + 4;
    assert_eq!(bytes.len(), expected_len);
}

#[test]
fn scale_factor_cannot_smuggle_bits_beyond_u32() {
    let plan = scale_plan(f32_type(vec![4]), Scalar::f32(1.5));
    let artifact = only_artifact(&plan);
    let mut bytes = artifact.encode();

    // A hypothetical wider factor would need 4 more bytes; the format has no
    // field to hold them, so they must be rejected as trailing bytes rather
    // than silently accepted or folded into the factor.
    bytes.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::TrailingBytes { remaining: 4 })
    );
}

// ---------------------------------------------------------------------------
// Order, identity and determinism
// ---------------------------------------------------------------------------

#[test]
fn kernel_order_matches_the_lowered_plan() {
    let mut graph = Graph::new();
    let input = graph.add_input("x", f32_type(vec![2])).unwrap();
    let relu = graph
        .add_node(Operation::Relu, vec![input], f32_type(vec![2]))
        .unwrap();
    let exp = graph
        .add_node(Operation::Exp, vec![relu], f32_type(vec![2]))
        .unwrap();
    let log = graph
        .add_node(Operation::Log, vec![exp], f32_type(vec![2]))
        .unwrap();
    graph.set_outputs(vec![log]).unwrap();

    let plan = lower(&graph);
    let set = ReferenceKernelGenerator::new().generate(&plan).unwrap();

    assert_eq!(
        set.artifacts()
            .iter()
            .map(ReferenceKernelArtifact::opcode)
            .collect::<Vec<_>>(),
        vec![
            ReferenceOpcode::Relu,
            ReferenceOpcode::Exp,
            ReferenceOpcode::Log
        ]
    );
    assert_eq!(
        plan.kernels()
            .iter()
            .map(|kernel| kernel.id())
            .collect::<Vec<_>>(),
        set.artifacts()
            .iter()
            .map(ReferenceKernelArtifact::kernel_id)
            .collect::<Vec<_>>()
    );
}

#[test]
fn artifact_lookup_by_logical_kernel_id_matches_its_source() {
    let plan = single_unary_plan(Operation::Relu, f32_type(vec![2]));
    let kernel_id = plan.kernels()[0].id();

    let set = ReferenceKernelGenerator::new().generate(&plan).unwrap();
    let artifact = set.artifact(kernel_id).unwrap();

    assert_eq!(artifact.kernel_id(), kernel_id);
}

#[test]
fn structurally_identical_kernels_of_different_types_keep_distinct_ids_and_bytes() {
    let mut graph = Graph::new();
    let small = graph.add_input("small", f32_type(vec![2])).unwrap();
    let large = graph.add_input("large", f32_type(vec![4])).unwrap();
    let first = graph
        .add_node(Operation::Relu, vec![small], f32_type(vec![2]))
        .unwrap();
    let second = graph
        .add_node(Operation::Relu, vec![large], f32_type(vec![4]))
        .unwrap();
    graph.set_outputs(vec![first, second]).unwrap();

    let plan = lower(&graph);
    let set = ReferenceKernelGenerator::new().generate(&plan).unwrap();

    assert_eq!(set.len(), 2);
    assert_ne!(
        set.artifacts()[0].kernel_id(),
        set.artifacts()[1].kernel_id()
    );
    assert_ne!(
        set.artifacts()[0].entry_point(),
        set.artifacts()[1].entry_point()
    );
    assert_ne!(set.artifacts()[0].encode(), set.artifacts()[1].encode());
}

#[test]
fn entry_point_is_derived_only_from_the_logical_kernel_id() {
    let plan = single_unary_plan(Operation::Relu, f32_type(vec![2]));
    let artifact = only_artifact(&plan);

    assert_eq!(
        artifact.entry_point(),
        format!("scirust_reference_kernel_{}", artifact.kernel_id().get())
    );
}

#[test]
fn entry_point_is_not_present_in_the_encoded_bytes() {
    let plan = single_unary_plan(Operation::Relu, f32_type(vec![2]));
    let artifact = only_artifact(&plan);
    let bytes = artifact.encode();
    let needle = artifact.entry_point().into_bytes();

    assert!(
        !bytes
            .windows(needle.len())
            .any(|window| window == needle.as_slice()),
        "the entry-point name must not appear anywhere in the encoded stream"
    );
}

#[test]
fn entry_point_survives_a_round_trip() {
    let plan = single_unary_plan(Operation::Relu, f32_type(vec![2]));
    let artifact = only_artifact(&plan);
    let decoded = ReferenceKernelArtifact::decode(&artifact.encode()).unwrap();

    assert_eq!(decoded.entry_point(), artifact.entry_point());
}

#[test]
fn repeated_encoding_is_byte_for_byte_identical() {
    let plan = single_binary_plan(Operation::Mul, f32_type(vec![3, 3]));

    let first = only_artifact(&plan).encode();
    let second = only_artifact(&plan).encode();

    assert_eq!(first, second);
}

#[test]
fn generation_is_deterministic_across_repeated_calls() {
    let plan = single_binary_plan(Operation::Add, f32_type(vec![2, 2]));

    let first = ReferenceKernelGenerator::new().generate(&plan).unwrap();
    let second = ReferenceKernelGenerator::new().generate(&plan).unwrap();

    assert_eq!(first, second);
    assert_eq!(
        first
            .artifacts()
            .iter()
            .map(ReferenceKernelArtifact::encode)
            .collect::<Vec<_>>(),
        second
            .artifacts()
            .iter()
            .map(ReferenceKernelArtifact::encode)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        first.to_kernel_modules().unwrap(),
        second.to_kernel_modules().unwrap()
    );
}

#[test]
fn encode_decode_round_trips_to_an_identical_artifact() {
    let plan = single_binary_plan(Operation::Div, f32_type(vec![2, 5]));
    let artifact = only_artifact(&plan);

    let bytes = artifact.encode();
    let decoded = ReferenceKernelArtifact::decode(&bytes).unwrap();

    assert_eq!(artifact, decoded);
    assert_eq!(bytes, decoded.encode());
}

#[test]
fn to_kernel_module_uses_reference_format_and_the_derived_entry_point() {
    let plan = single_unary_plan(Operation::Exp, f32_type(vec![2]));
    let artifact = only_artifact(&plan);

    let module = artifact.to_kernel_module().unwrap();

    assert_eq!(module.format, KernelFormat::Reference);
    assert_eq!(module.entry_point, artifact.entry_point());
    assert_eq!(module.code, artifact.encode());
}

#[test]
fn dimensions_beyond_u32_max_are_encoded_as_u64() {
    // 5_000_000_000 > u32::MAX (4_294_967_295); only representable on the wire
    // because dimensions are u64.
    const HUGE: usize = 5_000_000_000;

    let plan = single_unary_plan(Operation::Relu, f32_type(vec![HUGE]));
    let artifact = only_artifact(&plan);

    assert_eq!(artifact.operands()[0].dims(), &[HUGE as u64]);
    assert_eq!(artifact.operands()[0].elements(), HUGE as u64);

    let decoded = ReferenceKernelArtifact::decode(&artifact.encode()).unwrap();
    assert_eq!(decoded.operands()[0].dims(), &[HUGE as u64]);
    assert_eq!(decoded.result().elements(), HUGE as u64);
}

// ---------------------------------------------------------------------------
// Generation-time rejections reachable through the public API
// ---------------------------------------------------------------------------

#[test]
fn matmul_never_reaches_a_lowered_plan() {
    let mut graph = Graph::new();
    let lhs = graph.add_input("lhs", f32_type(vec![2, 2])).unwrap();
    let rhs = graph.add_input("rhs", f32_type(vec![2, 2])).unwrap();
    let product = graph
        .add_node(Operation::MatMul, vec![lhs, rhs], f32_type(vec![2, 2]))
        .unwrap();
    graph.set_outputs(vec![product]).unwrap();

    let plan = CanonicalCompiler::new().compile(&graph).unwrap();
    let bindings = ExternalBindings::derive(&plan);

    // `scirust_tensor_compile`'s lowering phase already rejects `MatMul`
    // before a `LoweredPlan` can exist, so `ReferenceKernelGenerator` never
    // has to reject it itself.
    assert!(KernelLowerer::new().lower(&plan, &bindings).is_err());
}

#[test]
fn unsupported_dtype_is_rejected() {
    let ty = TensorType::new(DType::U8, Shape::new(vec![2]));
    let plan = single_unary_plan(Operation::Relu, ty);

    let result = ReferenceKernelGenerator::new().generate(&plan);

    assert!(matches!(
        result,
        Err(ReferenceGenerationError::UnsupportedDType {
            dtype: DType::U8,
            ..
        })
    ));
}

#[test]
fn rank_exceeding_the_reference_limit_is_rejected() {
    let oversized_rank = REFERENCE_MAX_RANK as usize + 1;
    let ty = f32_type(vec![1; oversized_rank]);
    let plan = single_unary_plan(Operation::Relu, ty);

    let result = ReferenceKernelGenerator::new().generate(&plan);

    assert!(matches!(
        result,
        Err(ReferenceGenerationError::RankTooLarge { rank, limit, .. })
            if rank == oversized_rank && limit == REFERENCE_MAX_RANK
    ));
}

// ---------------------------------------------------------------------------
// Decoder: malformed streams
// ---------------------------------------------------------------------------

fn relu_baseline_bytes() -> Vec<u8> {
    let plan = single_unary_plan(Operation::Relu, f32_type(vec![2, 2]));
    only_artifact(&plan).encode()
}

#[test]
fn invalid_magic_is_rejected() {
    let mut bytes = relu_baseline_bytes();
    bytes[0] ^= 0xFF;

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::InvalidMagic)
    );
}

#[test]
fn incompatible_major_version_is_rejected() {
    let mut bytes = relu_baseline_bytes();
    let offset = REFERENCE_MAGIC.len();
    bytes[offset..offset + 2].copy_from_slice(&2u16.to_le_bytes());

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::IncompatibleMajorVersion {
            found: 2,
            supported: REFERENCE_FORMAT_VERSION.major(),
        })
    );
}

#[test]
fn unsupported_minor_version_is_rejected() {
    let mut bytes = relu_baseline_bytes();
    let offset = REFERENCE_MAGIC.len() + 2;
    bytes[offset..offset + 2].copy_from_slice(&7u16.to_le_bytes());

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::UnsupportedMinorVersion {
            found: 7,
            supported: REFERENCE_FORMAT_VERSION.minor(),
        })
    );
}

#[test]
fn unknown_opcode_is_rejected() {
    let mut bytes = relu_baseline_bytes();
    let offset = REFERENCE_MAGIC.len() + 2 + 2 + 4;
    bytes[offset..offset + 2].copy_from_slice(&0xFFFFu16.to_le_bytes());

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::UnknownOpcode { code: 0xFFFF })
    );
}

#[test]
fn unknown_dtype_tag_is_rejected() {
    let mut bytes = relu_baseline_bytes();
    let offset = REFERENCE_MAGIC.len() + 2 + 2 + 4 + 2;
    bytes[offset..offset + 2].copy_from_slice(&0xBEEFu16.to_le_bytes());

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::UnknownDType { tag: 0xBEEF })
    );
}

#[test]
fn known_but_unsupported_dtype_tag_is_distinguishable_from_an_unknown_one() {
    let mut bytes = relu_baseline_bytes();
    let offset = REFERENCE_MAGIC.len() + 2 + 2 + 4 + 2;
    bytes[offset..offset + 2].copy_from_slice(&0x0001u16.to_le_bytes()); // Bool's tag

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::UnsupportedDType { dtype: DType::Bool })
    );
}

#[test]
fn invalid_result_count_is_rejected() {
    let mut bytes = relu_baseline_bytes();
    let offset = REFERENCE_MAGIC.len() + 2 + 2 + 4 + 2 + 2 + 2;
    bytes[offset..offset + 2].copy_from_slice(&2u16.to_le_bytes());

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::InvalidResultCount { found: 2 })
    );
}

#[test]
fn unexpected_operand_count_is_rejected() {
    let mut bytes = relu_baseline_bytes();
    let offset = REFERENCE_MAGIC.len() + 2 + 2 + 4 + 2 + 2;
    bytes[offset..offset + 2].copy_from_slice(&2u16.to_le_bytes()); // Relu expects 1

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::UnexpectedOperandCount {
            opcode: ReferenceOpcode::Relu,
            expected: 1,
            actual: 2,
        })
    );
}

#[test]
fn empty_input_is_rejected_as_truncated() {
    assert!(matches!(
        ReferenceKernelArtifact::decode(&[]),
        Err(ReferenceDecodeError::TruncatedInput { .. })
    ));
}

#[test]
fn truncated_input_is_rejected() {
    let bytes = relu_baseline_bytes();
    let truncated = &bytes[..bytes.len() - 4];

    assert!(matches!(
        ReferenceKernelArtifact::decode(truncated),
        Err(ReferenceDecodeError::TruncatedInput { .. })
    ));
}

#[test]
fn trailing_bytes_are_rejected() {
    let mut bytes = relu_baseline_bytes();
    bytes.push(0);

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::TrailingBytes { remaining: 1 })
    );
}

#[test]
fn decoder_checks_the_rank_limit_before_any_rank_sized_allocation() {
    // A minimal, truncated-after-rank stream: if the rank check ran after
    // trying to read `rank` dimensions, this would report `TruncatedInput`
    // instead. It must report `RankTooLarge`.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(REFERENCE_MAGIC.as_slice());
    bytes.extend_from_slice(&REFERENCE_FORMAT_VERSION.major().to_le_bytes());
    bytes.extend_from_slice(&REFERENCE_FORMAT_VERSION.minor().to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&ReferenceOpcode::Relu.code().to_le_bytes());
    bytes.extend_from_slice(&0x000Au16.to_le_bytes()); // F32
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&(REFERENCE_MAX_RANK + 1).to_le_bytes());

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::RankTooLarge {
            rank: REFERENCE_MAX_RANK + 1,
            limit: REFERENCE_MAX_RANK,
        })
    );
}

#[test]
fn decoder_rejects_a_layout_whose_dimension_product_overflows_u64() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(REFERENCE_MAGIC.as_slice());
    bytes.extend_from_slice(&REFERENCE_FORMAT_VERSION.major().to_le_bytes());
    bytes.extend_from_slice(&REFERENCE_FORMAT_VERSION.minor().to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&ReferenceOpcode::Relu.code().to_le_bytes());
    bytes.extend_from_slice(&0x000Au16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    // rank 2, dims [u64::MAX, 2]: their product overflows u64 outright.
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());
    bytes.extend_from_slice(&2u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes()); // declared elements, moot: overflow trips first

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::ElementCountOverflow)
    );
}

#[test]
fn decoder_rejects_a_layout_whose_declared_element_count_disagrees_with_its_dimensions() {
    let mut bytes = relu_baseline_bytes();

    // The Relu baseline's operand[0] layout is rank 2, dims [2, 2]: its
    // `elements` field is the last 8 bytes of that layout.
    let elements_offset = header_len() + layout_total_len(2) - 8;
    bytes[elements_offset..elements_offset + 8].copy_from_slice(&999u64.to_le_bytes());

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::ElementCountMismatch)
    );
}

#[test]
fn decoder_rejects_an_elementwise_operand_shape_disagreeing_with_the_result() {
    let plan = single_binary_plan(Operation::Add, f32_type(vec![2, 2]));
    let artifact = only_artifact(&plan);
    let mut bytes = artifact.encode();

    // Corrupt operand[0]'s first dimension (2 -> 99), keeping its own layout
    // self-consistent (dims product == declared elements) so this exercises
    // the *cross*-layout `ShapeMismatch` check, not `ElementCountMismatch`.
    let first_dim_offset = header_len() + 2;
    bytes[first_dim_offset..first_dim_offset + 8].copy_from_slice(&99u64.to_le_bytes());
    let elements_offset = header_len() + layout_total_len(2) - 8;
    bytes[elements_offset..elements_offset + 8].copy_from_slice(&(99 * 2u64).to_le_bytes());

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::ShapeMismatch)
    );
}

fn permute_baseline() -> (Vec<u8>, ReferenceKernelArtifact) {
    let plan = transpose_plan(f32_type(vec![2, 3]), f32_type(vec![3, 2]), vec![1, 0]);
    let artifact = only_artifact(&plan);
    let bytes = artifact.encode();
    (bytes, artifact)
}

#[test]
fn decoder_rejects_a_permutation_with_a_repeated_axis() {
    let (mut bytes, artifact) = permute_baseline();
    let axes_start = bytes.len() - artifact.result().rank() * 2;
    bytes[axes_start..axes_start + 2].copy_from_slice(&0u16.to_le_bytes());
    bytes[axes_start + 2..axes_start + 4].copy_from_slice(&0u16.to_le_bytes());

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::InvalidPermutation)
    );
}

#[test]
fn decoder_rejects_a_permutation_axis_out_of_bounds() {
    let (mut bytes, artifact) = permute_baseline();
    let axes_start = bytes.len() - artifact.result().rank() * 2;
    bytes[axes_start..axes_start + 2].copy_from_slice(&99u16.to_le_bytes());

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::InvalidPermutation)
    );
}

#[test]
fn decoder_rejects_a_permutation_disagreeing_with_the_permuted_shape() {
    let (mut bytes, artifact) = permute_baseline();
    // [1, 0] -> [0, 1]: still a valid permutation, but now inconsistent with
    // the declared result shape [3, 2].
    let axes_start = bytes.len() - artifact.result().rank() * 2;
    bytes[axes_start..axes_start + 2].copy_from_slice(&0u16.to_le_bytes());
    bytes[axes_start + 2..axes_start + 4].copy_from_slice(&1u16.to_le_bytes());

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::ShapeMismatch)
    );
}

#[test]
fn decoder_rejects_an_attribute_tag_inconsistent_with_the_opcode() {
    let plan = single_binary_plan(Operation::Add, f32_type(vec![2, 2]));
    let artifact = only_artifact(&plan);
    let mut bytes = artifact.encode();

    // `Add` has no attribute payload, so its attribute tag is the final u16.
    let tag_offset = bytes.len() - 2;
    bytes[tag_offset..tag_offset + 2].copy_from_slice(&0x0001u16.to_le_bytes()); // Scale's tag

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::AttributeOpcodeMismatch {
            opcode: ReferenceOpcode::Add
        })
    );
}

#[test]
fn decoder_rejects_an_unknown_attribute_tag() {
    let plan = single_binary_plan(Operation::Add, f32_type(vec![2, 2]));
    let artifact = only_artifact(&plan);
    let mut bytes = artifact.encode();

    let tag_offset = bytes.len() - 2;
    bytes[tag_offset..tag_offset + 2].copy_from_slice(&0x00FFu16.to_le_bytes());

    assert_eq!(
        ReferenceKernelArtifact::decode(&bytes),
        Err(ReferenceDecodeError::UnknownAttributeTag { tag: 0x00FF })
    );
}
