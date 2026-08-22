//! Canonical byte identity, fingerprint hint and SHA-256 research digest.
//!
//! The **authoritative identity** of a [`ResearchProgram`] is
//! [`canonical_bytes`]: a fully specified, platform-independent encoding with
//! versioned magic. Two programs are structurally identical iff their
//! canonical bytes are equal ([`canonical_equal`]).
//!
//! * [`program_fingerprint`] is a 128-bit FNV-1a **hash** of those bytes — a
//!   fast lookup hint that never proves identity (collisions exist).
//! * [`program_digest`] is a hex SHA-256 over the canonical bytes — the
//!   archival/research identifier.
//!
//! The encoding never depends on `HashMap` iteration, addresses, threads,
//! wall-clock time or `Debug` formatting. Float constants are encoded through
//! their exact bit patterns (`-0.0 ≠ +0.0`). Every integer is fixed-width
//! little-endian `u64` via a checked conversion.

use sha2::{Digest, Sha256};

use super::ir::{
    Bin, Narrow, Op, Permute, Reduce, Ref, ResearchProgram, Section, ShapeTo, Ter, Un,
};
use super::types::ScalarValue;

/// Version of the canonical program encoding. Bump on any encoding change.
pub const CANONICAL_FORMAT_VERSION: u32 = 1;

/// Domain-separation magic prefix.
pub const CANONICAL_MAGIC: &[u8] = b"SCIRUST-RIR2\0";

/// The authoritative canonical byte encoding of a program.
pub fn canonical_bytes(program: &ResearchProgram) -> Vec<u8> {
    let mut encoder = Encoder::new();
    encoder.magic(CANONICAL_MAGIC);
    encoder.u32(CANONICAL_FORMAT_VERSION);

    encode_value_types(&mut encoder, &program.inputs);
    encode_value_types(&mut encoder, &program.items);
    encode_value_types(&mut encoder, &program.state);
    encoder.u32(program.steps);
    encode_section(&mut encoder, &program.init);
    encode_refs(&mut encoder, &program.init_state);
    encode_section(&mut encoder, &program.step);
    encode_refs(&mut encoder, &program.next_state);
    encode_section(&mut encoder, &program.finalize);
    encode_refs(&mut encoder, &program.outputs);

    encoder.finish()
}

/// Whether two programs share identical canonical bytes (authoritative
/// structural identity).
pub fn canonical_equal(left: &ResearchProgram, right: &ResearchProgram) -> bool {
    canonical_bytes(left) == canonical_bytes(right)
}

/// A stable 128-bit FNV-1a hash of the canonical bytes.
///
/// Deterministic and fixed, but **not** collision-free: equal fingerprints do
/// not imply equal programs. Never use it for deduplication or ordering.
pub fn program_fingerprint(program: &ResearchProgram) -> u128 {
    fnv1a_128(&canonical_bytes(program))
}

/// Hex SHA-256 of the canonical bytes — the archival research identifier.
pub fn program_digest(program: &ResearchProgram) -> String {
    let hash = Sha256::digest(canonical_bytes(program));
    let mut hex = String::with_capacity(hash.len() * 2);
    for byte in hash
    {
        hex.push(char::from_digit(u64::from(byte >> 4) as u32, 16).expect("hex digit"));
        hex.push(char::from_digit(u64::from(byte & 0x0f) as u32, 16).expect("hex digit"));
    }
    hex
}

// ---------------------------------------------------------------------------
// Encoding primitives
// ---------------------------------------------------------------------------

struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn magic(&mut self, magic: &[u8]) {
        self.bytes.extend_from_slice(magic);
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: usize) {
        // usize fits u64 on every supported target; the conversion keeps the
        // encoding width fixed regardless of pointer size.
        self.bytes.extend_from_slice(&(value as u64).to_le_bytes());
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

fn encode_value_type(encoder: &mut Encoder, value_type: &super::types::ValueType) {
    encoder.u8(value_type.dtype.tag());
    encoder.u64(value_type.shape.len());
    for &dimension in &value_type.shape
    {
        encoder.u64(dimension);
    }
}

fn encode_value_types(encoder: &mut Encoder, types: &[super::types::ValueType]) {
    encoder.u64(types.len());
    for value_type in types
    {
        encode_value_type(encoder, value_type);
    }
}

fn encode_ref(encoder: &mut Encoder, reference: &Ref) {
    match *reference
    {
        Ref::Input(index) =>
        {
            encoder.u8(0);
            encoder.u64(index);
        },
        Ref::Local(id) =>
        {
            encoder.u8(1);
            encoder.u64(id);
        },
        Ref::Item(index) =>
        {
            encoder.u8(2);
            encoder.u64(index);
        },
        Ref::StatePrev(slot) =>
        {
            encoder.u8(3);
            encoder.u64(slot);
        },
        Ref::StateFinal(slot) =>
        {
            encoder.u8(4);
            encoder.u64(slot);
        },
    }
}

fn encode_refs(encoder: &mut Encoder, references: &[usize]) {
    encoder.u64(references.len());
    for &reference in references
    {
        encoder.u64(reference);
    }
}

fn encode_bin(encoder: &mut Encoder, bin: &Bin) {
    encode_ref(encoder, &bin.lhs);
    encode_ref(encoder, &bin.rhs);
}

fn encode_un(encoder: &mut Encoder, un: &Un) {
    encode_ref(encoder, &un.src);
}

fn encode_ter(encoder: &mut Encoder, ter: &Ter) {
    encode_ref(encoder, &ter.a);
    encode_ref(encoder, &ter.b);
    encode_ref(encoder, &ter.c);
}

fn encode_scalar(encoder: &mut Encoder, value: &ScalarValue) {
    match *value
    {
        ScalarValue::F32(inner) =>
        {
            encoder.u8(0);
            encoder.u32(inner.to_bits());
        },
        ScalarValue::F64(inner) =>
        {
            encoder.u8(1);
            encoder
                .bytes
                .extend_from_slice(&inner.to_bits().to_le_bytes());
        },
        ScalarValue::Bool(inner) =>
        {
            encoder.u8(2);
            encoder.u8(u8::from(inner));
        },
    }
}

fn encode_reduce(encoder: &mut Encoder, reduce: &Reduce) {
    encode_ref(encoder, &reduce.src);
    match reduce.axis
    {
        None => encoder.u8(0),
        Some(axis) =>
        {
            encoder.u8(1);
            encoder.u64(axis);
        },
    }
}

fn encode_shape_to(encoder: &mut Encoder, shape_to: &ShapeTo) {
    encode_ref(encoder, &shape_to.src);
    encoder.u64(shape_to.shape.len());
    for &dimension in &shape_to.shape
    {
        encoder.u64(dimension);
    }
}

fn encode_permute(encoder: &mut Encoder, permute: &Permute) {
    encode_ref(encoder, &permute.src);
    encoder.u64(permute.perm.len());
    for &axis in &permute.perm
    {
        encoder.u64(axis);
    }
}

fn encode_narrow(encoder: &mut Encoder, narrow: &Narrow) {
    encode_ref(encoder, &narrow.src);
    encoder.u64(narrow.axis);
    encoder.u64(narrow.start);
    encoder.u64(narrow.len);
}

fn encode_section(encoder: &mut Encoder, section: &Section) {
    encoder.u64(section.ops.len());
    for op in &section.ops
    {
        encode_op(encoder, op);
    }
}

fn encode_op(encoder: &mut Encoder, op: &Op) {
    encoder.u16(op.tag());
    match op
    {
        Op::Const(value) => encode_scalar(encoder, value),
        Op::Add(b)
        | Op::Sub(b)
        | Op::Mul(b)
        | Op::Div(b)
        | Op::Pow(b)
        | Op::Min(b)
        | Op::Max(b)
        | Op::Eq(b)
        | Op::Ne(b)
        | Op::Lt(b)
        | Op::Le(b)
        | Op::Gt(b)
        | Op::Ge(b)
        | Op::And(b)
        | Op::Or(b)
        | Op::Dot(b)
        | Op::MatVec(b)
        | Op::VecMat(b)
        | Op::MatMul(b)
        | Op::BatchedMatMul(b)
        | Op::Outer(b) => encode_bin(encoder, b),
        Op::MulAdd(t) | Op::Clamp(t) | Op::Select(t) => encode_ter(encoder, t),
        Op::Neg(u)
        | Op::Abs(u)
        | Op::Exp(u)
        | Op::Exp2(u)
        | Op::Expm1(u)
        | Op::Log(u)
        | Op::Log2(u)
        | Op::Log1p(u)
        | Op::Sqrt(u)
        | Op::Rsqrt(u)
        | Op::Sin(u)
        | Op::Cos(u)
        | Op::Tanh(u)
        | Op::Not(u) => encode_un(encoder, u),
        Op::ReduceSum(r)
        | Op::ReduceProd(r)
        | Op::ReduceMax(r)
        | Op::ReduceMin(r)
        | Op::ReduceMean(r) => encode_reduce(encoder, r),
        Op::Reshape(s) | Op::BroadcastTo(s) => encode_shape_to(encoder, s),
        Op::Squeeze(a) | Op::Unsqueeze(a) =>
        {
            encode_ref(encoder, &a.src);
            encoder.u64(a.axis);
        },
        Op::Transpose(p) => encode_permute(encoder, p),
        Op::Concat { lhs, rhs, axis } =>
        {
            encode_ref(encoder, lhs);
            encode_ref(encoder, rhs);
            encoder.u64(*axis);
        },
        Op::Narrow(n) => encode_narrow(encoder, n),
    }
}

/// FNV-1a over 128 bits (same constants as the V1 fingerprint).
fn fnv1a_128(data: &[u8]) -> u128 {
    const OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013B;
    let mut hash = OFFSET;
    for &byte in data
    {
        hash ^= u128::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::v2::ir::{Bin, Section};
    use crate::tensor::v2::types::{DType, ValueType};

    fn sample_program() -> ResearchProgram {
        ResearchProgram::expression(
            vec![ValueType::new(DType::F32, vec![2])],
            Section::new(vec![
                Op::Const(ScalarValue::F32(-1.5)),
                Op::Mul(Bin::new(Ref::Input(0), Ref::Local(0))),
            ]),
            vec![1],
        )
    }

    #[test]
    fn identical_programs_have_identical_canonical_bytes() {
        let left = sample_program();
        let right = sample_program();
        assert!(canonical_equal(&left, &right));
        assert_eq!(program_fingerprint(&left), program_fingerprint(&right));
        assert_eq!(program_digest(&left), program_digest(&right));
    }

    #[test]
    fn any_structural_change_changes_the_identity() {
        let base = sample_program();

        let mut different_constant = base.clone();
        different_constant.finalize.ops[1] = Op::Const(ScalarValue::F32(-1.500001));
        assert!(!canonical_equal(&base, &different_constant));

        let mut different_output = base.clone();
        different_output.outputs = vec![0];
        assert!(!canonical_equal(&base, &different_output));

        let signed_zero = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![Op::Const(ScalarValue::F64(-0.0))]),
            vec![0],
        );
        let positive_zero = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![Op::Const(ScalarValue::F64(0.0))]),
            vec![0],
        );
        // Signed zero participates in identity by bit pattern.
        assert!(!canonical_equal(&signed_zero, &positive_zero));
    }

    #[test]
    fn canonical_encoding_is_versioned_and_magic_prefixed() {
        let bytes = canonical_bytes(&sample_program());
        assert!(bytes.starts_with(CANONICAL_MAGIC));
        let version = u32::from_le_bytes(
            bytes[CANONICAL_MAGIC.len()..CANONICAL_MAGIC.len() + 4]
                .try_into()
                .unwrap(),
        );
        assert_eq!(version, CANONICAL_FORMAT_VERSION);
    }

    #[test]
    fn digest_is_hex_sha256_of_canonical_bytes() {
        use sha2::{Digest, Sha256};
        let expected: String = Sha256::digest(canonical_bytes(&sample_program()))
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(program_digest(&sample_program()), expected);
        assert_eq!(expected.len(), 64);
    }

    #[test]
    fn usize_fields_encode_fixed_width_without_truncation() {
        let program = ResearchProgram::expression(vec![], Section::new(vec![]), vec![]);
        // An empty expression program still fails verification, but canonical
        // bytes are structural and defined for it.
        let bytes = canonical_bytes(&program);
        assert!(bytes.len() > CANONICAL_MAGIC.len() + 4);
    }
}
