//! Canonical byte identity and structural fingerprint of tensor programs.
//!
//! The **identity** of a program is its [`canonical_bytes`]: a fixed,
//! platform-independent serialisation of its instructions and output register.
//! Two programs are the same iff their canonical bytes are equal
//! ([`canonical_equal`]). The encoding never depends on `HashMap` iteration,
//! memory addresses, thread scheduling or wall-clock time, and `Scale` factors
//! are encoded through their exact `f32` bit pattern.
//! Every `usize` is widened losslessly to a fixed-width little-endian `u128`, so
//! the representation cannot truncate on targets whose pointer width exceeds
//! 64 bits.
//!
//! [`program_fingerprint`] is a 128-bit FNV-1a **hash** of those bytes. It is a
//! fast lookup hint, cache key and display identifier — not a proof of identity.
//! FNV-1a is not collision-free, so equal fingerprints do **not** imply equal
//! programs; callers that need identity must compare canonical bytes (or full
//! programs). Ordering and deduplication in this crate therefore use canonical
//! bytes as the authoritative comparison and treat the fingerprint only as a
//! first, fast comparison level.

use super::ir::{TensorInstruction, TensorProgram};

/// FNV-1a 128-bit offset basis.
const FNV_OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
/// FNV-1a 128-bit prime.
const FNV_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013B;

/// The canonical byte encoding of a program — its authoritative identity.
pub fn canonical_bytes(program: &TensorProgram) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_usize(&mut bytes, program.instructions.len());
    for instruction in &program.instructions
    {
        write_instruction(&mut bytes, instruction);
    }
    write_usize(&mut bytes, program.output);
    bytes
}

/// Whether two programs are structurally identical (equal canonical bytes).
///
/// This is the authoritative identity check; unlike deriving equality from a
/// fingerprint it cannot be defeated by a hash collision, and unlike `f32`
/// `PartialEq` it compares `Scale` factors by their exact bit pattern.
pub fn canonical_equal(left: &TensorProgram, right: &TensorProgram) -> bool {
    canonical_bytes(left) == canonical_bytes(right)
}

/// A stable 128-bit FNV-1a **hash** of a program's canonical bytes.
///
/// Deterministic and fixed (not the standard-library `Hash`), but not
/// collision-free: equal fingerprints do not prove equal programs. Use it as a
/// fast hint or identifier, never as an identity.
pub fn program_fingerprint(program: &TensorProgram) -> u128 {
    fnv1a_128(&canonical_bytes(program))
}

fn write_instruction(bytes: &mut Vec<u8>, instruction: &TensorInstruction) {
    match *instruction
    {
        TensorInstruction::Input { input } =>
        {
            bytes.push(0);
            write_usize(bytes, input);
        },
        TensorInstruction::Add { lhs, rhs } =>
        {
            bytes.push(1);
            write_usize(bytes, lhs);
            write_usize(bytes, rhs);
        },
        TensorInstruction::MatMul { lhs, rhs } =>
        {
            bytes.push(2);
            write_usize(bytes, lhs);
            write_usize(bytes, rhs);
        },
        TensorInstruction::Transpose2d { src } =>
        {
            bytes.push(3);
            write_usize(bytes, src);
        },
        TensorInstruction::Relu { src } =>
        {
            bytes.push(4);
            write_usize(bytes, src);
        },
        TensorInstruction::Scale { src, factor } =>
        {
            bytes.push(5);
            write_usize(bytes, src);
            write_u32(bytes, factor.to_bits());
        },
    }
}

fn write_usize(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u128).to_le_bytes());
}

fn write_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn fnv1a_128(data: &[u8]) -> u128 {
    let mut hash = FNV_OFFSET;
    for &byte in data
    {
        hash ^= byte as u128;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program(factor: f32, output: usize) -> TensorProgram {
        TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Scale { src: 0, factor },
            ],
            output,
        )
    }

    #[test]
    fn identical_programs_share_a_fingerprint() {
        assert_eq!(
            program_fingerprint(&program(2.0, 1)),
            program_fingerprint(&program(2.0, 1))
        );
    }

    #[test]
    fn a_single_changed_factor_changes_the_fingerprint() {
        assert_ne!(
            program_fingerprint(&program(2.0, 1)),
            program_fingerprint(&program(2.5, 1))
        );
    }

    #[test]
    fn a_changed_output_changes_the_fingerprint() {
        assert_ne!(
            program_fingerprint(&program(2.0, 0)),
            program_fingerprint(&program(2.0, 1))
        );
    }

    #[test]
    fn a_changed_instruction_changes_the_fingerprint() {
        let relu = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Relu { src: 0 },
            ],
            1,
        );
        let scale = program(1.0, 1);
        assert_ne!(program_fingerprint(&relu), program_fingerprint(&scale));
    }

    #[test]
    fn usize_fields_use_fixed_width_little_endian_without_truncation() {
        let program = TensorProgram::new(
            vec![TensorInstruction::Input { input: usize::MAX }],
            usize::MAX,
        );
        let bytes = canonical_bytes(&program);
        assert_eq!(&bytes[..16], &(1u128).to_le_bytes());
        assert_eq!(&bytes[17..33], &(usize::MAX as u128).to_le_bytes());
        assert_eq!(&bytes[33..49], &(usize::MAX as u128).to_le_bytes());
    }
}
