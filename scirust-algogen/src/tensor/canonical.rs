//! Canonical, deterministic identity of tensor programs.
//!
//! A program's canonical byte encoding is a fixed, platform-independent
//! serialisation of its instructions and output register. It is used to derive
//! a stable structural fingerprint that provides an order-independent
//! tie-breaker for ranking and a deduplication key for the hall of fame. The
//! encoding never depends on `HashMap` iteration, memory addresses, thread
//! scheduling or wall-clock time, and `Scale` factors are encoded through their
//! exact `f32` bit pattern so distinct programs always encode differently.

use super::ir::{TensorInstruction, TensorProgram};

/// FNV-1a 128-bit offset basis.
const FNV_OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
/// FNV-1a 128-bit prime.
const FNV_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013B;

/// The canonical byte encoding of a program.
pub fn canonical_bytes(program: &TensorProgram) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_u64(&mut bytes, program.instructions.len() as u64);
    for instruction in &program.instructions
    {
        write_instruction(&mut bytes, instruction);
    }
    write_u64(&mut bytes, program.output as u64);
    bytes
}

/// A stable 128-bit structural fingerprint of a program.
///
/// The algorithm (FNV-1a over the canonical bytes) is fixed and deterministic;
/// it is not the standard-library `Hash`, which is not guaranteed to be stable.
pub fn program_fingerprint(program: &TensorProgram) -> u128 {
    fnv1a_128(&canonical_bytes(program))
}

fn write_instruction(bytes: &mut Vec<u8>, instruction: &TensorInstruction) {
    match *instruction
    {
        TensorInstruction::Input { input } =>
        {
            bytes.push(0);
            write_u64(bytes, input as u64);
        },
        TensorInstruction::Add { lhs, rhs } =>
        {
            bytes.push(1);
            write_u64(bytes, lhs as u64);
            write_u64(bytes, rhs as u64);
        },
        TensorInstruction::MatMul { lhs, rhs } =>
        {
            bytes.push(2);
            write_u64(bytes, lhs as u64);
            write_u64(bytes, rhs as u64);
        },
        TensorInstruction::Transpose2d { src } =>
        {
            bytes.push(3);
            write_u64(bytes, src as u64);
        },
        TensorInstruction::Relu { src } =>
        {
            bytes.push(4);
            write_u64(bytes, src as u64);
        },
        TensorInstruction::Scale { src, factor } =>
        {
            bytes.push(5);
            write_u64(bytes, src as u64);
            write_u32(bytes, factor.to_bits());
        },
    }
}

fn write_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
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
}
