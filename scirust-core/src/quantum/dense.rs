//! Exact-in-model dense complex state-vector simulation on the CPU.
//!
//! Qubit `q` is bit `q` of a state-vector index (little endian). Thus, for two
//! qubits, index 1 is `|01>` (`q1=0, q0=1`) and index 2 is `|10>`. Matrices are
//! row-major with rows naming output basis states and columns input states.

use super::complex::Complex32;
use super::complex_gates;
use super::error::{QuantumError, QuantumResult};
use super::observable::{Observable, Pauli};
use crate::nn::PcgEngine;
use std::collections::BTreeMap;

/// Default tolerance for state normalization and unitary norm preservation.
pub const NORMALIZATION_TOLERANCE: f32 = 2.0e-5;
/// Maximum residual imaginary part accepted for a Hermitian expectation.
pub const EXPECTATION_IMAG_TOLERANCE: f32 = 2.0e-5;
/// Explicit dense-state allocation ceiling (one GiB).
pub const MAX_DENSE_STATE_BYTES: usize = 1 << 30;

/// An `n`-qubit dense state `|psi>` with `2^n` complex `f32` amplitudes.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseStateVector {
    num_qubits: usize,
    amplitudes: Vec<Complex32>,
}

impl DenseStateVector {
    /// Creates `|00...0>` for at least one qubit.
    pub fn zero(num_qubits: usize) -> QuantumResult<Self> {
        let dimension = checked_dimension(num_qubits)?;
        let mut amplitudes = Vec::new();
        amplitudes
            .try_reserve_exact(dimension)
            .map_err(|_| QuantumError::AllocationTooLarge {
                requested_bytes: dimension * core::mem::size_of::<Complex32>(),
                limit_bytes: MAX_DENSE_STATE_BYTES,
            })?;
        amplitudes.resize(dimension, Complex32::zero());
        amplitudes[0] = Complex32::one();
        Ok(Self {
            num_qubits,
            amplitudes,
        })
    }

    /// Constructs a state from little-endian-indexed amplitudes, requiring
    /// finite values and unit norm within [`NORMALIZATION_TOLERANCE`].
    pub fn from_amplitudes(num_qubits: usize, amplitudes: Vec<Complex32>) -> QuantumResult<Self> {
        let expected = checked_dimension(num_qubits)?;
        if amplitudes.len() != expected
        {
            return Err(QuantumError::InvalidStateDimension {
                expected,
                actual: amplitudes.len(),
            });
        }
        let state = Self {
            num_qubits,
            amplitudes,
        };
        state.validate_finite()?;
        state.ensure_normalized(NORMALIZATION_TOLERANCE)?;
        Ok(state)
    }

    /// Number of qubits.
    pub const fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Amplitudes in state-vector index order.
    pub fn amplitudes(&self) -> &[Complex32] {
        &self.amplitudes
    }

    /// Squared state norm, accumulated in deterministic index order.
    pub fn norm_sqr(&self) -> f32 {
        self.amplitudes.iter().map(|value| value.norm_sqr()).sum()
    }

    /// State norm.
    pub fn norm(&self) -> f32 {
        self.norm_sqr().sqrt()
    }

    /// Rejects non-finite amplitudes.
    pub fn validate_finite(&self) -> QuantumResult<()> {
        if self.amplitudes.iter().all(|value| value.is_finite())
        {
            Ok(())
        }
        else
        {
            Err(QuantumError::NumericalFailure {
                operation: "state finite-value validation",
            })
        }
    }

    /// Requires squared norm to be within `tolerance` of one.
    pub fn ensure_normalized(&self, tolerance: f32) -> QuantumResult<()> {
        if !tolerance.is_finite() || tolerance < 0.0
        {
            return Err(QuantumError::NonFiniteParameter {
                what: "normalization tolerance",
            });
        }
        let norm_sqr = self.norm_sqr();
        if !norm_sqr.is_finite()
        {
            return Err(QuantumError::NumericalFailure {
                operation: "state norm",
            });
        }
        if (norm_sqr - 1.0).abs() > tolerance
        {
            return Err(QuantumError::NonNormalizedState {
                norm_sqr,
                tolerance,
            });
        }
        Ok(())
    }

    /// Explicitly repairs normalization. A zero or non-finite norm is rejected.
    pub fn normalize(&mut self) -> QuantumResult<()> {
        let norm = self.norm();
        if !norm.is_finite() || norm <= 0.0
        {
            return Err(QuantumError::NumericalFailure {
                operation: "state normalization",
            });
        }
        for amplitude in &mut self.amplitudes
        {
            *amplitude = *amplitude / norm;
        }
        self.ensure_normalized(NORMALIZATION_TOLERANCE)
    }

    /// Applies a row-major 2x2 matrix to one qubit without constructing a full unitary.
    pub fn apply_matrix(&mut self, target: usize, matrix: &[Complex32; 4]) -> QuantumResult<()> {
        self.validate_qubit(target)?;
        if !matrix.iter().all(|value| value.is_finite())
        {
            return Err(QuantumError::NonFiniteParameter {
                what: "gate matrix",
            });
        }
        let mask = 1usize << target;
        for base in 0..self.amplitudes.len()
        {
            if base & mask == 0
            {
                let one = base | mask;
                let input_zero = self.amplitudes[base];
                let input_one = self.amplitudes[one];
                self.amplitudes[base] = matrix[0] * input_zero + matrix[1] * input_one;
                self.amplitudes[one] = matrix[2] * input_zero + matrix[3] * input_one;
            }
        }
        self.validate_after_unitary()
    }

    /// Applies a row-major 4x4 matrix to two distinct operands. Matrix local
    /// basis order is `|00>, |01>, |10>, |11>` in `(first, second)` order.
    pub fn apply_two_qubit_matrix(
        &mut self,
        first: usize,
        second: usize,
        matrix: &[Complex32; 16],
    ) -> QuantumResult<()> {
        self.validate_two_qubits(first, second)?;
        if !matrix.iter().all(|value| value.is_finite())
        {
            return Err(QuantumError::NonFiniteParameter {
                what: "two-qubit gate matrix",
            });
        }
        let first_mask = 1usize << first;
        let second_mask = 1usize << second;
        for base in 0..self.amplitudes.len()
        {
            if base & first_mask == 0 && base & second_mask == 0
            {
                let indices = [
                    base,
                    base | second_mask,
                    base | first_mask,
                    base | first_mask | second_mask,
                ];
                let input = indices.map(|index| self.amplitudes[index]);
                let mut output = [Complex32::zero(); 4];
                for row in 0..4
                {
                    for column in 0..4
                    {
                        output[row] += matrix[row * 4 + column] * input[column];
                    }
                }
                for (index, value) in indices.into_iter().zip(output)
                {
                    self.amplitudes[index] = value;
                }
            }
        }
        self.validate_after_unitary()
    }

    /// Identity gate.
    pub fn i(&mut self, target: usize) -> QuantumResult<()> {
        self.apply_matrix(target, &complex_gates::I)
    }

    /// Hadamard gate.
    pub fn h(&mut self, target: usize) -> QuantumResult<()> {
        self.apply_matrix(target, &complex_gates::H)
    }

    /// Pauli X gate.
    pub fn x(&mut self, target: usize) -> QuantumResult<()> {
        self.apply_matrix(target, &complex_gates::X)
    }

    /// Pauli Y gate.
    pub fn y(&mut self, target: usize) -> QuantumResult<()> {
        self.apply_matrix(target, &complex_gates::Y)
    }

    /// Pauli Z gate.
    pub fn z(&mut self, target: usize) -> QuantumResult<()> {
        self.apply_matrix(target, &complex_gates::Z)
    }

    /// S phase gate.
    pub fn s(&mut self, target: usize) -> QuantumResult<()> {
        self.apply_matrix(target, &complex_gates::S)
    }

    /// Inverse S phase gate.
    pub fn sdg(&mut self, target: usize) -> QuantumResult<()> {
        self.apply_matrix(target, &complex_gates::SDG)
    }

    /// T phase gate.
    pub fn t(&mut self, target: usize) -> QuantumResult<()> {
        self.apply_matrix(target, &complex_gates::t())
    }

    /// Inverse T phase gate.
    pub fn tdg(&mut self, target: usize) -> QuantumResult<()> {
        self.apply_matrix(target, &complex_gates::tdg())
    }

    /// X-axis rotation in radians.
    pub fn rx(&mut self, target: usize, theta: f32) -> QuantumResult<()> {
        validate_angle(theta)?;
        self.apply_matrix(target, &complex_gates::rx(theta))
    }

    /// Y-axis rotation in radians.
    pub fn ry(&mut self, target: usize, theta: f32) -> QuantumResult<()> {
        validate_angle(theta)?;
        self.apply_matrix(target, &complex_gates::ry(theta))
    }

    /// Z-axis rotation in radians.
    pub fn rz(&mut self, target: usize, theta: f32) -> QuantumResult<()> {
        validate_angle(theta)?;
        self.apply_matrix(target, &complex_gates::rz(theta))
    }

    /// Phase shift `diag(1, exp(i theta))` in radians.
    pub fn phase_shift(&mut self, target: usize, theta: f32) -> QuantumResult<()> {
        validate_angle(theta)?;
        self.apply_matrix(target, &complex_gates::phase_shift(theta))
    }

    /// Controlled-NOT, with explicit control and target ordering.
    pub fn cnot(&mut self, control: usize, target: usize) -> QuantumResult<()> {
        if control == target
        {
            return Err(QuantumError::InvalidControlTarget { control, target });
        }
        self.apply_two_qubit_matrix(control, target, &complex_gates::CNOT)
    }

    /// Controlled Z, with explicit control and target ordering.
    pub fn cz(&mut self, control: usize, target: usize) -> QuantumResult<()> {
        if control == target
        {
            return Err(QuantumError::InvalidControlTarget { control, target });
        }
        self.apply_two_qubit_matrix(control, target, &complex_gates::CZ)
    }

    /// Swaps any two distinct qubits; adjacency is not required.
    pub fn swap(&mut self, first: usize, second: usize) -> QuantumResult<()> {
        self.apply_two_qubit_matrix(first, second, &complex_gates::SWAP)
    }

    /// Computes a Hermitian Pauli-product expectation exactly from amplitudes.
    pub fn expectation(&self, observable: &Observable) -> QuantumResult<f32> {
        self.ensure_normalized(NORMALIZATION_TOLERANCE)?;
        for term in observable.terms()
        {
            self.validate_qubit(term.qubit)?;
        }

        let mut expectation = Complex32::zero();
        for (input_index, &input_amplitude) in self.amplitudes.iter().enumerate()
        {
            let mut output_index = input_index;
            let mut coefficient = Complex32::one();
            for term in observable.terms()
            {
                let mask = 1usize << term.qubit;
                let bit_is_one = input_index & mask != 0;
                match term.pauli
                {
                    Pauli::X => output_index ^= mask,
                    Pauli::Y =>
                    {
                        output_index ^= mask;
                        coefficient *= if bit_is_one
                        {
                            Complex32::new(0.0, -1.0)
                        }
                        else
                        {
                            Complex32::new(0.0, 1.0)
                        };
                    },
                    Pauli::Z =>
                    {
                        if bit_is_one
                        {
                            coefficient = -coefficient;
                        }
                    },
                }
            }
            expectation += self.amplitudes[output_index].conj() * coefficient * input_amplitude;
        }
        if expectation.im.abs() > EXPECTATION_IMAG_TOLERANCE
        {
            return Err(QuantumError::NumericalFailure {
                operation: "Hermitian expectation has a non-real residual",
            });
        }
        if !expectation.re.is_finite()
        {
            return Err(QuantumError::NumericalFailure {
                operation: "observable expectation",
            });
        }
        Ok(expectation.re)
    }

    /// Samples computational-basis outcomes from amplitude probabilities.
    ///
    /// The returned bit strings are written `q_(n-1)...q_0` (most significant
    /// displayed first), while state-vector indices remain little endian.
    /// Counts use a `BTreeMap` so iteration and serialization order are stable.
    pub fn sample(&self, shots: usize, seed: u64) -> QuantumResult<BTreeMap<String, usize>> {
        if shots == 0
        {
            return Err(QuantumError::BackendCapabilityMismatch {
                capability: "zero-shot sampling",
            });
        }
        self.ensure_normalized(NORMALIZATION_TOLERANCE)?;
        let mut cumulative = Vec::with_capacity(self.amplitudes.len());
        let mut total = 0.0f32;
        for amplitude in &self.amplitudes
        {
            total += amplitude.norm_sqr();
            cumulative.push(total);
        }
        if !total.is_finite() || total <= 0.0
        {
            return Err(QuantumError::NumericalFailure {
                operation: "sampling probability accumulation",
            });
        }

        let mut rng = PcgEngine::new(seed);
        let mut counts = BTreeMap::new();
        for _ in 0..shots
        {
            let draw = rng.float() * total;
            let index = cumulative.partition_point(|&boundary| boundary <= draw);
            let index = index.min(self.amplitudes.len() - 1);
            let bits = format!("{index:0width$b}", width = self.num_qubits);
            *counts.entry(bits).or_insert(0) += 1;
        }
        Ok(counts)
    }

    fn validate_qubit(&self, qubit: usize) -> QuantumResult<()> {
        if qubit < self.num_qubits
        {
            Ok(())
        }
        else
        {
            Err(QuantumError::InvalidQubitIndex {
                qubit,
                num_qubits: self.num_qubits,
            })
        }
    }

    fn validate_two_qubits(&self, first: usize, second: usize) -> QuantumResult<()> {
        self.validate_qubit(first)?;
        self.validate_qubit(second)?;
        if first == second
        {
            return Err(QuantumError::DuplicateQubit { qubit: first });
        }
        Ok(())
    }

    fn validate_after_unitary(&self) -> QuantumResult<()> {
        self.validate_finite()?;
        self.ensure_normalized(NORMALIZATION_TOLERANCE)
    }
}

fn validate_angle(theta: f32) -> QuantumResult<()> {
    if theta.is_finite()
    {
        Ok(())
    }
    else
    {
        Err(QuantumError::NonFiniteParameter { what: "gate angle" })
    }
}

fn checked_dimension(num_qubits: usize) -> QuantumResult<usize> {
    if num_qubits == 0 || num_qubits >= usize::BITS as usize
    {
        return Err(QuantumError::StateDimensionOverflow { num_qubits });
    }
    let dimension = 1usize
        .checked_shl(num_qubits as u32)
        .ok_or(QuantumError::StateDimensionOverflow { num_qubits })?;
    let requested_bytes = dimension
        .checked_mul(core::mem::size_of::<Complex32>())
        .ok_or(QuantumError::StateDimensionOverflow { num_qubits })?;
    if requested_bytes > MAX_DENSE_STATE_BYTES
    {
        return Err(QuantumError::AllocationTooLarge {
            requested_bytes,
            limit_bytes: MAX_DENSE_STATE_BYTES,
        });
    }
    Ok(dimension)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::observable::{Pauli, PauliTerm};

    const TOLERANCE: f32 = 3.0e-5;

    fn assert_amplitude(state: &DenseStateVector, index: usize, expected: Complex32) {
        assert!(
            state.amplitudes()[index].approx_eq(expected, TOLERANCE),
            "index {index}: {:?} != {expected:?}",
            state.amplitudes()[index]
        );
    }

    #[test]
    fn little_endian_basis_indices_are_explicit() {
        let mut state = DenseStateVector::zero(2).unwrap();
        state.x(0).unwrap();
        assert_amplitude(&state, 1, Complex32::one()); // |01>

        let mut state = DenseStateVector::zero(2).unwrap();
        state.x(1).unwrap();
        assert_amplitude(&state, 2, Complex32::one()); // |10>
    }

    #[test]
    fn all_one_qubit_basis_actions_preserve_norm() {
        let actions: &[fn(&mut DenseStateVector, usize) -> QuantumResult<()>] = &[
            DenseStateVector::i,
            DenseStateVector::h,
            DenseStateVector::x,
            DenseStateVector::y,
            DenseStateVector::z,
            DenseStateVector::s,
            DenseStateVector::sdg,
            DenseStateVector::t,
            DenseStateVector::tdg,
        ];
        for action in actions
        {
            let mut state = DenseStateVector::zero(1).unwrap();
            action(&mut state, 0).unwrap();
            assert!((state.norm_sqr() - 1.0).abs() <= TOLERANCE);
        }
        for matrix in [
            complex_gates::rx(0.4),
            complex_gates::ry(-0.7),
            complex_gates::rz(1.1),
            complex_gates::phase_shift(0.9),
        ]
        {
            let mut state = DenseStateVector::zero(1).unwrap();
            state.apply_matrix(0, &matrix).unwrap();
            assert!((state.norm_sqr() - 1.0).abs() <= TOLERANCE);
        }
    }

    #[test]
    fn plus_bell_and_ghz_states_match_manual_amplitudes() {
        let inv = core::f32::consts::FRAC_1_SQRT_2;
        let mut plus = DenseStateVector::zero(1).unwrap();
        plus.h(0).unwrap();
        assert_amplitude(&plus, 0, Complex32::new(inv, 0.0));
        assert_amplitude(&plus, 1, Complex32::new(inv, 0.0));

        let mut bell = DenseStateVector::zero(2).unwrap();
        bell.h(0).unwrap();
        bell.cnot(0, 1).unwrap();
        assert_amplitude(&bell, 0b00, Complex32::new(inv, 0.0));
        assert_amplitude(&bell, 0b11, Complex32::new(inv, 0.0));

        let mut ghz = DenseStateVector::zero(3).unwrap();
        ghz.h(0).unwrap();
        ghz.cnot(0, 1).unwrap();
        ghz.cnot(1, 2).unwrap();
        assert_amplitude(&ghz, 0b000, Complex32::new(inv, 0.0));
        assert_amplitude(&ghz, 0b111, Complex32::new(inv, 0.0));
    }

    #[test]
    fn h_then_s_requires_imaginary_amplitudes() {
        let mut state = DenseStateVector::zero(1).unwrap();
        state.h(0).unwrap();
        state.s(0).unwrap();
        let inv = core::f32::consts::FRAC_1_SQRT_2;
        assert_amplitude(&state, 0, Complex32::new(inv, 0.0));
        assert_amplitude(&state, 1, Complex32::new(0.0, inv));
        assert!((state.expectation(&Observable::y(0)).unwrap() - 1.0).abs() <= TOLERANCE);
    }

    #[test]
    fn swap_and_controlled_gates_have_known_basis_actions() {
        let mut state = DenseStateVector::zero(3).unwrap();
        state.x(0).unwrap(); // |001>
        state.swap(0, 2).unwrap(); // |100>
        assert_amplitude(&state, 0b100, Complex32::one());
        state.cnot(2, 1).unwrap(); // |110>
        assert_amplitude(&state, 0b110, Complex32::one());
        state.cz(2, 1).unwrap();
        assert_amplitude(&state, 0b110, Complex32::new(-1.0, 0.0));
    }

    #[test]
    fn pauli_expectations_match_manual_values() {
        let zero = DenseStateVector::zero(2).unwrap();
        assert!((zero.expectation(&Observable::z(0)).unwrap() - 1.0).abs() <= TOLERANCE);
        assert!(zero.expectation(&Observable::x(0)).unwrap().abs() <= TOLERANCE);
        assert!(zero.expectation(&Observable::y(0)).unwrap().abs() <= TOLERANCE);

        let mut bell = DenseStateVector::zero(2).unwrap();
        bell.h(0).unwrap();
        bell.cnot(0, 1).unwrap();
        let zz = Observable::new(vec![
            PauliTerm::new(0, Pauli::Z),
            PauliTerm::new(1, Pauli::Z),
        ])
        .unwrap();
        assert!((bell.expectation(&zz).unwrap() - 1.0).abs() <= TOLERANCE);
    }

    #[test]
    fn seeded_sampling_is_exactly_repeatable_and_counts_all_shots() {
        let mut state = DenseStateVector::zero(1).unwrap();
        state.h(0).unwrap();
        let first = state.sample(2_000, 17).unwrap();
        let second = state.sample(2_000, 17).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.values().sum::<usize>(), 2_000);
        let zeros = first.get("0").copied().unwrap_or(0);
        assert!(
            (850..=1_150).contains(&zeros),
            "unexpected counts: {first:?}"
        );
    }

    #[test]
    fn invalid_indices_duplicates_and_allocations_are_errors() {
        let mut state = DenseStateVector::zero(2).unwrap();
        assert!(matches!(
            state.x(2),
            Err(QuantumError::InvalidQubitIndex { .. })
        ));
        assert_eq!(
            state.swap(1, 1),
            Err(QuantumError::DuplicateQubit { qubit: 1 })
        );
        assert!(matches!(
            DenseStateVector::zero(40),
            Err(QuantumError::AllocationTooLarge { .. })
        ));
    }
}
