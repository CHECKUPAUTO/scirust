//! Complex gate matrices in row-major `(output, input)` layout.

use super::complex::Complex32;

const ZERO: Complex32 = Complex32::zero();
const ONE: Complex32 = Complex32::one();
const NEG_ONE: Complex32 = Complex32::new(-1.0, 0.0);
const I_UNIT: Complex32 = Complex32::new(0.0, 1.0);
const NEG_I: Complex32 = Complex32::new(0.0, -1.0);

/// Identity.
pub const I: [Complex32; 4] = [ONE, ZERO, ZERO, ONE];
/// Hadamard.
pub const H: [Complex32; 4] = [
    Complex32::new(core::f32::consts::FRAC_1_SQRT_2, 0.0),
    Complex32::new(core::f32::consts::FRAC_1_SQRT_2, 0.0),
    Complex32::new(core::f32::consts::FRAC_1_SQRT_2, 0.0),
    Complex32::new(-core::f32::consts::FRAC_1_SQRT_2, 0.0),
];
/// Pauli X.
pub const X: [Complex32; 4] = [ZERO, ONE, ONE, ZERO];
/// Pauli Y.
pub const Y: [Complex32; 4] = [ZERO, NEG_I, I_UNIT, ZERO];
/// Pauli Z.
pub const Z: [Complex32; 4] = [ONE, ZERO, ZERO, NEG_ONE];
/// Phase gate `diag(1, i)`.
pub const S: [Complex32; 4] = [ONE, ZERO, ZERO, I_UNIT];
/// Inverse phase gate `diag(1, -i)`.
pub const SDG: [Complex32; 4] = [ONE, ZERO, ZERO, NEG_I];
/// T gate `diag(1, exp(iπ/4))`.
pub fn t() -> [Complex32; 4] {
    [
        ONE,
        ZERO,
        ZERO,
        Complex32::from_phase(core::f32::consts::FRAC_PI_4),
    ]
}

/// Inverse T gate `diag(1, exp(-iπ/4))`.
pub fn tdg() -> [Complex32; 4] {
    [
        ONE,
        ZERO,
        ZERO,
        Complex32::from_phase(-core::f32::consts::FRAC_PI_4),
    ]
}

/// X-axis rotation `exp(-i θX/2)`.
pub fn rx(theta: f32) -> [Complex32; 4] {
    let (sine, cosine) = (0.5 * theta).sin_cos();
    let off_diagonal = Complex32::new(0.0, -sine);
    [
        Complex32::new(cosine, 0.0),
        off_diagonal,
        off_diagonal,
        Complex32::new(cosine, 0.0),
    ]
}

/// Y-axis rotation `exp(-i θY/2)`.
pub fn ry(theta: f32) -> [Complex32; 4] {
    let (sine, cosine) = (0.5 * theta).sin_cos();
    [
        Complex32::new(cosine, 0.0),
        Complex32::new(-sine, 0.0),
        Complex32::new(sine, 0.0),
        Complex32::new(cosine, 0.0),
    ]
}

/// Z-axis rotation `exp(-i θZ/2)`.
pub fn rz(theta: f32) -> [Complex32; 4] {
    [
        Complex32::from_phase(-0.5 * theta),
        ZERO,
        ZERO,
        Complex32::from_phase(0.5 * theta),
    ]
}

/// Phase shift `diag(1, exp(iθ))`.
pub fn phase_shift(theta: f32) -> [Complex32; 4] {
    [ONE, ZERO, ZERO, Complex32::from_phase(theta)]
}

/// Controlled-NOT with the first operand as control and the second as target.
/// Basis order is `|00>, |01>, |10>, |11>` in operand order.
pub const CNOT: [Complex32; 16] = [
    ONE, ZERO, ZERO, ZERO, // |00> -> |00>
    ZERO, ONE, ZERO, ZERO, // |01> -> |01>
    ZERO, ZERO, ZERO, ONE, // |11> -> |10>
    ZERO, ZERO, ONE, ZERO, // |10> -> |11>
];

/// Controlled Z in operand basis order `|00>, |01>, |10>, |11>`.
pub const CZ: [Complex32; 16] = [
    ONE, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, NEG_ONE,
];

/// Swap two qubits in operand basis order `|00>, |01>, |10>, |11>`.
pub const SWAP: [Complex32; 16] = [
    ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,
];

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 2.0e-6;

    fn assert_unitary(matrix: &[Complex32; 4]) {
        for row in 0..2
        {
            for column in 0..2
            {
                let mut value = Complex32::zero();
                for k in 0..2
                {
                    value += matrix[k * 2 + row].conj() * matrix[k * 2 + column];
                }
                let expected = if row == column
                {
                    Complex32::one()
                }
                else
                {
                    Complex32::zero()
                };
                assert!(
                    value.approx_eq(expected, TOLERANCE),
                    "{matrix:?}: {value:?}"
                );
            }
        }
    }

    fn assert_unitary_two_qubit(matrix: &[Complex32; 16]) {
        for row in 0..4
        {
            for column in 0..4
            {
                let mut value = Complex32::zero();
                for k in 0..4
                {
                    value += matrix[k * 4 + row].conj() * matrix[k * 4 + column];
                }
                let expected = if row == column
                {
                    Complex32::one()
                }
                else
                {
                    Complex32::zero()
                };
                assert!(
                    value.approx_eq(expected, TOLERANCE),
                    "{matrix:?}: {value:?}"
                );
            }
        }
    }

    #[test]
    fn every_one_qubit_gate_is_unitary() {
        let matrices = [
            I,
            H,
            X,
            Y,
            Z,
            S,
            SDG,
            t(),
            tdg(),
            rx(0.37),
            ry(-1.2),
            rz(2.1),
            phase_shift(-0.8),
        ];
        for matrix in &matrices
        {
            assert_unitary(matrix);
        }
    }

    #[test]
    fn y_has_the_expected_basis_action() {
        assert_eq!(Y[2], I_UNIT);
        assert_eq!(Y[1], NEG_I);
    }

    #[test]
    fn every_two_qubit_gate_is_unitary() {
        assert_unitary_two_qubit(&CNOT);
        assert_unitary_two_qubit(&CZ);
        assert_unitary_two_qubit(&SWAP);
    }
}
