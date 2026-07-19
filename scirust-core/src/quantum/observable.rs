//! Hermitian tensor products of single-qubit Pauli operators.

use super::error::{QuantumError, QuantumResult};
use std::collections::BTreeSet;

/// A Hermitian Pauli operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pauli {
    /// Pauli X.
    X,
    /// Pauli Y.
    Y,
    /// Pauli Z.
    Z,
}

/// One factor in a Pauli tensor product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PauliTerm {
    /// Qubit acted on by this factor.
    pub qubit: usize,
    /// Pauli operator applied to the qubit.
    pub pauli: Pauli,
}

impl PauliTerm {
    /// Constructs a Pauli factor on `qubit`.
    pub const fn new(qubit: usize, pauli: Pauli) -> Self {
        Self { qubit, pauli }
    }
}

/// A tensor product of Pauli factors on distinct qubits.
///
/// Omitted qubits carry the identity. Factors retain construction order for
/// stable serialization; because they act on distinct qubits, that order does
/// not change the operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observable {
    terms: Vec<PauliTerm>,
}

impl Observable {
    /// Creates a non-empty tensor product and rejects duplicate qubits.
    pub fn new(terms: Vec<PauliTerm>) -> QuantumResult<Self> {
        if terms.is_empty()
        {
            return Err(QuantumError::InvalidObservable {
                reason: "a Pauli observable needs at least one factor",
            });
        }
        let mut qubits = BTreeSet::new();
        for term in &terms
        {
            if !qubits.insert(term.qubit)
            {
                return Err(QuantumError::DuplicateQubit { qubit: term.qubit });
            }
        }
        Ok(Self { terms })
    }

    /// Convenience constructor for a one-qubit Pauli X observable.
    pub fn x(qubit: usize) -> Self {
        Self {
            terms: vec![PauliTerm::new(qubit, Pauli::X)],
        }
    }

    /// Convenience constructor for a one-qubit Pauli Y observable.
    pub fn y(qubit: usize) -> Self {
        Self {
            terms: vec![PauliTerm::new(qubit, Pauli::Y)],
        }
    }

    /// Convenience constructor for a one-qubit Pauli Z observable.
    pub fn z(qubit: usize) -> Self {
        Self {
            terms: vec![PauliTerm::new(qubit, Pauli::Z)],
        }
    }

    /// Factors in stable construction order.
    pub fn terms(&self) -> &[PauliTerm] {
        &self.terms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_qubits_are_rejected() {
        assert_eq!(
            Observable::new(vec![
                PauliTerm::new(1, Pauli::X),
                PauliTerm::new(1, Pauli::Z),
            ]),
            Err(QuantumError::DuplicateQubit { qubit: 1 })
        );
    }
}
