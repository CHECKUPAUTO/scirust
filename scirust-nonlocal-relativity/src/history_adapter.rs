//! Adapter between relativistic retained-history samples and `scirust-history`.
//!
//! This module deliberately adapts only domain-neutral history semantics:
//! retained value, true logical position, and reference-versus-approximation
//! classification. General-relativistic source coordinates and mutable
//! segment-by-segment transport remain owned by this crate.

use scirust_history::{HistoryEntry as GenericHistoryEntry, HistoryFidelity};

use crate::{HistoryApproximation, HistoryEntry};

/// Domain-neutral view of one relativistic velocity sample at its true parameter.
pub type PositionedVelocity<const D: usize> = GenericHistoryEntry<[f64; D], f64>;

/// Convert one typed relativistic history entry into the generic retained-history form.
///
/// The velocity components are copied bit-for-bit and `parameter` is used as the
/// original logical position. Coordinates are intentionally not transferred: they
/// are geometric metadata required by relativistic transport, not generic history
/// semantics.
#[must_use]
pub const fn positioned_velocity<const D: usize>(entry: &HistoryEntry<D>) -> PositionedVelocity<D> {
    GenericHistoryEntry::new(entry.velocity, entry.parameter)
}

/// Convert the generic history fidelity classification into this crate's legacy
/// compatibility classification.
#[must_use]
pub const fn approximation_from_fidelity(fidelity: HistoryFidelity) -> HistoryApproximation {
    match fidelity
    {
        HistoryFidelity::Reference => HistoryApproximation::Exact,
        HistoryFidelity::Approximation => HistoryApproximation::Approximate,
    }
}

/// Convert this crate's legacy compatibility classification into the generic
/// history fidelity classification.
#[must_use]
pub const fn fidelity_from_approximation(approximation: HistoryApproximation) -> HistoryFidelity {
    match approximation
    {
        HistoryApproximation::Exact => HistoryFidelity::Reference,
        HistoryApproximation::Approximate => HistoryFidelity::Approximation,
    }
}

#[cfg(test)]
mod tests {
    use scirust_history::HistoryFidelity;

    use super::{approximation_from_fidelity, fidelity_from_approximation, positioned_velocity};
    use crate::{HistoryApproximation, HistoryEntry};

    #[test]
    fn positioned_velocity_preserves_bits_and_true_parameter() {
        let entry = HistoryEntry::new(
            [1.0_f64, -2.0, 3.5, 4.0],
            [
                f64::from_bits(1),
                -0.0,
                7.25,
                f64::from_bits(0x7fe0_0000_0000_0001),
            ],
            0.375,
        );

        let generic = positioned_velocity(&entry);
        for (actual, expected) in generic.value().iter().zip(entry.velocity.iter())
        {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        assert_eq!(generic.position().to_bits(), entry.parameter.to_bits());
    }

    #[test]
    fn fidelity_mapping_is_exact_and_round_trips() {
        for approximation in [
            HistoryApproximation::Exact,
            HistoryApproximation::Approximate,
        ]
        {
            let fidelity = fidelity_from_approximation(approximation);
            assert_eq!(approximation_from_fidelity(fidelity), approximation);
        }

        assert_eq!(
            approximation_from_fidelity(HistoryFidelity::Reference),
            HistoryApproximation::Exact
        );
        assert_eq!(
            approximation_from_fidelity(HistoryFidelity::Approximation),
            HistoryApproximation::Approximate
        );
    }
}
