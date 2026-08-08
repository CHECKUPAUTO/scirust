//! Decomposition of real finite-group characters into irreducible multiplicities.
//!
//! The implementation consumes a validated [`crate::representation::CharacterTable`]
//! and writes multiplicities into caller-owned storage.

use crate::representation::CharacterTable;

/// Error returned while decomposing a reducible character.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CharacterDecompositionError
{
    /// The supplied character does not contain one value per conjugacy class.
    CharacterLengthMismatch,
    /// The caller-provided multiplicity buffer is too small.
    OutputTooSmall,
    /// The character table carries an invalid zero group order.
    InvalidGroupOrder,
    /// An inner product is not a non-negative integer within the requested tolerance.
    NonIntegralMultiplicity {
        /// Irreducible-representation row that failed validation.
        irrep: usize,
        /// Floating-point inner product obtained before integer validation.
        value: f64,
    },
}

/// Decompose a real class character into irreducible multiplicities.
///
/// For each irreducible row `rho`, computes
/// `n_rho = (1 / |G|) sum_C |C| chi(C) chi_rho(C)`.
/// The result must be a non-negative integer to `tolerance`; otherwise the input is
/// not accepted as a valid exact decomposition for this table.
pub fn decompose_character_into<const I: usize, const C: usize>(
    table: &CharacterTable<I, C>,
    character: &[f64],
    multiplicities: &mut [usize],
    tolerance: f64,
) -> Result<usize, CharacterDecompositionError>
{
    if character.len() != C
    {
        return Err(CharacterDecompositionError::CharacterLengthMismatch);
    }
    if multiplicities.len() < I
    {
        return Err(CharacterDecompositionError::OutputTooSmall);
    }
    if table.group_order() == 0
    {
        return Err(CharacterDecompositionError::InvalidGroupOrder);
    }

    let mut irrep = 0usize;
    while irrep < I
    {
        let mut weighted = 0.0;
        let mut class = 0usize;
        while class < C
        {
            weighted += table.class_size(class) as f64
                * character[class]
                * table.get(irrep, class);
            class += 1;
        }
        let value = weighted / table.group_order() as f64;
        let rounded = value.round();
        if !value.is_finite()
            || rounded < 0.0
            || (value - rounded).abs() > tolerance
            || rounded > usize::MAX as f64
        {
            return Err(CharacterDecompositionError::NonIntegralMultiplicity {
                irrep,
                value,
            });
        }
        multiplicities[irrep] = rounded as usize;
        irrep += 1;
    }

    Ok(I)
}

#[cfg(test)]
mod tests
{
    use super::*;

    fn s3_table() -> CharacterTable<3, 3>
    {
        CharacterTable::new(
            [[1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [2.0, 0.0, -1.0]],
            [1, 3, 2],
            6,
        )
    }

    #[test]
    fn regular_character_contains_each_irrep_by_its_dimension()
    {
        let table = s3_table();
        let mut multiplicities = [0usize; 3];
        assert_eq!(
            decompose_character_into(&table, &[6.0, 0.0, 0.0], &mut multiplicities, 1e-12),
            Ok(3)
        );
        assert_eq!(multiplicities, [1, 1, 2]);
    }

    #[test]
    fn natural_s3_permutation_character_is_trivial_plus_standard()
    {
        let table = s3_table();
        let mut multiplicities = [0usize; 3];
        decompose_character_into(&table, &[3.0, 1.0, 0.0], &mut multiplicities, 1e-12)
            .unwrap();
        assert_eq!(multiplicities, [1, 0, 1]);
    }

    #[test]
    fn invalid_character_is_rejected()
    {
        let table = s3_table();
        let mut multiplicities = [0usize; 3];
        assert!(matches!(
            decompose_character_into(&table, &[1.0, 0.0, 0.0], &mut multiplicities, 1e-12),
            Err(CharacterDecompositionError::NonIntegralMultiplicity { .. })
        ));
    }
}
