//! Experiment 1 — left/right multiplication matrices (spec §Experiment 1).
//!
//! For a fixed octonion `a`, `L_a(x) = a ⊗ x` and `R_a(x) = x ⊗ a` are linear
//! over `Z/2^k`. We build their `8×8` matrices from basis-vector evaluations and
//! verify `M · x == a ⊗ x` on the whole domain (`k = 2`) or a large sample.

use crate::algebra::Oct;
use crate::algebra::word::Word;
use crate::analysis::modmatrix::Mat8;
use crate::analysis::util::{Coverage, test_points};

/// Which side of the product a matrix represents.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Side {
    /// `L_a(x) = a ⊗ x`.
    Left,
    /// `R_a(x) = x ⊗ a`.
    Right,
}

/// Result of lifting one fixed `a` to its matrix and validating it.
#[derive(Clone, Debug)]
pub struct LiftResult<W: Word> {
    /// The fixed multiplier.
    pub a: Oct<W>,
    /// Which side.
    pub side: Side,
    /// The recovered `8×8` matrix.
    pub matrix: Mat8<W>,
    /// `det mod 2^k`.
    pub det_mod: u64,
    /// Whether the matrix (and hence the map) is invertible over `Z/2^k`.
    pub invertible: bool,
    /// Rank over `GF(2)` of the matrix reduced mod 2 (explicitly GF(2), not ring).
    pub gf2_rank: u32,
    /// `log2` of the exact kernel size over `Z/2^k`.
    pub kernel_log2: u32,
    /// Whether `M·x == a⊗x` held on every tested `x`.
    pub matrix_matches_oracle: bool,
    /// Coverage of the verification.
    pub coverage: Coverage,
}

/// Build the matrix of `L_a` or `R_a` from basis-vector evaluations.
pub fn build_matrix<W: Word>(a: Oct<W>, side: Side) -> Mat8<W> {
    let mut m = Mat8::<W>::zero();
    for j in 0..8
    {
        let ej = Oct::<W>::e(j);
        let col = match side
        {
            Side::Left => a.mul(ej),  // column j of L_a is a ⊗ e_j
            Side::Right => ej.mul(a), // column j of R_a is e_j ⊗ a
        };
        m.set_col(j, col.c);
    }
    m
}

/// Lift `a` to a matrix and validate it against the octonion oracle.
pub fn lift<W: Word>(a: Oct<W>, side: Side, seed: u64, sample: usize) -> LiftResult<W> {
    let matrix = build_matrix(a, side);
    let (points, coverage) = test_points::<W>(seed, sample);
    let mut matches = true;
    for x in &points
    {
        let lifted = matrix.matvec(&x.c);
        let oracle = match side
        {
            Side::Left => a.mul(*x),
            Side::Right => x.mul(a),
        };
        if lifted != oracle.c
        {
            matches = false;
            break;
        }
    }
    LiftResult {
        a,
        side,
        det_mod: matrix.det_mod().to_u64(),
        invertible: matrix.is_unit(),
        gf2_rank: matrix.gf2_rank(),
        kernel_log2: matrix.kernel_log2(),
        matrix_matches_oracle: matches,
        matrix,
        coverage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::word::{W2, W8};

    #[test]
    fn lifted_matrix_equals_oracle_exhaustive_w2() {
        // exhaustive over the W2 domain for a handful of multipliers, both sides
        for code in [0u64, 1, 2, 12345, 65535]
        {
            let a = Oct::<W2>::from_u64s(std::array::from_fn(|i| (code >> (2 * i)) & 3));
            for side in [Side::Left, Side::Right]
            {
                let r = lift(a, side, 1, 0);
                assert!(r.matrix_matches_oracle, "matrix != oracle for a={code:?}");
                assert_eq!(r.coverage, Coverage::Exhaustive);
            }
        }
    }

    #[test]
    fn identity_multiplier_is_invertible() {
        let a = Oct::<W8>::one();
        let r = lift(a, Side::Left, 7, 4096);
        assert!(r.matrix_matches_oracle);
        assert!(r.invertible, "L_1 must be invertible");
        assert_eq!(r.gf2_rank, 8);
        assert_eq!(r.kernel_log2, 0);
    }

    #[test]
    fn odd_norm_multiplier_invertible_even_norm_not() {
        // a with odd norm -> unit octonion -> L_a invertible.
        // a = e0 + e1 has norm 2 (even) -> non-unit; a = e0 has norm 1 (odd).
        let unit = Oct::<W8>::one(); // norm 1 (odd)
        assert!(lift(unit, Side::Left, 1, 1024).invertible);
        let nonunit = Oct::<W8>::from_u64s([1, 1, 0, 0, 0, 0, 0, 0]); // norm 2
        assert!(!lift(nonunit, Side::Left, 1, 1024).invertible);
    }
}
