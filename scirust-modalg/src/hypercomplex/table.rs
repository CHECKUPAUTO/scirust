//! The authoritative signed octonion multiplication table.
//!
//! Two independent representations are provided:
//!
//! * [`IDX`] / [`SIGN`] — the hardcoded table transcribed directly from spec
//!   §8.3(b). This is the **authoritative** routing used by the 64-term oracle
//!   in [`crate::hypercomplex::octonion`].
//! * [`table_from_triples`] — an independent generator that rebuilds the table
//!   from the seven Fano triples of . It exists ONLY as a
//!   differential oracle: a test asserts it equals the hardcoded table, so a
//!   transcription typo in `IDX`/`SIGN` cannot pass silently.
//!
//! Row `i`, column `j` describes `e_i · e_j = SIGN[i][j] · e_{IDX[i][j]}`.

/// Target basis index of `e_i · e_j`). Each row is a permutation
/// of `0..8`.
pub const IDX: [[usize; 8]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [1, 0, 4, 7, 2, 6, 5, 3],
    [2, 4, 0, 5, 1, 3, 7, 6],
    [3, 7, 5, 0, 6, 2, 4, 1],
    [4, 2, 1, 6, 0, 7, 3, 5],
    [5, 6, 3, 2, 7, 0, 1, 4],
    [6, 5, 7, 4, 3, 1, 0, 2],
    [7, 3, 6, 1, 5, 4, 2, 0],
];

/// Sign of `e_i · e_j`); `+1` or `-1`.
pub const SIGN: [[i8; 8]; 8] = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, -1, 1, 1, -1, 1, -1, -1],
    [1, -1, -1, 1, 1, -1, 1, -1],
    [1, -1, -1, -1, 1, 1, -1, 1],
    [1, 1, -1, -1, -1, 1, 1, -1],
    [1, -1, 1, -1, -1, -1, 1, 1],
    [1, 1, -1, 1, -1, -1, -1, 1],
    [1, 1, 1, -1, 1, -1, -1, -1],
];

/// The seven ordered Fano triples `(a, b, c)` of . Each means the
/// cyclic relations `a·b=c, b·c=a, c·a=b` (and reversing negates).
pub const TRIPLES: [(usize, usize, usize); 7] = [
    (1, 2, 4),
    (2, 3, 5),
    (3, 4, 6),
    (4, 5, 7),
    (5, 6, 1),
    (6, 7, 2),
    (7, 1, 3),
];

/// Independently rebuild `(IDX, SIGN)` from [`TRIPLES`] and the rules
/// `e_i·e_i = -e0`, `e0·e_x = e_x·e0 = e_x`). Used only as a
/// differential oracle against the hardcoded [`IDX`]/[`SIGN`].
pub fn table_from_triples() -> ([[usize; 8]; 8], [[i8; 8]; 8]) {
    // -1 sentinel marks "not yet filled" so gaps are detectable.
    let mut idx = [[usize::MAX; 8]; 8];
    let mut sign = [[0i8; 8]; 8];

    let set = |idx: &mut [[usize; 8]; 8],
               sign: &mut [[i8; 8]; 8],
               i: usize,
               j: usize,
               s: i8,
               k: usize| {
        idx[i][j] = k;
        sign[i][j] = s;
    };

    // identity row/column
    for x in 0..8
    {
        set(&mut idx, &mut sign, 0, x, 1, x);
        set(&mut idx, &mut sign, x, 0, 1, x);
    }
    // imaginary squares
    for x in 1..8
    {
        set(&mut idx, &mut sign, x, x, -1, 0);
    }
    // Fano triples (cyclic + reversed)
    for &(a, b, c) in TRIPLES.iter()
    {
        for &(x, y, z) in &[(a, b, c), (b, c, a), (c, a, b)]
        {
            set(&mut idx, &mut sign, x, y, 1, z); // x·y = +z
            set(&mut idx, &mut sign, y, x, -1, z); // y·x = -z
        }
    }
    (idx, sign)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardcoded_table_matches_triple_derivation() {
        let (idx, sign) = table_from_triples();
        // No gaps left.
        for i in 0..8
        {
            for j in 0..8
            {
                assert_ne!(idx[i][j], usize::MAX, "gap at ({i},{j})");
            }
        }
        assert_eq!(idx, IDX, "IDX disagrees with triple derivation");
        assert_eq!(sign, SIGN, "SIGN disagrees with triple derivation");
    }

    #[test]
    fn each_row_is_a_permutation() {
        for i in 0..8
        {
            let mut seen = [false; 8];
            for j in 0..8
            {
                assert!(!seen[IDX[i][j]], "row {i} routes twice to {}", IDX[i][j]);
                seen[IDX[i][j]] = true;
            }
        }
    }

    #[test]
    fn imaginary_units_are_antisymmetric() {
        for i in 1..8
        {
            for j in 1..8
            {
                if i == j
                {
                    continue;
                }
                // e_i·e_j and e_j·e_i target the same basis with opposite sign.
                assert_eq!(IDX[i][j], IDX[j][i], "target mismatch ({i},{j})");
                assert_eq!(SIGN[i][j], -SIGN[j][i], "sign not antisymmetric ({i},{j})");
            }
        }
    }
}
