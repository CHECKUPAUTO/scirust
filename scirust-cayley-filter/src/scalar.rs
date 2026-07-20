//! Scalar `f64` Cayley–Dickson reference implementation.

/// Number of real coordinates in a sedenion.
pub const SEDENION_DIMENSION: usize = 16;

/// Scalar reference representation of a sedenion.
pub type Sedenion = [f64; SEDENION_DIMENSION];

type Quaternion = [f64; 4];
type Octonion = [f64; 8];

/// Returns the basis vector `e_index`, or `None` when `index >= 16`.
#[must_use]
pub fn basis_vector(index: usize) -> Option<Sedenion> {
    if index >= SEDENION_DIMENSION
    {
        return None;
    }

    let mut result = [0.0; SEDENION_DIMENSION];
    result[index] = 1.0;
    Some(result)
}

/// Cayley–Dickson conjugation.
#[must_use]
pub fn conjugate(mut value: Sedenion) -> Sedenion {
    for component in value.iter_mut().skip(1)
    {
        *component = -*component;
    }
    value
}

/// Squared Euclidean norm with a fixed sequential accumulation order.
#[must_use]
pub fn squared_norm(value: &Sedenion) -> f64 {
    value
        .iter()
        .fold(0.0, |sum, component| sum + component * component)
}

/// Sedenion multiplication using the SciRust Cayley–Dickson convention:
///
/// `(a,b)(c,d) = (ac - conjugate(d)b, da + b conjugate(c))`.
#[must_use]
pub fn sedenion_mul(left: Sedenion, right: Sedenion) -> Sedenion {
    let mut a = [0.0; 8];
    let mut b = [0.0; 8];
    let mut c = [0.0; 8];
    let mut d = [0.0; 8];

    a.copy_from_slice(&left[..8]);
    b.copy_from_slice(&left[8..]);
    c.copy_from_slice(&right[..8]);
    d.copy_from_slice(&right[8..]);

    let ac = octonion_mul(a, c);
    let conjugate_d_b = octonion_mul(conjugate_array(d), b);
    let da = octonion_mul(d, a);
    let b_conjugate_c = octonion_mul(b, conjugate_array(c));

    let low: [f64; 8] = core::array::from_fn(|index| ac[index] - conjugate_d_b[index]);
    let high: [f64; 8] = core::array::from_fn(|index| da[index] + b_conjugate_c[index]);

    let mut result = [0.0; 16];
    result[..8].copy_from_slice(&low);
    result[8..].copy_from_slice(&high);
    result
}

fn octonion_mul(left: Octonion, right: Octonion) -> Octonion {
    let mut a = [0.0; 4];
    let mut b = [0.0; 4];
    let mut c = [0.0; 4];
    let mut d = [0.0; 4];

    a.copy_from_slice(&left[..4]);
    b.copy_from_slice(&left[4..]);
    c.copy_from_slice(&right[..4]);
    d.copy_from_slice(&right[4..]);

    let ac = quaternion_mul(a, c);
    let conjugate_d_b = quaternion_mul(conjugate_array(d), b);
    let da = quaternion_mul(d, a);
    let b_conjugate_c = quaternion_mul(b, conjugate_array(c));

    let low: [f64; 4] = core::array::from_fn(|index| ac[index] - conjugate_d_b[index]);
    let high: [f64; 4] = core::array::from_fn(|index| da[index] + b_conjugate_c[index]);

    let mut result = [0.0; 8];
    result[..4].copy_from_slice(&low);
    result[4..].copy_from_slice(&high);
    result
}

fn quaternion_mul(left: Quaternion, right: Quaternion) -> Quaternion {
    [
        left[0] * right[0] - left[1] * right[1] - left[2] * right[2] - left[3] * right[3],
        left[0] * right[1] + left[1] * right[0] + left[2] * right[3] - left[3] * right[2],
        left[0] * right[2] - left[1] * right[3] + left[2] * right[0] + left[3] * right[1],
        left[0] * right[3] + left[1] * right[2] - left[2] * right[1] + left[3] * right[0],
    ]
}

fn conjugate_array<const N: usize>(mut value: [f64; N]) -> [f64; N] {
    for component in value.iter_mut().skip(1)
    {
        *component = -*component;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::{
        SEDENION_DIMENSION, Sedenion, basis_vector, conjugate, sedenion_mul, squared_norm,
    };

    const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];

    #[test]
    fn basis_vectors_cover_all_sixteen_coordinates() {
        for index in 0..SEDENION_DIMENSION
        {
            let basis = basis_vector(index).expect("index is inside the basis");
            assert_eq!(basis[index], 1.0);
            assert_eq!(squared_norm(&basis), 1.0);
        }

        assert!(basis_vector(SEDENION_DIMENSION).is_none());
    }

    #[test]
    fn conjugation_negates_only_imaginary_coordinates() {
        let mut value = [0.0; SEDENION_DIMENSION];
        value[0] = 2.0;
        value[1] = 3.0;
        value[15] = -4.0;

        let result = conjugate(value);

        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], -3.0);
        assert_eq!(result[15], 4.0);
    }

    #[test]
    fn real_unit_is_a_two_sided_identity() {
        let one = basis_vector(0).expect("e0 exists");

        for index in 0..SEDENION_DIMENSION
        {
            let value = basis_vector(index).expect("basis element exists");
            assert_eq!(sedenion_mul(one, value), value);
            assert_eq!(sedenion_mul(value, one), value);
        }
    }

    #[test]
    fn fixed_product_matches_exact_oracle() {
        let left = [
            1.0, -1.0, 2.0, 0.0, 3.0, -2.0, 1.0, 1.0, 0.0, 2.0, -1.0, 1.0, -3.0, 0.0, 2.0, -1.0,
        ];
        let right = [
            2.0, 1.0, 0.0, -1.0, 1.0, 3.0, -2.0, 0.0, 1.0, -1.0, 2.0, 0.0, 1.0, -2.0, 0.0, 3.0,
        ];
        let expected = [
            18.0, 6.0, -1.0, 3.0, 9.0, -4.0, -11.0, 10.0, -20.0, 15.0, -3.0, -12.0, 3.0, -7.0,
            14.0, 2.0,
        ];

        assert_eq!(sedenion_mul(left, right), expected);
    }

    #[test]
    fn known_zero_divisor_pair_multiplies_to_exact_zero() {
        let mut left = ZERO;
        left[1] = 1.0;
        left[10] = 1.0;

        let mut right = ZERO;
        right[4] = 1.0;
        right[15] = -1.0;

        assert_eq!(squared_norm(&left), 2.0);
        assert_eq!(squared_norm(&right), 2.0);
        assert_eq!(sedenion_mul(left, right), ZERO);
    }

    #[test]
    fn norm_is_not_multiplicative_for_zero_divisors() {
        let mut left = ZERO;
        left[1] = 1.0;
        left[10] = 1.0;

        let mut right = ZERO;
        right[4] = 1.0;
        right[15] = -1.0;

        let product = sedenion_mul(left, right);

        assert_eq!(squared_norm(&product), 0.0);
        assert_ne!(
            squared_norm(&product),
            squared_norm(&left) * squared_norm(&right)
        );
    }
}
