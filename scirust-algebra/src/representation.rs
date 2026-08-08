//! Representation theory and harmonic-analysis primitives.

/// Dense square matrix with compile-time dimension and row-major storage.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct SquareMatrix<const N: usize> {
    data: [[f64; N]; N],
}

impl<const N: usize> SquareMatrix<N> {
    /// Construct from rows.
    pub const fn from_rows(data: [[f64; N]; N]) -> Self { Self { data } }
    /// Zero matrix.
    pub const fn zero() -> Self { Self { data: [[0.0; N]; N] } }
    /// Identity matrix.
    pub fn identity() -> Self {
        let mut out = Self::zero();
        let mut i = 0;
        while i < N { out.data[i][i] = 1.0; i += 1; }
        out
    }
    /// Matrix entry.
    #[inline]
    pub const fn get(&self, row: usize, col: usize) -> f64 { self.data[row][col] }
    /// Trace / character value for this representation matrix.
    pub fn trace(&self) -> f64 {
        let mut s = 0.0;
        let mut i = 0;
        while i < N { s += self.data[i][i]; i += 1; }
        s
    }
    /// Matrix product.
    pub fn mul(&self, rhs: &Self) -> Self {
        let mut out = Self::zero();
        let mut i = 0;
        while i < N {
            let mut k = 0;
            while k < N {
                let a = self.data[i][k];
                let mut j = 0;
                while j < N { out.data[i][j] += a * rhs.data[k][j]; j += 1; }
                k += 1;
            }
            i += 1;
        }
        out
    }
    /// Scalar multiplication.
    pub fn scale(&self, alpha: f64) -> Self {
        let mut out = *self;
        let mut i = 0;
        while i < N {
            let mut j = 0;
            while j < N { out.data[i][j] *= alpha; j += 1; }
            i += 1;
        }
        out
    }
    /// Add another matrix in place.
    pub fn add_assign(&mut self, rhs: &Self) {
        let mut i = 0;
        while i < N {
            let mut j = 0;
            while j < N { self.data[i][j] += rhs.data[i][j]; j += 1; }
            i += 1;
        }
    }
}

/// Error returned while reducing element-wise representation traces to classes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CharacterClassError {
    /// The representation and class-label slices describe different element counts.
    LengthMismatch,
    /// A class label does not fit in the caller-provided output storage.
    ClassOutOfRange { class: usize },
    /// Two elements assigned to one conjugacy class have different traces.
    NonConstantTrace {
        class: usize,
        expected: f64,
        actual: f64,
    },
}

/// Compute one real character value per conjugacy class from representation matrices.
///
/// `representation[i]` and `class_of[i]` must refer to the same group element. The
/// output is caller-owned and no heap allocation is performed. A representation
/// character must be constant on conjugacy classes; this routine checks that invariant
/// to `tolerance` rather than silently accepting an inconsistent enumeration.
pub fn character_on_classes_into<const N: usize>(
    representation: &[SquareMatrix<N>],
    class_of: &[usize],
    values: &mut [f64],
    tolerance: f64,
) -> Result<usize, CharacterClassError> {
    if representation.len() != class_of.len() {
        return Err(CharacterClassError::LengthMismatch);
    }

    let mut seen = 0usize;
    let mut class_count = 0usize;
    while seen < values.len() {
        values[seen] = f64::NAN;
        seen += 1;
    }

    let mut i = 0usize;
    while i < representation.len() {
        let class = class_of[i];
        if class >= values.len() {
            return Err(CharacterClassError::ClassOutOfRange { class });
        }
        let trace = representation[i].trace();
        if values[class].is_nan() {
            values[class] = trace;
        } else if (values[class] - trace).abs() > tolerance {
            return Err(CharacterClassError::NonConstantTrace {
                class,
                expected: values[class],
                actual: trace,
            });
        }
        class_count = class_count.max(class + 1);
        i += 1;
    }

    Ok(class_count)
}

/// Character table indexed by irreducible representation and conjugacy class.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharacterTable<const IRREPS: usize, const CLASSES: usize> {
    values: [[f64; CLASSES]; IRREPS],
    class_sizes: [usize; CLASSES],
    group_order: usize,
}

impl<const I: usize, const C: usize> CharacterTable<I, C> {
    /// Construct a character table with explicit class sizes and group order.
    pub const fn new(values: [[f64; C]; I], class_sizes: [usize; C], group_order: usize) -> Self {
        Self { values, class_sizes, group_order }
    }
    /// Character value.
    pub const fn get(&self, irrep: usize, class: usize) -> f64 { self.values[irrep][class] }
    /// Size of one conjugacy class.
    pub const fn class_size(&self, class: usize) -> usize { self.class_sizes[class] }
    /// Group order carried by this table.
    pub const fn group_order(&self) -> usize { self.group_order }
    /// Dimension of an irrep, read from the identity conjugacy class.
    pub fn irrep_dimension(&self, irrep: usize, identity_class: usize) -> Option<usize> {
        if irrep >= I || identity_class >= C {
            return None;
        }
        let value = self.values[irrep][identity_class];
        if !value.is_finite() || value < 0.0 {
            return None;
        }
        let rounded = value.round();
        if (value - rounded).abs() > 1e-12 || rounded > usize::MAX as f64 {
            return None;
        }
        Some(rounded as usize)
    }
    /// Inner product of two real-valued characters.
    pub fn inner_product(&self, lhs: usize, rhs: usize) -> f64 {
        let mut sum = 0.0;
        let mut c = 0;
        while c < C {
            sum += self.class_sizes[c] as f64 * self.values[lhs][c] * self.values[rhs][c];
            c += 1;
        }
        sum / self.group_order as f64
    }
    /// Verify row orthonormality to an absolute tolerance.
    pub fn rows_orthonormal(&self, tolerance: f64) -> bool {
        if self.group_order == 0 {
            return false;
        }
        let mut i = 0;
        while i < I {
            let mut j = 0;
            while j < I {
                let target = if i == j { 1.0 } else { 0.0 };
                if (self.inner_product(i, j) - target).abs() > tolerance { return false; }
                j += 1;
            }
            i += 1;
        }
        true
    }
    /// Verify the real-character column orthogonality relations.
    ///
    /// For a complete irreducible table,
    /// `sum_rho chi_rho(C_a) chi_rho(C_b) = |G| / |C_a|` when `a == b`
    /// and is zero otherwise.
    pub fn columns_orthogonal(&self, tolerance: f64) -> bool {
        if self.group_order == 0 || I != C {
            return false;
        }
        let mut a = 0usize;
        while a < C {
            if self.class_sizes[a] == 0 || self.group_order % self.class_sizes[a] != 0 {
                return false;
            }
            let mut b = 0usize;
            while b < C {
                let mut sum = 0.0;
                let mut irrep = 0usize;
                while irrep < I {
                    sum += self.values[irrep][a] * self.values[irrep][b];
                    irrep += 1;
                }
                let target = if a == b {
                    (self.group_order / self.class_sizes[a]) as f64
                } else {
                    0.0
                };
                if (sum - target).abs() > tolerance {
                    return false;
                }
                b += 1;
            }
            a += 1;
        }
        true
    }
    /// Verify the degree sum rule `sum_rho dim(rho)^2 = |G|`.
    pub fn dimensions_complete(&self, identity_class: usize) -> bool {
        if identity_class >= C || self.group_order == 0 {
            return false;
        }
        let mut sum = 0usize;
        let mut irrep = 0usize;
        while irrep < I {
            let Some(dimension) = self.irrep_dimension(irrep, identity_class) else {
                return false;
            };
            let Some(square) = dimension.checked_mul(dimension) else {
                return false;
            };
            let Some(next) = sum.checked_add(square) else {
                return false;
            };
            sum = next;
            irrep += 1;
        }
        sum == self.group_order
    }
}

/// Compute the finite-group irrep projector
/// `P = d/|G| Σ_g χ(g^{-1}) ρ(g)` for real characters.
///
/// Matrices and character values must be aligned in the same group-element order.
pub fn irrep_projector<const N: usize>(
    irrep_dimension: usize,
    group_order: usize,
    characters_on_inverse: &[f64],
    representation: &[SquareMatrix<N>],
) -> Option<SquareMatrix<N>> {
    if group_order == 0 || characters_on_inverse.len() != representation.len() || representation.len() != group_order {
        return None;
    }
    let mut out = SquareMatrix::zero();
    let mut i = 0;
    while i < group_order {
        out.add_assign(&representation[i].scale(characters_on_inverse[i]));
        i += 1;
    }
    Some(out.scale(irrep_dimension as f64 / group_order as f64))
}

/// Naive finite-group Fourier transform for a matrix-valued representation.
///
/// Computes `Σ_g signal[g] ρ(g)` without allocation. This is the deterministic
/// reference kernel against which specialised non-abelian FFT factorizations can
/// be validated.
pub fn group_fourier<const N: usize>(signal: &[f64], representation: &[SquareMatrix<N>]) -> Option<SquareMatrix<N>> {
    if signal.len() != representation.len() { return None; }
    let mut out = SquareMatrix::zero();
    let mut i = 0;
    while i < signal.len() {
        out.add_assign(&representation[i].scale(signal[i]));
        i += 1;
    }
    Some(out)
}

/// Associated Legendre polynomial `P_l^m(x)` for `0 <= m <= l`.
pub fn associated_legendre(l: usize, m: usize, x: f64) -> Option<f64> {
    if m > l || x.abs() > 1.0 { return None; }
    let mut pmm = 1.0;
    if m > 0 {
        let somx2 = (1.0 - x * x).sqrt();
        let mut fact = 1.0;
        let mut i = 1;
        while i <= m { pmm *= -fact * somx2; fact += 2.0; i += 1; }
    }
    if l == m { return Some(pmm); }
    let pmmp1 = x * (2 * m + 1) as f64 * pmm;
    if l == m + 1 { return Some(pmmp1); }
    let mut pprev = pmm;
    let mut pcur = pmmp1;
    let mut ll = m + 2;
    while ll <= l {
        let next = (((2 * ll - 1) as f64 * x * pcur) - ((ll + m - 1) as f64 * pprev)) / (ll - m) as f64;
        pprev = pcur;
        pcur = next;
        ll += 1;
    }
    Some(pcur)
}

/// Real spherical harmonic cosine branch `Y_l^m(theta, phi)` without the
/// normalisation constant. Useful as a low-level SO(3) basis primitive.
pub fn spherical_harmonic_cosine(l: usize, m: usize, theta: f64, phi: f64) -> Option<f64> {
    associated_legendre(l, m, theta.cos()).map(|p| p * (m as f64 * phi).cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c2_character_table_is_orthonormal() {
        let table = CharacterTable::new([[1.0, 1.0], [1.0, -1.0]], [1, 1], 2);
        assert!(table.rows_orthonormal(1e-12));
        assert!(table.columns_orthogonal(1e-12));
        assert!(table.dimensions_complete(0));
    }

    #[test]
    fn s3_character_table_satisfies_both_orthogonality_relations() {
        let table = CharacterTable::new(
            [[1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [2.0, 0.0, -1.0]],
            [1, 3, 2],
            6,
        );
        assert!(table.rows_orthonormal(1e-12));
        assert!(table.columns_orthogonal(1e-12));
        assert!(table.dimensions_complete(0));
        assert_eq!(table.irrep_dimension(2, 0), Some(2));
    }

    #[test]
    fn character_reduction_requires_class_constant_trace() {
        let representation = [
            SquareMatrix::from_rows([[1.0]]),
            SquareMatrix::from_rows([[-1.0]]),
            SquareMatrix::from_rows([[-1.0]]),
        ];
        let classes = [0usize, 1, 1];
        let mut values = [0.0; 2];
        assert_eq!(
            character_on_classes_into(&representation, &classes, &mut values, 1e-12),
            Ok(2)
        );
        assert_eq!(values, [1.0, -1.0]);

        let inconsistent = [
            SquareMatrix::from_rows([[1.0]]),
            SquareMatrix::from_rows([[-1.0]]),
            SquareMatrix::from_rows([[1.0]]),
        ];
        assert!(matches!(
            character_on_classes_into(&inconsistent, &classes, &mut values, 1e-12),
            Err(CharacterClassError::NonConstantTrace { class: 1, .. })
        ));
    }

    #[test]
    fn matrix_identity_multiplies() {
        let a = SquareMatrix::from_rows([[2.0, 3.0], [5.0, 7.0]]);
        assert_eq!(a.mul(&SquareMatrix::identity()), a);
    }
}
