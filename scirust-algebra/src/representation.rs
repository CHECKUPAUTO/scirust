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
    }

    #[test]
    fn matrix_identity_multiplies() {
        let a = SquareMatrix::from_rows([[2.0, 3.0], [5.0, 7.0]]);
        assert_eq!(a.mul(&SquareMatrix::identity()), a);
    }
}
