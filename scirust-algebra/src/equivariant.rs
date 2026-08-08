//! Equivariant operators and symmetry-aware tensor coupling.

use core::marker::PhantomData;

use crate::angular_momentum::clebsch_gordan;

/// Static representation action used to express equivariance without dynamic dispatch.
pub trait Representation<G, X> {
    /// Apply the representation of `g` to `x`.
    fn act(g: &G, x: &X) -> X;
}

/// Marker for an operator that is intended to be equivariant from `InRep` to `OutRep`.
#[derive(Clone, Copy, Debug, Default)]
pub struct EquivariantMap<G, X, Y, InRep, OutRep, F> {
    map: F,
    marker: PhantomData<(G, X, Y, InRep, OutRep)>,
}

impl<G, X, Y, InRep, OutRep, F> EquivariantMap<G, X, Y, InRep, OutRep, F>
where
    F: Fn(&X) -> Y,
{
    /// Construct a typed equivariant-map wrapper.
    pub const fn new(map: F) -> Self { Self { map, marker: PhantomData } }
    /// Evaluate the wrapped map.
    #[inline]
    pub fn apply(&self, x: &X) -> Y { (self.map)(x) }
}

impl<G, X, Y, InRep, OutRep, F> EquivariantMap<G, X, Y, InRep, OutRep, F>
where
    X: Clone,
    Y: PartialEq,
    InRep: Representation<G, X>,
    OutRep: Representation<G, Y>,
    F: Fn(&X) -> Y,
{
    /// Check `f(rho_in(g)x) == rho_out(g)f(x)` for one test point.
    pub fn verify_exact(&self, g: &G, x: &X) -> bool {
        let lhs = (self.map)(&InRep::act(g, x));
        let rhs = OutRep::act(g, &(self.map)(x));
        lhs == rhs
    }
}

/// Compile-time irrep label. `L` can encode angular momentum/order and `P`
/// parity or another problem-specific discrete label.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Irrep<const L: usize, const P: i8>;

/// Tensor value carrying a representation label at the type level.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct TypedTensor<T, Rep> {
    value: T,
    marker: PhantomData<Rep>,
}

impl<T, Rep> TypedTensor<T, Rep> {
    /// Wrap a tensor/value in a representation marker.
    pub const fn new(value: T) -> Self { Self { value, marker: PhantomData } }
    /// Borrow the payload.
    pub const fn value(&self) -> &T { &self.value }
    /// Consume the wrapper.
    pub fn into_inner(self) -> T { self.value }
}

/// Clebsch-Gordan coefficient table for fixed input/output dimensions.
///
/// The coefficient tensor is stored as `[OUT][LEFT][RIGHT]`, so coupling is a
/// deterministic allocation-free triple contraction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClebschGordan<const LEFT: usize, const RIGHT: usize, const OUT: usize> {
    coeff: [[[f64; RIGHT]; LEFT]; OUT],
}

impl<const L: usize, const R: usize, const O: usize> ClebschGordan<L, R, O> {
    /// Construct from a precomputed coefficient tensor.
    pub const fn new(coeff: [[[f64; R]; L]; O]) -> Self { Self { coeff } }

    /// Generate the standard integer-spin SO(3) coupling table.
    ///
    /// Dimensions must match `L = 2*j_left + 1`, `R = 2*j_right + 1` and
    /// `O = 2*j_out + 1`; incompatible angular momenta or dimensions return `None`.
    /// Basis indices are ordered by magnetic quantum number from `-j` through `+j`.
    pub fn from_so3(j_left: usize, j_right: usize, j_out: usize) -> Option<Self> {
        let expected_left = j_left.checked_mul(2)?.checked_add(1)?;
        let expected_right = j_right.checked_mul(2)?.checked_add(1)?;
        let expected_out = j_out.checked_mul(2)?.checked_add(1)?;
        if L != expected_left || R != expected_right || O != expected_out {
            return None;
        }
        if j_left.abs_diff(j_right) > j_out || j_out > j_left.checked_add(j_right)? {
            return None;
        }

        let jl = isize::try_from(j_left).ok()?;
        let jr = isize::try_from(j_right).ok()?;
        let jo = isize::try_from(j_out).ok()?;
        let mut coeff = [[[0.0; R]; L]; O];
        let mut out = 0usize;
        while out < O {
            let m_out = isize::try_from(out).ok()? - jo;
            let mut left = 0usize;
            while left < L {
                let m_left = isize::try_from(left).ok()? - jl;
                let mut right = 0usize;
                while right < R {
                    let m_right = isize::try_from(right).ok()? - jr;
                    coeff[out][left][right] =
                        clebsch_gordan(j_left, m_left, j_right, m_right, j_out, m_out)?;
                    right += 1;
                }
                left += 1;
            }
            out += 1;
        }
        Some(Self { coeff })
    }

    /// Return one coefficient by output, left and right basis indices.
    #[inline]
    pub const fn coefficient(&self, out: usize, left: usize, right: usize) -> f64 {
        self.coeff[out][left][right]
    }

    /// Couple two vectors into the output irrep.
    pub fn couple(&self, left: &[f64; L], right: &[f64; R]) -> [f64; O] {
        let mut out = [0.0; O];
        let mut o = 0;
        while o < O {
            let mut i = 0;
            while i < L {
                let mut j = 0;
                while j < R {
                    out[o] += self.coeff[o][i][j] * left[i] * right[j];
                    j += 1;
                }
                i += 1;
            }
            o += 1;
        }
        out
    }
}

/// Contract two fixed vectors with an invariant bilinear form.
pub fn invariant_bilinear<const N: usize>(metric: &[[f64; N]; N], lhs: &[f64; N], rhs: &[f64; N]) -> f64 {
    let mut sum = 0.0;
    let mut i = 0;
    while i < N {
        let mut j = 0;
        while j < N {
            sum += lhs[i] * metric[i][j] * rhs[j];
            j += 1;
        }
        i += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clebsch_gordan_scalar_dot_product() {
        let cg = ClebschGordan::<2, 2, 1>::new([[[1.0, 0.0], [0.0, 1.0]]]);
        assert_eq!(cg.couple(&[2.0, 3.0], &[5.0, 7.0]), [31.0]);
    }

    #[test]
    fn generated_vector_to_scalar_table_has_standard_coefficients() {
        let cg = ClebschGordan::<3, 3, 1>::from_so3(1, 1, 0).unwrap();
        let normalization = 3.0_f64.sqrt();
        assert!((cg.coefficient(0, 2, 0) - 1.0 / normalization).abs() < 1e-14);
        assert!((cg.coefficient(0, 1, 1) + 1.0 / normalization).abs() < 1e-14);
        assert!((cg.coefficient(0, 0, 2) - 1.0 / normalization).abs() < 1e-14);
    }

    #[test]
    fn generated_table_rejects_wrong_dimensions() {
        assert!(ClebschGordan::<2, 3, 1>::from_so3(1, 1, 0).is_none());
    }

    #[test]
    fn generated_highest_weight_coupling_is_one() {
        let cg = ClebschGordan::<3, 3, 5>::from_so3(1, 1, 2).unwrap();
        assert!((cg.coefficient(4, 2, 2) - 1.0).abs() < 1e-14);
    }
}
