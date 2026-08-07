//! Equivariant operators and symmetry-aware tensor coupling.

use core::marker::PhantomData;

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
}
