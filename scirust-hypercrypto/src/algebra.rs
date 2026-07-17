//! Algebra facade for the harness: re-exports the shared, general-purpose
//! [`scirust_modalg`] algebra (the `Z/2^k` rings, exact octonions/quaternions,
//! and modular linear algebra), and adds the **v0.1-specific** diffusion
//! constants `λ`, `π` and the `ROT_λ` / `PERM_π` convenience layers (spec §12.1)
//! that are meaningful only to this construction, not to general hypercomplex
//! algebra.

pub use scirust_modalg::hypercomplex::{Oct, Quat};
pub use scirust_modalg::linalg::ModMatrix;
pub use scirust_modalg::ring::{W2, W4, W8, W16, W64, WidthTag, Word};

/// Back-compatible path `crate::algebra::word::*` for the ring layer.
pub mod word {
    pub use scirust_modalg::ring::{W2, W4, W8, W16, W64, WidthTag, Word};
}

/// Per-lane left-rotation amounts `λ` (spec §12.1, `lid = 0`). For a reduced
/// width `k`, lane `j` is rotated by `λ[j] mod k` (spec §16.1).
pub const LAMBDA: [u32; 8] = [7, 19, 31, 47, 11, 23, 37, 53];

/// Coefficient-slot permutation `π` (spec §12.1, `pid = 0`), a derangement:
/// `(PERM_π(x))_i = x_{π[i]}`.
pub const PI: [usize; 8] = [3, 6, 1, 4, 7, 2, 5, 0];

/// The v0.1 diffusion layers `ROT_λ` and `PERM_π` as convenience methods on the
/// shared octonion type. These bind the specific published constants; the
/// general per-lane rotate / slot-permute live on [`Oct`] itself.
pub trait OctLayers {
    /// `ROT_λ`: rotate lane `j` left by `λ[j] mod BITS`.
    fn rot_lambda(self) -> Self;
    /// `PERM_π`: `(PERM_π(x))_i = x_{π[i]}`.
    fn perm_pi(self) -> Self;
}

impl<W: Word> OctLayers for Oct<W> {
    fn rot_lambda(self) -> Self {
        self.rotate_lanes(&LAMBDA)
    }
    fn perm_pi(self) -> Self {
        self.permute_slots(&PI)
    }
}
