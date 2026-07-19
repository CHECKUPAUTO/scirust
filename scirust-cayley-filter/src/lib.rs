//! Experimental Cayley–Dickson filtering for SciRust.
//!
//! This crate studies whether real-linear multiplication operators derived
//! from Cayley–Dickson algebras can reject identified noise subspaces while
//! preserving useful signal components.
//!
//! No superiority over established denoising methods is assumed. Every
//! filtering claim must be established through reproducible measurements
//! against explicit baselines.

#![forbid(unsafe_code)]

/// Number of real components in a sedenion.
pub const SEDENION_DIMENSION: usize = 16;

/// Scalar reference representation used during mathematical validation.
///
/// The first implementation intentionally uses `f64` and a fixed-size array.
/// Architecture-specific `f32` SIMD kernels will only be introduced after
/// their numerical parity with the scalar reference has been demonstrated.
pub type SedenionVector = [f64; SEDENION_DIMENSION];

/// Computes the squared Euclidean norm in a fixed accumulation order.
///
/// The fixed sequential order makes the scalar reference reproducible on a
/// given IEEE-754 target and suitable as an oracle for later SIMD kernels.
#[must_use]
pub fn squared_norm(vector: &SedenionVector) -> f64 {
    let mut sum = 0.0;

    for component in vector
    {
        sum += component * component;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::{SEDENION_DIMENSION, SedenionVector, squared_norm};

    #[test]
    fn dimension_is_sixteen() {
        assert_eq!(SEDENION_DIMENSION, 16);
    }

    #[test]
    fn squared_norm_uses_all_components() {
        let mut vector: SedenionVector = [0.0; SEDENION_DIMENSION];
        vector[0] = 3.0;
        vector[15] = 4.0;

        assert_eq!(squared_norm(&vector), 25.0);
    }
}
