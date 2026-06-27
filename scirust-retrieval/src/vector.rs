//! Deterministic dense-vector primitives.
//!
//! All reductions accumulate left-to-right in a single `f32` so a run is
//! bit-reproducible — the determinism the rest of SciRust guarantees. (A SIMD
//! dot product lives in `scirust-simd`; it is bit-identical per IEEE-754 but we
//! keep the scalar fixed-order version here so the index is trivially auditable.)

/// Dot product `Σ aᵢ·bᵢ`, summed in index order. Panics in debug if the lengths
/// differ; in release the shorter length wins.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot: length mismatch");
    let mut acc = 0.0f32;
    for (x, y) in a.iter().zip(b)
    {
        acc += x * y;
    }
    acc
}

/// Euclidean (L2) norm `√(Σ aᵢ²)`.
pub fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// Cosine similarity in `[-1, 1]`. Returns `0.0` when either operand is the zero
/// vector (rather than `NaN`).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = norm(a);
    let nb = norm(b);
    // norm is non-negative, so `<= 0.0` means exactly zero (no float-eq lint).
    if na <= 0.0 || nb <= 0.0
    {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

/// Return an L2-normalised copy. The zero vector maps to itself.
pub fn normalized(a: &[f32]) -> Vec<f32> {
    let n = norm(a);
    if n <= 0.0
    {
        return a.to_vec();
    }
    let inv = 1.0 / n;
    a.iter().map(|&x| x * inv).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_and_norm_match_hand_values() {
        // a·b = 1·4 + 2·5 + 3·6 = 32; |a| = √14.
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert!((dot(&a, &b) - 32.0).abs() < 1e-6, "dot {}", dot(&a, &b));
        assert!(
            (norm(&a) - 14.0_f32.sqrt()).abs() < 1e-6,
            "norm {}",
            norm(&a)
        );
    }

    #[test]
    fn cosine_of_known_geometry() {
        // identical -> 1, orthogonal -> 0, opposite -> -1.
        assert!((cosine(&[1.0, 0.0], &[3.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 5.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[-2.0, 0.0]) + 1.0).abs() < 1e-6);
        // 45° between [1,0] and [1,1] -> cos = 1/√2.
        let c = cosine(&[1.0, 0.0], &[1.0, 1.0]);
        assert!(
            (c - core::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6,
            "cos {c}"
        );
    }

    #[test]
    fn the_zero_vector_never_produces_nan() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert_eq!(normalized(&[0.0, 0.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn normalized_has_unit_norm_and_preserves_direction() {
        let v = normalized(&[3.0, 4.0]); // |[3,4]| = 5 -> [0.6, 0.8]
        assert!(
            (v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6,
            "{v:?}"
        );
        assert!((norm(&v) - 1.0).abs() < 1e-6);
    }
}
