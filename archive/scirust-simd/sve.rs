//! # ARM SVE Intrinsics — Pilier 4
//!
//! Scalable Vector Extension — SIMD à longueur scalable sur ARMv8.2+.
//! Contrairement à NEON (4-lanes fixes), SVE peut contenir N×64 bits
//! où N dépend du matériel (256-bit sur Jetson Thor, 512-bit sur Ampere Altra).
//!
//! ## Avantages
//!
//! - Adaptatif: le même code fonctionne sur tous les processeurs ARM SVE
//! - Plus d'overhead de tail handling (la longueur est connue à l'exécution)
//! - PREDICAT: opérations conditionnelles lane-par-lane
//!
//! ## Limitations
//!
//! - Nécessite `feature = "arm_sve"` au runtime
//! - Support matériel rare: Ampere Altra, AWS Graviton 3, Fujitsu A64FX
//! - Compiler: nightly Rust avec `-C target-feature=+sve`

use std::arch::aarch64::*;

/// Longueur vectorielle SVE en bits (déterminée au runtime).
#[inline]
fn sve_vl_bits() -> usize {
    // SVE VL (Vector Length) en bits
    unsafe { sve_vl() * 8 }
}

/// SVE vector length en éléments de type T.
#[inline]
fn sve_vl_elements<T>() -> usize {
    sve_vl_bits() / (std::mem::size_of::<T>() * 8)
}

/// AXPY: y = alpha * x + y (SVE, scalable)
#[inline]
#[cfg(target_arch = "aarch64")]
pub fn saxpy_f32_sve(alpha: f32, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    let n = x.len();

    // Utiliser svwhilelt pour le predicate
    let idx = svwhilelt_b32(0u32, n as u32);
    let alpha_vec = svdup_f32(alpha);

    unsafe {
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_mut_ptr();

        // Premier bloc avec predicate SVE
        if sve_vl_bits() >= 256 {
            let vl = sve_vl_elements::<f32>();
            let mut i = 0;

            while i + vl <= n {
                let mask = svwhilelt_b32(i as u32, n as u32);
                let vx = svld1_f32(mask, x_ptr.add(i));
                let vy = svld1_f32(mask, y_ptr.add(i));
                let result = svmla_f32(mask, vy, vx, alpha_vec);
                svst1_f32(mask, y_ptr.add(i), result);
                i += vl;
            }
        }
    }

    // Fallback scalar
    while n > 0 {
        y[n - 1] = alpha * x[n - 1] + y[n - 1];
        n -= 1;
    }
}

/// Addition SVE scalable
#[inline]
pub fn add_f32_sve(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    let n = a.len();

    unsafe {
        let vl = sve_vl_elements::<f32>();
        let mut i = 0;

        while i + vl <= n {
            let mask = svwhilelt_b32(i as u32, n as u32);
            let va = svld1_f32(mask, a.as_ptr().add(i));
            let vb = svld1_f32(mask, b.as_ptr().add(i));
            let vr = svadd_f32(mask, va, vb);
            svst1_f32(mask, out.as_mut_ptr().add(i), vr);
            i += vl;
        }
    }

    while i < n {
        out[i] = a[i] + b[i];
        i += 1;
    }
}

/// Multiplication SVE scalable
#[inline]
pub fn mul_f32_sve(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    let n = a.len();

    unsafe {
        let vl = sve_vl_elements::<f32>();
        let mut i = 0;

        while i + vl <= n {
            let mask = svwhilelt_b32(i as u32, n as u32);
            let va = svld1_f32(mask, a.as_ptr().add(i));
            let vb = svld1_f32(mask, b.as_ptr().add(i));
            let vr = svmul_f32(mask, va, vb);
            svst1_f32(mask, out.as_mut_ptr().add(i), vr);
            i += vl;
        }
    }

    while i < n {
        out[i] = a[i] * b[i];
        i += 1;
    }
}

/// SiLU SVE scalable
#[inline]
pub fn silu_f32_sve(input: &[f32], output: &mut [f32]) {
    let n = input.len().min(output.len());
    let mut i = 0;

    unsafe {
        let vl = sve_vl_elements::<f32>();

        while i + vl <= n {
            let mask = svwhilelt_b32(i as u32, n as u32);
            let vx = svld1_f32(mask, input.as_ptr().add(i));

            // sigmoid: 1 / (1 + exp(-x))
            let neg_x = svneg_f32(mask, vx);
            let exp_neg = svexp_f32(mask, neg_x);
            let one = svdup_f32(1.0);
            let sigmoid = svsdiv_f32(mask, one, svadd_f32(mask, one, exp_neg));

            let result = svmul_f32(mask, vx, sigmoid);
            svst1_f32(mask, output.as_mut_ptr().add(i), result);

            i += vl;
        }
    }

    while i < n {
        let s = 1.0 / (1.0 + (-input[i]).exp());
        output[i] = input[i] * s;
        i += 1;
    }
}

/// ReLU SVE
#[inline]
pub fn relu_f32_sve(input: &[f32], output: &mut [f32]) {
    let n = input.len().min(output.len());
    let mut i = 0;

    unsafe {
        let vl = sve_vl_elements::<f32>();

        while i + vl <= n {
            let mask = svwhilelt_b32(i as u32, n as u32);
            let vx = svld1_f32(mask, input.as_ptr().add(i));
            let zero = svdup_f32(0.0);
            let result = svmax_f32(mask, vx, zero);
            svst1_f32(mask, output.as_mut_ptr().add(i), result);

            i += vl;
        }
    }

    while i < n {
        output[i] = input[i].max(0.0);
        i += 1;
    }
}

/// Vérifie si le processeur supporte SVE.
pub fn has_sve() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let hwcap = libc::getauxval(33); // AT_HWCAP
            (hwcap & (1 << 31)) != 0
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    false
}

/// Retourne la longueur vectorielle SVE en éléments.
pub fn sve_vector_length_elements<T>() -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        sve_vl_elements::<T>()
    }

    #[cfg(not(target_arch = "aarch64"))]
    0
}
