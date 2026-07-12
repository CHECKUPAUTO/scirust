//! # Extensions x86_64 avancées
//!
//! Kernels x86-spécifiques qui vont au-delà du SIMD flottant « classique »
//! (AVX2/AVX-512F) et exploitent des fonctionnalités plus pointues de la
//! micro-architecture :
//!
//! * **Masques `k` AVX-512** — mise à jour *conditionnelle* par voie
//!   ([`axpy_masked_f32`]), sans branche ni écriture des voies inactives.
//! * **Prefetch logiciel + stores non-temporels** — pour les flux qui
//!   dépassent le cache ([`scale_stream_f32`], [`axpy_prefetch_f32`]).
//!
//! La **quantification** (int8 VNNI/dotprod, bf16), qui est cross-arch, vit
//! désormais dans [`crate::quant`].
//!
//! Chaque entrée publique fait de la détection *runtime* et retombe sur une
//! implémentation scalaire portable, donc l'appel est sûr sur n'importe quel
//! CPU (le binaire reste unique).
//!
//! ## Safety
//!
//! Les fonctions `#[target_feature(enable = ...)]` ne sont appelées qu'après
//! `is_x86_feature_detected!` du même jeu d'instructions. Les intrinsèques de
//! chargement/écriture utilisés sont *unaligned* (`loadu`/`storeu`) ou masqués,
//! donc aucune contrainte d'alignement au-delà de 1 octet ; l'arithmétique de
//! pointeurs reste bornée par les conditions de boucle et les longueurs de
//! slices. Les stores non-temporels sont suivis d'un `sfence` avant tout retour.

#![allow(clippy::missing_safety_doc)]

// ===================================================================== //
//  Masque k AVX-512 — axpy conditionnel par voie                         //
// ===================================================================== //

/// AXPY **conditionnel** : `y[i] += alpha * x[i]` uniquement là où `keep[i]`
/// est vrai ; les autres `y[i]` sont laissés intacts. Démontre les masques `k`
/// AVX-512 (mise à jour partielle par voie, sans branche). Repli scalaire sinon.
pub fn axpy_masked_f32(alpha: f32, x: &[f32], keep: &[bool], y: &mut [f32]) {
    assert_eq!(x.len(), y.len(), "axpy_masked_f32: x/y length mismatch");
    assert_eq!(x.len(), keep.len(), "axpy_masked_f32: keep length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
        {
            // SAFETY: gated by the runtime detection just above.
            unsafe { axpy_masked_f32_avx512(alpha, x, keep, y) };
            return;
        }
    }
    for i in 0..x.len()
    {
        if keep[i]
        {
            y[i] += alpha * x[i];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn axpy_masked_f32_avx512(alpha: f32, x: &[f32], keep: &[bool], y: &mut [f32]) {
    use core::arch::x86_64::*;
    let a16 = _mm512_set1_ps(alpha);
    let n = x.len();
    let mut i = 0;
    while i + 16 <= n
    {
        // Construit le masque k depuis les 16 booléens.
        let mut m: u16 = 0;
        for (j, k) in keep[i..i + 16].iter().enumerate()
        {
            m |= (*k as u16) << j;
        }
        let xv = _mm512_loadu_ps(x.as_ptr().add(i));
        let yv = _mm512_loadu_ps(y.as_ptr().add(i));
        // mask3-fmadd : voies actives = a16*xv + yv (alpha*x + y), voies
        // inactives conservent yv (le 3ᵉ opérande sert de source de merge).
        let res = _mm512_mask3_fmadd_ps(a16, xv, yv, m);
        _mm512_storeu_ps(y.as_mut_ptr().add(i), res);
        i += 16;
    }
    for j in i..n
    {
        if keep[j]
        {
            y[j] += alpha * x[j];
        }
    }
}

// ===================================================================== //
//  Prefetch logiciel + stores non-temporels                             //
// ===================================================================== //

/// Met à l'échelle `data[i] *= s` avec des **stores non-temporels**
/// (`_mm256_stream_ps`, MOVNTPS) : les écritures contournent le cache, idéal
/// pour un gros tableau qu'on ne relira pas de suite (évite la pollution du
/// cache et le trafic RFO). Un `sfence` garantit la visibilité avant retour.
pub fn scale_stream_f32(s: f32, data: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx")
        {
            // SAFETY: gated by the runtime detection just above.
            unsafe { scale_stream_f32_avx(s, data) };
            return;
        }
    }
    for x in data.iter_mut()
    {
        *x *= s;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn scale_stream_f32_avx(s: f32, data: &mut [f32]) {
    use core::arch::x86_64::*;
    let sv = _mm256_set1_ps(s);
    let n = data.len();
    let ptr = data.as_mut_ptr();
    // MOVNTPS exige un alignement 32 octets ; on traite le préambule non aligné
    // en scalaire jusqu'à tomber sur une frontière de 32 octets.
    let mut i = 0;
    while i < n && !(ptr.add(i) as usize).is_multiple_of(32)
    {
        *ptr.add(i) *= s;
        i += 1;
    }
    while i + 8 <= n
    {
        let v = _mm256_load_ps(ptr.add(i)); // aligné ici par construction
        _mm256_stream_ps(ptr.add(i), _mm256_mul_ps(v, sv));
        i += 8;
    }
    // Rends visibles les stores non-temporels avant toute lecture ultérieure.
    _mm_sfence();
    for j in i..n
    {
        *ptr.add(j) *= s;
    }
}

/// `y[i] += alpha * x[i]` (AVX2/FMA) avec **prefetch logiciel** : chaque
/// itération précharge une ligne de cache en avance (`_mm_prefetch`, distance
/// réglée sur ~512 octets) pour masquer la latence mémoire sur de longs
/// vecteurs qui dépassent le L2. Repli scalaire hors AVX2/FMA.
pub fn axpy_prefetch_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len(), "axpy_prefetch_f32: length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
        {
            // SAFETY: gated by the runtime detection just above.
            unsafe { axpy_prefetch_f32_avx2(alpha, x, y) };
            return;
        }
    }
    for i in 0..x.len()
    {
        y[i] += alpha * x[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_prefetch_f32_avx2(alpha: f32, x: &[f32], y: &mut [f32]) {
    use core::arch::x86_64::*;
    const PF_AHEAD: usize = 128; // 128 f32 = 512 octets = 8 lignes de cache
    let a8 = _mm256_set1_ps(alpha);
    let n = x.len();
    let mut i = 0;
    while i + 8 <= n
    {
        if i + PF_AHEAD < n
        {
            _mm_prefetch(x.as_ptr().add(i + PF_AHEAD) as *const i8, _MM_HINT_T0);
            _mm_prefetch(y.as_ptr().add(i + PF_AHEAD) as *const i8, _MM_HINT_T0);
        }
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let yv = _mm256_loadu_ps(y.as_ptr().add(i));
        _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_fmadd_ps(a8, xv, yv));
        i += 8;
    }
    for j in i..n
    {
        y[j] += alpha * x[j];
    }
}

// ===================================================================== //
//  Tests                                                                 //
// ===================================================================== //
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axpy_masked_matches_scalar() {
        for len in 0..=70usize
        {
            let x: Vec<f32> = (0..len).map(|i| (i as f32) * 0.3 - 5.0).collect();
            let keep: Vec<bool> = (0..len).map(|i| i % 3 != 0).collect();
            let y0: Vec<f32> = (0..len).map(|i| (i as f32) * -0.2 + 1.0).collect();

            let mut got = y0.clone();
            axpy_masked_f32(2.5, &x, &keep, &mut got);

            let mut want = y0.clone();
            for i in 0..len
            {
                if keep[i]
                {
                    want[i] += 2.5 * x[i];
                }
            }
            for i in 0..len
            {
                assert!((got[i] - want[i]).abs() <= 1e-5, "len={len} i={i}");
            }
        }
    }

    #[test]
    fn scale_stream_matches_scalar() {
        for len in 0..=100usize
        {
            let base: Vec<f32> = (0..len).map(|i| (i as f32) * 0.7 - 3.0).collect();
            let mut got = base.clone();
            scale_stream_f32(1.5, &mut got);
            for i in 0..len
            {
                assert!((got[i] - base[i] * 1.5).abs() <= 1e-4, "len={len} i={i}");
            }
        }
    }

    #[test]
    fn axpy_prefetch_matches_scalar() {
        for len in [0usize, 1, 7, 8, 9, 130, 300]
        {
            let x: Vec<f32> = (0..len).map(|i| (i as f32).sin()).collect();
            let y0: Vec<f32> = (0..len).map(|i| (i as f32).cos()).collect();
            let mut got = y0.clone();
            axpy_prefetch_f32(-0.75, &x, &mut got);
            for i in 0..len
            {
                let want = y0[i] + -0.75 * x[i];
                assert!((got[i] - want).abs() <= 1e-5, "len={len} i={i}");
            }
        }
    }
}
