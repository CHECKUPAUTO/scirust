//! # Extensions x86_64 avancées
//!
//! Kernels qui vont au-delà du SIMD flottant « classique » (AVX2/AVX-512F) et
//! exploitent des fonctionnalités plus pointues de la micro-architecture :
//!
//! * **AVX-512 VNNI** (`_mm512_dpbusd_epi32`) — produit scalaire entier `u8·i8`
//!   accumulé en `i32`, socle de l'inférence **quantifiée** (int8).
//! * **BF16 mixed-precision** — stockage en `bfloat16` (moitié de la bande
//!   passante mémoire), calcul accumulé en `f32` (le repli matériel exact de
//!   `VDPBF16PS` quand l'ISA `avx512bf16` est absente).
//! * **Masques `k` AVX-512** — mise à jour *conditionnelle* par voie
//!   (`axpy_masked_f32`), sans branche ni écriture des voies inactives.
//! * **Prefetch logiciel + stores non-temporels** — pour les flux qui
//!   dépassent le cache (`scale_stream_f32`, `axpy_prefetch_f32`).
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
//  BF16 (bfloat16) — conversions round-to-nearest-even                   //
// ===================================================================== //

/// Convertit un `f32` en `bfloat16` (stocké dans les 16 bits de poids fort),
/// avec arrondi *round-to-nearest-even* — identique au comportement matériel.
/// Les `NaN` sont préservés (quiet).
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000
    {
        // NaN → NaN quiet, on garde le signe.
        return ((bits >> 16) as u16) | 0x0040;
    }
    // round-to-nearest-even : on ajoute le bit de garde + le biais pair.
    let rounding_bias = 0x0000_7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// Convertit un `bfloat16` (16 bits de poids fort) en `f32` (exact, sans perte).
#[inline]
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// Convertit une tranche `f32 → bf16`.
pub fn f32_to_bf16_slice(src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len(), "f32_to_bf16_slice: length mismatch");
    for (d, &s) in dst.iter_mut().zip(src)
    {
        *d = f32_to_bf16(s);
    }
}

/// Convertit une tranche `bf16 → f32`.
pub fn bf16_to_f32_slice(src: &[u16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len(), "bf16_to_f32_slice: length mismatch");
    for (d, &s) in dst.iter_mut().zip(src)
    {
        *d = bf16_to_f32(s);
    }
}

/// Produit scalaire **BF16 mixed-precision** : entrées stockées en `bf16`,
/// produits et accumulation en `f32` — exactement la sémantique de `VDPBF16PS`.
/// Divise par deux la bande passante mémoire par rapport au `f32` plein tout en
/// gardant une accumulation `f32` précise.
///
/// Chemin AVX-512 (élargissement `bf16 → f32` par `_mm512_cvtepu16_epi32` +
/// décalage de 16 bits, puis FMA `f32`) quand `avx512f` est présent — 16 voies
/// par pas ; repli scalaire sinon. Sur une puce dotée de l'ISA `avx512bf16`, la
/// boucle interne se remplacerait un-pour-un par `_mm512_dpbf16_ps` (un seul
/// produit-fusionné par paire de `bf16`) ; le présent chemin `f32` est le repli
/// exact et portable de cette instruction.
pub fn dot_bf16(a: &[u16], b: &[u16]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_bf16: length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
        {
            // SAFETY: gated by the runtime detection just above.
            return unsafe { dot_bf16_avx512(a, b) };
        }
    }
    dot_bf16_scalar(a, b)
}

/// Repli scalaire de [`dot_bf16`] (aussi la référence des tests).
pub fn dot_bf16_scalar(a: &[u16], b: &[u16]) -> f32 {
    let mut acc = 0.0f32;
    for (&x, &y) in a.iter().zip(b)
    {
        acc += bf16_to_f32(x) * bf16_to_f32(y);
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_bf16_avx512(a: &[u16], b: &[u16]) -> f32 {
    use core::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= n
    {
        // 16×bf16 (u16) → 16×f32 : zero-extend u16→u32 puis <<16 (place la
        // mantisse/exposant bf16 en tête d'un f32), reinterprété en f32.
        let ai = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let bi = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        let af = _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(ai)));
        let bf = _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(bi)));
        acc = _mm512_fmadd_ps(af, bf, acc);
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(acc);
    for j in i..n
    {
        sum += bf16_to_f32(a[j]) * bf16_to_f32(b[j]);
    }
    sum
}

// ===================================================================== //
//  VNNI — produit scalaire entier quantifié (u8 · i8 → i32)              //
// ===================================================================== //

/// Produit scalaire quantifié `sum(a[i] as i32 * b[i] as i32)` avec activations
/// `u8` (non signées) et poids `i8` (signés) — la convention usuelle de
/// l'inférence int8. Utilise **AVX-512 VNNI** (`VPDPBUSD`, 4 MAC entiers par
/// voie et par instruction) quand disponible, sinon repli scalaire.
///
/// L'accumulation `i32` est exacte (aucun débordement pour des longueurs
/// réalistes : borne `|somme| ≤ len · 255 · 128`).
pub fn dot_u8i8_i32(a: &[u8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len(), "dot_u8i8_i32: length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512vnni") && std::is_x86_feature_detected!("avx512bw")
        {
            // SAFETY: gated by the runtime detection just above.
            return unsafe { dot_u8i8_i32_vnni(a, b) };
        }
    }
    dot_u8i8_i32_scalar(a, b)
}

/// Repli scalaire portable de [`dot_u8i8_i32`] (aussi la référence des tests).
pub fn dot_u8i8_i32_scalar(a: &[u8], b: &[i8]) -> i32 {
    let mut acc: i32 = 0;
    for (&x, &y) in a.iter().zip(b)
    {
        acc += (x as i32) * (y as i32);
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512bw,avx512f")]
unsafe fn dot_u8i8_i32_vnni(a: &[u8], b: &[i8]) -> i32 {
    use core::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm512_setzero_si512();
    let mut i = 0;
    while i + 64 <= n
    {
        let va = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
        // dpbusd : a non signé (u8), b signé (i8), 4 produits/voie → i32.
        acc = _mm512_dpbusd_epi32(acc, va, vb);
        i += 64;
    }
    let r = n - i;
    if r > 0
    {
        let mask: u64 = (1u64 << r) - 1;
        let va = _mm512_maskz_loadu_epi8(mask, a.as_ptr().add(i) as *const i8);
        let vb = _mm512_maskz_loadu_epi8(mask, b.as_ptr().add(i));
        acc = _mm512_dpbusd_epi32(acc, va, vb);
    }
    _mm512_reduce_add_epi32(acc)
}

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
    fn bf16_roundtrip_exact_powers_of_two() {
        for e in -30i32..30
        {
            let v = 2f32.powi(e);
            assert_eq!(bf16_to_f32(f32_to_bf16(v)), v, "2^{e}");
        }
        assert_eq!(bf16_to_f32(f32_to_bf16(0.0)), 0.0);
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0)), 1.0);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    }

    #[test]
    fn bf16_conversion_within_relative_tolerance() {
        // bf16 a 8 bits de mantisse effectifs → erreur relative < 2^-8.
        for k in 1..1000
        {
            let v = (k as f32) * 0.123 - 40.0;
            if v == 0.0
            {
                continue;
            }
            let r = bf16_to_f32(f32_to_bf16(v));
            let rel = ((r - v) / v).abs();
            assert!(rel <= 2f32.powi(-8), "v={v} rel={rel}");
        }
    }

    #[test]
    fn dot_bf16_matches_f32_reference_loosely() {
        let a: Vec<f32> = (0..500).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..500).map(|i| (i as f32 * 0.02).cos()).collect();
        let mut ab = vec![0u16; a.len()];
        let mut bb = vec![0u16; b.len()];
        f32_to_bf16_slice(&a, &mut ab);
        f32_to_bf16_slice(&b, &mut bb);
        let got = dot_bf16(&ab, &bb);
        let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!(
            (got - want).abs() <= 1e-1 * (1.0 + want.abs()),
            "{got} vs {want}"
        );
    }

    #[test]
    fn vnni_dot_matches_scalar_all_lengths() {
        for len in 0..=200usize
        {
            let a: Vec<u8> = (0..len).map(|i| ((i * 7 + 3) % 256) as u8).collect();
            let b: Vec<i8> = (0..len)
                .map(|i| ((i as i32 * 5 - 17) % 128) as i8)
                .collect();
            let got = dot_u8i8_i32(&a, &b);
            let want = dot_u8i8_i32_scalar(&a, &b);
            assert_eq!(got, want, "len={len}");
        }
    }

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
