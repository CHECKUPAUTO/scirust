//! # Quantification matérielle réelle (x86_64 **et** aarch64)
//!
//! Produits scalaires basse-précision qui sont le socle de l'inférence
//! quantifiée sur l'embarqué. Chaque entrée choisit au *runtime* le meilleur
//! chemin **matériel** disponible, avec repli scalaire portable garanti :
//!
//! * **BF16 mixed-precision** ([`dot_bf16`]) — stockage `bfloat16` (moitié de la
//!   bande passante mémoire), produits/accumulation `f32`. Chemin natif
//!   `avx512bf16` (`VDPBF16PS`) ou élargissement AVX-512F numériquement
//!   identique.
//! * **int8 `u8·i8→i32`** ([`dot_u8i8_i32`]) — convention activation non signée /
//!   poids signé. **AVX-512 VNNI** (`VPDPBUSD`) sur x86, **`i8mm` USDOT**
//!   (`vusdotq_s32`, l'analogue ARM exact) sur aarch64.
//! * **int8 signé·signé `i8·i8→i32`** ([`dot_i8i8_i32`]) — quantification centrée
//!   sur zéro. **`dotprod` SDOT** (`vdotq_s32`) sur ARM — l'instruction
//!   dot-product de base d'ARMv8.2, largement déployée sur l'embarqué —,
//!   `_mm512_madd_epi16` (extension de signe `i8→i16`) sur x86.
//!
//! Le binaire reste **unique** : le même code source couvre le datacenter x86 et
//! le SoC ARM, la détection runtime aiguille vers l'ISA présente.
//!
//! ## Safety
//!
//! Les fonctions `#[target_feature(enable = ...)]` ne sont appelées qu'après la
//! détection runtime du même jeu d'instructions (`is_x86_feature_detected!` /
//! `is_aarch64_feature_detected!`). Les chargements sont *unaligned* (`loadu` /
//! `vld1q`), donc aucune contrainte d'alignement ; l'arithmétique de pointeurs
//! reste bornée par les conditions de boucle et les longueurs de slices.

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
/// Trois chemins, choisis au runtime :
///
/// 1. **`avx512bf16` natif** — `_mm512_dpbf16_ps` : une seule instruction
///    consomme 32 `bf16` par opérande, multiplie et accumule en `f32` (produit
///    matériel exact `VDPBF16PS`). C'est la quantification **matérielle réelle**
///    dès que la puce l'expose (Sapphire Rapids, Zen 4…).
/// 2. **`avx512f` (élargissement)** — `bf16 → f32` par `_mm512_cvtepu16_epi32` +
///    décalage de 16 bits puis FMA `f32`, 16 voies/pas. Repli **numériquement
///    identique** au chemin natif (mêmes produits/accumulation `f32`) sur les
///    puces AVX-512 sans l'ISA `bf16`.
/// 3. **Scalaire** — portable, référence des tests.
pub fn dot_bf16(a: &[u16], b: &[u16]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_bf16: length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512bf16") && std::is_x86_feature_detected!("avx512f")
        {
            // SAFETY: gated by the runtime detection just above.
            return unsafe { dot_bf16_avx512bf16(a, b) };
        }
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

/// Chemin **`avx512bf16` natif** de [`dot_bf16`] : `_mm512_dpbf16_ps` accumule
/// 32 produits `bf16·bf16→f32` par pas dans 16 voies `f32`, réduites à la fin.
///
/// Les 32 `u16` bruts sont chargés puis réinterprétés en `__m512bh` (même
/// disposition binaire) : chaque `bf16[k]` reste apparié à son homologue, et
/// comme toutes les voies sont additionnées en sortie, l'agencement interne
/// paire/voie de `dpbf16` n'affecte pas le résultat (chaque `a[k]·b[k]` est
/// compté une fois). Bord `< 32` traité en scalaire.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bf16,avx512f")]
unsafe fn dot_bf16_avx512bf16(a: &[u16], b: &[u16]) -> f32 {
    use core::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i + 32 <= n
    {
        let va = _mm512_loadu_si512(a.as_ptr().add(i).cast());
        let vb = _mm512_loadu_si512(b.as_ptr().add(i).cast());
        // Réinterprétation bit-à-bit u16×32 → bf16×32 (taille identique, 512 b).
        let abh: __m512bh = core::mem::transmute(va);
        let bbh: __m512bh = core::mem::transmute(vb);
        acc = _mm512_dpbf16_ps(acc, abh, bbh);
        i += 32;
    }
    let mut sum = _mm512_reduce_add_ps(acc);
    for j in i..n
    {
        sum += bf16_to_f32(a[j]) * bf16_to_f32(b[j]);
    }
    sum
}

// ===================================================================== //
//  int8 u8·i8 → i32 (VNNI x86 / USDOT ARM)                               //
// ===================================================================== //

/// Produit scalaire quantifié `sum(a[i] as i32 * b[i] as i32)` avec activations
/// `u8` (non signées) et poids `i8` (signés) — la convention usuelle de
/// l'inférence int8.
///
/// Chemins matériels réels, choisis au runtime :
/// * **x86_64** : **AVX-512 VNNI** (`VPDPBUSD`, 4 MAC entiers/voie/instruction),
///   qui prend précisément un opérande `u8` non signé et un `i8` signé.
/// * **aarch64** : **`i8mm` USDOT** (`vusdotq_s32`), l'analogue ARM exact de
///   `VPDPBUSD` (`u8·i8→i32`, 4 MAC/voie). C'est la quantification int8 native de
///   l'embarqué récent (Cortex-A710/X2, Neoverse V2…).
/// * repli scalaire sinon.
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
    #[cfg(all(feature = "nightly-simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("i8mm")
        {
            // SAFETY: gated by the runtime detection just above.
            return unsafe { dot_u8i8_i32_usdot(a, b) };
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
        let va = _mm512_loadu_si512(a.as_ptr().add(i).cast());
        let vb = _mm512_loadu_si512(b.as_ptr().add(i).cast());
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

/// Chemin **`i8mm` USDOT** de [`dot_u8i8_i32`] (aarch64) : `vusdotq_s32` prend
/// `u8` (non signé) et `i8` (signé), 4 MAC/voie → 4 accumulateurs `i32`, 16
/// éléments/pas ; réduction horizontale + bord scalaire.
#[cfg(all(feature = "nightly-simd", target_arch = "aarch64"))]
#[target_feature(enable = "i8mm")]
unsafe fn dot_u8i8_i32_usdot(a: &[u8], b: &[i8]) -> i32 {
    use core::arch::aarch64::*;
    let n = a.len();
    let mut acc = vdupq_n_s32(0);
    let mut i = 0;
    while i + 16 <= n
    {
        let va = vld1q_u8(a.as_ptr().add(i));
        let vb = vld1q_s8(b.as_ptr().add(i));
        acc = vusdotq_s32(acc, va, vb);
        i += 16;
    }
    let mut sum = vaddvq_s32(acc);
    for j in i..n
    {
        sum += (a[j] as i32) * (b[j] as i32);
    }
    sum
}

// ===================================================================== //
//  int8 signé·signé i8·i8 → i32 (SDOT ARM / madd x86)                    //
// ===================================================================== //

/// Produit scalaire int8 **signé·signé** `sum(a[i] as i32 * b[i] as i32)` — la
/// convention *symétrique* (poids **et** activations signés), courante quand la
/// quantification est centrée sur zéro.
///
/// Chemins matériels réels, choisis au runtime :
/// * **aarch64** : **`dotprod` SDOT** (`vdotq_s32`) — l'instruction dot-product
///   de base d'ARMv8.2, présente sur tout l'embarqué inférence moderne
///   (Cortex-A76 du RK3588, A78 de l'Orin…), donc plus largement déployée que
///   `i8mm`.
/// * **x86_64** : AVX-512BW `_mm512_madd_epi16` après extension de signe
///   `i8→i16` (VNNI `VPDPBUSD` ne couvre pas le cas signé·signé) — 32
///   éléments/pas.
/// * repli scalaire sinon.
pub fn dot_i8i8_i32(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len(), "dot_i8i8_i32: length mismatch");
    #[cfg(all(feature = "nightly-simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod")
        {
            // SAFETY: gated by the runtime detection just above.
            return unsafe { dot_i8i8_i32_sdot(a, b) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512bw") && std::is_x86_feature_detected!("avx512f")
        {
            // SAFETY: gated by the runtime detection just above.
            return unsafe { dot_i8i8_i32_avx512(a, b) };
        }
    }
    dot_i8i8_i32_scalar(a, b)
}

/// Repli scalaire portable de [`dot_i8i8_i32`] (aussi la référence des tests).
pub fn dot_i8i8_i32_scalar(a: &[i8], b: &[i8]) -> i32 {
    let mut acc: i32 = 0;
    for (&x, &y) in a.iter().zip(b)
    {
        acc += (x as i32) * (y as i32);
    }
    acc
}

/// Chemin **`dotprod` SDOT** de [`dot_i8i8_i32`] (aarch64) : `vdotq_s32`,
/// 4 MAC signés/voie → 4 accumulateurs `i32`, 16 éléments/pas.
#[cfg(all(feature = "nightly-simd", target_arch = "aarch64"))]
#[target_feature(enable = "dotprod")]
unsafe fn dot_i8i8_i32_sdot(a: &[i8], b: &[i8]) -> i32 {
    use core::arch::aarch64::*;
    let n = a.len();
    let mut acc = vdupq_n_s32(0);
    let mut i = 0;
    while i + 16 <= n
    {
        let va = vld1q_s8(a.as_ptr().add(i));
        let vb = vld1q_s8(b.as_ptr().add(i));
        acc = vdotq_s32(acc, va, vb);
        i += 16;
    }
    let mut sum = vaddvq_s32(acc);
    for j in i..n
    {
        sum += (a[j] as i32) * (b[j] as i32);
    }
    sum
}

/// Chemin **AVX-512BW** de [`dot_i8i8_i32`] (x86_64) : extension de signe
/// `i8→i16` (`_mm512_cvtepi8_epi16`) puis `_mm512_madd_epi16` (produits appariés
/// `i16·i16→i32`), 32 éléments/pas ; bord `< 32` en scalaire.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512f")]
unsafe fn dot_i8i8_i32_avx512(a: &[i8], b: &[i8]) -> i32 {
    use core::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm512_setzero_si512();
    let mut i = 0;
    while i + 32 <= n
    {
        let va = _mm512_cvtepi8_epi16(_mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i));
        let vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i));
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(va, vb));
        i += 32;
    }
    let mut sum = _mm512_reduce_add_epi32(acc);
    for j in i..n
    {
        sum += (a[j] as i32) * (b[j] as i32);
    }
    sum
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
        // Longueur > 32 pour traverser le chemin natif avx512bf16 quand présent.
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
        // Exerce VNNI (x86) / USDOT (aarch64 i8mm) / scalaire, tous bords.
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
    fn i8i8_dot_matches_scalar_all_lengths() {
        // Couvre le chemin SDOT (aarch64/dotprod), AVX-512BW madd (x86) et le
        // repli scalaire, sur toutes les longueurs (multiples de 16/32 + bords).
        for len in 0..=200usize
        {
            let a: Vec<i8> = (0..len)
                .map(|i| ((i as i32 * 7 - 61) % 128) as i8)
                .collect();
            let b: Vec<i8> = (0..len)
                .map(|i| ((i as i32 * -5 + 23) % 128) as i8)
                .collect();
            let got = dot_i8i8_i32(&a, &b);
            let want = dot_i8i8_i32_scalar(&a, &b);
            assert_eq!(got, want, "len={len}");
        }
    }

    #[test]
    fn i8i8_dot_extreme_values_no_overflow() {
        // Valeurs extrêmes signées (-128, 127) sur une longueur significative.
        let a: Vec<i8> = (0..64)
            .map(|i| if i % 2 == 0 { -128 } else { 127 })
            .collect();
        let b: Vec<i8> = (0..64)
            .map(|i| if i % 3 == 0 { 127 } else { -128 })
            .collect();
        assert_eq!(dot_i8i8_i32(&a, &b), dot_i8i8_i32_scalar(&a, &b));
    }
}
