// scirust-simd/src/hypercomplex/asm_probe.rs
//
// Sondes assembleur : points d'entrée autonomes, NON inlinés, qui forcent
// chaque noyau de multiplication hypercomplexe à être émis comme son propre
// symbole. Le script de régression assembleur
// (`scripts/asm_spill_check.sh`) compile ce module pour AArch64 et compte
// le trafic vectoriel référençant `sp` (spills/reloads) dans la boucle chaude.
//
// Chaque sonde reproduit EXACTEMENT la forme du benchmark : une boucle
// d'accumulation `acc = acc + x * y` sur deux slices. C'est le corps réel
// dont on veut prouver (ou réfuter) la résidence-registre.
//
// Gardé derrière la feature `asm-probe` — jamais compilé dans les builds,
// tests ou lints ordinaires. Le `#[no_mangle]` garantit un symbole stable
// à griller par le script ; `#[inline(never)]` empêche la fusion dans un
// éventuel appelant.

use super::octonion::OctonionSimd;
use super::quat::quat_mul;
use super::sedenion::SedenionSimd;
use std::simd::f32x4;

/// Boucle chaude quaternion : `Σ xᵢ · yᵢ` (produit de Hamilton).
#[no_mangle]
#[inline(never)]
pub fn scirust_probe_quat_mul(xs: &[f32x4], ys: &[f32x4]) -> f32x4 {
    let mut acc = f32x4::splat(0.0);
    for (&x, &y) in xs.iter().zip(ys)
    {
        acc += quat_mul(x, y);
    }
    acc
}

/// Boucle chaude octonion : `Σ xᵢ · yᵢ` (Cayley-Dickson sur ℍ).
#[no_mangle]
#[inline(never)]
pub fn scirust_probe_oct_mul(xs: &[OctonionSimd], ys: &[OctonionSimd]) -> OctonionSimd {
    let mut acc = OctonionSimd::ZERO;
    for (&x, &y) in xs.iter().zip(ys)
    {
        acc = acc + x * y;
    }
    acc
}

/// Boucle chaude sédénion : `Σ xᵢ · yᵢ` (Cayley-Dickson sur 𝕆).
#[no_mangle]
#[inline(never)]
pub fn scirust_probe_sed_mul(xs: &[SedenionSimd], ys: &[SedenionSimd]) -> SedenionSimd {
    let mut acc = SedenionSimd::ZERO;
    for (&x, &y) in xs.iter().zip(ys)
    {
        acc = acc + x * y;
    }
    acc
}
