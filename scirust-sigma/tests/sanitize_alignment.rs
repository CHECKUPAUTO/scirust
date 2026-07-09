//! Test d'alignement : le seuil de `sanitize_f32` (dans
//! `scirust-gpu/src/deterministic.rs`) DOIT être bit-identique à
//! [`scirust_sigma::SIGMA_SANITIZED_F32`].
//!
//! Aucune dépendance de crate vers `scirust-gpu` (chemin chaud GPU, à ne pas
//! coupler) : on recopie ici, en dur, la constante utilisée par `sanitize_f32`
//! (`f32::MIN_POSITIVE`) et on l'affirme par `to_bits()`. Ce test casse
//! volontairement si l'un des deux seuils bouge sans l'autre — c'est le rappel
//! qu'ils encodent le même contrat σ (le « couvercle de zéro » de la voie 3).

use scirust_sigma::SIGMA_SANITIZED_F32;

/// Recopie littérale du seuil de `sanitize_f32` (voie 3, GPU) :
/// `if x.abs() < f32::MIN_POSITIVE { 0.0 } else { x }`.
///
/// Si `deterministic.rs` change ce seuil, il faut mettre à jour ici ET σ, sinon
/// ce test échoue (volontairement).
const SANITIZE_F32_THRESHOLD: f32 = f32::MIN_POSITIVE;

#[test]
fn sanitize_threshold_matches_sigma_sanitized_f32() {
    assert_eq!(
        SANITIZE_F32_THRESHOLD.to_bits(),
        SIGMA_SANITIZED_F32.to_bits(),
        "le seuil de sanitize_f32 et SIGMA_SANITIZED_F32 ont divergé bit-à-bit"
    );
    // Ancrage bit-à-bit en dur : le plus petit f32 normal.
    assert_eq!(SIGMA_SANITIZED_F32.to_bits(), 0x0080_0000);
}
