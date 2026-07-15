// scirust-simd/src/hypercomplex/mod.rs
//
// # Algèbres hypercomplexes SIMD — Octonions (𝕆) et Sédénions (𝕊)
//
// Implémentation register-résidente des algèbres de Cayley-Dickson de
// dimension 8 (octonions) et 16 (sédénions) au-dessus de `std::simd`
// (nightly, feature `portable_simd` déclarée à la racine du crate).
//
// ## Convention Cayley-Dickson
//
// Toute la pile utilise la convention :
//
// ```text
//   (a, b) * (c, d) = (a·c − d̄·b,  d·a + b·c̄)
//   conj((a, b))    = (ā, −b)
// ```
//
// où `a, b, c, d` appartiennent à l'algèbre de dimension moitié et `x̄`
// dénote la conjugaison. Cette convention est appliquée récursivement :
//
// ```text
//   ℝ (f32) ──CD──▶ ℂ ──CD──▶ ℍ (quaternions, f32x4)
//     ──CD──▶ 𝕆 (octonions, f32x8) ──CD──▶ 𝕊 (sédénions, f32x16)
// ```
//
// Le niveau quaternion est le « cas de base » vectorisé : sa multiplication
// est écrite directement en permutations de registres + FMA (voir `quat.rs`),
// sans dérouler la récursion jusqu'aux scalaires.
//
// ## Garanties d'exécution
//
// * **Zéro allocation** : tous les types sont `Copy`, à taille fixe, et
//   toutes les opérations sont `#[inline(always)]`. Après inlining, une
//   multiplication d'octonions est une pure séquence
//   shuffle/FMA sur registres YMM (x86_64/AVX2) ou paires de registres
//   Q NEON (ARM64) — aucune écriture mémoire intermédiaire.
// * **Alignement strict** : `OctonionSimd` est `#[repr(C, align(32))]`
//   (un registre 256 bits), `SedenionSimd` est `#[repr(C, align(64))]`
//   (un registre 512 bits AVX-512, ou 2×YMM / 4×Q NEON après lowering).
// * **Portabilité** : un seul source `std::simd` ; le backend LLVM émet
//   le meilleur jeu d'instructions disponible sous
//   `RUSTFLAGS="-C target-cpu=native"` (AVX-512/AVX2 sur x86_64,
//   NEON sur ARM64 via registres 128 bits jumelés).
//
// ## Contenu
//
// * [`quat`]      — noyau quaternionique `f32x4` (mul, conj) en shuffle+FMA.
// * [`octonion`]  — [`OctonionSimd`], produit 𝕆 par Cayley-Dickson sur ℍ.
// * [`sedenion`]  — [`SedenionSimd`], produit 𝕊 par Cayley-Dickson sur 𝕆.
// * [`dual`]      — [`DualOctonion`] / [`DualSedenion`], différenciation
//                   automatique forward-mode (nombres duaux ε² = 0).
// * [`scalar`]    — implémentations scalaires de référence (récursives et
//                   par table de constantes de structure) pour la validation
//                   croisée et les benchmarks comparatifs.
//
// ## Rappels mathématiques vérifiés par les tests
//
// * 𝕆 est **non associatif** mais **alternatif** : x(xy) = (xx)y.
// * 𝕆 est une algèbre de composition : ‖xy‖² = ‖x‖²‖y‖².
// * 𝕊 perd l'alternativité **et** possède des diviseurs de zéro :
//   (e₁ + e₁₀)(e₄ − e₁₅) = 0 avec les deux facteurs non nuls.

pub mod dual;
pub mod octonion;
pub mod quat;
pub mod scalar;
pub mod sedenion;

pub use dual::{DualOctonion, DualSedenion};
pub use octonion::OctonionSimd;
pub use sedenion::SedenionSimd;

#[cfg(test)]
mod tests;
