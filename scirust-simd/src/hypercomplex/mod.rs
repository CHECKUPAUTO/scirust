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
//   toutes les opérations sont `#[inline(always)]`. Aucun accès au tas, jamais.
// * **Résidence registre** (mesurée, pas supposée — cf.
//   [`scripts/asm_spill_check.sh`](../../scripts/asm_spill_check.sh)) :
//   - Les noyaux **quaternion** et **octonion** sont entièrement
//     register-résidents sur toutes les cibles testées (x86_64 AVX2/AVX-512
//     et AArch64 generic/Neoverse/Apple) : **zéro spill** de boucle chaude.
//   - Le noyau **sédénion** (16 produits de Hamilton) est plus lourd. Grâce à
//     l'accumulation séquentielle (voir [`sedenion`]), il est register-résident
//     sur x86_64 AVX-512 et sur les cœurs AArch64 out-of-order **Neoverse
//     N1/V1 (Graviton 2/3) et Apple Silicon** — les cibles visées. Sur les
//     profils AArch64 `generic`/petit cœur in-order (ex. Cortex-A72) et en
//     AVX2 (16 registres seulement), la pression dépasse le fichier de
//     registres et quelques spills subsistent. Ceci est **documenté et
//     mesuré**, pas masqué.
// * **Alignement strict** : `OctonionSimd` est `#[repr(C, align(32))]`
//   (256 bits), `SedenionSimd` est `#[repr(C, align(64))]` (512 bits). Sur
//   NEON (pas de registre > 128 bits) ces types sont abaissés en 2×Q /
//   4×Q registres ; l'alignement sert alors aux chargements de tableaux.
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
//                   automatique forward-mode (nombres duaux ε² = 0) :
//                   `conj`/`norm`/`normalize`/`inverse` et, comme leurs
//                   bases, `exp`/`ln`/`powf` (dérivée de Fréchet).
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
//
// [`OctonionSimd::exp`]/[`OctonionSimd::ln`]/[`OctonionSimd::powf`] (et leurs
// pendants [`SedenionSimd`]) généralisent l'exponentielle complexe/
// quaternionique : `exp(w·e₀ + v) = eʷ·(cos‖v‖·e₀ + (v/‖v‖)·sin‖v‖)`. La
// formule ne dépend que de l'identité `v̄·v = ‖v‖²·1` (donc `v·v = −‖v‖²·1`
// pour `v` pur), qui tient à tout niveau de Cayley-Dickson — **y compris**
// 𝕊, malgré sa non-associativité et ses diviseurs de zéro, car la série
// `Σ vⁿ/n!` ne met en jeu que les puissances d'un seul élément.

pub mod dual;
pub mod octonion;
pub mod quat;
pub mod scalar;
pub mod sedenion;

pub use dual::{DualOctonion, DualSedenion};
pub use octonion::OctonionSimd;
pub use sedenion::SedenionSimd;

// Sondes assembleur (symboles autonomes pour la régression de spills),
// compilées uniquement avec la feature `asm-probe`.
#[cfg(feature = "asm-probe")]
pub mod asm_probe;

#[cfg(test)]
mod tests;
