// scirust-simd/src/dsp/mod.rs
//
// # Traitement du signal — filtres déterministes génériques
//
// Filtres numériques **génériques sur le scalaire**, construits sur les traits
// [`NumericScalar`](crate::fixed::NumericScalar) /
// [`RealScalar`](crate::fixed::RealScalar). Le même code sert le flottant
// (`f32`/`f64`) **et** la virgule fixe déterministe (`FixedI32<FRAC>`) — un
// filtre en virgule fixe produit les **mêmes bits** sur toute architecture,
// indépendamment du matériel flottant.
//
// ## Contenu
//
// * [`Biquad`] — section du second ordre (IIR) en forme directe II transposée,
//   avec conception « cookbook » RBJ (passe-bas / passe-haut / passe-bande).
// * [`Fir`] — filtre à réponse impulsionnelle finie, ligne à retard circulaire
//   sans allocation, phase linéaire pour des coefficients symétriques.
// * [`fft`] — transformée de Fourier rapide radix-2 (Cooley–Tukey) en place,
//   avec le complexe générique [`fft::Complex`].
// * [`window`] — fenêtres d'apodisation (Hann, Hamming, Blackman,
//   Blackman-Harris, Kaiser) pour réduire la fuite spectrale avant une FFT.
// * [`stft`] — transformée de Fourier à court terme ([`stft::stft`]/
//   [`stft::istft`]) : spectrogramme par recouvrement-fenêtrage, reconstruction
//   par recouvrement-addition (COLA).
// * [`mel`] — banque de filtres mel ([`mel::MelFilterbank`]) : spectrogramme
//   mel standard en reconnaissance vocale et classification audio.
//
// ## Pourquoi la virgule fixe pour le DSP ?
//
// Un pipeline audio ou image en `FixedI32<FRAC>` est **rejouable au bit près** :
// utile pour la reproductibilité scientifique, la vérification formelle et
// l'embarqué sans FPU. Les mêmes structures, instanciées sur `f32`, servent de
// référence et de comparaison.

pub mod biquad;
pub mod fft;
pub mod fir;
pub mod mel;
pub mod stft;
pub mod window;

pub use biquad::Biquad;
pub use fft::{Complex, Plan, fft, ifft, irfft, rfft};
pub use fir::Fir;

#[cfg(test)]
mod tests;
