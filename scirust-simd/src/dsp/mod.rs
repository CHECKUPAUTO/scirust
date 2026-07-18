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
//   [`biquad::BiquadCascade`] enchaîne plusieurs sections pour des filtres de
//   Butterworth d'ordre pair quelconque ([`biquad::BiquadCascade::butterworth_lowpass`]/
//   [`butterworth_highpass`](biquad::BiquadCascade::butterworth_highpass)).
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
// * [`resample`] — ré-échantillonnage rationnel `L/M` ([`resample::resample`]),
//   filtre passe-bas prototype (sinus cardinal fenêtré) décomposé en `L`
//   sous-filtres polyphase : change la fréquence d'échantillonnage sans
//   matérialiser le signal suréchantillonné.
// * [`fftconv`] — convolution linéaire rapide ([`fftconv::fft_convolve`]),
//   recouvrement-addition via `rfft`/`irfft` : `O(longueur·log(bloc))` contre
//   `O(longueur·N)` en temps direct ([`Fir`]) pour un noyau de `N` prises —
//   avantageux pour les noyaux longs.
//
// ## Pourquoi la virgule fixe pour le DSP ?
//
// Un pipeline audio ou image en `FixedI32<FRAC>` est **rejouable au bit près** :
// utile pour la reproductibilité scientifique, la vérification formelle et
// l'embarqué sans FPU. Les mêmes structures, instanciées sur `f32`, servent de
// référence et de comparaison.

pub mod biquad;
pub mod fft;
pub mod fftconv;
pub mod fir;
pub mod mel;
pub mod resample;
pub mod stft;
pub mod window;

pub use biquad::{Biquad, BiquadCascade};
pub use fft::{Complex, Plan, fft, ifft, irfft, rfft};
pub use fftconv::fft_convolve;
pub use fir::Fir;
pub use resample::resample;

#[cfg(test)]
mod tests;
