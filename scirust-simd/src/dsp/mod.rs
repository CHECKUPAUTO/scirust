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
//   [`biquad::BiquadCascade`] enchaîne plusieurs sections pour des filtres
//   d'ordre pair quelconque : Butterworth ([`biquad::BiquadCascade::butterworth_lowpass`]/
//   [`butterworth_highpass`](biquad::BiquadCascade::butterworth_highpass), platitude
//   maximale) ou Chebyshev de type I ([`biquad::BiquadCascade::chebyshev1_lowpass`]/
//   [`chebyshev1_highpass`](biquad::BiquadCascade::chebyshev1_highpass), ondulation
//   contrôlée contre une coupure plus raide au même ordre).
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
// * [`mfcc`] — coefficients cepstraux sur l'échelle mel ([`mfcc::Mfcc`],
//   [`mfcc::dct2`]) : dernière étape standard au-dessus de [`mel`]
//   (`ln` des énergies mel, puis DCT-II tronquée aux coefficients les plus
//   bas — l'enveloppe spectrale, information clé en reconnaissance vocale).
// * [`resample`] — ré-échantillonnage rationnel `L/M` ([`resample::resample`]),
//   filtre passe-bas prototype (sinus cardinal fenêtré) décomposé en `L`
//   sous-filtres polyphase : change la fréquence d'échantillonnage sans
//   matérialiser le signal suréchantillonné.
// * [`fftconv`] — convolution linéaire rapide ([`fftconv::fft_convolve`]),
//   recouvrement-addition via `rfft`/`irfft` : `O(longueur·log(bloc))` contre
//   `O(longueur·N)` en temps direct ([`Fir`]) pour un noyau de `N` prises —
//   avantageux pour les noyaux longs.
// * [`adaptive`] — filtres adaptatifs ([`adaptive::Lms`], [`adaptive::Nlms`],
//   [`adaptive::Rls`]) : contrairement à [`Biquad`]/[`Fir`]/[`BiquadCascade`],
//   les coefficients ne sont **pas conçus a priori** mais **appris en ligne**
//   à partir d'un signal d'erreur — identification de système, annulation
//   d'écho, égalisation de canal.
// * [`freqz`] — réponse en fréquence (`Biquad::frequency_response`,
//   [`BiquadCascade::frequency_response`], [`Fir::frequency_response`]) :
//   évaluation de `H(e^{jω})`, magnitude ([`freqz::magnitude`]/
//   [`freqz::magnitude_db`]), phase ([`freqz::phase`]/[`freqz::unwrap_phase`])
//   et délai de groupe ([`freqz::group_delay`]) — permet de **vérifier** ce
//   que [`Biquad`]/[`BiquadCascade`]/[`Fir`] ont conçu.
// * [`pll`] — boucle à verrouillage de phase ([`pll::Pll`], oscillateur
//   commandé [`pll::Nco`] + filtre de boucle proportionnel-intégral
//   [`pll::PiLoopFilter`]) : contrairement à [`adaptive`] (erreur
//   instantanée, pas d'état de phase), une PLL **suit** une porteuse/horloge
//   de fréquence potentiellement variable — récupération de porteuse,
//   démodulation FM, synthèse de fréquence.
// * [`timing`] — récupération d'horloge **symbole** ([`timing::SymbolTimingLoop`],
//   détecteur de Gardner [`timing::gardner_ted`]) : le pendant « instant de
//   décision fractionnaire dans un flux échantillonné » de [`pll`] (porteuse
//   continue) — les deux boucles indépendantes de toute chaîne de réception
//   numérique. [`timing::mueller_muller_ted`] (piloté par décision) est
//   fourni comme brique indépendante pour un synchroniseur personnalisé.
//
// ## Pourquoi la virgule fixe pour le DSP ?
//
// Un pipeline audio ou image en `FixedI32<FRAC>` est **rejouable au bit près** :
// utile pour la reproductibilité scientifique, la vérification formelle et
// l'embarqué sans FPU. Les mêmes structures, instanciées sur `f32`, servent de
// référence et de comparaison.

pub mod adaptive;
pub mod biquad;
pub mod fft;
pub mod fftconv;
pub mod fir;
pub mod freqz;
pub mod mel;
pub mod mfcc;
pub mod pll;
pub mod resample;
pub mod stft;
pub mod timing;
pub mod window;

pub use adaptive::{Lms, Nlms, Rls};
pub use biquad::{Biquad, BiquadCascade};
pub use fft::{Complex, Plan, fft, ifft, irfft, rfft};
pub use fftconv::fft_convolve;
pub use fir::Fir;
pub use freqz::{group_delay, magnitude, magnitude_db, phase, unwrap_phase};
pub use mfcc::{Mfcc, dct2};
pub use pll::{Nco, PiLoopFilter, Pll};
pub use resample::resample;
pub use timing::{SymbolTimingLoop, gardner_ted, mueller_muller_ted};

#[cfg(test)]
mod tests;
