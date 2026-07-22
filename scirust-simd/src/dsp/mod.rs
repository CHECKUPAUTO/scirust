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
// * [`kalman`] — filtre de Kalman linéaire et étendu ([`kalman::KalmanFilter`]) :
//   contrairement à [`adaptive`] (aucun modèle a priori, coefficients appris
//   depuis l'erreur instantanée), connaît un modèle de transition **et** de
//   mesure et calcule l'estimateur bayésien de variance minimale à chaque
//   pas — le « gain de Kalman » que [`adaptive::Rls`] mentionne déjà dans sa
//   mise à jour de covariance. [`kalman::KalmanFilter::predict_nonlinear`]/
//   [`update_nonlinear`](kalman::KalmanFilter::update_nonlinear) généralisent
//   aux modèles non linéaires (EKF) par linéarisation à chaque pas.
//   [`kalman::UnscentedKalmanFilter`] évite même cette linéarisation : il
//   propage des points sigma **directement** à travers `f`/`h` non linéaires
//   (aucune jacobienne requise), exact au second ordre pour toute
//   non-linéarité et exact pour tout système linéaire (l'EKF n'est exact
//   qu'au premier ordre).
// * [`wavelet`] — transformée en ondelettes discrète par schéma de lifting
//   ([`wavelet::dwt_decompose`]/[`wavelet::dwt_reconstruct`], Haar ou CDF 5/3
//   « LeGall ») : complète [`fft`]/[`stft`] (résolution temps-fréquence
//   **fixe**) par une décomposition **multirésolution**. Réversible
//   **exactement** (au bit près) quel que soit `T` — la prédiction/mise à
//   jour de lifting s'annule par télescopage à la reconstruction, sans
//   arrondi requis, contrairement à un banc de filtres flottant classique.
// * [`hilbert`] — signal analytique par FFT ([`hilbert::analytic_signal`],
//   [`hilbert::hilbert`]) et quantités instantanées qui en découlent :
//   enveloppe ([`hilbert::envelope`], détection AM), phase
//   ([`hilbert::instantaneous_phase`]) et fréquence
//   ([`hilbert::instantaneous_frequency`], démodulation FM) instantanées —
//   la brique de démodulation qui manquait au-dessus de [`fft`], en amont de
//   [`pll`].
// * [`goertzel`] — DFT à **une seule fréquence** ([`goertzel::goertzel`],
//   [`goertzel::goertzel_power`]) par récurrence en `O(N)` sans table :
//   détection de tonalité/DTMF, mesure de puissance à une raie — le
//   complément « un bin » de la [`fft`] complète.
// * [`welch`] — densité spectrale de puissance par la méthode de Welch
//   ([`welch::welch`]) : périodogrammes fenêtrés moyennés (au-dessus de
//   [`stft`]), estimateur de PSD consistant là où un périodogramme brut ne
//   l'est pas.
// * [`savgol`] — filtre de Savitzky–Golay ([`savgol::savgol_filter`],
//   [`savgol::savgol_coeffs`]) : lissage et différentiation par ajustement
//   polynomial glissant au sens des moindres carrés — préserve pics et
//   moments, contrairement à une moyenne glissante.
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
pub mod goertzel;
pub mod hilbert;
pub mod kalman;
pub mod mel;
pub mod mfcc;
pub mod pll;
pub mod resample;
pub mod savgol;
pub mod stft;
pub mod timing;
pub mod wavelet;
pub mod welch;
pub mod window;

pub use adaptive::{Lms, Nlms, Rls};
pub use biquad::{Biquad, BiquadCascade};
pub use fft::{Complex, Plan, fft, ifft, irfft, rfft};
pub use fftconv::fft_convolve;
pub use fir::Fir;
pub use freqz::{group_delay, magnitude, magnitude_db, phase, unwrap_phase};
pub use goertzel::{goertzel, goertzel_power};
pub use hilbert::{
    analytic_signal, envelope, hilbert, instantaneous_frequency, instantaneous_phase,
};
pub use kalman::{KalmanFilter, UnscentedKalmanFilter};
pub use mfcc::{Mfcc, dct2};
pub use pll::{Nco, PiLoopFilter, Pll};
pub use resample::resample;
pub use savgol::{savgol_coeffs, savgol_filter};
pub use timing::{SymbolTimingLoop, gardner_ted, mueller_muller_ted};
pub use wavelet::{Wavelet, dwt_decompose, dwt_reconstruct};
pub use welch::welch;

#[cfg(test)]
mod tests;
