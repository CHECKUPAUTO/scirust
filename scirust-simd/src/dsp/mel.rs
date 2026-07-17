// scirust-simd/src/dsp/mel.rs
//
// # Banque de filtres mel (mel-spectrogramme)
//
// [`MelFilterbank`] convertit un spectrogramme de puissance/magnitude linéaire
// en fréquence ([`super::stft::power_spectrogram`]) en spectrogramme **mel** :
// `n_mels` bandes réparties selon l'échelle perceptuelle mel (convention
// HTK/O'Shaughnessy), standard en reconnaissance vocale et classification
// audio (MFCC, spectrogrammes mel en entrée d'un CNN).
//
// ## Échelle mel
//
// `mel(f) = 2595·log₁₀(1 + f/700)`, inverse `hz(m) = 700·(10^(m/2595) − 1)`.
// Calculée entièrement en `T` (`ln`/`exp`/division, **aucune conversion
// flottante intermédiaire**) : générique sur [`RealScalar`] `+ Div`, donc
// `f32`, `f64` **et** `FixedI32<FRAC>` — même technique que le reste du
// module `dsp`. Les dénominateurs ici (`700`, `2595`, ...) ne sont **pas**
// des puissances de deux : contrairement à `window::angle` (dénominateur
// `len`, toujours une puissance de 2 pour `stft`), `x * y.recip()` perdrait
// ici trop de précision en virgule fixe (le calcul du réciproque isolé
// arrondit une première fois avant même la multiplication). On utilise donc
// la division réelle (`/`), qui accumule sur une largeur double avant un
// unique arrondi final.
//
// ## Filtres triangulaires
//
// `n_mels + 2` points également espacés en échelle mel entre `f_min` et
// `f_max` définissent `n_mels` filtres triangulaires qui se chevauchent à
// moitié. Le poids du filtre `m` au bin linéaire `k` (fréquence `f_k`) :
//
// * `0` hors de `[f_{m-1}, f_{m+1}]` ;
// * pente montante entre `f_{m-1}` et `f_m` ;
// * pente descendante entre `f_m` et `f_{m+1}`.
//
// ## Application
//
// `mel[frame, m] = Σ_k filterbank[m, k] · power[frame, k]` : la banque de
// filtres (matrice `n_mels × bins`) est appliquée à chaque trame du
// spectrogramme d'entrée (`num_frames × bins`, comme produit par
// [`super::stft::stft`] + [`super::stft::power_spectrogram`]), donnant un
// spectrogramme mel `num_frames × n_mels`.

use core::ops::Div;

use crate::fixed::RealScalar;

/// `2595·log₁₀(1 + hz/700)`, calculé via `ln` (`log₁₀(x) = ln(x)/ln(10)`).
fn hz_to_mel<T: RealScalar + Div<Output = T>>(hz: T) -> T {
    let seven_hundred = T::from_i32(700);
    let arg = T::one() + hz / seven_hundred;
    let ln10 = T::from_i32(10).ln();
    T::from_i32(2595) * arg.ln() / ln10
}

/// `700·(10^(mel/2595) − 1)`, calculé via `exp` (`10^x = exp(x·ln 10)`).
fn mel_to_hz<T: RealScalar + Div<Output = T>>(mel: T) -> T {
    let ln10 = T::from_i32(10).ln();
    let exponent = mel / T::from_i32(2595) * ln10;
    T::from_i32(700) * (exponent.exp() - T::one())
}

/// Banque de `n_mels` filtres triangulaires mel, précalculée pour un nombre de
/// bins linéaires et une fréquence d'échantillonnage donnés.
#[derive(Clone, Debug)]
pub struct MelFilterbank<T> {
    n_mels: usize,
    bins: usize,
    /// `n_mels × bins`, row-major.
    weights: Vec<T>,
}

impl<T: RealScalar + Div<Output = T>> MelFilterbank<T> {
    /// Construit une banque de `n_mels` filtres pour un spectre de `bins =
    /// n_fft/2 + 1` bins linéaires, fréquence d'échantillonnage
    /// `sample_rate` (Hz), bande `[f_min, f_max]` (Hz).
    ///
    /// Panique si `n_mels == 0`, `bins < 2`, ou `f_min >= f_max`.
    #[must_use]
    pub fn new(n_mels: usize, bins: usize, sample_rate: T, f_min: T, f_max: T) -> Self {
        assert!(n_mels >= 1, "MelFilterbank::new : n_mels doit être ≥ 1");
        assert!(bins >= 2, "MelFilterbank::new : bins doit être ≥ 2");
        assert!(
            f_min < f_max,
            "MelFilterbank::new : f_min doit être < f_max"
        );

        let mel_min = hz_to_mel(f_min);
        let mel_max = hz_to_mel(f_max);
        let n_points = n_mels + 2;
        let step = (mel_max - mel_min) / T::from_i32((n_points - 1) as i32);

        // n_mels+2 fréquences de coupure (Hz), également espacées en mel.
        let hz_points: Vec<T> = (0..n_points)
            .map(|i| mel_to_hz(mel_min + T::from_i32(i as i32) * step))
            .collect();

        // Fréquence centrale de chaque bin linéaire (n_fft = 2·(bins−1)).
        // Diviser *avant* de multiplier par `k` : le résultat final
        // (`k·sample_rate/n_fft ≤ sample_rate/2`) tient dans `T`, mais le
        // produit intermédiaire `k·sample_rate` peut le dépasser largement
        // (ex. Q16.16, `sample_rate = 16000`, `k = 3` : `48000` hors plage).
        let n_fft = 2 * (bins - 1);
        let hz_per_bin = sample_rate / T::from_i32(n_fft as i32);
        let bin_hz: Vec<T> = (0..bins)
            .map(|k| T::from_i32(k as i32) * hz_per_bin)
            .collect();

        let mut weights = vec![T::zero(); n_mels * bins];
        for m in 0..n_mels
        {
            let (left, center, right) = (hz_points[m], hz_points[m + 1], hz_points[m + 2]);
            for (k, &f) in bin_hz.iter().enumerate()
            {
                let w = if f <= left || f >= right
                {
                    T::zero()
                }
                else if f <= center
                {
                    (f - left) / (center - left)
                }
                else
                {
                    (right - f) / (right - center)
                };
                weights[m * bins + k] = w;
            }
        }

        Self {
            n_mels,
            bins,
            weights,
        }
    }
}

impl<T: RealScalar> MelFilterbank<T> {
    /// Nombre de bandes mel.
    #[inline(always)]
    #[must_use]
    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Nombre de bins linéaires attendus en entrée.
    #[inline(always)]
    #[must_use]
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// Applique la banque de filtres à un spectrogramme de puissance/magnitude
    /// `num_frames × bins()` (row-major, comme produit par
    /// [`super::stft::power_spectrogram`]/[`super::stft::magnitude_spectrogram`]).
    /// Retourne `num_frames × n_mels()`.
    ///
    /// Panique si `spectrogram.len()` n'est pas un multiple de `bins()`.
    #[must_use]
    pub fn apply(&self, spectrogram: &[T]) -> Vec<T> {
        assert_eq!(
            spectrogram.len() % self.bins,
            0,
            "MelFilterbank::apply : spectrogramme de longueur {} non multiple de {} bins",
            spectrogram.len(),
            self.bins
        );
        let frames = spectrogram.len() / self.bins;
        let mut out = vec![T::zero(); frames * self.n_mels];
        for f in 0..frames
        {
            let frame = &spectrogram[f * self.bins..(f + 1) * self.bins];
            for m in 0..self.n_mels
            {
                let filt = &self.weights[m * self.bins..(m + 1) * self.bins];
                let mut acc = T::zero();
                for k in 0..self.bins
                {
                    acc = acc + filt[k] * frame[k];
                }
                out[f * self.n_mels + m] = acc;
            }
        }
        out
    }
}
