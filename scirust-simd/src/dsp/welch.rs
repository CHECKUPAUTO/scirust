// scirust-simd/src/dsp/welch.rs
//
// # Densité spectrale de puissance — méthode de Welch
//
// Le périodogramme brut (`|FFT(x)|²` d'un seul bloc) est un estimateur de PSD
// **non consistant** : sa variance ne décroît pas avec la longueur du signal.
// La méthode de Welch la réduit en **moyennant** les périodogrammes de
// segments **chevauchants** et **fenêtrés** — l'équivalent de
// `scipy.signal.welch`. **Générique sur le scalaire** comme le reste de `dsp`.
//
// ## Estimation
//
// Le signal est découpé en trames de longueur `window.len()` (puissance de 2),
// espacées de `hop` (le recouvrement est `frame − hop`). Chaque trame est
// multipliée par la fenêtre puis passée à [`super::fft::rfft`] — exactement le
// pipeline de [`super::stft::stft`], réutilisé tel quel. On accumule `|X|²` par
// bin, on moyenne sur les trames, puis on applique la mise à l'échelle
// **densité** de `scipy` :
//
// ```text
// Pxx[k] = (1/(fs·U))·⟨|X[k]|²⟩   avec  U = Σₙ w[n]²  (puissance de fenêtre)
// ```
//
// La PSD renvoyée est **unilatérale** (`bins = frame/2 + 1`, de `0` à Nyquist) :
// les bins intérieurs sont **doublés** pour compter l'énergie du spectre miroir
// négatif, mais **pas** le continu (`k=0`) ni Nyquist (`k=frame/2`), qui n'ont
// pas de jumeau. L'intégrale de `Pxx` sur `[0, fs/2]` estime alors la variance
// du signal (théorème de Parseval).
//
// ## Précision en virgule fixe
//
// Hérite de [`super::stft`]/[`super::fft`] (arrondi des twiddles). La
// normalisation `1/(fs·U)` fait un `recip` par appel, pas par bin. Aucun
// `unsafe`.

use crate::fixed::RealScalar;

/// Densité spectrale de puissance **unilatérale** par la méthode de Welch (cf.
/// en-tête de module) : moyenne des périodogrammes fenêtrés de trames espacées
/// de `hop`, échelle densité `1/(fs·U)`. Renvoie `frame/2 + 1` bins (de `0` à
/// Nyquist), où `frame = window.len()`.
///
/// `window` est la fenêtre d'apodisation (par ex. [`super::window::hann`]) ;
/// sa longueur fixe la taille de trame et doit être une puissance de 2 `≥ 2`.
/// `sample_rate` est en Hz.
///
/// Panique si `frame` n'est pas une puissance de 2 `≥ 2`, si `hop == 0`, ou si
/// le signal est plus court qu'une trame (préconditions de
/// [`super::stft::num_frames`]).
#[must_use]
pub fn welch<T: RealScalar>(signal: &[T], window: &[T], hop: usize, sample_rate: T) -> Vec<T> {
    let frame = window.len();
    // `stft` valide frame (puissance de 2 ≥ 2), hop ≥ 1 et signal ≥ frame.
    let spec = super::stft::stft(signal, window, hop);
    let bins = frame / 2 + 1;
    let frames = spec.len() / bins;

    // Puissance de fenêtre U = Σ w².
    let u = window.iter().fold(T::zero(), |acc, &w| acc + w * w);
    // Échelle densité 1/(fs·U), moyennée sur les trames.
    let inv_frames = T::from_i32(frames as i32).recip();
    let scale = (sample_rate * u).recip() * inv_frames;

    let two = T::from_i32(2);
    let nyquist = frame / 2;

    let mut psd = vec![T::zero(); bins];
    for f in 0..frames
    {
        for (k, p) in psd.iter_mut().enumerate()
        {
            *p = *p + spec[f * bins + k].norm_sqr();
        }
    }
    for (k, p) in psd.iter_mut().enumerate()
    {
        let mut v = *p * scale;
        // Unilatéral : doubler les bins intérieurs (ni continu ni Nyquist).
        if k != 0 && k != nyquist
        {
            v = v * two;
        }
        *p = v;
    }
    psd
}

/// Fréquences centrales (Hz) des bins renvoyés par [`welch`] : `f[k] =
/// k·sample_rate/frame`, `k = 0..=frame/2`. Pratique pour étiqueter/localiser
/// un pic.
///
/// Panique si `frame` n'est pas une puissance de 2 `≥ 2`.
#[must_use]
pub fn welch_freqs<T: RealScalar>(frame: usize, sample_rate: T) -> Vec<T> {
    assert!(
        frame.is_power_of_two() && frame >= 2,
        "welch_freqs : frame = puissance de 2 ≥ 2"
    );
    let inv = sample_rate * T::from_i32(frame as i32).recip();
    (0..=frame / 2)
        .map(|k| T::from_i32(k as i32) * inv)
        .collect()
}
