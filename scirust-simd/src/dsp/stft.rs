// scirust-simd/src/dsp/stft.rs
//
// # Transformée de Fourier à court terme (STFT) — spectrogramme
//
// [`stft`] découpe un signal en trames chevauchantes, applique une fenêtre
// d'apodisation ([`super::window`]) à chacune, puis calcule son spectre réel
// ([`super::fft::rfft`]). Le résultat — un spectrogramme — est une matrice
// `num_frames × bins` row-major : directement utilisable comme entrée 2D pour
// [`crate::fixed::conv2d`] (temps × fréquence, comme une image).
//
// [`istft`] reconstruit le signal par recouvrement-addition (overlap-add) :
// chaque trame est resynthétisée par [`super::fft::irfft`] puis additionnée à
// la position qui lui correspond, **sans** refenêtrage de synthèse. C'est
// suffisant (reconstruction exacte, aux bords près) car la fenêtre d'analyse
// périodique de Hann à 50 % de recouvrement vérifie la propriété **COLA**
// (constant overlap-add) : `w(i) + w(i + N/2) = 1` pour tout `i` (identité
// trigonométrique `cos(θ) + cos(θ+π) = 0`), donc la somme des copies décalées
// de la fenêtre vaut exactement `1`, sans normalisation ni refenêtrage requis.
//
// ## Disposition mémoire
//
// `frame_size` doit être une puissance de 2 (contrainte de [`super::fft::rfft`]).
// `bins = frame_size/2 + 1`. La sortie de `stft` a pour forme `num_frames ×
// bins`, row-major : la trame `f`, bin `k`, est à l'indice `f·bins + k`.
//
// ## Spectrogramme réel : [`power_spectrogram`] / [`magnitude_spectrogram`]
//
// L'entrée usuelle d'une couche convolutive (`fixed::conv2d`) en
// reconnaissance audio est un spectrogramme **réel** (puissance ou
// magnitude), pas le spectre complexe brut. Ces deux fonctions transforment
// élément par élément la sortie de [`stft`] en conservant sa forme `num_frames
// × bins` — directement enchaînables avec `fixed::conv2d`.

use super::fft::{Complex, irfft, rfft};
use crate::fixed::RealScalar;

/// Nombre de trames pour un signal de longueur `signal_len`, trame
/// `frame_size`, saut `hop`.
///
/// Panique si `hop == 0` ou `signal_len < frame_size` (aucune trame ne tient
/// dans le signal).
#[must_use]
pub fn num_frames(signal_len: usize, frame_size: usize, hop: usize) -> usize {
    assert!(hop >= 1, "stft : hop doit être ≥ 1");
    assert!(
        signal_len >= frame_size,
        "stft : signal de longueur {signal_len} < trame {frame_size}"
    );
    (signal_len - frame_size) / hop + 1
}

/// STFT : spectrogramme `num_frames × bins` (row-major), `bins = frame_size/2
/// + 1`, `frame_size = window.len()`.
///
/// Panique si `frame_size` n'est pas une puissance de 2 `≥ 2` (contrainte de
/// [`super::fft::rfft`]), ou selon les préconditions de [`num_frames`].
#[must_use]
pub fn stft<T: RealScalar>(signal: &[T], window: &[T], hop: usize) -> Vec<Complex<T>> {
    let frame_size = window.len();
    assert!(
        frame_size.is_power_of_two() && frame_size >= 2,
        "stft : taille de trame {frame_size} doit être une puissance de 2 ≥ 2"
    );
    let frames = num_frames(signal.len(), frame_size, hop);
    let bins = frame_size / 2 + 1;
    let mut out = Vec::with_capacity(frames * bins);
    let mut buf = vec![T::zero(); frame_size];
    for f in 0..frames
    {
        let start = f * hop;
        for i in 0..frame_size
        {
            buf[i] = signal[start + i] * window[i];
        }
        out.extend_from_slice(&rfft(&buf));
    }
    out
}

/// Reconstruction par recouvrement-addition (overlap-add) à partir d'un
/// spectrogramme `num_frames × bins` (comme produit par [`stft`]).
///
/// Retourne un signal de longueur `(num_frames−1)·hop + frame_size` (`Vec`
/// vide si `spectrogram` est vide). Exact aux bords près si la fenêtre
/// d'analyse utilisée pour [`stft`] vérifie la propriété COLA à ce `hop`
/// (Hann périodique + recouvrement 50 %, par exemple — voir la documentation
/// de ce module).
///
/// Panique si `spectrogram.len()` n'est pas un multiple de `bins =
/// frame_size/2 + 1`, ou si `frame_size` n'est pas une puissance de 2 `≥ 2`.
#[must_use]
pub fn istft<T: RealScalar>(spectrogram: &[Complex<T>], frame_size: usize, hop: usize) -> Vec<T> {
    assert!(
        frame_size.is_power_of_two() && frame_size >= 2,
        "istft : taille de trame {frame_size} doit être une puissance de 2 ≥ 2"
    );
    if spectrogram.is_empty()
    {
        return Vec::new();
    }
    let bins = frame_size / 2 + 1;
    assert_eq!(
        spectrogram.len() % bins,
        0,
        "istft : spectrogramme de longueur {} non multiple de {bins} bins",
        spectrogram.len()
    );
    let frames = spectrogram.len() / bins;
    let out_len = (frames - 1) * hop + frame_size;
    let mut out = vec![T::zero(); out_len];
    for f in 0..frames
    {
        let frame_time = irfft(&spectrogram[f * bins..(f + 1) * bins], frame_size);
        let start = f * hop;
        for i in 0..frame_size
        {
            out[start + i] = out[start + i] + frame_time[i];
        }
    }
    out
}

/// Spectrogramme de puissance `|X|² = re² + im²`, élément par élément.
///
/// Conserve la forme `num_frames × bins` de [`stft`]. Moins coûteux que
/// [`magnitude_spectrogram`] (pas de racine carrée) et suffisant pour la
/// plupart des pipelines de reconnaissance audio (souvent suivi d'un
/// logarithme, non fourni ici).
#[must_use]
pub fn power_spectrogram<T: RealScalar>(spectrogram: &[Complex<T>]) -> Vec<T> {
    spectrogram
        .iter()
        .map(|c| c.re * c.re + c.im * c.im)
        .collect()
}

/// Spectrogramme de magnitude `|X| = √(re² + im²)`, élément par élément.
///
/// Conserve la forme `num_frames × bins` de [`stft`]. Voir aussi
/// [`power_spectrogram`] (sans racine carrée, moins coûteux).
#[must_use]
pub fn magnitude_spectrogram<T: RealScalar>(spectrogram: &[Complex<T>]) -> Vec<T> {
    spectrogram
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect()
}
