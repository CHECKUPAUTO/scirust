// scirust-simd/src/dsp/fftconv.rs
//
// # Convolution linéaire rapide (recouvrement-addition), déterministe
//
// [`fft_convolve`] calcule la convolution linéaire complète de `signal` par
// `kernel` via [`super::fft::rfft`]/[`super::fft::irfft`] et la méthode
// **recouvrement-addition** (« overlap-add ») — **aucune nouvelle
// transformée**, uniquement l'assemblage : découpage en blocs, produit
// spectral bloc par bloc (multiplication ponctuelle de [`super::fft::Complex`],
// déjà munie de l'opérateur `Mul`), puis recouvrement des blocs successifs.
//
// **Générique sur le scalaire** comme le reste de `dsp` : la même
// implémentation sert `f32`/`f64` et la virgule fixe déterministe.
//
// ## Pourquoi, en plus de [`super::Fir`] ?
//
// [`super::Fir`] convolue en temps direct : `O(longueur·N)` opérations pour un
// noyau de `N` prises — adapté aux noyaux **courts** (quelques dizaines de
// prises), coûteux au-delà. `fft_convolve` ramène le coût à
// `O(longueur·log(bloc))` via la FFT — l'avantage classique de la convolution
// rapide pour des noyaux **longs** (réverbération, filtres FIR de plusieurs
// centaines/milliers de prises).
//
// ## Recouvrement-addition
//
// La transformée radix-2 existante exige une puissance de deux : `fft_size`
// doit donc être une puissance de deux strictement supérieure à
// `kernel.len()`. Chaque bloc du signal, de longueur `block_size = fft_size −
// kernel.len() + 1`, est zéro-complété à `fft_size`, transformé, multiplié
// point à point par le spectre du noyau (lui-même zéro-complété à
// `fft_size`, calculé **une seule fois**), puis retransformé — `fft_size`
// couvre exactement `block_size + kernel.len() − 1`, donc **aucun repliement
// circulaire** ne contamine le résultat (contrairement à une FFT de la seule
// longueur du bloc). Les blocs successifs se chevauchent sur les
// `kernel.len() − 1` derniers échantillons : on les **additionne**
// (recouvrement-addition), reconstituant exactement la convolution linéaire
// complète, bloc par bloc.
//
// Retourne la convolution **complète** (longueur `signal.len() + kernel.len()
// − 1`, comme la définition mathématique du produit de convolution), pas une
// troncature « same »/« valid ».

use crate::fixed::RealScalar;

use super::fft::{irfft, rfft};

/// Convolution linéaire rapide de `signal` par `kernel`, recouvrement-addition
/// via `rfft`/`irfft` (cf. en-tête de module).
///
/// `fft_size` doit être une puissance de deux strictement supérieure à
/// `kernel.len()` — plus grand, moins de blocs mais plus de travail par
/// bloc ; un choix courant est `4×` à `8×` la longueur du noyau.
///
/// Retourne la convolution complète, longueur `signal.len() + kernel.len() −
/// 1`. `signal` peut être vide (renvoie `kernel.len() − 1` zéros).
///
/// Panique si `fft_size` n'est pas une puissance de deux, si `fft_size <=
/// kernel.len()`, ou si `kernel` est vide.
#[must_use]
pub fn fft_convolve<T: RealScalar>(signal: &[T], kernel: &[T], fft_size: usize) -> Vec<T> {
    assert!(
        fft_size.is_power_of_two(),
        "fft_convolve : fft_size ({fft_size}) doit être une puissance de deux"
    );
    assert!(!kernel.is_empty(), "fft_convolve : noyau vide");
    assert!(
        fft_size > kernel.len(),
        "fft_convolve : fft_size ({fft_size}) doit être > kernel.len() ({})",
        kernel.len()
    );

    let block_size = fft_size - kernel.len() + 1;
    let out_len = signal.len() + kernel.len() - 1;
    let mut out = vec![T::zero(); out_len];

    let mut kernel_padded = vec![T::zero(); fft_size];
    kernel_padded[..kernel.len()].copy_from_slice(kernel);
    let kernel_spec = rfft(&kernel_padded);

    let mut pos = 0;
    while pos < signal.len()
    {
        let end = (pos + block_size).min(signal.len());
        let mut block = vec![T::zero(); fft_size];
        block[..end - pos].copy_from_slice(&signal[pos..end]);

        let mut block_spec = rfft(&block);
        for (bs, &ks) in block_spec.iter_mut().zip(&kernel_spec)
        {
            *bs = *bs * ks;
        }
        let block_conv = irfft(&block_spec, fft_size);

        for (i, &v) in block_conv.iter().enumerate()
        {
            let dst = pos + i;
            if dst < out_len
            {
                out[dst] = out[dst] + v;
            }
        }
        pos += block_size;
    }
    out
}
