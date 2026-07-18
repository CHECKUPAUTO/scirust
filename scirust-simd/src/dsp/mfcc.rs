// scirust-simd/src/dsp/mfcc.rs
//
// # Coefficients cepstraux sur l'échelle mel (MFCC)
//
// [`Mfcc`] complète la chaîne [`super::stft::stft`] → [`super::stft::power_spectrogram`]
// → [`super::mel::MelFilterbank`] par la dernière étape standard en
// reconnaissance vocale et classification audio : `log` des énergies mel,
// puis **DCT-II** (transformée en cosinus discrète, type II), dont on ne
// garde que les `n_coeffs` premiers coefficients (les plus basses
// « quéfrences », qui capturent l'enveloppe spectrale — les coefficients
// suivants portent l'information fine de hauteur/harmonique, généralement
// écartée pour la reconnaissance vocale).
//
// ## DCT-II orthonormée
//
// [`dct2`] calcule la DCT-II normalisée « ortho » (convention SciPy/librosa) :
//
// ```text
//   Xₖ = αₖ · Σₘ xₘ · cos(π/N · (m + 1/2) · k),   α₀ = √(1/N), αₖ = √(2/N) (k > 0)
// ```
//
// Cette normalisation rend la transformée **orthogonale** (`C·Cᵀ = I`) :
// l'inverse (DCT-III) est simplement la **transposée** de la même matrice de
// base, propriété exploitée par les tests de reconstruction plutôt qu'une
// DCT-III séparée (non nécessaire ici, MFCC n'utilise que le sens direct).
//
// Générique sur [`RealScalar`] `+ Div` (`cos`, racine carrée, division
// réelle par `N` — pas une puissance de deux) comme le reste du module `dsp` :
// même code pour `f32`/`f64` et la virgule fixe déterministe.
//
// ## [`Mfcc`]
//
// Précalcule la banque de filtres mel ([`super::mel::MelFilterbank`]) **et**
// la base DCT-II tronquée aux `n_coeffs` premières lignes (`n_coeffs ≤
// n_mels` — calculer la DCT complète puis tronquer serait un gaspillage,
// la plupart des usages ne gardent que `13`–`20` coefficients sur une
// banque de `40` filtres mel). [`Mfcc::apply`] applique la chaîne complète à
// un spectrogramme de puissance/magnitude déjà calculé.
//
// Le logarithme d'une énergie mel nulle (silence) suit la convention déjà
// établie de [`RealScalar::ln`] (saturation au minimum représentable en
// virgule fixe, `−∞`/`NaN` en flottant comme `f64::ln`) : aucun plancher
// (« epsilon ») arbitraire n'est ajouté ici, pour rester cohérent avec le
// reste du crate plutôt que d'introduire une constante de réglage propre à
// ce module.

use core::ops::Div;

use crate::fixed::RealScalar;

use super::mel::MelFilterbank;

/// Base DCT-II orthonormée `n_coeffs × n_mels` (cf. en-tête de module) :
/// ligne `k`, colonne `m` = `αₖ · cos(π/n_mels · (m + 1/2) · k)`.
pub(crate) fn dct2_basis<T: RealScalar + Div<Output = T>>(
    n_coeffs: usize,
    n_mels: usize,
) -> Vec<T> {
    let n_mels_t = T::from_i32(n_mels as i32);
    let pi_over_n = T::pi() / n_mels_t;
    let half = T::from_i32(2).recip(); // 1/2 : puissance de deux, recip() exact.
    let alpha0 = (T::one() / n_mels_t).sqrt();
    let alpha_k = (T::from_i32(2) / n_mels_t).sqrt();

    let mut basis = vec![T::zero(); n_coeffs * n_mels];
    for k in 0..n_coeffs
    {
        let alpha = if k == 0 { alpha0 } else { alpha_k };
        let k_t = T::from_i32(k as i32);
        for m in 0..n_mels
        {
            let angle = pi_over_n * (T::from_i32(m as i32) + half) * k_t;
            basis[k * n_mels + m] = alpha * angle.cos();
        }
    }
    basis
}

/// DCT-II orthonormée (cf. en-tête de module) d'un vecteur de longueur `n`.
/// Renvoie les `n` coefficients (aucune troncature — pour la troncature aux
/// `n_coeffs` premiers, utiliser [`Mfcc`]).
///
/// Panique si `input` est vide.
#[must_use]
pub fn dct2<T: RealScalar + Div<Output = T>>(input: &[T]) -> Vec<T> {
    assert!(!input.is_empty(), "dct2 : entrée vide");
    let n = input.len();
    let basis = dct2_basis::<T>(n, n);
    (0..n)
        .map(|k| {
            let row = &basis[k * n..(k + 1) * n];
            let mut acc = T::zero();
            for (&b, &x) in row.iter().zip(input)
            {
                acc = acc + b * x;
            }
            acc
        })
        .collect()
}

/// Coefficients cepstraux sur l'échelle mel (MFCC), précalculés pour une
/// banque de filtres mel et un nombre de coefficients donnés.
#[derive(Clone, Debug)]
pub struct Mfcc<T> {
    filterbank: MelFilterbank<T>,
    /// `n_coeffs × n_mels`, row-major (base DCT-II tronquée).
    dct_basis: Vec<T>,
    n_coeffs: usize,
}

impl<T: RealScalar + Div<Output = T>> Mfcc<T> {
    /// Construit pour `n_mels` filtres mel (cf. [`MelFilterbank::new`]) et
    /// `n_coeffs` coefficients cepstraux conservés (`1 ≤ n_coeffs ≤ n_mels`).
    ///
    /// Panique si `n_coeffs == 0`, `n_coeffs > n_mels`, ou dans les mêmes
    /// conditions que [`MelFilterbank::new`].
    #[must_use]
    pub fn new(
        n_mels: usize,
        n_coeffs: usize,
        bins: usize,
        sample_rate: T,
        f_min: T,
        f_max: T,
    ) -> Self {
        assert!(n_coeffs >= 1, "Mfcc::new : n_coeffs doit être ≥ 1");
        assert!(
            n_coeffs <= n_mels,
            "Mfcc::new : n_coeffs ({n_coeffs}) doit être ≤ n_mels ({n_mels})"
        );
        let filterbank = MelFilterbank::new(n_mels, bins, sample_rate, f_min, f_max);
        let dct_basis = dct2_basis::<T>(n_coeffs, n_mels);
        Self {
            filterbank,
            dct_basis,
            n_coeffs,
        }
    }
}

impl<T: RealScalar> Mfcc<T> {
    /// Nombre de coefficients cepstraux conservés par trame.
    #[inline(always)]
    #[must_use]
    pub fn n_coeffs(&self) -> usize {
        self.n_coeffs
    }

    /// Nombre de bandes mel de la banque de filtres sous-jacente.
    #[inline(always)]
    #[must_use]
    pub fn n_mels(&self) -> usize {
        self.filterbank.n_mels()
    }

    /// Nombre de bins linéaires attendus en entrée.
    #[inline(always)]
    #[must_use]
    pub fn bins(&self) -> usize {
        self.filterbank.bins()
    }

    /// Applique la chaîne complète (banque mel → `ln` → DCT-II tronquée) à un
    /// spectrogramme de puissance/magnitude `num_frames × bins()` (row-major,
    /// comme produit par [`super::stft::power_spectrogram`]). Renvoie
    /// `num_frames × n_coeffs()`.
    ///
    /// Panique si `spectrogram.len()` n'est pas un multiple de `bins()`.
    #[must_use]
    pub fn apply(&self, spectrogram: &[T]) -> Vec<T> {
        let mel = self.filterbank.apply(spectrogram);
        let n_mels = self.filterbank.n_mels();
        let frames = mel.len() / n_mels;

        let mut out = vec![T::zero(); frames * self.n_coeffs];
        for f in 0..frames
        {
            let log_mel: Vec<T> = mel[f * n_mels..(f + 1) * n_mels]
                .iter()
                .map(|&e| e.ln())
                .collect();
            for k in 0..self.n_coeffs
            {
                let basis_row = &self.dct_basis[k * n_mels..(k + 1) * n_mels];
                let mut acc = T::zero();
                for (&b, &x) in basis_row.iter().zip(&log_mel)
                {
                    acc = acc + b * x;
                }
                out[f * self.n_coeffs + k] = acc;
            }
        }
        out
    }
}
