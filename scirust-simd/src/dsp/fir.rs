// scirust-simd/src/dsp/fir.rs
//
// # Filtre à réponse impulsionnelle finie `Fir<T, N>`
//
// Convolution `y[n] = Σₖ hₖ · x[n−k]` avec `N` coefficients (`taps`),
// **générique sur le scalaire** (`f32`/`f64`/`FixedI32<FRAC>`) et **sans
// allocation** : la ligne à retard est un tableau de taille fixe `[T; N]` géré
// en tampon circulaire. En virgule fixe, le filtrage est **déterministe
// bit-à-bit**.
//
// Un FIR à coefficients symétriques est à **phase linéaire** (retard de groupe
// constant `= (N−1)/2` échantillons), propriété clé en traitement d'image et
// audio.

use crate::fixed::NumericScalar;

/// Filtre FIR à `N` coefficients, ligne à retard circulaire (zéro allocation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fir<T, const N: usize> {
    taps: [T; N],
    delay: [T; N],
    pos: usize,
}

impl<T: NumericScalar, const N: usize> Fir<T, N> {
    /// Construit depuis les coefficients `taps` (`h₀ … h_{N−1}`). Ligne à retard
    /// initialisée à zéro.
    #[inline]
    pub fn new(taps: [T; N]) -> Self {
        const {
            assert!(N >= 1, "Fir: au moins un coefficient");
        }
        Self {
            taps,
            delay: [T::zero(); N],
            pos: 0,
        }
    }

    /// Remet la ligne à retard à zéro.
    #[inline]
    pub fn reset(&mut self) {
        self.delay = [T::zero(); N];
        self.pos = 0;
    }

    /// Traite un échantillon : insère `x`, renvoie `Σₖ hₖ · x[n−k]`.
    #[inline]
    pub fn process(&mut self, x: T) -> T {
        self.delay[self.pos] = x;
        let mut acc = T::zero();
        let mut idx = self.pos;
        for &h in &self.taps
        {
            acc = acc + h * self.delay[idx];
            idx = if idx == 0 { N - 1 } else { idx - 1 };
        }
        self.pos = if self.pos + 1 == N { 0 } else { self.pos + 1 };
        acc
    }

    /// Filtre un bloc `input` vers `out` (même longueur). Panique sinon.
    #[inline]
    pub fn process_block(&mut self, input: &[T], out: &mut [T]) {
        assert_eq!(
            input.len(),
            out.len(),
            "process_block: longueurs différentes"
        );
        for (o, &x) in out.iter_mut().zip(input)
        {
            *o = self.process(x);
        }
    }

    /// Les coefficients du filtre.
    #[inline]
    pub fn taps(&self) -> &[T; N] {
        &self.taps
    }
}
