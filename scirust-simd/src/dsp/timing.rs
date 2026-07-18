// scirust-simd/src/dsp/timing.rs
//
// # Récupération d'horloge symbole (synchronisation de timing)
//
// [`super::pll`] suit une porteuse/horloge **continue** (phase/fréquence d'un
// oscillateur). La récupération d'horloge symbole résout un problème voisin
// mais distinct : retrouver, dans un flux échantillonné à un rythme fixe
// (`sps` échantillons/symbole, généralement non entier idéalement, mais fixé
// ici à un entier — voir plus bas), l'**instant fractionnaire** optimal de
// décision de chaque symbole — récupération de porteuse et récupération
// d'horloge sont les deux boucles indépendantes de toute chaîne de réception
// numérique.
//
// ## Détecteurs d'erreur de timing (TED)
//
// * [`gardner_ted`] — détecteur de Gardner (1986), **indépendant de la
//   porteuse** (ne requiert aucune décision de symbole, contrairement à
//   Mueller-Müller) : `e = (late − early)·punctual`, où `early`/`punctual`/
//   `late` sont trois échantillons interpolés espacés d'une demi-période
//   symbole (`early = punctual − T/2`, `late = punctual + T/2`).
// * [`mueller_muller_ted`] — détecteur de Mueller-Müller, **piloté par
//   décision** : `e = prev_decision·curr_symbol − curr_decision·prev_symbol`,
//   à partir de deux symboles consécutifs et de leurs décisions démodulées
//   (ex. signe, pour une modulation binaire). Fourni comme brique
//   **indépendante** (pas intégrée à [`SymbolTimingLoop`], qui utilise
//   Gardner) : un appelant construisant son propre synchroniseur piloté par
//   décision peut la combiner à sa propre logique de décision/interpolation.
//
// ## [`SymbolTimingLoop`] — boucle complète (Gardner)
//
// Combine un interpolateur linéaire sur une ligne à retard circulaire (même
// technique que [`super::adaptive`]) et un [`super::pll::PiLoopFilter`]
// (réutilisé tel quel) pilotant un compte à rebours d'échantillons jusqu'au
// prochain symbole — le pendant, en unités d'échantillons plutôt que de
// radians, de [`super::pll::Nco`].
//
// **`samples_per_symbol` est un entier (`usize`), pas un scalaire
// générique `T`.** [`RealScalar`] n'expose aucune opération de partie
// entière (`floor`/`trunc`) : décomposer un décalage fractionnaire
// quelconque en indice de tampon + reste generique en `T` exigerait une
// telle opération, absente du trait. Fixer l'espacement des prises
// early/punctual/late à un nombre entier d'échantillons (choix de
// suréchantillonnage classique — 2×, 4×, 8× — décidé à l'étage
// d'acquisition) contourne entièrement le problème : les décalages
// entiers (`0`, `sps/2`, `sps`) sont des `usize` connus, et **seule** la
// partie difficile — le déphasage **sous-échantillon** (`mu`) — reste
// suivie en `T`, avec toute la précision (et le déterminisme virgule
// fixe) du reste du crate.
//
// ## Anti-débordement du compte à rebours
//
// Le compte à rebours (`strobe_countdown`, en unités d'échantillons) est
// décrémenté de `1` à chaque échantillon ; un symbole est émis dès qu'il
// devient `≤ 0`, et le **dépassement** (`overshoot = −strobe_countdown`,
// dans `[0, 1)` tant que `sps ≥ 1` et la correction reste petite) est
// **conservé** lors de la remise à niveau (`+= sps + correction`) — même
// technique « accumulateur avec retenue » qu'un compteur de phase NCO
// classique, mais en unités d'échantillons plutôt que de radians.

use core::ops::Div;

use crate::fixed::{NumericScalar, RealScalar};

use super::pll::PiLoopFilter;

/// Détecteur d'erreur de timing de Gardner (cf. en-tête de module) :
/// `(late − early)·punctual`. `early`/`punctual`/`late` : échantillons
/// interpolés à `−T/2`, `0`, `+T/2` autour du point de décision courant.
#[inline]
pub fn gardner_ted<T: NumericScalar>(early: T, punctual: T, late: T) -> T {
    (late - early) * punctual
}

/// Détecteur d'erreur de timing de Mueller-Müller (cf. en-tête de module) :
/// `prev_decision·curr_symbol − curr_decision·prev_symbol`.
#[inline]
pub fn mueller_muller_ted<T: NumericScalar>(
    prev_symbol: T,
    curr_symbol: T,
    prev_decision: T,
    curr_decision: T,
) -> T {
    prev_decision * curr_symbol - curr_decision * prev_symbol
}

/// Interpolation linéaire `a + (b − a)·t`.
#[inline]
fn lerp<T: NumericScalar>(a: T, b: T, t: T) -> T {
    a + (b - a) * t
}

/// Fenêtre glissante circulaire : insère `x`, renvoie `[x[n], x[n−1], …,
/// x[n−N+1]]` (indice 0 = le plus récent), avance la position. Même
/// technique que [`super::adaptive::slide_window`] (privée à ce module,
/// donc reproduite localement plutôt que partagée).
#[inline]
fn slide_window<T: NumericScalar, const N: usize>(
    delay: &mut [T; N],
    pos: &mut usize,
    x: T,
) -> [T; N] {
    delay[*pos] = x;
    let mut window = [T::zero(); N];
    let mut idx = *pos;
    for w in &mut window
    {
        *w = delay[idx];
        idx = if idx == 0 { N - 1 } else { idx - 1 };
    }
    *pos = if *pos + 1 == N { 0 } else { *pos + 1 };
    window
}

/// Boucle de récupération d'horloge symbole par détecteur de Gardner (cf.
/// en-tête de module). `N` (tampon circulaire, comme [`super::Fir<T, N>`])
/// doit satisfaire `N ≥ samples_per_symbol + 2` (vérifié à la construction).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymbolTimingLoop<T, const N: usize> {
    history: [T; N],
    pos: usize,
    filled: usize,
    sps: usize,
    sps_t: T,
    half_sps: usize,
    strobe_countdown: T,
    filter: PiLoopFilter<T>,
    last_mu: T,
    last_correction: T,
}

impl<T: NumericScalar, const N: usize> SymbolTimingLoop<T, N> {
    fn from_filter(samples_per_symbol: usize, filter: PiLoopFilter<T>) -> Self {
        assert!(
            samples_per_symbol >= 2,
            "SymbolTimingLoop: samples_per_symbol doit être ≥ 2"
        );
        assert!(
            N >= samples_per_symbol + 2,
            "SymbolTimingLoop: tampon N={N} trop petit pour samples_per_symbol={samples_per_symbol} (N ≥ samples_per_symbol + 2 requis)"
        );
        let mut sps_t = T::zero();
        for _ in 0..samples_per_symbol
        {
            sps_t = sps_t + T::one();
        }
        Self {
            history: [T::zero(); N],
            pos: 0,
            filled: 0,
            sps: samples_per_symbol,
            sps_t,
            half_sps: samples_per_symbol / 2,
            strobe_countdown: sps_t,
            filter,
            last_mu: T::zero(),
            last_correction: T::zero(),
        }
    }

    /// Construit depuis un [`PiLoopFilter`] déjà conçu (gains bruts connus).
    #[inline]
    pub fn new_with_filter(samples_per_symbol: usize, filter: PiLoopFilter<T>) -> Self {
        Self::from_filter(samples_per_symbol, filter)
    }

    /// Remet la ligne à retard, le compte à rebours et le filtre de boucle à
    /// leur état initial (redémarre l'acquisition).
    #[inline]
    pub fn reset(&mut self) {
        self.history = [T::zero(); N];
        self.pos = 0;
        self.filled = 0;
        self.strobe_countdown = self.sps_t;
        self.filter.reset();
        self.last_mu = T::zero();
        self.last_correction = T::zero();
    }

    /// Déphasage sous-échantillon courant (`mu`, dans `[0, 1)` échantillons),
    /// mesuré au dernier symbole émis.
    #[inline]
    pub fn mu(&self) -> T {
        self.last_mu
    }

    /// Estimation courante du nombre d'échantillons par symbole
    /// (`samples_per_symbol` nominal + dernière correction de boucle).
    #[inline]
    pub fn samples_per_symbol_estimate(&self) -> T {
        self.sps_t + self.last_correction
    }

    /// Traite un échantillon : renvoie `Some(symbole interpolé)` sur un
    /// « strobe » (une fois tous les `≈ samples_per_symbol` échantillons),
    /// `None` sinon (y compris pendant le remplissage initial de la ligne à
    /// retard).
    #[inline]
    pub fn step(&mut self, x: T) -> Option<T> {
        let window = slide_window(&mut self.history, &mut self.pos, x);
        if self.filled < N
        {
            self.filled += 1;
            return None;
        }

        self.strobe_countdown = self.strobe_countdown - T::one();
        if self.strobe_countdown > T::zero()
        {
            return None;
        }

        let overshoot = -self.strobe_countdown; // dans [0, 1), cf. en-tête de module.
        self.last_mu = overshoot;

        let late = lerp(window[0], window[1], overshoot);
        let punctual = lerp(window[self.half_sps], window[self.half_sps + 1], overshoot);
        let early = lerp(window[self.sps], window[self.sps + 1], overshoot);

        let error = gardner_ted(early, punctual, late);
        let correction = self.filter.step(error);
        self.last_correction = correction;
        self.strobe_countdown = self.strobe_countdown + self.sps_t + correction;

        Some(punctual)
    }
}

impl<T: RealScalar + Div<Output = T>, const N: usize> SymbolTimingLoop<T, N> {
    /// Construction « cookbook » : conçoit le filtre de boucle interne
    /// depuis `loop_bandwidth` et `damping` (cf.
    /// [`super::pll::PiLoopFilter::design`]), exprimés comme fraction du
    /// rythme d'échantillonnage (le filtre opère directement en unités
    /// d'échantillons, donc `sample_rate = 1` dans la formule de Gardner —
    /// pas de fréquence physique distincte ici, contrairement à [`Pll`]
    /// (porteuse) qui a une fréquence centrale séparée du rythme
    /// d'échantillonnage).
    ///
    /// [`Pll`]: super::pll::Pll
    #[inline]
    pub fn new(samples_per_symbol: usize, loop_bandwidth: T, damping: T) -> Self {
        let filter = PiLoopFilter::design(loop_bandwidth, damping, T::one());
        Self::from_filter(samples_per_symbol, filter)
    }
}
