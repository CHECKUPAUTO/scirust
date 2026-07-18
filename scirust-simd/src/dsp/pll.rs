// scirust-simd/src/dsp/pll.rs
//
// # Boucle à verrouillage de phase (PLL) générique `Pll<T>`
//
// Contrairement aux filtres du reste de [`super`] — conçus une fois pour
// toutes ([`super::Biquad`]/[`super::Fir`]) ou appris en ligne par descente
// de gradient sur une erreur **instantanée** ([`super::adaptive`]) — une PLL
// est une boucle de rétroaction qui **suit une phase/fréquence porteuse
// inconnue et potentiellement variable** : un oscillateur commandé
// numériquement ([`Nco`]) génère une référence locale, un détecteur de phase
// compare cette référence à l'entrée, et un filtre de boucle
// ([`PiLoopFilter`]) intègre l'erreur pour corriger la fréquence de l'OCN —
// jusqu'à ce que la référence locale « verrouille » sur l'entrée. Cas
// d'usage classiques : récupération de porteuse/horloge, démodulation FM,
// synthèse de fréquence.
//
// **Générique sur le scalaire** comme le reste de `dsp` : la même
// implémentation sert `f32`/`f64` et la virgule fixe déterministe
// (`FixedI32<FRAC>`) — une PLL en virgule fixe reproduit la **même**
// trajectoire de verrouillage bit à bit sur toute architecture.
//
// ## [`Nco`] — oscillateur commandé numériquement
//
// Maintient un accumulateur de phase (radians) et produit `(sin, cos)` à
// chaque [`Nco::tick`], puis avance la phase de son incrément courant
// (radians/échantillon, réglable en ligne par [`Nco::set_frequency`]).
//
// **Rembobinage explicite de la phase, pas un débordement d'accumulateur.**
// L'astuce matérielle classique (« laisser l'accumulateur entier déborder »)
// ne fonctionne qu'en arithmétique **enveloppante** ; elle serait invisible
// et fausse pour `f32`/`f64` (aucun débordement, la phase croîtrait sans
// borne, perdant toute précision après quelques millions d'échantillons) —
// et [`FixedI32`](crate::fixed::FixedI32) elle-même n'expose pas cette
// sémantique via [`RealScalar`]. [`Nco::tick`] rembobine donc explicitement
// dans `(−π, π]` par une simple comparaison/soustraction (jamais une boucle
// non bornée : l'incrément par échantillon est toujours petit devant `2π`,
// un seul rembobinage suffit).
//
// ## [`PiLoopFilter`] — filtre de boucle proportionnel-intégral
//
// [`PiLoopFilter::design`] calcule les gains `(Kp, Ki)` depuis des
// paramètres physiques — bande passante de boucle et facteur
// d'amortissement — plutôt que des gains bruts, même esprit « cookbook » que
// [`super::Biquad::lowpass`]. Formule classique (Gardner, *Phaselock
// Techniques*) : soit `θ = (bw/fs) / (ζ + 1/(4ζ))` la bande passante
// normalisée. Alors `Kp = 4ζθ/(1+2ζθ+θ²)`, `Ki = 4θ²/(1+2ζθ+θ²)`.
//
// **Anti-saturation de l'intégrateur** : rien n'empêche l'intégrateur de
// dériver largement pendant l'acquisition (avant verrouillage, l'erreur de
// phase peut rester grande pendant de nombreux échantillons) — même
// précaution que documentée pour un PID classique. [`PiLoopFilter::step`]
// ne clampe **pas** lui-même (aucune borne universelle n'existe : elle
// dépend de la plage utile de fréquence de l'application) ; un appelant
// avec des contraintes de plage doit clamper la sortie de [`Pll::process`]/
// [`Pll::process_quadrature`] avant de la réinjecter si nécessaire — limite
// documentée plutôt qu'une lacune silencieuse.
//
// ## [`Pll`] — boucle complète
//
// Combine [`Nco`] + [`PiLoopFilter`] autour d'une fréquence centrale
// (`center_freq`, Hz) : la fréquence de l'OCN à l'échantillon `n` est
// `center_freq_per_sample + correction[n]`, `correction` étant la sortie du
// filtre de boucle. Deux détecteurs de phase :
//
// * [`Pll::process`] — entrée **réelle** (porteuse `A·cos(ωt+φ)`) : produit
//   d'entrée par la référence en quadrature de l'OCN, standard PLL/Costas.
//   Le terme à `2ω` du produit **n'est pas filtré séparément** — il est
//   supposé atténué par la dynamique propre de la boucle (bande passante
//   très inférieure à la fréquence porteuse), approximation classique valide
//   **seulement** si `loop_bandwidth ≪ center_freq` ; hors de ce régime, une
//   section passe-bas explicite ([`super::Biquad::lowpass`]) serait
//   nécessaire — non ajoutée ici pour rester un module ciblé.
// * [`Pll::process_quadrature`] — entrée **complexe en bande de base**
//   (`I + j·Q`, déjà démodulée) : détecteur de phase par `atan2`, exact et
//   linéaire même loin du verrouillage (pas de terme battant à filtrer),
//   standard en récupération de porteuse numérique.
//
// [`Pll::locked`] rapporte si la dernière erreur de phase observée est sous
// un seuil donné — indicateur de verrouillage simple, pas un détecteur de
// verrouillage à hystérésis (laissé à l'appelant si nécessaire).

use core::ops::Div;

use crate::fixed::{NumericScalar, RealScalar};

/// Rembobine `phase` dans `(−π, π]` par une comparaison/soustraction unique
/// (cf. en-tête de module : jamais un débordement d'accumulateur implicite).
#[inline]
fn wrap_phase<T: RealScalar>(phase: T) -> T {
    let pi = T::pi();
    let two_pi = T::from_i32(2) * pi;
    if phase > pi
    {
        phase - two_pi
    }
    else if phase <= -pi
    {
        phase + two_pi
    }
    else
    {
        phase
    }
}

/// Oscillateur commandé numériquement (cf. en-tête de module).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nco<T> {
    phase: T,
    freq_per_sample: T,
}

impl<T: RealScalar> Nco<T> {
    /// Construit avec une phase initiale nulle et un incrément
    /// `freq_per_sample` (radians/échantillon).
    #[inline]
    pub fn new(freq_per_sample: T) -> Self {
        Self {
            phase: T::zero(),
            freq_per_sample,
        }
    }

    /// Phase courante (radians, dans `(−π, π]`).
    #[inline]
    pub fn phase(&self) -> T {
        self.phase
    }

    /// Incrément de phase courant (radians/échantillon).
    #[inline]
    pub fn frequency(&self) -> T {
        self.freq_per_sample
    }

    /// Change l'incrément de phase (radians/échantillon) — effectif à partir
    /// du prochain [`Nco::tick`].
    #[inline]
    pub fn set_frequency(&mut self, freq_per_sample: T) {
        self.freq_per_sample = freq_per_sample;
    }

    /// Renvoie `(sin, cos)` de la phase **courante**, puis l'avance de
    /// [`Nco::frequency`] (rembobinée dans `(−π, π]`).
    #[inline]
    pub fn tick(&mut self) -> (T, T) {
        let out = (self.phase.sin(), self.phase.cos());
        self.phase = wrap_phase(self.phase + self.freq_per_sample);
        out
    }
}

/// Filtre de boucle proportionnel-intégral (cf. en-tête de module).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PiLoopFilter<T> {
    kp: T,
    ki: T,
    integrator: T,
}

impl<T: NumericScalar> PiLoopFilter<T> {
    /// Construit depuis des gains bruts déjà connus (intégrateur à zéro).
    #[inline]
    pub fn new(kp: T, ki: T) -> Self {
        Self {
            kp,
            ki,
            integrator: T::zero(),
        }
    }

    /// Remet l'intégrateur à zéro.
    #[inline]
    pub fn reset(&mut self) {
        self.integrator = T::zero();
    }

    /// Intégrateur courant (lecture seule).
    #[inline]
    pub fn integrator(&self) -> T {
        self.integrator
    }

    /// Gain proportionnel courant.
    #[inline]
    pub fn kp(&self) -> T {
        self.kp
    }

    /// Gain intégral courant.
    #[inline]
    pub fn ki(&self) -> T {
        self.ki
    }

    /// Applique le filtre à une erreur de phase : avance l'intégrateur de
    /// `Ki·erreur`, renvoie `Kp·erreur + intégrateur` (après avance).
    #[inline]
    pub fn step(&mut self, phase_error: T) -> T {
        self.integrator = self.integrator + self.ki * phase_error;
        self.kp * phase_error + self.integrator
    }
}

impl<T: RealScalar + Div<Output = T>> PiLoopFilter<T> {
    /// Conception « cookbook » depuis la bande passante de boucle
    /// `loop_bandwidth` (Hz), le facteur d'amortissement `damping`
    /// (`≈ 1/√2` pour une réponse critique usuelle) et la fréquence
    /// d'échantillonnage `sample_rate` (Hz) — cf. en-tête de module pour la
    /// formule.
    #[inline]
    pub fn design(loop_bandwidth: T, damping: T, sample_rate: T) -> Self {
        let one = T::one();
        let two = T::from_i32(2);
        let four = T::from_i32(4);
        let theta = (loop_bandwidth / sample_rate) / (damping + (four * damping).recip());
        let denom = (one + two * damping * theta) + theta * theta;
        let kp = (four * damping * theta) / denom;
        let ki = (four * theta * theta) / denom;
        Self::new(kp, ki)
    }
}

/// Boucle à verrouillage de phase complète (cf. en-tête de module).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pll<T> {
    nco: Nco<T>,
    filter: PiLoopFilter<T>,
    center_freq_per_sample: T,
    last_error: T,
}

impl<T: RealScalar + Div<Output = T>> Pll<T> {
    /// Construit une PLL autour de `center_freq` (Hz), échantillonnée à
    /// `sample_rate` (Hz), avec un filtre de boucle conçu depuis
    /// `loop_bandwidth` (Hz) et `damping` (cf. [`PiLoopFilter::design`]).
    #[inline]
    pub fn new(sample_rate: T, center_freq: T, loop_bandwidth: T, damping: T) -> Self {
        let two = T::from_i32(2);
        let center_freq_per_sample = (two * T::pi() * center_freq) / sample_rate;
        Self {
            nco: Nco::new(center_freq_per_sample),
            filter: PiLoopFilter::design(loop_bandwidth, damping, sample_rate),
            center_freq_per_sample,
            last_error: T::zero(),
        }
    }

    /// Remet l'OCN (à sa fréquence centrale) et le filtre de boucle à leur
    /// état initial (redémarre l'acquisition).
    #[inline]
    pub fn reset(&mut self) {
        self.nco = Nco::new(self.center_freq_per_sample);
        self.filter.reset();
        self.last_error = T::zero();
    }

    /// Traite un échantillon **réel** (cf. en-tête de module) : renvoie la
    /// référence en phase (`cos`) de l'OCN à cet échantillon.
    #[inline]
    pub fn process(&mut self, input: T) -> T {
        let (sin_ref, cos_ref) = self.nco.tick();
        let phase_error = input * sin_ref;
        self.last_error = phase_error;
        let correction = self.filter.step(phase_error);
        self.nco
            .set_frequency(self.center_freq_per_sample + correction);
        cos_ref
    }

    /// Traite un échantillon **complexe en bande de base** `i + j·q` (cf.
    /// en-tête de module) : renvoie l'erreur de phase (`atan2`) mesurée à cet
    /// échantillon.
    #[inline]
    pub fn process_quadrature(&mut self, i: T, q: T) -> T {
        let (sin_ref, cos_ref) = self.nco.tick();
        let real = i * cos_ref + q * sin_ref;
        let imag = q * cos_ref - i * sin_ref;
        let phase_error = imag.atan2(real);
        self.last_error = phase_error;
        let correction = self.filter.step(phase_error);
        self.nco
            .set_frequency(self.center_freq_per_sample + correction);
        phase_error
    }

    /// Fréquence courante estimée de l'OCN (radians/échantillon).
    #[inline]
    pub fn frequency_estimate(&self) -> T {
        self.nco.frequency()
    }

    /// Phase courante de l'OCN (radians, dans `(−π, π]`).
    #[inline]
    pub fn phase_estimate(&self) -> T {
        self.nco.phase()
    }

    /// `true` si `|dernière erreur de phase| < threshold` — indicateur de
    /// verrouillage simple (cf. en-tête de module).
    #[inline]
    pub fn locked(&self, threshold: T) -> bool {
        self.last_error.abs() < threshold
    }
}
