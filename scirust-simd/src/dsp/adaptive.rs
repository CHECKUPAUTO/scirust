// scirust-simd/src/dsp/adaptive.rs
//
// # Filtres adaptatifs `Lms<T, N>` / `Nlms<T, N>` / `Rls<T, N>`
//
// Tous les filtres du module [`super`] vus jusqu'ici ([`super::Biquad`],
// [`super::Fir`], [`super::BiquadCascade`]) ont des coefficients **conçus une
// fois pour toutes** (à partir d'une fréquence de coupure, d'un facteur de
// qualité…). Un filtre **adaptatif** fait l'inverse : il n'a **aucun modèle a
// priori** du système à identifier — ses coefficients s'ajustent
// échantillon par échantillon à partir d'un signal d'erreur, jusqu'à
// converger vers le filtre optimal. Cas d'usage classiques : annulation
// d'écho, identification de système, égalisation de canal, débruitage
// adaptatif.
//
// **Générique sur le scalaire** comme le reste de `dsp` : la même
// implémentation sert `f32`/`f64` et la virgule fixe déterministe
// (`FixedI32<FRAC>`) — un filtre adaptatif en virgule fixe converge vers
// les **mêmes bits** sur toute architecture, propriété précieuse pour
// reproduire un résultat d'apprentissage en ligne.
//
// ## [`Lms`] — Least Mean Squares
//
// Le plus simple et le moins coûteux (`O(N)` par échantillon, uniquement
// l'anneau [`NumericScalar`], aucune division). Prédit `y = wᵀ·x`, calcule
// l'erreur `e = d − y` (`d` = signal désiré), puis met à jour chaque poids
// par descente de gradient stochastique : `wₖ ← wₖ + μ·e·x[n−k]`. Le pas `μ`
// est fixe et doit être choisi assez petit pour garantir la convergence
// (borne classique : `0 < μ < 2/(N·E[x²])`) — trop grand, le filtre diverge ;
// trop petit, la convergence est lente.
//
// ## [`Nlms`] — Normalized LMS
//
// Même récurrence que [`Lms`], mais le pas est normalisé par l'énergie de la
// fenêtre d'entrée courante : `μ_eff = μ/(‖x‖² + ε)` (`ε` évite la division
// par zéro sur un signal nul). Corrige le principal défaut de LMS — un pas
// fixe mal choisi si l'amplitude du signal varie — au prix d'une division par
// échantillon (`O(N)`, requiert seulement [`NumericScalar`] + division réelle,
// pas besoin de racine ni de transcendantes). Converge plus vite et plus
// uniformément que LMS sur des signaux d'amplitude non stationnaire.
//
// ## [`Rls`] — Recursive Least Squares
//
// Converge beaucoup plus vite que LMS/NLMS (en variance, il approche la
// solution des moindres carrés exacte dès que la fenêtre a été vue une fois)
// au prix d'un coût `O(N²)` par échantillon : il maintient la matrice de
// covariance inverse `P` (`N×N`) et la met à jour par la formule de
// Sherman-Morrison (rang 1), évitant toute inversion de matrice explicite à
// chaque pas :
//
// ```text
//   k = P·x / (λ + xᵀ·P·x)     (gain de Kalman)
//   e = d − wᵀ·x
//   w ← w + k·e
//   P ← (P − k·(xᵀ·P)) / λ
// ```
//
// `λ` (`0 < λ ≤ 1`) est le facteur d'oubli : `λ = 1` pondère également tout
// l'historique ; `λ < 1` privilégie les échantillons récents (utile si le
// système identifié varie lentement dans le temps). `P` est initialisée à
// `δ⁻¹·I` (`δ` petit et positif) : une covariance initiale large traduit une
// confiance initiale faible dans les poids (tous nuls).
//
// ## Division réelle, pas d'inverse mise en cache
//
// [`Nlms`] et [`Rls`] divisent par des quantités qui ne sont **jamais** des
// puissances de deux (`‖x‖² + ε`, `λ + xᵀ·P·x`, `λ`) : comme partout ailleurs
// dans le crate (cf. [`super::biquad::butterworth_qs`],
// [`crate::geometry::DualQuaternion::pow`]), on utilise l'opérateur `/`
// (`Div<Output = T>`) directement sur chaque valeur, jamais un `.recip()` mis
// en cache puis multiplié — cette dernière voie cumule deux arrondis
// (l'inverse, puis le produit) au lieu d'un seul.

use core::ops::Div;

use crate::fixed::NumericScalar;

/// Fenêtre d'entrée circulaire commune aux trois filtres : insère `x`,
/// renvoie la fenêtre `[x[n], x[n−1], …, x[n−N+1]]` dans l'ordre naturel
/// (indice `0` = échantillon le plus récent), et avance la position.
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

/// Filtre adaptatif LMS (Least Mean Squares), `N` poids, ligne à retard
/// circulaire sans allocation (cf. en-tête de module).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lms<T, const N: usize> {
    weights: [T; N],
    delay: [T; N],
    pos: usize,
    mu: T,
}

impl<T: NumericScalar, const N: usize> Lms<T, N> {
    /// Construit avec des poids initiaux nuls et un pas d'adaptation `mu`.
    #[inline]
    pub fn new(mu: T) -> Self {
        const {
            assert!(N >= 1, "Lms: au moins un poids");
        }
        Self {
            weights: [T::zero(); N],
            delay: [T::zero(); N],
            pos: 0,
            mu,
        }
    }

    /// Remet les poids et la ligne à retard à zéro (redémarre l'adaptation).
    #[inline]
    pub fn reset(&mut self) {
        self.weights = [T::zero(); N];
        self.delay = [T::zero(); N];
        self.pos = 0;
    }

    /// Poids courants (coefficients du filtre appris jusqu'ici).
    #[inline]
    pub fn weights(&self) -> &[T; N] {
        &self.weights
    }

    /// Traite un échantillon : insère `x`, prédit `y = wᵀ·x`, met à jour les
    /// poids par descente de gradient stochastique à partir de l'erreur
    /// `desired − y`. Renvoie `(y, erreur)`.
    #[inline]
    pub fn update(&mut self, x: T, desired: T) -> (T, T) {
        let window = slide_window(&mut self.delay, &mut self.pos, x);

        let mut y = T::zero();
        for (&w, &xi) in self.weights.iter().zip(&window)
        {
            y = y + w * xi;
        }
        let error = desired - y;

        let step = self.mu * error;
        for (w, &xi) in self.weights.iter_mut().zip(&window)
        {
            *w = *w + step * xi;
        }
        (y, error)
    }
}

/// Filtre adaptatif NLMS (Normalized LMS), `N` poids — comme [`Lms`] mais le
/// pas est normalisé par l'énergie de la fenêtre courante (cf. en-tête de
/// module).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nlms<T, const N: usize> {
    weights: [T; N],
    delay: [T; N],
    pos: usize,
    mu: T,
    eps: T,
}

impl<T: NumericScalar + Div<Output = T>, const N: usize> Nlms<T, N> {
    /// Construit avec des poids initiaux nuls, un pas `mu` (typiquement
    /// `0 < mu < 2` pour la stabilité) et une constante `eps > 0` évitant la
    /// division par zéro sur un signal nul.
    #[inline]
    pub fn new(mu: T, eps: T) -> Self {
        const {
            assert!(N >= 1, "Nlms: au moins un poids");
        }
        Self {
            weights: [T::zero(); N],
            delay: [T::zero(); N],
            pos: 0,
            mu,
            eps,
        }
    }

    /// Remet les poids et la ligne à retard à zéro (redémarre l'adaptation).
    #[inline]
    pub fn reset(&mut self) {
        self.weights = [T::zero(); N];
        self.delay = [T::zero(); N];
        self.pos = 0;
    }

    /// Poids courants.
    #[inline]
    pub fn weights(&self) -> &[T; N] {
        &self.weights
    }

    /// Traite un échantillon (cf. [`Lms::update`]), avec un pas normalisé
    /// `μ/(‖x‖² + ε)` au lieu d'un pas fixe.
    #[inline]
    pub fn update(&mut self, x: T, desired: T) -> (T, T) {
        let window = slide_window(&mut self.delay, &mut self.pos, x);

        let mut y = T::zero();
        let mut energy = T::zero();
        for &xi in &window
        {
            energy = energy + xi * xi;
        }
        for (&w, &xi) in self.weights.iter().zip(&window)
        {
            y = y + w * xi;
        }
        let error = desired - y;

        let step = (self.mu * error) / (energy + self.eps);
        for (w, &xi) in self.weights.iter_mut().zip(&window)
        {
            *w = *w + step * xi;
        }
        (y, error)
    }
}

/// Filtre adaptatif RLS (Recursive Least Squares), `N` poids — convergence
/// bien plus rapide que [`Lms`]/[`Nlms`] au prix d'un coût `O(N²)` par
/// échantillon (mise à jour de la covariance inverse, cf. en-tête de module).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rls<T, const N: usize> {
    weights: [T; N],
    delay: [T; N],
    pos: usize,
    lambda: T,
    delta: T,
    p: [[T; N]; N],
}

impl<T: NumericScalar + Div<Output = T>, const N: usize> Rls<T, N> {
    /// Construit avec des poids initiaux nuls, un facteur d'oubli `lambda`
    /// (`0 < lambda ≤ 1`) et une covariance initiale `P₀ = δ⁻¹·I` (`delta`
    /// petit et positif — plus petit, plus grande confiance initiale accordée
    /// aux premiers échantillons).
    #[inline]
    pub fn new(lambda: T, delta: T) -> Self {
        const {
            assert!(N >= 1, "Rls: au moins un poids");
        }
        Self {
            weights: [T::zero(); N],
            delay: [T::zero(); N],
            pos: 0,
            lambda,
            delta,
            p: Self::initial_p(delta),
        }
    }

    #[inline]
    fn initial_p(delta: T) -> [[T; N]; N] {
        let inv_delta = T::one() / delta;
        let mut p = [[T::zero(); N]; N];
        for (i, row) in p.iter_mut().enumerate()
        {
            row[i] = inv_delta;
        }
        p
    }

    /// Remet les poids, la ligne à retard et la covariance à leur état
    /// initial (redémarre l'adaptation).
    #[inline]
    pub fn reset(&mut self) {
        self.weights = [T::zero(); N];
        self.delay = [T::zero(); N];
        self.pos = 0;
        self.p = Self::initial_p(self.delta);
    }

    /// Poids courants.
    #[inline]
    pub fn weights(&self) -> &[T; N] {
        &self.weights
    }

    /// Covariance inverse courante `P` (lecture seule) — permet d'inspecter
    /// la confiance restante par direction de l'espace des poids.
    #[inline]
    pub fn covariance(&self) -> &[[T; N]; N] {
        &self.p
    }

    /// Traite un échantillon : insère `x`, prédit `y = wᵀ·x`, met à jour les
    /// poids et la covariance inverse par la récurrence RLS (cf. en-tête de
    /// module). Renvoie `(y, erreur)`.
    #[inline]
    pub fn update(&mut self, x: T, desired: T) -> (T, T) {
        let u = slide_window(&mut self.delay, &mut self.pos, x);

        let mut y = T::zero();
        for (&w, &ui) in self.weights.iter().zip(&u)
        {
            y = y + w * ui;
        }
        let error = desired - y;

        // p_u = P·u
        let mut p_u = [T::zero(); N];
        for (i, row) in self.p.iter().enumerate()
        {
            let mut acc = T::zero();
            for (&pij, &uj) in row.iter().zip(&u)
            {
                acc = acc + pij * uj;
            }
            p_u[i] = acc;
        }

        // denom = lambda + u^T P u
        let mut u_p_u = T::zero();
        for (&ui, &pui) in u.iter().zip(&p_u)
        {
            u_p_u = u_p_u + ui * pui;
        }
        let denom = self.lambda + u_p_u;

        // gain = p_u / denom (gain de Kalman)
        let mut gain = [T::zero(); N];
        for (g, &pui) in gain.iter_mut().zip(&p_u)
        {
            *g = pui / denom;
        }

        for (w, &g) in self.weights.iter_mut().zip(&gain)
        {
            *w = *w + g * error;
        }

        // P = (P - gain * (u^T P)) / lambda ; u^T P = p_uᵀ (P symétrique).
        for (i, row) in self.p.iter_mut().enumerate()
        {
            for (j, cell) in row.iter_mut().enumerate()
            {
                *cell = (*cell - gain[i] * p_u[j]) / self.lambda;
            }
        }

        (y, error)
    }
}
