//! Durée de vie des roulements — durée nominale de base **ISO 281** (L10) et
//! charge dynamique équivalente, dimensionnement de base d'un roulement sous
//! charge combinée radiale/axiale.
//!
//! La durée nominale de base (fiabilité 90 %) s'écrit :
//!
//! ```text
//! L10 = (C / P)^p          (millions de tours)
//! ```
//!
//! `C` charge dynamique de base (catalogue, N), `P` charge dynamique
//! équivalente (N), `p` exposant de durée : `3` pour les roulements à billes,
//! `10/3` pour les roulements à rouleaux. En heures, à `n` tr/min :
//!
//! ```text
//! L10h = (10⁶ / (60·n)) · L10
//! ```
//!
//! La charge équivalente combine radial et axial :
//!
//! ```text
//! P = X·Fr + Y·Fa
//! ```
//!
//! et la durée corrigée (ISO 281) module la fiabilité (`a₁`) et les conditions
//! (`a_ISO`) : `Lnm = a₁ · a_ISO · L10`.
//!
//! **Limite honnête** : `C`, ainsi que les facteurs `X`, `Y` (dépendant de la
//! géométrie et du rapport `Fa/Fr`), et le facteur `a_ISO` (propreté du
//! lubrifiant, contamination, limite de fatigue) sont des données de catalogue
//! ou d'un calcul détaillé que l'appelant fournit — ce module ne les invente
//! pas. Il assemble la durée de vie à partir de ces entrées.

/// Type de roulement, fixant l'exposant de durée `p` de la loi ISO 281.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BearingType {
    /// Roulement à billes : `p = 3`.
    Ball,
    /// Roulement à rouleaux : `p = 10/3`.
    Roller,
}

impl BearingType {
    /// Exposant de durée `p` de la loi `L10 = (C/P)^p`.
    pub fn life_exponent(self) -> f64 {
        match self
        {
            BearingType::Ball => 3.0,
            BearingType::Roller => 10.0 / 3.0,
        }
    }
}

/// Charge dynamique équivalente `P = X·Fr + Y·Fa` (N), à partir des charges
/// radiale `fr` et axiale `fa` (N) et des facteurs `x`, `y` (catalogue).
pub fn equivalent_dynamic_load(fr_n: f64, fa_n: f64, x: f64, y: f64) -> f64 {
    x * fr_n + y * fa_n
}

/// Durée nominale de base `L10` en **millions de tours** :
/// `L10 = (C/P)^p`, charge dynamique de base `c` (N) et charge équivalente
/// `p_load` (N).
///
/// Panique si `c <= 0` ou `p_load <= 0`.
pub fn basic_rating_life_revs(c_n: f64, p_load_n: f64, bearing: BearingType) -> f64 {
    assert!(
        c_n > 0.0 && p_load_n > 0.0,
        "les charges C et P doivent être strictement positives"
    );
    (c_n / p_load_n).powf(bearing.life_exponent())
}

/// Durée nominale de base en **heures** : `L10h = (10⁶/(60·n)) · L10`, à partir
/// de `l10` (millions de tours) et de la vitesse `n` (tr/min).
///
/// Panique si `n <= 0`.
pub fn basic_rating_life_hours(l10_mrev: f64, n_rpm: f64) -> f64 {
    assert!(
        n_rpm > 0.0,
        "la vitesse de rotation doit être strictement positive"
    );
    1.0e6 / (60.0 * n_rpm) * l10_mrev
}

/// Niveau de fiabilité normalisé (ISO 281) et son facteur de correction `a₁`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reliability {
    /// 90 % — durée nominale de base (`a₁ = 1`).
    R90,
    /// 95 % (`a₁ = 0,64`).
    R95,
    /// 96 % (`a₁ = 0,55`).
    R96,
    /// 97 % (`a₁ = 0,47`).
    R97,
    /// 98 % (`a₁ = 0,37`).
    R98,
    /// 99 % (`a₁ = 0,25`).
    R99,
}

impl Reliability {
    /// Facteur de fiabilité `a₁` de la durée corrigée (ISO 281:2007, tableau 1).
    pub fn a1(self) -> f64 {
        match self
        {
            Reliability::R90 => 1.0,
            Reliability::R95 => 0.64,
            Reliability::R96 => 0.55,
            Reliability::R97 => 0.47,
            Reliability::R98 => 0.37,
            Reliability::R99 => 0.25,
        }
    }
}

/// Durée de vie corrigée `Lnm = a₁ · a_ISO · L10` (mêmes unités que `l10`).
///
/// `a_iso` est le facteur de conditions de fonctionnement (lubrification,
/// contamination, limite de fatigue), fourni par l'appelant. Panique si
/// `a_iso <= 0`.
pub fn adjusted_rating_life(l10: f64, reliability: Reliability, a_iso: f64) -> f64 {
    assert!(
        a_iso > 0.0,
        "le facteur a_ISO doit être strictement positif"
    );
    reliability.a1() * a_iso * l10
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ball_bearing_life_uses_cubic_exponent() {
        // C=50 kN, P=10 kN, billes → L10 = 5³ = 125 millions de tours.
        assert_relative_eq!(
            basic_rating_life_revs(50_000.0, 10_000.0, BearingType::Ball),
            125.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn roller_bearing_uses_10_over_3_exponent() {
        // billes^3 vs rouleaux^(10/3) : les rouleaux vivent plus longtemps à charge égale.
        let ball = basic_rating_life_revs(50_000.0, 10_000.0, BearingType::Ball);
        let roller = basic_rating_life_revs(50_000.0, 10_000.0, BearingType::Roller);
        assert!(roller > ball);
        assert_relative_eq!(roller, 5f64.powf(10.0 / 3.0), epsilon = 1e-9);
    }

    #[test]
    fn life_in_hours_matches_the_iso281_relation() {
        // L10=125 Mrev à 1500 tr/min → (1e6/90000)·125 ≈ 1388,9 h.
        assert_relative_eq!(
            basic_rating_life_hours(125.0, 1500.0),
            1.0e6 / 90_000.0 * 125.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn equivalent_load_combines_radial_and_axial() {
        // Fr=8000, Fa=4000, X=0,56, Y=1,5 → P = 4480 + 6000 = 10480 N.
        assert_relative_eq!(
            equivalent_dynamic_load(8000.0, 4000.0, 0.56, 1.5),
            10_480.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn higher_reliability_shortens_rated_life() {
        // a₁ décroît de 90 % à 99 % : la durée exigée à haute fiabilité est plus courte.
        let l90 = adjusted_rating_life(125.0, Reliability::R90, 1.0);
        let l99 = adjusted_rating_life(125.0, Reliability::R99, 1.0);
        assert_relative_eq!(l90, 125.0, epsilon = 1e-9);
        assert_relative_eq!(l99, 0.25 * 125.0, epsilon = 1e-9);
        assert!(l99 < l90);
    }

    #[test]
    #[should_panic(expected = "C et P")]
    fn zero_load_panics() {
        basic_rating_life_revs(50_000.0, 0.0, BearingType::Ball);
    }
}
