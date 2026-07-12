//! RDM — **réactions d'appui** et efforts internes maximaux de poutres isostatiques
//! sous cas de charge usuels (charge ponctuelle excentrée, charge répartie,
//! console).
//!
//! ```text
//! deux appuis, charge P en a (b = L − a)
//!   réactions   R1 = P·b/L   R2 = P·a/L
//!   moment max  M = P·a·b/L               (sous la charge)
//! deux appuis, charge répartie w
//!   réaction    R = w·L/2 (chaque appui)
//! console, charge en bout P     encastrement : R = P,  M = P·L
//! console, charge répartie w    encastrement : R = w·L, M = w·L²/2
//! ```
//!
//! `P` charge ponctuelle (N), `a` position depuis l'appui gauche (m), `L` portée
//! (m), `w` charge répartie (N/m). Les réactions équilibrent la charge ; le moment
//! maximal sert au dimensionnement en flexion.
//!
//! **Convention** : SI cohérent. **Limite honnête** : poutres **isostatiques**
//! (statiquement déterminées), charges dans le plan ; les flèches sont dans
//! [`crate::beams`] et [`crate::deflection_cases`], les cas hyperstatiques ne sont
//! pas résolus ici.

/// Réactions `(R1, R2)` — deux appuis, charge `P` à la distance `a` de l'appui
/// gauche : `R1 = P·(L−a)/L`, `R2 = P·a/L`.
///
/// Panique si `a` n'est pas dans `[0, L]` ou `L <= 0`.
pub fn ss_point_load_reactions(load: f64, a: f64, span: f64) -> (f64, f64) {
    assert!(
        span > 0.0 && (0.0..=span).contains(&a),
        "0 ≤ a ≤ L et L > 0 requis"
    );
    let b = span - a;
    (load * b / span, load * a / span)
}

/// Moment fléchissant maximal (sous la charge) `M = P·a·(L−a)/L`.
///
/// Panique si `L <= 0`. Se réduit à `P·L/4` pour une charge centrée.
pub fn ss_point_load_max_moment(load: f64, a: f64, span: f64) -> f64 {
    assert!(
        span > 0.0 && (0.0..=span).contains(&a),
        "0 ≤ a ≤ L et L > 0 requis"
    );
    load * a * (span - a) / span
}

/// Réaction à chaque appui — deux appuis, charge répartie `w` : `R = w·L/2`.
pub fn ss_udl_reaction(w: f64, span: f64) -> f64 {
    w * span / 2.0
}

/// Moment d'encastrement d'une **console** — charge en bout `P` : `M = P·L`.
pub fn cantilever_point_load_moment(load: f64, length: f64) -> f64 {
    load * length
}

/// Moment d'encastrement d'une **console** — charge répartie `w` : `M = w·L²/2`.
pub fn cantilever_udl_moment(w: f64, length: f64) -> f64 {
    w * length * length / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reactions_sum_to_load_and_balance_moments() {
        // P=1000 N à a=2 m sur L=5 m → R1=600, R2=400 (somme = charge).
        let (r1, r2) = ss_point_load_reactions(1000.0, 2.0, 5.0);
        assert_relative_eq!(r1, 600.0, epsilon = 1e-9);
        assert_relative_eq!(r2, 400.0, epsilon = 1e-9);
        assert_relative_eq!(r1 + r2, 1000.0, epsilon = 1e-9);
    }

    #[test]
    fn centred_load_recovers_pl_over_4() {
        // a = L/2 → M = P·L/4.
        assert_relative_eq!(
            ss_point_load_max_moment(1000.0, 2.5, 5.0),
            1000.0 * 5.0 / 4.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn udl_reaction_is_half_total() {
        // w=400 N/m sur 5 m → total 2000 N → 1000 N par appui.
        assert_relative_eq!(ss_udl_reaction(400.0, 5.0), 1000.0, epsilon = 1e-9);
    }

    #[test]
    fn cantilever_moments() {
        // Console : charge bout P=500 sur 2 m → M=1000 ; répartie w=300 → M = 300·4/2 = 600.
        assert_relative_eq!(
            cantilever_point_load_moment(500.0, 2.0),
            1000.0,
            epsilon = 1e-9
        );
        assert_relative_eq!(cantilever_udl_moment(300.0, 2.0), 600.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "0 ≤ a ≤ L")]
    fn load_outside_span_panics() {
        ss_point_load_reactions(1000.0, 6.0, 5.0);
    }
}
