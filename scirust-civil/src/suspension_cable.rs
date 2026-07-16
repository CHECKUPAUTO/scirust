//! **Câble porteur parabolique** sous charge uniformément répartie (statique
//! des câbles, ouvrages suspendus) : composante horizontale de la tension,
//! tension maximale aux appuis, longueur développée approchée du câble,
//! flèche déduite de la tension et réaction verticale d'appui.
//!
//! ```text
//! composante horizontale     H    = w·L² / (8·f)
//! tension maximale (appuis)  Tmax = √(H² + (w·L/2)²)
//! longueur développée        Lc   = L·(1 + (8/3)·(f/L)²)
//! flèche depuis la tension    f    = w·L² / (8·H)
//! réaction verticale d'appui  V    = w·L / 2
//! ```
//!
//! `w` = `load_per_length` charge uniformément répartie **en projection
//! horizontale** (N/m), `L` = `span` portée entre appuis de même niveau (m),
//! `f` = `sag` flèche à mi-portée (m), `H` = `horizontal_tension` composante
//! horizontale de la tension, **constante le long du câble** (N), `Tmax` =
//! tension maximale, atteinte **aux appuis** où la pente est maximale (N),
//! `Lc` = longueur développée du câble (m) et `V` = réaction verticale à
//! chaque appui (N).
//!
//! **Convention** : unités **SI** cohérentes **N, m** (avec `w` en N/m, donc
//! `w·L²/f` est en N, `w·L/2` en N et la longueur en m) ; les appuis sont de
//! **même niveau** et la flèche est comptée positive vers le bas. Types `f64`.
//!
//! **Limite honnête** : ce module traite un câble **parabolique** sous charge
//! **uniformément répartie en projection horizontale** (tablier de pont
//! suspendu), avec des **appuis de même niveau**. Le poids propre d'un câble
//! suit en toute rigueur une **chaînette** ; l'assimilation à une parabole
//! n'est valable que pour une **flèche faible** (rapport `f/L` petit), et la
//! longueur développée `Lc` est un **développement approché** au premier terme.
//! La charge répartie `w`, la portée `L`, la flèche `f` (ou la tension `H`)
//! sont **fournies par l'appelant** — jamais inventées. Le câble est supposé
//! parfaitement **flexible** (aucune rigidité de flexion) ; ce module ne
//! traite ni l'allongement élastique, ni les appuis dénivelés, ni les charges
//! concentrées, ni l'interaction câble-suspentes-tablier.

/// Composante horizontale de la tension `H = w·L² / (8·f)` (N).
///
/// Cette composante est **constante** sur toute la longueur d'un câble
/// parabolique sous charge uniforme.
///
/// Panique si `load_per_length <= 0`, `span <= 0` ou `sag <= 0`.
pub fn suscab_horizontal_tension(load_per_length: f64, span: f64, sag: f64) -> f64 {
    assert!(
        load_per_length > 0.0,
        "la charge répartie w doit être strictement positive"
    );
    assert!(span > 0.0, "la portée L doit être strictement positive");
    assert!(sag > 0.0, "la flèche f doit être strictement positive");
    load_per_length * span * span / (8.0 * sag)
}

/// Tension maximale aux appuis `Tmax = √(H² + (w·L/2)²)` (N).
///
/// La tension est maximale là où la pente du câble est la plus forte,
/// c'est-à-dire aux **appuis**, où la composante verticale vaut `w·L/2`.
///
/// Panique si `horizontal_tension <= 0`, `load_per_length <= 0` ou `span <= 0`.
pub fn suscab_max_tension(horizontal_tension: f64, load_per_length: f64, span: f64) -> f64 {
    assert!(
        horizontal_tension > 0.0,
        "la composante horizontale H doit être strictement positive"
    );
    assert!(
        load_per_length > 0.0,
        "la charge répartie w doit être strictement positive"
    );
    assert!(span > 0.0, "la portée L doit être strictement positive");
    (horizontal_tension * horizontal_tension + (load_per_length * span / 2.0).powi(2)).sqrt()
}

/// Longueur développée approchée du câble parabolique
/// `Lc = L·(1 + (8/3)·(f/L)²)` (m).
///
/// Développement limité au premier terme, valable pour une **flèche faible**.
///
/// Panique si `span <= 0` ou `sag <= 0`.
pub fn suscab_length_parabolic(span: f64, sag: f64) -> f64 {
    assert!(span > 0.0, "la portée L doit être strictement positive");
    assert!(sag > 0.0, "la flèche f doit être strictement positive");
    span * (1.0 + (8.0 / 3.0) * (sag / span).powi(2))
}

/// Flèche à mi-portée déduite de la tension horizontale
/// `f = w·L² / (8·H)` (m).
///
/// Réciproque de [`suscab_horizontal_tension`].
///
/// Panique si `load_per_length <= 0`, `span <= 0` ou `horizontal_tension <= 0`.
pub fn suscab_sag_from_tension(load_per_length: f64, span: f64, horizontal_tension: f64) -> f64 {
    assert!(
        load_per_length > 0.0,
        "la charge répartie w doit être strictement positive"
    );
    assert!(span > 0.0, "la portée L doit être strictement positive");
    assert!(
        horizontal_tension > 0.0,
        "la composante horizontale H doit être strictement positive"
    );
    load_per_length * span * span / (8.0 * horizontal_tension)
}

/// Réaction verticale à chaque appui `V = w·L / 2` (N).
///
/// Par symétrie, chaque appui reprend la moitié de la charge totale `w·L`.
///
/// Panique si `load_per_length <= 0` ou `span <= 0`.
pub fn suscab_support_reaction_vertical(load_per_length: f64, span: f64) -> f64 {
    assert!(
        load_per_length > 0.0,
        "la charge répartie w doit être strictement positive"
    );
    assert!(span > 0.0, "la portée L doit être strictement positive");
    load_per_length * span / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn tension_and_sag_are_reciprocal() {
        // Réciprocité : H = w·L²/(8f) puis f = w·L²/(8H) redonne la flèche.
        let w = 20_000.0_f64;
        let l = 100.0_f64;
        let f = 10.0_f64;
        let h = suscab_horizontal_tension(w, l, f);
        assert_relative_eq!(suscab_sag_from_tension(w, l, h), f, epsilon = 1e-9);
    }

    #[test]
    fn horizontal_tension_worked_case() {
        // Cas chiffré : w = 20 000 N/m, L = 100 m, f = 10 m.
        // H = 20000·100² / (8·10) = 20000·10000 / 80 = 2,0e8 / 80 = 2,5e6 N.
        // Recalcul : 20000·10000 = 2e8 ; 8·10 = 80 ; 2e8/80 = 2,5e6.
        let h = suscab_horizontal_tension(20_000.0, 100.0, 10.0);
        assert_relative_eq!(h, 2.5e6, epsilon = 1e-3);
    }

    #[test]
    fn max_tension_worked_case() {
        // H = 2,5e6 N, w·L/2 = 20000·100/2 = 1,0e6 N.
        // Tmax = √((2,5e6)² + (1,0e6)²) = √(6,25e12 + 1,0e12) = √(7,25e12).
        // √7,25 = 2,692582403…  =>  Tmax = 2,692582403…e6 N.
        // Recalcul : 2,5² = 6,25 ; 6,25 + 1 = 7,25 ; √7,25 = 2,6925824035…
        let tmax = suscab_max_tension(2.5e6, 20_000.0, 100.0);
        assert_relative_eq!(tmax, 2.692_582_403_567_252e6, epsilon = 1e-3);
        // Tmax est toujours supérieure à la composante horizontale H.
        assert!(tmax > 2.5e6);
    }

    #[test]
    fn vertical_reaction_pairs_the_total_load() {
        // Les deux réactions d'appui équilibrent la charge totale w·L.
        let w = 15_000.0_f64;
        let l = 80.0_f64;
        let v = suscab_support_reaction_vertical(w, l);
        assert_relative_eq!(2.0 * v, w * l, epsilon = 1e-6);
    }

    #[test]
    fn cable_length_exceeds_span_and_worked_case() {
        // Lc = L·(1 + (8/3)·(f/L)²) avec L = 100 m, f = 10 m.
        // f/L = 0,1 ; (f/L)² = 0,01 ; (8/3)·0,01 = 0,0266666… ;
        // Lc = 100·1,0266666… = 102,6666666… m.
        // Recalcul : 8/3 = 2,6666… ; ·0,01 = 0,026666… ; +1 = 1,026666… ; ·100.
        let lc = suscab_length_parabolic(100.0, 10.0);
        assert_relative_eq!(lc, 102.666_666_666_666_67, epsilon = 1e-3);
        // La longueur développée dépasse toujours la portée (câble tendu = corde).
        assert!(lc > 100.0);
    }

    #[test]
    fn max_tension_reduces_to_horizontal_for_vanishing_load() {
        // Limite : quand w → 0, la part verticale s'annule et Tmax → H.
        let h = 3.0e6_f64;
        let tmax = suscab_max_tension(h, 1.0e-9, 100.0);
        assert_relative_eq!(tmax, h, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "la flèche f doit être strictement positive")]
    fn zero_sag_panics() {
        let _ = suscab_horizontal_tension(20_000.0, 100.0, 0.0);
    }
}
