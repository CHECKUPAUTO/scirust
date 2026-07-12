//! RDM — **flèches et pentes** de cas de poutres complémentaires, et
//! **superposition** de plusieurs cas de charge.
//!
//! ```text
//! console, charge répartie w      δ = w·L⁴/(8·E·I)
//! console, charge en bout P       θ_bout = P·L²/(2·E·I)     (pente)
//! deux appuis, charge centrale P  θ_appui = P·L²/(16·E·I)   (pente aux appuis)
//! encastrée-encastrée, P central  δ = P·L³/(192·E·I)
//! superposition                   δ_tot = Σ δi
//! ```
//!
//! `w` charge répartie (N/m), `P` charge ponctuelle (N), `L` portée (m), `E`
//! module de Young (Pa), `I` moment quadratique (m⁴), `θ` pente (rad). La
//! superposition additionne les flèches de cas de charge indépendants (linéarité).
//!
//! **Convention** : SI cohérent. **Limite honnête** : théorie d'Euler-Bernoulli
//! **élastique linéaire**, petites déformations ; complète les cas de
//! [`crate::beams`] (charges centrées, appuis simples). L'encastrée-encastrée est
//! hyperstatique mais son résultat fermé est fourni.

/// Flèche en bout d'une **console sous charge répartie** `δ = w·L⁴/(8·E·I)` (m).
///
/// Panique si `e*i <= 0`.
pub fn cantilever_udl_deflection(w: f64, length: f64, e: f64, i: f64) -> f64 {
    assert!(
        e * i > 0.0,
        "la rigidité E·I doit être strictement positive"
    );
    w * length.powi(4) / (8.0 * e * i)
}

/// Pente en bout d'une **console sous charge en bout** `θ = P·L²/(2·E·I)` (rad).
///
/// Panique si `e*i <= 0`.
pub fn cantilever_end_slope(p: f64, length: f64, e: f64, i: f64) -> f64 {
    assert!(
        e * i > 0.0,
        "la rigidité E·I doit être strictement positive"
    );
    p * length * length / (2.0 * e * i)
}

/// Pente aux appuis — **deux appuis, charge centrale** `θ = P·L²/(16·E·I)` (rad).
///
/// Panique si `e*i <= 0`.
pub fn simply_supported_center_slope(p: f64, length: f64, e: f64, i: f64) -> f64 {
    assert!(
        e * i > 0.0,
        "la rigidité E·I doit être strictement positive"
    );
    p * length * length / (16.0 * e * i)
}

/// Flèche centrale — **encastrée-encastrée, charge centrale**
/// `δ = P·L³/(192·E·I)` (m).
///
/// Panique si `e*i <= 0`.
pub fn fixed_fixed_center_deflection(p: f64, length: f64, e: f64, i: f64) -> f64 {
    assert!(
        e * i > 0.0,
        "la rigidité E·I doit être strictement positive"
    );
    p * length.powi(3) / (192.0 * e * i)
}

/// Flèche résultante par **superposition** `δ_tot = Σ δi` (m).
pub fn superpose_deflections(deflections: &[f64]) -> f64 {
    deflections.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cantilever_udl_softer_than_end_load() {
        // δ_udl = wL⁴/8EI. Vérification directe.
        let (w, l, e, i) = (500.0, 2.0, 2.0e11, 1.0e-6);
        assert_relative_eq!(
            cantilever_udl_deflection(w, l, e, i),
            w * l.powi(4) / (8.0 * e * i),
            epsilon = 1e-15
        );
    }

    #[test]
    fn fixed_fixed_stiffer_than_simply_supported() {
        // À charge/portée/EI égaux, l'encastrée-encastrée (PL³/192) fléchit bien
        // moins que le cas sur deux appuis (PL³/48) : rapport 4.
        let (p, l, e, i) = (1000.0, 2.0, 2.0e11, 1.0e-6);
        let ff = fixed_fixed_center_deflection(p, l, e, i);
        let ss = p * l.powi(3) / (48.0 * e * i);
        assert_relative_eq!(ss / ff, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn slopes_are_positive() {
        assert!(cantilever_end_slope(1000.0, 2.0, 2.0e11, 1.0e-6) > 0.0);
        assert!(simply_supported_center_slope(1000.0, 2.0, 2.0e11, 1.0e-6) > 0.0);
    }

    #[test]
    fn superposition_adds_cases() {
        // Deux contributions se somment (linéarité).
        assert_relative_eq!(
            superpose_deflections(&[1e-3, 2e-3, -0.5e-3]),
            2.5e-3,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "rigidité E·I")]
    fn zero_stiffness_panics() {
        cantilever_udl_deflection(500.0, 2.0, 0.0, 1e-6);
    }
}
