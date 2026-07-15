//! RDM — **flèche maximale** de poutres prismatiques pour les cas de charge
//! standards de la théorie d'Euler-Bernoulli.
//!
//! ```text
//! console (encastrée-libre), charge en bout P     δ = P·L³/(3·E·I)
//! console (encastrée-libre), charge répartie w    δ = w·L⁴/(8·E·I)
//! deux appuis, charge centrale P                  δ = P·L³/(48·E·I)
//! deux appuis, charge répartie w                  δ = 5·w·L⁴/(384·E·I)
//! ```
//!
//! `P` charge ponctuelle (N), `w` charge répartie linéique (N/m), `L` portée ou
//! longueur (m), `E` module de Young (Pa = N/m²), `I` moment quadratique de la
//! section (m⁴), `δ` flèche maximale (m).
//!
//! **Convention** : SI cohérent, flèches comptées positives dans le sens de la
//! charge. **Limite honnête** : poutre **prismatique** (section constante),
//! matériau **élastique linéaire**, **petites flèches**, cisaillement négligé
//! (poutre élancée). Cas de charge **classiques** uniquement. `E` (matériau) et
//! `I` (géométrie de section) sont **fournis par l'appelant** ; aucune valeur
//! « par défaut » de matériau ou de procédé n'est inventée ici.

/// Flèche au bout d'une **console** (encastrée-libre) sous **charge en bout**
/// `δ = P·L³/(3·E·I)` (m).
///
/// Panique si `length < 0`, ou si `youngs_modulus <= 0` ou `second_moment <= 0`.
pub fn beam_cantilever_end_load(
    load: f64,
    length: f64,
    youngs_modulus: f64,
    second_moment: f64,
) -> f64 {
    assert!(length >= 0.0, "la longueur L doit être positive ou nulle");
    assert!(
        youngs_modulus > 0.0,
        "le module de Young E doit être strictement positif"
    );
    assert!(
        second_moment > 0.0,
        "le moment quadratique I doit être strictement positif"
    );
    load * length.powi(3) / (3.0 * youngs_modulus * second_moment)
}

/// Flèche au bout d'une **console** (encastrée-libre) sous **charge répartie**
/// uniforme `δ = w·L⁴/(8·E·I)` (m).
///
/// Panique si `length < 0`, ou si `youngs_modulus <= 0` ou `second_moment <= 0`.
pub fn beam_cantilever_udl(
    distributed_load: f64,
    length: f64,
    youngs_modulus: f64,
    second_moment: f64,
) -> f64 {
    assert!(length >= 0.0, "la longueur L doit être positive ou nulle");
    assert!(
        youngs_modulus > 0.0,
        "le module de Young E doit être strictement positif"
    );
    assert!(
        second_moment > 0.0,
        "le moment quadratique I doit être strictement positif"
    );
    distributed_load * length.powi(4) / (8.0 * youngs_modulus * second_moment)
}

/// Flèche à mi-portée d'une poutre sur **deux appuis simples** sous **charge
/// centrale** `δ = P·L³/(48·E·I)` (m).
///
/// Panique si `length < 0`, ou si `youngs_modulus <= 0` ou `second_moment <= 0`.
pub fn beam_simply_supported_center_load(
    load: f64,
    length: f64,
    youngs_modulus: f64,
    second_moment: f64,
) -> f64 {
    assert!(length >= 0.0, "la portée L doit être positive ou nulle");
    assert!(
        youngs_modulus > 0.0,
        "le module de Young E doit être strictement positif"
    );
    assert!(
        second_moment > 0.0,
        "le moment quadratique I doit être strictement positif"
    );
    load * length.powi(3) / (48.0 * youngs_modulus * second_moment)
}

/// Flèche à mi-portée d'une poutre sur **deux appuis simples** sous **charge
/// répartie** uniforme `δ = 5·w·L⁴/(384·E·I)` (m).
///
/// Panique si `length < 0`, ou si `youngs_modulus <= 0` ou `second_moment <= 0`.
pub fn beam_simply_supported_udl(
    distributed_load: f64,
    length: f64,
    youngs_modulus: f64,
    second_moment: f64,
) -> f64 {
    assert!(length >= 0.0, "la portée L doit être positive ou nulle");
    assert!(
        youngs_modulus > 0.0,
        "le module de Young E doit être strictement positif"
    );
    assert!(
        second_moment > 0.0,
        "le moment quadratique I doit être strictement positif"
    );
    5.0 * distributed_load * length.powi(4) / (384.0 * youngs_modulus * second_moment)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Jeu d'essai commun : acier E = 200 GPa, section I = 1·10⁻⁶ m⁴, L = 2 m.
    const E: f64 = 2.0e11;
    const I: f64 = 1.0e-6;
    const L: f64 = 2.0;

    #[test]
    fn cantilever_end_load_reference_value() {
        // δ = P·L³/(3·E·I) = 1000·8 / (3·2e11·1e-6)
        //   = 8000 / 600000 = 0.013333… m.
        let delta = beam_cantilever_end_load(1000.0, L, E, I);
        assert_relative_eq!(delta, 8000.0 / 600_000.0, epsilon = 1e-12);
        assert_relative_eq!(delta, 0.013_333_333_333_333, epsilon = 1e-12);
    }

    #[test]
    fn simply_supported_center_reference_value() {
        // δ = P·L³/(48·E·I) = 1000·8 / (48·2e11·1e-6)
        //   = 8000 / 9_600_000 = 8.3333…e-4 m.
        let delta = beam_simply_supported_center_load(1000.0, L, E, I);
        assert_relative_eq!(delta, 8000.0 / 9_600_000.0, epsilon = 1e-15);
    }

    #[test]
    fn point_load_stiffness_ratio_is_16() {
        // À P, L, E, I égaux : δ_console / δ_appuis = (1/3)/(1/48) = 16.
        let cantilever = beam_cantilever_end_load(1000.0, L, E, I);
        let simply = beam_simply_supported_center_load(1000.0, L, E, I);
        assert_relative_eq!(cantilever / simply, 16.0, epsilon = 1e-12);
    }

    #[test]
    fn udl_stiffness_ratio_is_9_point_6() {
        // À w, L, E, I égaux : δ_console / δ_appuis = (1/8)/(5/384) = 48/5 = 9.6.
        let cantilever = beam_cantilever_udl(500.0, L, E, I);
        let simply = beam_simply_supported_udl(500.0, L, E, I);
        assert_relative_eq!(cantilever / simply, 9.6, epsilon = 1e-12);
    }

    #[test]
    fn deflection_scales_with_load_cube_of_length() {
        // Linéarité en charge et δ ∝ L³ (charge ponctuelle) : doubler L → ×8.
        let base = beam_simply_supported_center_load(1000.0, L, E, I);
        let double_len = beam_simply_supported_center_load(1000.0, 2.0 * L, E, I);
        assert_relative_eq!(double_len / base, 8.0, epsilon = 1e-12);
        let double_load = beam_simply_supported_center_load(2000.0, L, E, I);
        assert_relative_eq!(double_load / base, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "module de Young E")]
    fn non_positive_modulus_panics() {
        beam_cantilever_end_load(1000.0, L, 0.0, I);
    }
}
