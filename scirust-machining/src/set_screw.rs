//! Vis de pression (*set screw*) — maintien d'un moyeu sur un arbre par
//! pénétration/frottement de la pointe, relation empirique **couple-effort**.
//!
//! ```text
//! couple de mise en place    T  = K·d·F
//! effort de maintien axial    F  = T/(K·d)
//! couple transmis à l'arbre    C  = F·ds/2
//! nombre de vis requis         n  = C_req/C_1
//! ```
//!
//! `T` couple de mise en place de la vis (N·m), `K` coefficient de couple
//! (*seating/torque coefficient*, sans dimension, empirique), `d` diamètre
//! nominal de la vis (m), `F` effort de maintien axial de la pointe (N), `ds`
//! diamètre de l'arbre (m), `C` couple transmissible par frottement (N·m),
//! `C_req` couple requis (N·m), `C_1` couple tenu par une seule vis (N·m), `n`
//! nombre de vis (sans dimension).
//!
//! **Convention** : SI cohérent (N, m, N·m), efforts et diamètres positifs.
//!
//! **Limite honnête** : maintien par **pénétration/frottement de la pointe** ;
//! le coefficient de couple `K` est **empirique** (dépend du filet, de la
//! lubrification et du procédé) et **FOURNI par l'appelant**, de même que les
//! efforts et les diamètres — ce module n'invente aucune valeur « par défaut ».
//! Approximation **statique** qui ne tient pas compte des chocs ni des
//! vibrations ; le nombre de vis est renvoyé en `f64` (ratio brut, à arrondir
//! par l'appelant au moyen d'un `ceil`).

/// Couple de mise en place `T = K·d·F` (N·m).
///
/// `torque_coefficient` = `K` (sans dimension), `nominal_diameter` = `d` (m),
/// `axial_holding_force` = `F` (N).
///
/// Panique si `torque_coefficient <= 0`, si `nominal_diameter <= 0` ou si
/// `axial_holding_force < 0`.
pub fn setscrew_seating_torque(
    torque_coefficient: f64,
    nominal_diameter: f64,
    axial_holding_force: f64,
) -> f64 {
    assert!(
        torque_coefficient > 0.0,
        "le coefficient de couple K doit être strictement positif"
    );
    assert!(
        nominal_diameter > 0.0,
        "le diamètre nominal doit être strictement positif"
    );
    assert!(
        axial_holding_force >= 0.0,
        "l'effort de maintien axial doit être positif ou nul"
    );
    torque_coefficient * nominal_diameter * axial_holding_force
}

/// Effort de maintien axial `F = T/(K·d)` (N).
///
/// `seating_torque` = `T` (N·m), `nominal_diameter` = `d` (m),
/// `torque_coefficient` = `K` (sans dimension).
///
/// Panique si `seating_torque < 0`, si `nominal_diameter <= 0` ou si
/// `torque_coefficient <= 0`.
pub fn setscrew_axial_holding_force(
    seating_torque: f64,
    nominal_diameter: f64,
    torque_coefficient: f64,
) -> f64 {
    assert!(
        seating_torque >= 0.0,
        "le couple de mise en place doit être positif ou nul"
    );
    assert!(
        nominal_diameter > 0.0,
        "le diamètre nominal doit être strictement positif"
    );
    assert!(
        torque_coefficient > 0.0,
        "le coefficient de couple K doit être strictement positif"
    );
    seating_torque / (torque_coefficient * nominal_diameter)
}

/// Couple transmissible à l'arbre par frottement de la pointe `C = F·ds/2` (N·m).
///
/// `axial_holding_force` = `F` (N), `shaft_diameter` = `ds` (m).
///
/// Panique si `axial_holding_force < 0` ou si `shaft_diameter <= 0`.
pub fn setscrew_holding_torque_on_shaft(axial_holding_force: f64, shaft_diameter: f64) -> f64 {
    assert!(
        axial_holding_force >= 0.0,
        "l'effort de maintien axial doit être positif ou nul"
    );
    assert!(
        shaft_diameter > 0.0,
        "le diamètre de l'arbre doit être strictement positif"
    );
    axial_holding_force * shaft_diameter / 2.0
}

/// Nombre de vis requis `n = C_req/C_1` (sans dimension, ratio brut).
///
/// `required_torque` = `C_req` (N·m), `single_screw_torque` = `C_1` (N·m). Le
/// résultat est un ratio `f64` que l'appelant arrondira (typiquement `ceil`).
///
/// Panique si `required_torque < 0` ou si `single_screw_torque <= 0`.
pub fn setscrew_required_count(required_torque: f64, single_screw_torque: f64) -> f64 {
    assert!(
        required_torque >= 0.0,
        "le couple requis doit être positif ou nul"
    );
    assert!(
        single_screw_torque > 0.0,
        "le couple tenu par une seule vis doit être strictement positif"
    );
    required_torque / single_screw_torque
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn axial_holding_force_reference_case() {
        // M6 (d=0,006 m), K=0,2, T=1,2 N·m → F = 1,2/(0,2·0,006) = 1000 N.
        assert_relative_eq!(
            setscrew_axial_holding_force(1.2, 0.006, 0.2),
            1000.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn seating_torque_and_holding_force_are_reciprocal() {
        // Réciprocité : F → T = K·d·F → F retrouvé par T/(K·d).
        let k = 0.22_f64;
        let d = 0.008_f64;
        let f = 1500.0_f64;
        let t = setscrew_seating_torque(k, d, f);
        assert_relative_eq!(setscrew_axial_holding_force(t, d, k), f, epsilon = 1e-9);
    }

    #[test]
    fn holding_torque_reference_case() {
        // F=1000 N sur un arbre ds=0,020 m → C = 1000·0,020/2 = 10 N·m.
        assert_relative_eq!(
            setscrew_holding_torque_on_shaft(1000.0, 0.020),
            10.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn holding_torque_scales_linearly_with_force() {
        // Proportionnalité : doubler l'effort double le couple transmis (ds fixé).
        let c1 = setscrew_holding_torque_on_shaft(1000.0, 0.020);
        let c2 = setscrew_holding_torque_on_shaft(2000.0, 0.020);
        assert_relative_eq!(c2, 2.0 * c1, epsilon = 1e-9);
    }

    #[test]
    fn required_count_reference_case() {
        // C_req=25 N·m tenu par des vis à C_1=10 N·m → n = 2,5 (soit 3 après ceil).
        assert_relative_eq!(setscrew_required_count(25.0, 10.0), 2.5, epsilon = 1e-9);
    }

    #[test]
    fn required_count_is_one_when_torques_match() {
        // Cas limite : une seule vis suffit exactement quand C_req = C_1.
        assert_relative_eq!(setscrew_required_count(10.0, 10.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "coefficient de couple K doit être strictement positif")]
    fn zero_coefficient_panics() {
        setscrew_axial_holding_force(1.2, 0.006, 0.0);
    }
}
