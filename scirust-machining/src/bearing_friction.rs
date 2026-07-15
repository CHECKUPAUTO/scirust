//! **Couple de frottement d'un roulement (modèle de Palmgren)** — couple dû à la
//! charge, couple visqueux de lubrification, couple total et puissance dissipée.
//!
//! ```text
//! couple dû à la charge   M1 = 0,5·μ·F·dm
//! couple visqueux         M0 = 1e-7·f0·(ν·n)^(2/3)·dm^3   (si ν·n ≥ 2000)
//! couple total            M  = M1 + M0
//! puissance dissipée      P  = M·ω
//! ```
//!
//! `μ` coefficient de frottement du roulement (sans dimension), `F` charge
//! appliquée (N), `dm` diamètre moyen du roulement `dm = (d+D)/2`, `f0` facteur
//! visqueux (sans dimension), `ν·n` produit viscosité cinématique × vitesse de
//! rotation, `M1`/`M0`/`M` couples (couple dû à la charge, couple visqueux, couple
//! total), `ω` vitesse angulaire (rad/s), `P` puissance dissipée par frottement.
//!
//! **Unités.** Le terme visqueux de Palmgren est **empirique** : la constante
//! `1e-7` correspond à la forme classique où `ν` est en mm²/s (cSt), `n` en tr/min
//! et `dm` en mm, ce qui donne `M0` en N·mm. Le terme de charge `M1` est purement
//! géométrique et suit l'unité choisie pour `dm` et `F`. **L'appelant est
//! responsable de la cohérence des unités** avant d'additionner `M1` et `M0` ou de
//! calculer `P = M·ω` (convertir dans un système commun, N·mm ou N·m).
//!
//! **Limite honnête.** Modèle **empirique** de Palmgren : le coefficient de
//! frottement `μ` et le facteur visqueux `f0` dépendent du **type de roulement** et
//! de la **lubrification** et sont **fournis par l'appelant** ; la viscosité `ν` est
//! **fournie**. La condition de validité du terme visqueux (`ν·n ≥ 2000`) est
//! imposée ; en deçà une autre expression s'applique. Aucune valeur de `μ`, `f0`,
//! matériau, lubrifiant ou procédé n'est inventée — c'est une **estimation** (SKF
//! propose des modèles plus fins, avec termes de roulement/glissement et joints).

/// Couple de frottement dû à la charge `M1 = 0,5·μ·F·dm`.
///
/// Panique si `friction_coefficient < 0`, `load < 0` ou `mean_diameter < 0`.
pub fn bearingfric_load_torque(friction_coefficient: f64, load: f64, mean_diameter: f64) -> f64 {
    assert!(
        friction_coefficient >= 0.0,
        "le coefficient de frottement μ doit être positif"
    );
    assert!(load >= 0.0, "la charge F doit être positive");
    assert!(
        mean_diameter >= 0.0,
        "le diamètre moyen dm doit être positif"
    );
    0.5 * friction_coefficient * load * mean_diameter
}

/// Couple visqueux de Palmgren `M0 = 1e-7·f0·(ν·n)^(2/3)·dm^3`.
///
/// Valable pour `ν·n ≥ 2000` (forme classique) ; en deçà une autre expression
/// s'applique et cette fonction panique.
///
/// Panique si `viscous_factor < 0`, `mean_diameter < 0` ou
/// `kinematic_viscosity_speed_product < 2000`.
pub fn bearingfric_viscous_torque(
    viscous_factor: f64,
    kinematic_viscosity_speed_product: f64,
    mean_diameter: f64,
) -> f64 {
    assert!(
        viscous_factor >= 0.0,
        "le facteur visqueux f0 doit être positif"
    );
    assert!(
        mean_diameter >= 0.0,
        "le diamètre moyen dm doit être positif"
    );
    assert!(
        kinematic_viscosity_speed_product >= 2000.0,
        "le produit ν·n doit être ≥ 2000 (domaine de validité de la forme classique)"
    );
    1e-7 * viscous_factor
        * kinematic_viscosity_speed_product.powf(2.0 / 3.0)
        * mean_diameter.powi(3)
}

/// Couple de frottement total `M = M1 + M0`.
///
/// Panique si `load_torque < 0` ou `viscous_torque < 0`.
pub fn bearingfric_total_torque(load_torque: f64, viscous_torque: f64) -> f64 {
    assert!(
        load_torque >= 0.0,
        "le couple dû à la charge M1 doit être positif"
    );
    assert!(
        viscous_torque >= 0.0,
        "le couple visqueux M0 doit être positif"
    );
    load_torque + viscous_torque
}

/// Puissance dissipée par frottement `P = M·ω`.
///
/// Panique si `total_torque < 0` ou `angular_speed_rad < 0`.
pub fn bearingfric_power_loss(total_torque: f64, angular_speed_rad: f64) -> f64 {
    assert!(total_torque >= 0.0, "le couple total M doit être positif");
    assert!(
        angular_speed_rad >= 0.0,
        "la vitesse angulaire ω doit être positive"
    );
    total_torque * angular_speed_rad
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn load_torque_realistic_case() {
        // μ=0,002 ; F=5000 N ; dm=0,05 m → M1 = 0,5·0,002·5000·0,05 = 0,25 N·m.
        let m1 = bearingfric_load_torque(0.002, 5000.0, 0.05);
        assert_relative_eq!(m1, 0.25, epsilon = 1e-12);
    }

    #[test]
    fn load_torque_scales_linearly_with_load() {
        // M1 ∝ F : doubler la charge double le couple dû à la charge.
        let a = bearingfric_load_torque(0.0018, 4000.0, 0.06);
        let b = bearingfric_load_torque(0.0018, 8000.0, 0.06);
        assert_relative_eq!(b, 2.0 * a, epsilon = 1e-12);
    }

    #[test]
    fn viscous_torque_clean_case() {
        // f0=2 ; ν·n=8000 (⇒ (8000)^(2/3)=20²=400) ; dm=10 (⇒ dm³=1000)
        // → M0 = 1e-7·2·400·1000 = 0,08.
        let m0 = bearingfric_viscous_torque(2.0, 8000.0, 10.0);
        assert_relative_eq!(m0, 0.08, epsilon = 1e-12);
    }

    #[test]
    fn viscous_torque_scales_with_diameter_cubed() {
        // M0 ∝ dm³ : doubler dm multiplie le couple visqueux par 8.
        let small = bearingfric_viscous_torque(2.0, 8000.0, 10.0);
        let large = bearingfric_viscous_torque(2.0, 8000.0, 20.0);
        assert_relative_eq!(large, 8.0 * small, epsilon = 1e-9);
    }

    #[test]
    fn total_torque_is_sum_reciprocity() {
        // Identité de décomposition : M − M1 = M0.
        let m1 = bearingfric_load_torque(0.002, 5000.0, 0.05);
        let m0 = bearingfric_viscous_torque(2.0, 8000.0, 10.0);
        let m = bearingfric_total_torque(m1, m0);
        assert_relative_eq!(m, m1 + m0, epsilon = 1e-12);
        assert_relative_eq!(m - m1, m0, epsilon = 1e-12);
    }

    #[test]
    fn power_loss_realistic_case() {
        // M=0,33 N·m à ω=100 rad/s → P = 0,33·100 = 33 W.
        let p = bearingfric_power_loss(0.33, 100.0);
        assert_relative_eq!(p, 33.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "domaine de validité")]
    fn viscous_below_threshold_panics() {
        bearingfric_viscous_torque(2.0, 1999.0, 10.0);
    }
}
