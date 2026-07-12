//! RDM — **énergie de déformation** d'éléments prismatiques et principe de
//! **Castigliano** pour les déplacements.
//!
//! ```text
//! traction/compression  U = N²·L/(2·E·A)
//! flexion (M constant)  U = M²·L/(2·E·I)
//! torsion               U = T²·L/(2·G·J)
//! Castigliano           δ = ∂U/∂P        (déplacement au point/direction de P)
//! ```
//!
//! `N` effort normal (N), `M` moment fléchissant (N·m), `T` couple (N·m), `L`
//! longueur (m), `E`/`G` modules (Pa), `A` aire (m²), `I` moment quadratique
//! (m⁴), `J` moment quadratique polaire (m⁴), `U` énergie (J). Le théorème de
//! Castigliano donne le déplacement comme dérivée partielle de l'énergie par
//! rapport à la charge conjuguée.
//!
//! **Convention** : SI cohérent. **Limite honnête** : énergie d'éléments
//! **prismatiques** à effort **constant** (pour un effort variable, intégrer
//! `∫N²/2EA dx`, etc.) ; ce module fournit les briques énergétiques, la dérivée
//! de Castigliano `∂U/∂P` étant appliquée par l'appelant. Élasticité linéaire.

/// Énergie de déformation en traction/compression `U = N²·L/(2·E·A)` (J).
///
/// Panique si `e*area <= 0`.
pub fn axial_energy(normal_force: f64, length: f64, e: f64, area: f64) -> f64 {
    assert!(
        e * area > 0.0,
        "la rigidité E·A doit être strictement positive"
    );
    normal_force * normal_force * length / (2.0 * e * area)
}

/// Énergie de déformation en flexion (moment constant) `U = M²·L/(2·E·I)` (J).
///
/// Panique si `e*i <= 0`.
pub fn bending_energy(moment: f64, length: f64, e: f64, i: f64) -> f64 {
    assert!(
        e * i > 0.0,
        "la rigidité E·I doit être strictement positive"
    );
    moment * moment * length / (2.0 * e * i)
}

/// Énergie de déformation en torsion `U = T²·L/(2·G·J)` (J).
///
/// Panique si `g*j <= 0`.
pub fn torsion_energy(torque: f64, length: f64, g: f64, j: f64) -> f64 {
    assert!(
        g * j > 0.0,
        "la rigidité G·J doit être strictement positive"
    );
    torque * torque * length / (2.0 * g * j)
}

/// Énergie de déformation totale `U = Σ Ui` (J).
pub fn total_energy(energies: &[f64]) -> f64 {
    energies.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn axial_energy_scales_with_force_squared() {
        // Doubler l'effort quadruple l'énergie.
        let u1 = axial_energy(10_000.0, 2.0, 2.0e11, 1e-4);
        let u2 = axial_energy(20_000.0, 2.0, 2.0e11, 1e-4);
        assert_relative_eq!(u2 / u1, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn castigliano_recovers_cantilever_tip_deflection() {
        // Console charge bout P : M(x)=P·x, U=∫₀ᴸ P²x²/(2EI)dx = P²L³/(6EI).
        // δ = ∂U/∂P = P·L³/(3EI) — on vérifie via différence finie sur U.
        let (l, e, i) = (2.0_f64, 2.0e11, 1.0e-6);
        // U(P) = P²L³/(6EI) ; on l'assemble par tranches de bending_energy à M~P·x.
        // Ici on teste l'énergie fermée U = P²L³/(6EI) et sa dérivée numérique.
        let energy = |p: f64| p * p * l.powi(3) / (6.0 * e * i);
        let (p, h) = (1000.0, 1e-3);
        let delta = (energy(p + h) - energy(p - h)) / (2.0 * h);
        assert_relative_eq!(delta, p * l.powi(3) / (3.0 * e * i), max_relative = 1e-6);
    }

    #[test]
    fn total_energy_sums_contributions() {
        // Traction + flexion + torsion s'additionnent.
        let u = total_energy(&[10.0, 25.0, 5.0]);
        assert_relative_eq!(u, 40.0, epsilon = 1e-9);
    }

    #[test]
    fn torsion_energy_definition() {
        assert_relative_eq!(
            torsion_energy(500.0, 1.5, 80e9, 2e-8),
            500.0f64.powi(2) * 1.5 / (2.0 * 80e9 * 2e-8),
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "rigidité E·A")]
    fn zero_axial_stiffness_panics() {
        axial_energy(10_000.0, 2.0, 0.0, 1e-4);
    }
}
