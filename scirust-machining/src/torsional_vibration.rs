//! Vibration de **torsion** d'un système à deux disques montés sur un arbre —
//! raideur en torsion de l'arbre, pulsation propre du système à deux inerties et
//! position du **nœud** de vibration.
//!
//! ```text
//! raideur torsion      k = G·Jp / L
//! inertie équivalente  Je = J1·J2 / (J1 + J2)
//! pulsation propre     ωn = √( k·(J1 + J2) / (J1·J2) ) = √( k / Je )
//! fréquence propre     fn = ωn / (2π)
//! position du nœud     l1 = L·J2 / (J1 + J2)   (mesurée depuis le disque 1)
//! ```
//!
//! `G` module de cisaillement (Pa), `Jp` moment quadratique **polaire** de la
//! section (m⁴), `L` longueur de l'arbre (m), `k` raideur en torsion (N·m/rad),
//! `J1`, `J2` moments d'inertie de masse des deux disques (kg·m²), `ωn` pulsation
//! propre (rad/s), `fn` fréquence propre (Hz). Le nœud est le point de
//! l'arbre qui reste immobile pendant l'oscillation ; il vérifie `l1·J1 = l2·J2`.
//!
//! **Convention** : SI cohérent. **Limite honnête** : arbre supposé **sans
//! masse** (inertie répartie négligée), deux inerties **concentrées**, régime de
//! **petit angle** (raideur linéaire), amortissement négligé, premier mode
//! torsionnel uniquement. Les constantes physiques et propriétés matériau
//! (module de cisaillement, moment polaire, inerties) sont **fournies par
//! l'appelant** — aucune valeur « par défaut » n'est supposée.

use core::f64::consts::PI;

/// Raideur en torsion d'un arbre `k = G·Jp / L` (N·m/rad).
///
/// `shear_modulus` en Pa, `polar_inertia_area` (moment quadratique polaire de
/// section `Jp`) en m⁴, `length` en m.
///
/// Panique si `length <= 0`, `shear_modulus < 0` ou `polar_inertia_area < 0`.
pub fn torsional_stiffness(shear_modulus: f64, polar_inertia_area: f64, length: f64) -> f64 {
    assert!(
        length > 0.0,
        "la longueur de l'arbre doit être strictement positive"
    );
    assert!(
        shear_modulus >= 0.0,
        "le module de cisaillement doit être ≥ 0"
    );
    assert!(
        polar_inertia_area >= 0.0,
        "le moment quadratique polaire doit être ≥ 0"
    );
    shear_modulus * polar_inertia_area / length
}

/// Inertie **équivalente** du système à deux disques `Je = J1·J2 / (J1 + J2)`
/// (kg·m²).
///
/// Panique si `inertia1 <= 0` ou `inertia2 <= 0`.
pub fn two_disc_equivalent_inertia(inertia1: f64, inertia2: f64) -> f64 {
    assert!(
        inertia1 > 0.0 && inertia2 > 0.0,
        "les deux inerties doivent être strictement positives"
    );
    inertia1 * inertia2 / (inertia1 + inertia2)
}

/// Pulsation propre torsionnelle du système à deux disques
/// `ωn = √( k·(J1 + J2) / (J1·J2) )` (rad/s).
///
/// `stiffness` raideur en torsion `k` (N·m/rad), `inertia1`/`inertia2` les
/// moments d'inertie des disques (kg·m²).
///
/// Panique si `stiffness < 0`, `inertia1 <= 0` ou `inertia2 <= 0`.
pub fn two_disc_natural_frequency_rad(stiffness: f64, inertia1: f64, inertia2: f64) -> f64 {
    assert!(stiffness >= 0.0, "la raideur en torsion doit être ≥ 0");
    let equivalent = two_disc_equivalent_inertia(inertia1, inertia2);
    (stiffness / equivalent).sqrt()
}

/// Fréquence propre torsionnelle `fn = ωn / (2π)` (Hz) du système à deux disques.
///
/// Panique si `stiffness < 0`, `inertia1 <= 0` ou `inertia2 <= 0`.
pub fn two_disc_natural_frequency_hz(stiffness: f64, inertia1: f64, inertia2: f64) -> f64 {
    two_disc_natural_frequency_rad(stiffness, inertia1, inertia2) / (2.0 * PI)
}

/// Position du **nœud** de vibration `l1 = L·J2 / (J1 + J2)` (m), distance
/// mesurée depuis le disque 1 (celui portant `inertia1`).
///
/// Le nœud vérifie `l1·J1 = l2·J2` avec `l1 + l2 = L` : le disque de plus forte
/// inertie est le plus proche du nœud.
///
/// Panique si `length <= 0`, `inertia1 <= 0` ou `inertia2 <= 0`.
pub fn two_disc_node_position(length: f64, inertia1: f64, inertia2: f64) -> f64 {
    assert!(
        length > 0.0,
        "la longueur de l'arbre doit être strictement positive"
    );
    assert!(
        inertia1 > 0.0 && inertia2 > 0.0,
        "les deux inerties doivent être strictement positives"
    );
    length * inertia2 / (inertia1 + inertia2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn stiffness_is_inversely_proportional_to_length() {
        // k ∝ 1/L : doubler la longueur divise la raideur par deux.
        let k1 = torsional_stiffness(80e9, 6.0e-7, 1.0);
        let k2 = torsional_stiffness(80e9, 6.0e-7, 2.0);
        assert_relative_eq!(k1, 2.0 * k2, epsilon = 1e-6);
    }

    #[test]
    fn frequency_matches_equivalent_inertia_form() {
        // ωn = √(k/Je) doit coïncider avec la forme développée √(k(J1+J2)/(J1 J2)).
        let (k, j1, j2) = (48_660.0, 0.10, 0.15);
        let je = two_disc_equivalent_inertia(j1, j2);
        assert_relative_eq!(
            two_disc_natural_frequency_rad(k, j1, j2),
            (k / je).sqrt(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn heavy_second_disc_reduces_to_single_disc() {
        // Quand J2 ≫ J1, le disque 2 devient un encastrement : ωn → √(k/J1).
        let (k, j1) = (10_000.0, 0.05);
        let big = 1.0e9_f64;
        assert_relative_eq!(
            two_disc_natural_frequency_rad(k, j1, big),
            (k / j1).sqrt(),
            epsilon = 1e-2
        );
    }

    #[test]
    fn node_is_at_midspan_for_equal_inertias() {
        // J1 = J2 → nœud au milieu de l'arbre.
        assert_relative_eq!(two_disc_node_position(1.2, 0.2, 0.2), 0.6, epsilon = 1e-12);
    }

    #[test]
    fn node_reciprocity_and_stiffness_balance() {
        // l1 (depuis disque 1) + l2 (depuis disque 2, inerties permutées) = L,
        // et le nœud vérifie l1·J1 = l2·J2.
        let (l, j1, j2) = (1.0, 0.1, 0.3);
        let l1 = two_disc_node_position(l, j1, j2);
        let l2 = two_disc_node_position(l, j2, j1);
        assert_relative_eq!(l1 + l2, l, epsilon = 1e-12);
        assert_relative_eq!(l1 * j1, l2 * j2, epsilon = 1e-12);
        // Le disque le plus lourd (J2) est le plus proche du nœud.
        assert!(l2 < l1);
    }

    #[test]
    fn realistic_steel_shaft_two_disc() {
        // Arbre acier G=79,3 GPa, d=50 mm → Jp=π d⁴/32, L=1 m ; J1=0,1, J2=0,15 kg·m².
        let d = 0.050_f64;
        let jp = PI * d.powi(4) / 32.0;
        let k = torsional_stiffness(79.3e9, jp, 1.0);
        let wn = two_disc_natural_frequency_rad(k, 0.10, 0.15);
        // fn = ωn/(2π) doit être cohérente avec la forme en rad/s.
        assert_relative_eq!(
            two_disc_natural_frequency_hz(k, 0.10, 0.15),
            wn / (2.0 * PI),
            epsilon = 1e-9
        );
        // Ordre de grandeur attendu : ~900 rad/s pour ces valeurs.
        assert!((850.0..950.0).contains(&wn));
    }

    #[test]
    #[should_panic(expected = "longueur de l'arbre")]
    fn zero_length_panics() {
        torsional_stiffness(80e9, 6.0e-7, 0.0);
    }
}
