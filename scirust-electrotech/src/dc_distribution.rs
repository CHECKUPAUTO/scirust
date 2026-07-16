//! **Distribution en courant continu** — chute de tension d'un feeder CC deux
//! fils, section de conducteur requise pour une chute admissible, pertes Joule
//! dans la boucle, rendement de la ligne et chute pour une charge uniformément
//! répartie, à partir de la résistivité et de la géométrie du conducteur.
//!
//! ```text
//! chute deux fils (charge en bout)   ΔU = 2 · I · ρ · L / A
//! section pour chute admissible       A  = 2 · I · ρ · L / ΔU_adm
//! pertes Joule dans la boucle         P  = I² · R_boucle
//! rendement de la ligne               η  = P_reçue / (P_reçue + P_pertes)
//! chute charge répartie (deux fils)   ΔU = i · ρ · L² / (2 · A)
//! ```
//!
//! `I` courant de service (A), `ρ` résistivité du conducteur (Ω·m), `L`
//! longueur simple du feeder (m), `A` section du conducteur (m²), `ΔU` chute de
//! tension aller-retour (V), `ΔU_adm` chute admissible imposée (V), `R_boucle`
//! résistance totale de la boucle aller-retour (Ω), `P` pertes Joule (W),
//! `P_reçue` puissance utile livrée à la charge (W), `P_pertes` pertes dans la
//! ligne (W), `η` rendement (sans dimension, `∈ ]0, 1]`), `i` courant par unité
//! de longueur d'une charge uniformément répartie (A/m).
//!
//! **Convention** : SI ; tensions et chutes en V, courants en A, courant
//! linéique en A/m, sections en m², longueurs en m, résistivités en Ω·m,
//! résistances en Ω, puissances en W. Le facteur `2` des formules deux fils
//! compte **l'aller ET le retour** du conducteur ; `L` est la longueur simple
//! (aller). **Limite honnête** : distribution en courant **continu** ; la
//! résistivité `ρ`, la géométrie (`L`, `A`) et les courants sont **fournis par
//! l'appelant** — aucune valeur « typique » n'est inventée. Il n'y a **aucune
//! réactance** en continu : seule la **résistance** intervient (régime
//! **permanent**). Pour une charge **uniformément répartie**, la chute vaut la
//! **moitié** de celle d'une charge de même courant total **concentrée en
//! bout** de ligne (le courant décroît linéairement le long du feeder).

/// Chute de tension aller-retour d'un feeder CC **deux fils** avec charge
/// concentrée en bout `ΔU = 2 · I · ρ · L / A` (V).
///
/// Le facteur `2` compte les conducteurs **aller et retour** ; `L` est la
/// longueur simple du feeder.
///
/// Panique si `current < 0`, si `resistivity < 0`, si `length < 0` ou si
/// `cross_section_area <= 0` (division par zéro).
pub fn dcdist_two_wire_drop(
    current: f64,
    resistivity: f64,
    length: f64,
    cross_section_area: f64,
) -> f64 {
    assert!(current >= 0.0, "le courant current doit être ≥ 0");
    assert!(
        resistivity >= 0.0,
        "la résistivité resistivity doit être ≥ 0"
    );
    assert!(length >= 0.0, "la longueur length doit être ≥ 0");
    assert!(
        cross_section_area > 0.0,
        "la section cross_section_area doit être strictement positive"
    );
    2.0 * current * resistivity * length / cross_section_area
}

/// Section de conducteur requise pour respecter une chute admissible sur un
/// feeder CC deux fils `A = 2 · I · ρ · L / ΔU_adm` (m²).
///
/// Inverse de [`dcdist_two_wire_drop`] par rapport à la section : c'est la
/// section minimale telle que la chute aller-retour n'excède pas `allowable_drop`.
///
/// Panique si `current < 0`, si `resistivity < 0`, si `length < 0` ou si
/// `allowable_drop <= 0` (division par zéro ou chute admissible non physique).
pub fn dcdist_conductor_size_for_drop(
    current: f64,
    resistivity: f64,
    length: f64,
    allowable_drop: f64,
) -> f64 {
    assert!(current >= 0.0, "le courant current doit être ≥ 0");
    assert!(
        resistivity >= 0.0,
        "la résistivité resistivity doit être ≥ 0"
    );
    assert!(length >= 0.0, "la longueur length doit être ≥ 0");
    assert!(
        allowable_drop > 0.0,
        "la chute admissible allowable_drop doit être strictement positive"
    );
    2.0 * current * resistivity * length / allowable_drop
}

/// Pertes Joule dissipées dans la boucle aller-retour `P = I² · R_boucle` (W),
/// à partir du courant et de la résistance **totale** de la boucle.
///
/// `loop_resistance` est la résistance de l'ensemble aller + retour ; le facteur
/// `2` de la géométrie deux fils est déjà **inclus** dans cette résistance.
///
/// Panique si `loop_resistance < 0` (résistance non physique). Un `current`
/// négatif est accepté : `I²` étant pair, la puissance reste bien définie.
pub fn dcdist_power_loss(current: f64, loop_resistance: f64) -> f64 {
    assert!(
        loop_resistance >= 0.0,
        "la résistance de boucle loop_resistance doit être ≥ 0"
    );
    current * current * loop_resistance
}

/// Rendement de la ligne `η = P_reçue / (P_reçue + P_pertes)` (sans dimension),
/// rapport de la puissance utile livrée à la puissance envoyée (utile + pertes).
///
/// Panique si `receiving_power < 0`, si `power_loss < 0` ou si la somme
/// `receiving_power + power_loss <= 0` (division par zéro : aucune puissance
/// transitée).
pub fn dcdist_efficiency(receiving_power: f64, power_loss: f64) -> f64 {
    assert!(
        receiving_power >= 0.0,
        "la puissance reçue receiving_power doit être ≥ 0"
    );
    assert!(power_loss >= 0.0, "les pertes power_loss doivent être ≥ 0");
    let sent = receiving_power + power_loss;
    assert!(
        sent > 0.0,
        "la puissance envoyée receiving_power + power_loss doit être strictement positive"
    );
    receiving_power / sent
}

/// Chute de tension d'une charge **uniformément répartie** sur un feeder CC
/// deux fils `ΔU = i · ρ · L² / A` (V).
///
/// `current_per_length` est le courant linéique `i` (A/m) supposé constant le
/// long de la ligne ; le courant total est `i · L`. La chute vaut la **moitié**
/// de celle qu'on obtiendrait avec le même courant total concentré en bout.
///
/// Panique si `current_per_length < 0`, si `resistivity < 0`, si `length < 0`
/// ou si `cross_section_area <= 0` (division par zéro).
pub fn dcdist_distributed_load_drop(
    current_per_length: f64,
    resistivity: f64,
    length: f64,
    cross_section_area: f64,
) -> f64 {
    assert!(
        current_per_length >= 0.0,
        "le courant linéique current_per_length doit être ≥ 0"
    );
    assert!(
        resistivity >= 0.0,
        "la résistivité resistivity doit être ≥ 0"
    );
    assert!(length >= 0.0, "la longueur length doit être ≥ 0");
    assert!(
        cross_section_area > 0.0,
        "la section cross_section_area doit être strictement positive"
    );
    current_per_length * resistivity * length * length / cross_section_area
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn two_wire_drop_numeric() {
        // Cas chiffré : I = 20 A, ρ = 1,72e-8 Ω·m (cuivre fourni), L = 50 m,
        //   A = 4e-6 m² (4 mm²).
        //   ΔU = 2 · 20 · 1,72e-8 · 50 / 4e-6
        //      = 2 · 20 · 1,72e-8 · 50 = 3,44e-5 ; / 4e-6 = 8,6 V.
        // Recalcul indépendant : 20·1,72e-8 = 3,44e-7 ; ·50 = 1,72e-5 ;
        //   ·2 = 3,44e-5 ; /4e-6 = 8,6.
        let du = dcdist_two_wire_drop(20.0, 1.72e-8, 50.0, 4.0e-6);
        assert_relative_eq!(du, 8.6, epsilon = 1e-3);
    }

    #[test]
    fn conductor_size_inverts_two_wire_drop() {
        // Réciprocité : la section dimensionnée pour une chute admissible donnée
        // reproduit exactement cette chute quand on la réinjecte dans la formule
        // directe.
        let (i, rho, l, du_adm) = (20.0, 1.72e-8, 50.0, 8.6);
        let a = dcdist_conductor_size_for_drop(i, rho, l, du_adm);
        let du = dcdist_two_wire_drop(i, rho, l, a);
        assert_relative_eq!(du, du_adm, epsilon = 1e-9);
    }

    #[test]
    fn drop_proportional_to_length() {
        // Proportionnalité : à section et courant fixés, la chute deux fils est
        // proportionnelle à la longueur — doubler L double ΔU.
        let du1 = dcdist_two_wire_drop(10.0, 1.72e-8, 30.0, 6.0e-6);
        let du2 = dcdist_two_wire_drop(10.0, 1.72e-8, 60.0, 6.0e-6);
        assert_relative_eq!(du2, 2.0 * du1, epsilon = 1e-12);
    }

    #[test]
    fn distributed_load_is_half_of_concentrated() {
        // Identité clé : une charge de courant total I_tot = i·L uniformément
        // répartie provoque la moitié de la chute d'une charge I_tot concentrée
        // en bout, sur la même géométrie deux fils.
        let (i_lin, rho, l, a) = (2.0, 1.72e-8, 40.0, 5.0e-6);
        let i_tot = i_lin * l; // 80 A concentrés
        let du_distributed = dcdist_distributed_load_drop(i_lin, rho, l, a);
        let du_concentrated = dcdist_two_wire_drop(i_tot, rho, l, a);
        assert_relative_eq!(du_distributed, 0.5 * du_concentrated, epsilon = 1e-9);
    }

    #[test]
    fn power_loss_and_efficiency_consistent() {
        // Cas chiffré pertes : I = 25 A, R_boucle = 0,4 Ω → P = 25²·0,4
        //   = 625·0,4 = 250 W. Recalcul : 25·25 = 625 ; ·0,4 = 250.
        let p_loss = dcdist_power_loss(25.0, 0.4);
        assert_relative_eq!(p_loss, 250.0, epsilon = 1e-9);
        // Rendement avec P_reçue = 4750 W : η = 4750/(4750+250) = 4750/5000
        //   = 0,95.
        let eta = dcdist_efficiency(4750.0, p_loss);
        assert_relative_eq!(eta, 0.95, epsilon = 1e-9);
        assert!(eta > 0.0 && eta <= 1.0);
    }

    #[test]
    fn efficiency_unity_without_loss() {
        // Limite : sans pertes, le rendement vaut exactement 1.
        assert_relative_eq!(dcdist_efficiency(1000.0, 0.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "la section cross_section_area doit être strictement positive")]
    fn zero_section_panics() {
        dcdist_two_wire_drop(20.0, 1.72e-8, 50.0, 0.0);
    }
}
