//! **Diviseurs de tension et de courant** — module de répartition d'une tension
//! entre deux résistances (pont diviseur, à vide ou chargé) et de répartition
//! d'un courant entre branches parallèles dans un réseau résistif linéaire.
//!
//! ```text
//! diviseur à vide       V_out = V_s·R_low/(R_low + R_up)
//! diviseur chargé       V_out = V_s·R_p/(R_p + R_up)   avec R_p = R_low·R_L/(R_low + R_L)
//! diviseur de courant   I_x   = I_tot·G_x/G_tot
//! diviseur (2 résist.)  I_x   = I_tot·R_autre/(R_x + R_autre)
//! ```
//!
//! `V_s` tension de source idéale (V), `R_low` résistance inférieure du pont
//! (Ω), `R_up` résistance supérieure du pont (Ω), `V_out` tension aux bornes de
//! la résistance inférieure (V), `R_L` résistance de charge branchée en
//! parallèle sur `R_low` (Ω), `R_p` résistance équivalente parallèle
//! `R_low‖R_L` (Ω), `I_tot` courant total entrant dans le nœud (A), `I_x`
//! courant dérivé dans la branche considérée (A), `G_x` conductance de la
//! branche considérée (S), `G_tot` conductance totale du nœud parallèle (S),
//! `R_x` résistance de la branche considérée (Ω), `R_autre` résistance de
//! l'autre branche parallèle (Ω).
//!
//! **Convention** : SI ; tensions en V, courants en A, résistances en Ω,
//! conductances en S. Types `f64`, arithmétique réelle. **Limite honnête** :
//! réseau **résistif linéaire** en régime continu (ou valeurs efficaces d'un
//! régime sinusoïdal en phase), **source de tension idéale** (impédance interne
//! nulle) et source de courant idéale pour le diviseur de courant. Le diviseur
//! **à vide** suppose une charge d'impédance **infinie** (aucun courant
//! prélevé) ; dès qu'une charge finie `R_L` est branchée, elle **abaisse** la
//! tension de sortie et doit être prise en compte (`vdiv_loaded`). Toutes les
//! grandeurs réseau (tensions, courants) et de composant (résistances,
//! conductances) sont **fournies par l'appelant** (mesures, valeurs nominales)
//! — aucune valeur « par défaut » n'est inventée.

/// Tension de sortie d'un pont diviseur **à vide**
/// `V_out = V_s·R_low/(R_low + R_up)` (V), aux bornes de la résistance
/// inférieure, charge supposée d'impédance infinie.
///
/// Panique si `resistance_lower < 0`, si `resistance_upper < 0` ou si leur
/// somme est nulle (division par zéro).
pub fn vdiv_unloaded(source_voltage: f64, resistance_lower: f64, resistance_upper: f64) -> f64 {
    assert!(
        resistance_lower >= 0.0,
        "la résistance inférieure R_low doit être ≥ 0"
    );
    assert!(
        resistance_upper >= 0.0,
        "la résistance supérieure R_up doit être ≥ 0"
    );
    let sum = resistance_lower + resistance_upper;
    assert!(
        sum > 0.0,
        "la somme R_low + R_up doit être strictement positive"
    );
    source_voltage * resistance_lower / sum
}

/// Tension de sortie d'un pont diviseur **chargé**
/// `V_out = V_s·R_p/(R_p + R_up)` avec `R_p = R_low·R_L/(R_low + R_L)` (V) :
/// la charge finie `R_L` en parallèle sur `R_low` abaisse la tension.
///
/// Panique si `resistance_lower <= 0`, si `load_resistance <= 0` ou si
/// `resistance_upper < 0` (grandeurs de composant non physiques ou division
/// par zéro).
pub fn vdiv_loaded(
    source_voltage: f64,
    resistance_lower: f64,
    resistance_upper: f64,
    load_resistance: f64,
) -> f64 {
    assert!(
        resistance_lower > 0.0,
        "la résistance inférieure R_low doit être strictement positive"
    );
    assert!(
        load_resistance > 0.0,
        "la résistance de charge R_L doit être strictement positive"
    );
    assert!(
        resistance_upper >= 0.0,
        "la résistance supérieure R_up doit être ≥ 0"
    );
    let parallel = resistance_lower * load_resistance / (resistance_lower + load_resistance);
    source_voltage * parallel / (parallel + resistance_upper)
}

/// Courant dérivé dans une branche par la **loi du diviseur de courant**
/// `I_x = I_tot·G_x/G_tot` (A), à partir des conductances de branche.
///
/// Panique si `this_branch_conductance < 0` ou si `total_conductance <= 0`
/// (division par zéro).
pub fn vdiv_current_divider(
    total_current: f64,
    this_branch_conductance: f64,
    total_conductance: f64,
) -> f64 {
    assert!(
        this_branch_conductance >= 0.0,
        "la conductance de branche G_x doit être ≥ 0"
    );
    assert!(
        total_conductance > 0.0,
        "la conductance totale G_tot doit être strictement positive"
    );
    total_current * this_branch_conductance / total_conductance
}

/// Courant dérivé dans une branche pour **deux résistances en parallèle**
/// `I_x = I_tot·R_autre/(R_x + R_autre)` (A) : la branche considérée prend une
/// part inversement proportionnelle à sa résistance.
///
/// Panique si `this_resistance < 0`, si `other_resistance < 0` ou si leur
/// somme est nulle (division par zéro).
pub fn vdiv_current_divider_two_resistors(
    total_current: f64,
    this_resistance: f64,
    other_resistance: f64,
) -> f64 {
    assert!(
        this_resistance >= 0.0,
        "la résistance de branche R_x doit être ≥ 0"
    );
    assert!(
        other_resistance >= 0.0,
        "la résistance de l'autre branche R_autre doit être ≥ 0"
    );
    let sum = this_resistance + other_resistance;
    assert!(
        sum > 0.0,
        "la somme R_x + R_autre doit être strictement positive"
    );
    total_current * other_resistance / sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn unloaded_realistic_case() {
        // Cas chiffré réaliste : V_s = 12 V, R_low = 1 kΩ, R_up = 2 kΩ.
        //   V_out = 12·1000/(1000 + 2000) = 12·1000/3000 = 12/3 = 4,0 V.
        let v_out = vdiv_unloaded(12.0, 1_000.0, 2_000.0);
        assert_relative_eq!(v_out, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn unloaded_equal_resistors_halves_source() {
        // Cas limite : deux résistances égales → la sortie vaut la moitié de la
        // source, indépendamment de la valeur commune.
        let r = 4_700.0_f64;
        assert_relative_eq!(vdiv_unloaded(9.0, r, r), 4.5, epsilon = 1e-9);
    }

    #[test]
    fn loaded_reduces_output_versus_unloaded() {
        // Une charge finie ne peut qu'abaisser (ou laisser égale) la tension de
        // sortie par rapport au diviseur à vide, jamais l'augmenter.
        let vs = 10.0_f64;
        let r_low = 1_000.0_f64;
        let r_up = 1_000.0_f64;
        let unloaded = vdiv_unloaded(vs, r_low, r_up);
        let loaded = vdiv_loaded(vs, r_low, r_up, 1_000.0);
        assert!(
            loaded < unloaded,
            "la charge doit abaisser la tension de sortie"
        );
    }

    #[test]
    fn loaded_realistic_case() {
        // Cas chiffré réaliste : V_s = 10 V, R_low = 1 kΩ, R_up = 1 kΩ,
        // R_L = 1 kΩ.
        //   R_p = 1000·1000/(1000 + 1000) = 1e6/2000 = 500 Ω
        //   V_out = 10·500/(500 + 1000) = 10·500/1500 = 5000/1500 = 3,333 33 V
        let v_out = vdiv_loaded(10.0, 1_000.0, 1_000.0, 1_000.0);
        assert_relative_eq!(v_out, 10.0 / 3.0, epsilon = 1e-9);
    }

    #[test]
    fn loaded_tends_to_unloaded_for_large_load() {
        // Cohérence : une charge d'impédance très grande devant R_low rend le
        // diviseur chargé quasi identique au diviseur à vide.
        let vs = 15.0_f64;
        let r_low = 2_200.0_f64;
        let r_up = 3_300.0_f64;
        let unloaded = vdiv_unloaded(vs, r_low, r_up);
        let loaded = vdiv_loaded(vs, r_low, r_up, 1.0e9);
        assert_relative_eq!(loaded, unloaded, epsilon = 1e-3);
    }

    #[test]
    fn current_divider_branches_sum_to_total() {
        // Réciprocité / conservation : dans deux branches parallèles, la somme
        // des courants dérivés vaut le courant total (loi des nœuds).
        // I_tot = 1 A, R1 = 100 Ω, R2 = 300 Ω :
        //   I1 = 1·300/(100 + 300) = 300/400 = 0,75 A
        //   I2 = 1·100/(100 + 300) = 100/400 = 0,25 A ; somme = 1,0 A.
        let i1 = vdiv_current_divider_two_resistors(1.0, 100.0, 300.0);
        let i2 = vdiv_current_divider_two_resistors(1.0, 300.0, 100.0);
        assert_relative_eq!(i1, 0.75, epsilon = 1e-9);
        assert_relative_eq!(i2, 0.25, epsilon = 1e-9);
        assert_relative_eq!(i1 + i2, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn current_divider_conductance_matches_two_resistor_form() {
        // Identité : la forme en conductances `I·G_x/G_tot` coïncide avec la
        // forme à deux résistances `I·R_autre/(R_x + R_autre)`.
        let i_tot = 2.0_f64;
        let r1 = 220.0_f64;
        let r2 = 470.0_f64;
        let g1 = 1.0 / r1;
        let g2 = 1.0 / r2;
        let via_conductance = vdiv_current_divider(i_tot, g1, g1 + g2);
        let via_resistors = vdiv_current_divider_two_resistors(i_tot, r1, r2);
        assert_relative_eq!(via_conductance, via_resistors, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "la conductance totale G_tot doit être strictement positive")]
    fn zero_total_conductance_panics() {
        vdiv_current_divider(1.0, 0.0, 0.0);
    }
}
