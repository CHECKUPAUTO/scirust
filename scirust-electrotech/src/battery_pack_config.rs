//! **Configuration d'un pack de batteries** — module d'assemblage série/parallèle
//! d'un ensemble de cellules identiques : tension de pack, capacité, énergie
//! stockée, nombre total de cellules et courant admissible à un régime C donné.
//!
//! ```text
//! tension du pack       V_pack = V_cell·N_s
//! capacité du pack      Q_pack = Q_cell·N_p
//! énergie du pack       E_pack = V_pack·Q_pack
//! nombre de cellules    N      = N_s·N_p
//! courant continu       I      = C_rate·Q_pack
//! ```
//!
//! `V_cell` tension nominale d'une cellule (V), `N_s` nombre de cellules en
//! série (—), `V_pack` tension nominale du pack (V), `Q_cell` capacité d'une
//! cellule (Ah), `N_p` nombre de branches en parallèle (—), `Q_pack` capacité
//! du pack (Ah), `E_pack` énergie stockée du pack (Wh), `N` nombre total de
//! cellules (—), `C_rate` régime de décharge/charge (1/h, ex. 1 C = pleine
//! capacité en 1 h), `I` courant admissible du pack (A).
//!
//! **Convention** : SI ; tensions en V, capacités en Ah, énergies en Wh,
//! courants en A ; le régime C est exprimé en multiples de la capacité par
//! heure (h⁻¹). **Limite honnête** : pack de cellules **identiques et
//! équilibrées**, chaque branche parallèle portant la même tension et chaque
//! étage série le même courant ; le modèle **ne représente ni le déséquilibre**
//! entre cellules, **ni la température, ni la dégradation** (pour l'état de
//! charge SoC et l'état de santé SoH, voir la crate `scirust-bms`). L'énergie
//! `E_pack = V_pack·Q_pack` suppose une **tension nominale constante** (elle
//! surestime l'énergie réelle, la tension chutant en décharge). La tension et
//! la capacité de cellule, ainsi que le régime C, sont **fournis par la fiche
//! cellule/appelant** — aucune valeur « par défaut » n'est inventée.

/// Tension nominale du pack `V_pack = V_cell·N_s` (V).
///
/// Panique si `cell_voltage < 0` ou si `series_count < 1` (un pack compte au
/// moins une cellule en série).
pub fn packcfg_pack_voltage(cell_voltage: f64, series_count: f64) -> f64 {
    assert!(
        cell_voltage >= 0.0,
        "la tension de cellule V_cell doit être ≥ 0"
    );
    assert!(
        series_count >= 1.0,
        "le nombre de cellules en série N_s doit être ≥ 1"
    );
    cell_voltage * series_count
}

/// Capacité du pack `Q_pack = Q_cell·N_p` (Ah).
///
/// Panique si `cell_capacity < 0` ou si `parallel_count < 1` (un pack compte
/// au moins une branche en parallèle).
pub fn packcfg_pack_capacity(cell_capacity: f64, parallel_count: f64) -> f64 {
    assert!(
        cell_capacity >= 0.0,
        "la capacité de cellule Q_cell doit être ≥ 0"
    );
    assert!(
        parallel_count >= 1.0,
        "le nombre de branches en parallèle N_p doit être ≥ 1"
    );
    cell_capacity * parallel_count
}

/// Énergie stockée du pack `E_pack = V_pack·Q_pack` (Wh), à tension nominale
/// constante.
///
/// Panique si `pack_voltage < 0` ou si `pack_capacity < 0`.
pub fn packcfg_pack_energy(pack_voltage: f64, pack_capacity: f64) -> f64 {
    assert!(
        pack_voltage >= 0.0,
        "la tension du pack V_pack doit être ≥ 0"
    );
    assert!(
        pack_capacity >= 0.0,
        "la capacité du pack Q_pack doit être ≥ 0"
    );
    pack_voltage * pack_capacity
}

/// Nombre total de cellules `N = N_s·N_p` (—).
///
/// Panique si `series_count < 1` ou si `parallel_count < 1`.
pub fn packcfg_cell_count(series_count: f64, parallel_count: f64) -> f64 {
    assert!(
        series_count >= 1.0,
        "le nombre de cellules en série N_s doit être ≥ 1"
    );
    assert!(
        parallel_count >= 1.0,
        "le nombre de branches en parallèle N_p doit être ≥ 1"
    );
    series_count * parallel_count
}

/// Courant admissible du pack à un régime C donné `I = C_rate·Q_pack` (A).
///
/// Panique si `c_rate < 0` ou si `pack_capacity < 0`.
pub fn packcfg_continuous_current(c_rate: f64, pack_capacity: f64) -> f64 {
    assert!(c_rate >= 0.0, "le régime C C_rate doit être ≥ 0");
    assert!(
        pack_capacity >= 0.0,
        "la capacité du pack Q_pack doit être ≥ 0"
    );
    c_rate * pack_capacity
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn pack_voltage_is_linear_in_series_count() {
        // Proportionnalité : V_pack = V_cell·N_s est linéaire en N_s ; doubler
        // le nombre de cellules en série double la tension du pack.
        let v_cell = 3.7_f64;
        let v1 = packcfg_pack_voltage(v_cell, 10.0);
        let v2 = packcfg_pack_voltage(v_cell, 20.0);
        assert_relative_eq!(v2, 2.0 * v1, epsilon = 1e-12);
    }

    #[test]
    fn capacity_is_linear_in_parallel_count() {
        // Proportionnalité : Q_pack = Q_cell·N_p est linéaire en N_p ; tripler
        // le nombre de branches parallèles triple la capacité du pack.
        let q_cell = 3.4_f64;
        let q1 = packcfg_pack_capacity(q_cell, 1.0);
        let q3 = packcfg_pack_capacity(q_cell, 3.0);
        assert_relative_eq!(q3, 3.0 * q1, epsilon = 1e-12);
    }

    #[test]
    fn energy_factorizes_through_cell_grandeurs() {
        // Cohérence : E_pack = V_pack·Q_pack = (V_cell·N_s)·(Q_cell·N_p), soit
        // aussi (V_cell·Q_cell)·(N_s·N_p) = énergie d'une cellule × N. On relie
        // l'énergie du pack à l'énergie unitaire et au nombre de cellules.
        let v_cell = 3.6_f64;
        let q_cell = 2.5_f64;
        let n_s = 12.0_f64;
        let n_p = 6.0_f64;
        let v_pack = packcfg_pack_voltage(v_cell, n_s);
        let q_pack = packcfg_pack_capacity(q_cell, n_p);
        let e_pack = packcfg_pack_energy(v_pack, q_pack);
        let n = packcfg_cell_count(n_s, n_p);
        assert_relative_eq!(e_pack, (v_cell * q_cell) * n, epsilon = 1e-9);
    }

    #[test]
    fn c_rate_reciprocity_on_current() {
        // Réciprocité : à 1 C le courant vaut numériquement la capacité, et le
        // courant est proportionnel au régime C. À 2 C il double celui de 1 C.
        let q_pack = 13.6_f64;
        let i_1c = packcfg_continuous_current(1.0, q_pack);
        assert_relative_eq!(i_1c, q_pack, epsilon = 1e-12);
        let i_2c = packcfg_continuous_current(2.0, q_pack);
        assert_relative_eq!(i_2c, 2.0 * i_1c, epsilon = 1e-12);
    }

    #[test]
    fn realistic_21s4p_pack_case() {
        // Cas chiffré réaliste : pack 21S4P de cellules 18650 (3,7 V, 3,4 Ah),
        // régime 2 C.
        //   V_pack = 3,7·21             = 77,7 V
        //   Q_pack = 3,4·4              = 13,6 Ah
        //   E_pack = 77,7·13,6          = 1056,72 Wh
        //   N      = 21·4               = 84 cellules
        //   I      = 2·13,6             = 27,2 A
        let v_cell = 3.7_f64;
        let q_cell = 3.4_f64;
        let n_s = 21.0_f64;
        let n_p = 4.0_f64;
        let v_pack = packcfg_pack_voltage(v_cell, n_s);
        let q_pack = packcfg_pack_capacity(q_cell, n_p);
        assert_relative_eq!(v_pack, 77.7, epsilon = 1e-3);
        assert_relative_eq!(q_pack, 13.6, epsilon = 1e-3);
        assert_relative_eq!(
            packcfg_pack_energy(v_pack, q_pack),
            1_056.72,
            epsilon = 1e-3
        );
        assert_relative_eq!(packcfg_cell_count(n_s, n_p), 84.0, epsilon = 1e-9);
        assert_relative_eq!(
            packcfg_continuous_current(2.0, q_pack),
            27.2,
            epsilon = 1e-3
        );
    }

    #[test]
    #[should_panic(expected = "le nombre de cellules en série N_s doit être ≥ 1")]
    fn zero_series_count_panics() {
        packcfg_pack_voltage(3.7, 0.0);
    }
}
