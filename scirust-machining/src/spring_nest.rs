//! Ressorts hélicoïdaux **concentriques** (imbriqués, montés en **parallèle**) :
//! raideur combinée, répartition de la charge et flèche commune.
//!
//! ```text
//! raideur combinée   k = k₁ + k₂
//! part ressort ext.  F₁ = F·k₁/(k₁ + k₂)
//! part ressort int.  F₂ = F·k₂/(k₁ + k₂)
//! flèche commune     δ = F/(k₁ + k₂)
//! ```
//!
//! `k₁` (`rate_outer`) raideur du ressort extérieur (N/m), `k₂` (`rate_inner`)
//! raideur du ressort intérieur (N/m), `F` (`total_load`) charge totale appliquée
//! à l'ensemble (N), `δ` flèche partagée par les deux ressorts (m). Les deux
//! ressorts étant contraints à la **même flèche**, ils se comportent comme deux
//! raideurs en parallèle : les raideurs s'ajoutent et la charge se répartit au
//! prorata des raideurs.
//!
//! **Convention** : SI cohérent (N, m, N/m). **Limite honnête** : modèle de deux
//! ressorts en parallèle de **même longueur libre** et de même point d'appui,
//! donc de flèche commune, dans le **domaine linéaire** (loi de Hooke, avant
//! bloc à spires jointives). Il ne traite ni les longueurs libres inégales (mise
//! en charge décalée), ni la précontrainte, ni le flambage, ni les effets de fin
//! de course. Les raideurs `k₁`, `k₂` sont **fournies par l'appelant** ; aucune
//! valeur de matériau ou de géométrie n'est supposée « par défaut ».

/// Raideur combinée `k = k₁ + k₂` de deux ressorts imbriqués en parallèle (N/m).
///
/// Panique si `rate_outer < 0`, `rate_inner < 0`, ou si les deux sont nuls.
pub fn nested_spring_combined_rate(rate_outer: f64, rate_inner: f64) -> f64 {
    assert!(
        rate_outer >= 0.0,
        "la raideur du ressort extérieur doit être positive ou nulle"
    );
    assert!(
        rate_inner >= 0.0,
        "la raideur du ressort intérieur doit être positive ou nulle"
    );
    assert!(
        rate_outer + rate_inner > 0.0,
        "k₁ + k₂ doit être strictement positif"
    );
    rate_outer + rate_inner
}

/// Charge reprise par le ressort **extérieur** `F₁ = F·k₁/(k₁ + k₂)` (N).
///
/// Panique si `rate_outer < 0`, `rate_inner < 0`, ou si `k₁ + k₂ <= 0`.
pub fn nested_spring_load_share_outer(total_load: f64, rate_outer: f64, rate_inner: f64) -> f64 {
    assert!(
        rate_outer >= 0.0,
        "la raideur du ressort extérieur doit être positive ou nulle"
    );
    assert!(
        rate_inner >= 0.0,
        "la raideur du ressort intérieur doit être positive ou nulle"
    );
    let combined = rate_outer + rate_inner;
    assert!(combined > 0.0, "k₁ + k₂ doit être strictement positif");
    total_load * rate_outer / combined
}

/// Charge reprise par le ressort **intérieur** `F₂ = F·k₂/(k₁ + k₂)` (N).
///
/// Panique si `rate_outer < 0`, `rate_inner < 0`, ou si `k₁ + k₂ <= 0`.
pub fn nested_spring_load_share_inner(total_load: f64, rate_outer: f64, rate_inner: f64) -> f64 {
    assert!(
        rate_outer >= 0.0,
        "la raideur du ressort extérieur doit être positive ou nulle"
    );
    assert!(
        rate_inner >= 0.0,
        "la raideur du ressort intérieur doit être positive ou nulle"
    );
    let combined = rate_outer + rate_inner;
    assert!(combined > 0.0, "k₁ + k₂ doit être strictement positif");
    total_load * rate_inner / combined
}

/// Flèche commune `δ = F/(k₁ + k₂)` de l'ensemble imbriqué (m).
///
/// Panique si `rate_outer < 0`, `rate_inner < 0`, ou si `k₁ + k₂ <= 0`.
pub fn nested_spring_deflection(total_load: f64, rate_outer: f64, rate_inner: f64) -> f64 {
    assert!(
        rate_outer >= 0.0,
        "la raideur du ressort extérieur doit être positive ou nulle"
    );
    assert!(
        rate_inner >= 0.0,
        "la raideur du ressort intérieur doit être positive ou nulle"
    );
    let combined = rate_outer + rate_inner;
    assert!(combined > 0.0, "k₁ + k₂ doit être strictement positif");
    total_load / combined
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn combined_rate_is_sum() {
        // Deux raideurs en parallèle s'ajoutent : k = k₁ + k₂.
        let k = nested_spring_combined_rate(30_000.0, 20_000.0);
        assert_relative_eq!(k, 50_000.0, max_relative = 1e-12);
    }

    #[test]
    fn load_shares_sum_to_total() {
        // Conservation de la charge : F₁ + F₂ = F.
        let (f, k1, k2) = (1000.0, 30_000.0, 20_000.0);
        let f1 = nested_spring_load_share_outer(f, k1, k2);
        let f2 = nested_spring_load_share_inner(f, k1, k2);
        assert_relative_eq!(f1 + f2, f, max_relative = 1e-12);
    }

    #[test]
    fn shared_deflection_is_consistent() {
        // Chaque ressort à sa propre part suit F_i = k_i·δ : même flèche δ.
        let (f, k1, k2) = (1000.0, 30_000.0, 20_000.0);
        let delta = nested_spring_deflection(f, k1, k2);
        let f1 = nested_spring_load_share_outer(f, k1, k2);
        let f2 = nested_spring_load_share_inner(f, k1, k2);
        assert_relative_eq!(f1, k1 * delta, max_relative = 1e-12);
        assert_relative_eq!(f2, k2 * delta, max_relative = 1e-12);
    }

    #[test]
    fn load_share_ratio_follows_rate_ratio() {
        // F₁/F₂ = k₁/k₂ : la répartition suit exactement le rapport des raideurs.
        let (f, k1, k2) = (500.0, 45_000.0, 15_000.0);
        let f1 = nested_spring_load_share_outer(f, k1, k2);
        let f2 = nested_spring_load_share_inner(f, k1, k2);
        assert_relative_eq!(f1 / f2, k1 / k2, max_relative = 1e-12);
    }

    #[test]
    fn deflection_matches_series_of_definitions() {
        // Cas chiffré : F=1200 N, k₁=40 000, k₂=60 000 N/m ⇒ k=100 000, δ=0,012 m.
        let (f, k1, k2) = (1200.0, 40_000.0, 60_000.0);
        let k = nested_spring_combined_rate(k1, k2);
        let delta = nested_spring_deflection(f, k1, k2);
        assert_relative_eq!(k, 100_000.0, max_relative = 1e-12);
        assert_relative_eq!(delta, 0.012, max_relative = 1e-12);
        // δ redonne bien F par le produit k·δ.
        assert_relative_eq!(k * delta, f, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "k₁ + k₂ doit être strictement positif")]
    fn zero_total_rate_panics() {
        nested_spring_deflection(1000.0, 0.0, 0.0);
    }
}
