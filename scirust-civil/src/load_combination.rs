//! **Actions — combinaisons d'actions** selon l'**Eurocode 0** (EN 1990) :
//! combinaison **fondamentale** à l'état-limite ultime (ELU), combinaisons
//! **caractéristique** et **quasi-permanente** à l'état-limite de service (ELS),
//! et combinaison **accidentelle**. Chaque fonction assemble une valeur de
//! calcul à partir d'actions caractéristiques et de coefficients fournis.
//!
//! ```text
//! ELU fondamentale     Ed = γG·Gk + γQ·Qk,1 + Σ γQ,i·ψ0,i·Qk,i
//! ELS caractéristique  Ed = Gk + Qk,1 + Σ ψ0,i·Qk,i
//! ELS quasi-permanente Ed = Gk + Σ ψ2,i·Qk,i
//! accidentelle         Ed = Gk + Ad + ψ1,1·Qk,1 + Σ ψ2,i·Qk,i
//! ```
//!
//! `Gk` action permanente caractéristique, `Qk,1` action variable dominante
//! caractéristique, `Qk,i` actions variables d'accompagnement caractéristiques,
//! `Ad` action accidentelle de calcul, `γG` coefficient partiel des actions
//! permanentes (sans dimension), `γQ` / `γQ,i` coefficients partiels des actions
//! variables (sans dimension), `ψ0,i` coefficient de combinaison, `ψ1,1`
//! coefficient de valeur fréquente, `ψ2,i` coefficient de valeur quasi-permanente
//! (tous sans dimension), `Ed` valeur de calcul de la sollicitation résultante.
//!
//! **Convention** : **SI cohérent** — toutes les actions caractéristiques
//! (`Gk`, `Qk,i`, `Ad`) sont exprimées dans **la même unité** (par exemple des
//! **forces en N**, ou des **charges linéiques en N/m**, ou des **moments en
//! N·m**) ; la combinaison est **linéaire** et renvoie la valeur de calcul dans
//! **cette même unité**. Les coefficients `γ` et `ψ` sont **sans dimension**.
//! Types `f64`.
//!
//! **Limite honnête** : simple **assemblage linéaire** des combinaisons de
//! l'**Eurocode 0** (EN 1990, § 6.4.3 et 6.5.3). Les **coefficients partiels**
//! `γG` / `γQ` et les **coefficients** `ψ0` / `ψ1` / `ψ2` sont **fournis par
//! l'appelant** d'après l'**EN 1990** et l'**Annexe Nationale** (tableaux A1.1
//! et A1.2) — aucune valeur « par défaut » n'est inventée. Les **actions
//! caractéristiques** (`Gk`, `Qk,i`, `Ad`) sont elles aussi **fournies**. Ce
//! module **ne recherche pas** l'action variable **dominante** : c'est à
//! l'appelant de **tester tour à tour** chaque action variable comme dominante
//! et de retenir l'enveloppe la plus défavorable. La **distinction actions
//! favorables / défavorables** (choix de `γG,sup` ou `γG,inf`) relève également
//! du choix de l'appelant, tout comme les combinaisons **sismiques** ou aux
//! **fatigue**, non traitées ici.

/// Combinaison **fondamentale** à l'ELU (EN 1990 § 6.4.3.2, éq. 6.10) :
/// `Ed = γG·Gk + γQ·Qk,1 + Σ γQ,i·ψ0,i·Qk,i`.
///
/// `permanent_factor` = `γG` coefficient partiel de l'action permanente,
/// `permanent_load` = `Gk` action permanente caractéristique, `variable_factor`
/// = `γQ` coefficient partiel de l'action variable dominante, `leading_variable`
/// = `Qk,1` action variable dominante caractéristique, `accompanying` = tranche
/// des actions variables d'accompagnement, chacune sous la forme
/// `(γQ,i, ψ0,i, Qk,i)` ; renvoie la valeur de calcul `Ed` dans l'unité commune
/// des actions.
///
/// Panique si un coefficient (`permanent_factor`, `variable_factor`, ou un `γQ,i`
/// ou `ψ0,i` d'accompagnement) est négatif ou non fini, ou si une action fournie
/// est non finie (coefficients partiels physiquement ≥ 0).
pub fn loadcomb_uls_fundamental(
    permanent_factor: f64,
    permanent_load: f64,
    variable_factor: f64,
    leading_variable: f64,
    accompanying: &[(f64, f64, f64)],
) -> f64 {
    assert!(
        permanent_factor >= 0.0 && permanent_factor.is_finite(),
        "le coefficient partiel γG doit être fini et ≥ 0"
    );
    assert!(
        variable_factor >= 0.0 && variable_factor.is_finite(),
        "le coefficient partiel γQ doit être fini et ≥ 0"
    );
    assert!(
        permanent_load.is_finite(),
        "l'action permanente Gk doit être finie"
    );
    assert!(
        leading_variable.is_finite(),
        "l'action variable dominante Qk,1 doit être finie"
    );
    let mut total = permanent_factor * permanent_load + variable_factor * leading_variable;
    for &(gamma, psi0, q) in accompanying
    {
        assert!(
            gamma >= 0.0 && gamma.is_finite(),
            "un coefficient partiel γQ,i d'accompagnement doit être fini et ≥ 0"
        );
        assert!(
            psi0 >= 0.0 && psi0.is_finite(),
            "un coefficient ψ0,i d'accompagnement doit être fini et ≥ 0"
        );
        assert!(
            q.is_finite(),
            "une action Qk,i d'accompagnement doit être finie"
        );
        total += gamma * psi0 * q;
    }
    total
}

/// Combinaison **caractéristique** (rare) à l'ELS (EN 1990 § 6.5.3, éq. 6.14b) :
/// `Ed = Gk + Qk,1 + Σ ψ0,i·Qk,i`. Tous les coefficients partiels `γ` valent 1 à
/// l'ELS.
///
/// `permanent_load` = `Gk` action permanente caractéristique, `leading_variable`
/// = `Qk,1` action variable dominante caractéristique, `accompanying` = tranche
/// des actions d'accompagnement, chacune sous la forme `(ψ0,i, Qk,i)` ; renvoie
/// la valeur de calcul `Ed` dans l'unité commune des actions.
///
/// Panique si un coefficient `ψ0,i` d'accompagnement est négatif ou non fini, ou
/// si une action fournie est non finie (coefficients de combinaison ≥ 0).
pub fn loadcomb_sls_characteristic(
    permanent_load: f64,
    leading_variable: f64,
    accompanying: &[(f64, f64)],
) -> f64 {
    assert!(
        permanent_load.is_finite(),
        "l'action permanente Gk doit être finie"
    );
    assert!(
        leading_variable.is_finite(),
        "l'action variable dominante Qk,1 doit être finie"
    );
    let mut total = permanent_load + leading_variable;
    for &(psi0, q) in accompanying
    {
        assert!(
            psi0 >= 0.0 && psi0.is_finite(),
            "un coefficient ψ0,i d'accompagnement doit être fini et ≥ 0"
        );
        assert!(
            q.is_finite(),
            "une action Qk,i d'accompagnement doit être finie"
        );
        total += psi0 * q;
    }
    total
}

/// Combinaison **quasi-permanente** à l'ELS (EN 1990 § 6.5.3, éq. 6.16b) :
/// `Ed = Gk + Σ ψ2,i·Qk,i`. Toutes les actions variables sont pondérées par leur
/// coefficient quasi-permanent `ψ2,i` (l'action dominante n'y est pas
/// distinguée).
///
/// `permanent_load` = `Gk` action permanente caractéristique, `variables` =
/// tranche des actions variables, chacune sous la forme `(ψ2,i, Qk,i)` ; renvoie
/// la valeur de calcul `Ed` dans l'unité commune des actions.
///
/// Panique si un coefficient `ψ2,i` est négatif ou non fini, ou si une action
/// fournie est non finie (coefficients quasi-permanents ≥ 0).
pub fn loadcomb_sls_quasi_permanent(permanent_load: f64, variables: &[(f64, f64)]) -> f64 {
    assert!(
        permanent_load.is_finite(),
        "l'action permanente Gk doit être finie"
    );
    let mut total = permanent_load;
    for &(psi2, q) in variables
    {
        assert!(
            psi2 >= 0.0 && psi2.is_finite(),
            "un coefficient ψ2,i doit être fini et ≥ 0"
        );
        assert!(q.is_finite(), "une action Qk,i doit être finie");
        total += psi2 * q;
    }
    total
}

/// Combinaison **accidentelle** (EN 1990 § 6.4.3.3, éq. 6.11b) :
/// `Ed = Gk + Ad + ψ1,1·Qk,1 + Σ ψ2,i·Qk,i`. L'action accidentelle `Ad` s'ajoute
/// directement (son coefficient partiel vaut 1) ; l'action variable dominante est
/// prise à sa valeur **fréquente** `ψ1,1·Qk,1`, les autres à leur valeur
/// **quasi-permanente** `ψ2,i·Qk,i`.
///
/// `permanent_load` = `Gk` action permanente caractéristique, `accidental_action`
/// = `Ad` action accidentelle de calcul, `leading_psi` = `ψ1,1` coefficient de
/// valeur fréquente de l'action dominante, `leading_variable` = `Qk,1` action
/// variable dominante caractéristique, `accompanying` = tranche des actions
/// d'accompagnement, chacune sous la forme `(ψ2,i, Qk,i)` ; renvoie la valeur de
/// calcul `Ed` dans l'unité commune des actions.
///
/// Panique si `leading_psi` ou un `ψ2,i` d'accompagnement est négatif ou non
/// fini, ou si une action fournie est non finie (coefficients ψ ≥ 0).
pub fn loadcomb_accidental(
    permanent_load: f64,
    accidental_action: f64,
    leading_psi: f64,
    leading_variable: f64,
    accompanying: &[(f64, f64)],
) -> f64 {
    assert!(
        permanent_load.is_finite(),
        "l'action permanente Gk doit être finie"
    );
    assert!(
        accidental_action.is_finite(),
        "l'action accidentelle Ad doit être finie"
    );
    assert!(
        leading_psi >= 0.0 && leading_psi.is_finite(),
        "le coefficient ψ1,1 doit être fini et ≥ 0"
    );
    assert!(
        leading_variable.is_finite(),
        "l'action variable dominante Qk,1 doit être finie"
    );
    let mut total = permanent_load + accidental_action + leading_psi * leading_variable;
    for &(psi2, q) in accompanying
    {
        assert!(
            psi2 >= 0.0 && psi2.is_finite(),
            "un coefficient ψ2,i d'accompagnement doit être fini et ≥ 0"
        );
        assert!(
            q.is_finite(),
            "une action Qk,i d'accompagnement doit être finie"
        );
        total += psi2 * q;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn fundamental_reduces_to_sum_with_unit_factors() {
        // Avec γG = γQ = 1 et aucune action d'accompagnement, la combinaison
        // fondamentale se réduit à la simple somme Gk + Qk,1.
        let ed = loadcomb_uls_fundamental(1.0, 50_000.0, 1.0, 30_000.0, &[]);
        assert_relative_eq!(ed, 80_000.0, epsilon = 1e-6);
    }

    #[test]
    fn characteristic_reduces_to_sum_without_accompanying() {
        // Sans action d'accompagnement, la combinaison caractéristique ELS vaut
        // exactement Gk + Qk,1.
        let ed = loadcomb_sls_characteristic(50_000.0, 30_000.0, &[]);
        assert_relative_eq!(ed, 80_000.0, epsilon = 1e-6);
    }

    #[test]
    fn quasi_permanent_reduces_to_permanent_alone() {
        // Sans action variable, la combinaison quasi-permanente vaut Gk seul ;
        // et avec ψ2 = 1 pour chaque variable elle redonne la somme brute.
        assert_relative_eq!(
            loadcomb_sls_quasi_permanent(50_000.0, &[]),
            50_000.0,
            epsilon = 1e-6
        );
        let ed = loadcomb_sls_quasi_permanent(50_000.0, &[(1.0, 30_000.0), (1.0, 20_000.0)]);
        assert_relative_eq!(ed, 100_000.0, epsilon = 1e-6);
    }

    #[test]
    fn accompanying_terms_scale_linearly() {
        // La combinaison est linéaire : doubler une action d'accompagnement
        // augmente Ed exactement de γQ,i·ψ0,i·ΔQk,i.
        let base = loadcomb_uls_fundamental(1.35, 40_000.0, 1.5, 25_000.0, &[(1.5, 0.7, 10_000.0)]);
        let more = loadcomb_uls_fundamental(1.35, 40_000.0, 1.5, 25_000.0, &[(1.5, 0.7, 20_000.0)]);
        // Δ = 1,5 · 0,7 · (20 000 − 10 000) = 1,05 · 10 000 = 10 500.
        assert_relative_eq!(more - base, 10_500.0, epsilon = 1e-6);
    }

    #[test]
    fn realistic_office_floor_case() {
        // Plancher de bureau, actions en N (efforts cohérents) :
        //   Gk = 50 000, Qk,1 (exploitation) = 30 000, Qk,2 (neige/vent) = 20 000.
        //   Coefficients EN 1990 / AN : γG = 1,35, γQ = 1,5, ψ0 = 0,7,
        //   ψ1 = 0,5, ψ2 = 0,3, action accidentelle Ad = 100 000.
        let gk = 50_000.0_f64;
        let q1 = 30_000.0_f64;
        let q2 = 20_000.0_f64;

        // ELU fondamentale :
        //   1,35·50 000 + 1,5·30 000 + 1,5·0,7·20 000
        //   = 67 500 + 45 000 + 21 000 = 133 500.
        let elu = loadcomb_uls_fundamental(1.35, gk, 1.5, q1, &[(1.5, 0.7, q2)]);
        assert_relative_eq!(elu, 133_500.0, epsilon = 1e-3);

        // ELS caractéristique :
        //   50 000 + 30 000 + 0,7·20 000 = 50 000 + 30 000 + 14 000 = 94 000.
        let els_c = loadcomb_sls_characteristic(gk, q1, &[(0.7, q2)]);
        assert_relative_eq!(els_c, 94_000.0, epsilon = 1e-3);

        // ELS quasi-permanente :
        //   50 000 + 0,3·30 000 + 0,3·20 000 = 50 000 + 9 000 + 6 000 = 65 000.
        let els_qp = loadcomb_sls_quasi_permanent(gk, &[(0.3, q1), (0.3, q2)]);
        assert_relative_eq!(els_qp, 65_000.0, epsilon = 1e-3);

        // Accidentelle :
        //   50 000 + 100 000 + 0,5·30 000 + 0,3·20 000
        //   = 50 000 + 100 000 + 15 000 + 6 000 = 171 000.
        let acc = loadcomb_accidental(gk, 100_000.0, 0.5, q1, &[(0.3, q2)]);
        assert_relative_eq!(acc, 171_000.0, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "le coefficient partiel γG doit être fini et ≥ 0")]
    fn negative_permanent_factor_panics() {
        loadcomb_uls_fundamental(-1.35, 50_000.0, 1.5, 30_000.0, &[]);
    }
}
