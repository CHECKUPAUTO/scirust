//! **Retrait du béton (Eurocode 2)** : déformation totale de retrait comme
//! somme du retrait de dessiccation et du retrait endogène, valeur finale du
//! retrait endogène, fonction temporelle du retrait de dessiccation, rayon
//! moyen (« notional size ») de la section et contrainte engendrée par un
//! retrait gêné.
//!
//! ```text
//! retrait total          εcs   = εcd + εca
//! retrait endogène final εca(∞) = 2,5 · (fck − 10) · 1e−6
//! fonction temporelle    βds   = (t − ts) / ((t − ts) + 0,04 · √(h0³))
//! rayon moyen            h0    = 2 · Ac / u
//! contrainte gênée       σ     = kr · εcs · Ec,eff
//! ```
//!
//! `εcs` déformation totale de retrait (sans dimension), `εcd` retrait de
//! dessiccation (sans dimension), `εca` retrait endogène (sans dimension),
//! `εca(∞)` valeur finale du retrait endogène (sans dimension), `fck`
//! résistance caractéristique en compression du béton (MPa), `βds` fonction
//! temporelle du retrait de dessiccation (sans dimension, comprise entre 0 et
//! 1), `t` âge du béton considéré (jours), `ts` âge du béton au début du
//! séchage (jours), `h0` rayon moyen de la section (mm), `Ac` aire de la
//! section transversale (mm²), `u` périmètre exposé au séchage (mm), `σ`
//! contrainte de traction engendrée par le retrait gêné (même unité que
//! `Ec,eff`), `kr` facteur de bridage/degré de gêne (sans dimension), `Ec,eff`
//! module effectif du béton (tenant compte du fluage, en Pa ou MPa).
//!
//! **Convention** : SI cohérent. Les déformations et facteurs (`εcs`, `εcd`,
//! `εca`, `βds`, `kr`) sont **sans dimension** ; les âges (`t`, `ts`) en
//! **jours** ; les longueurs, aires et périmètres (`h0`, `Ac`, `u`) en **mm**,
//! **mm²** et **mm** de façon cohérente (le rapport `2·Ac/u` restitue alors des
//! **mm**) ; `fck` en **MPa** ; la contrainte `σ` est exprimée dans la **même
//! unité que le module effectif `Ec,eff`** fourni (Pa ou MPa). Types `f64`.
//!
//! **Limite honnête** : formules de l'**Annexe B de l'Eurocode 2 simplifiées**.
//! Le retrait de dessiccation nominal `εcd,0` (dépendant de `fck`, de l'humidité
//! relative et du type de ciment via les coefficients de l'EC2), le retrait de
//! dessiccation `εcd = kh · βds · εcd,0` avec le coefficient `kh` fonction de
//! `h0`, ainsi que le facteur de bridage `kr` et le module effectif `Ec,eff`
//! (couplant fluage et retrait) sont **fournis par l'appelant** d'après
//! l'Eurocode 2 et son Annexe Nationale ; aucune valeur « par défaut » n'est
//! inventée. Le **couplage fluage/retrait** complet relève du module dédié.

/// Déformation totale de retrait `εcs = εcd + εca`, somme du retrait de
/// dessiccation `drying_shrinkage` et du retrait endogène `autogenous_shrinkage`
/// (toutes deux sans dimension).
///
/// `εcs = εcd + εca`
///
/// Panique si l'un des retraits est négatif (`drying_shrinkage < 0` ou
/// `autogenous_shrinkage < 0`).
pub fn shrink_total_strain(drying_shrinkage: f64, autogenous_shrinkage: f64) -> f64 {
    assert!(
        drying_shrinkage >= 0.0,
        "le retrait de dessiccation drying_shrinkage doit être positif ou nul"
    );
    assert!(
        autogenous_shrinkage >= 0.0,
        "le retrait endogène autogenous_shrinkage doit être positif ou nul"
    );
    drying_shrinkage + autogenous_shrinkage
}

/// Valeur finale du retrait endogène `εca(∞) = 2,5 · (fck − 10) · 1e−6`, avec la
/// résistance caractéristique `characteristic_strength` (`fck`) en MPa.
///
/// `εca(∞) = 2,5 · (fck − 10) · 1e−6`
///
/// Note : pour `fck < 10 MPa` la formule renvoie une valeur négative sans
/// signification physique ; le domaine d'application de l'EC2 vise les bétons
/// structuraux (`fck ≥ 12 MPa`).
///
/// Panique si `characteristic_strength <= 0`.
pub fn shrink_autogenous_final(characteristic_strength: f64) -> f64 {
    assert!(
        characteristic_strength > 0.0,
        "la résistance caractéristique characteristic_strength doit être strictement positive"
    );
    2.5 * (characteristic_strength - 10.0) * 1e-6
}

/// Fonction temporelle du retrait de dessiccation
/// `βds = (t − ts) / ((t − ts) + 0,04 · √(h0³))`, avec `time_days` l'âge `t` du
/// béton (jours), `age_at_drying_days` l'âge `ts` au début du séchage (jours) et
/// `notional_size` le rayon moyen `h0` de la section (mm). Le résultat est sans
/// dimension et croît de 0 (à `t = ts`) vers 1 (aux grands âges).
///
/// `βds = (t − ts) / ((t − ts) + 0,04 · √(h0³))`
///
/// Panique si `notional_size <= 0`, si `age_at_drying_days < 0` ou si
/// `time_days < age_at_drying_days`.
pub fn shrink_drying_time_function(
    time_days: f64,
    age_at_drying_days: f64,
    notional_size: f64,
) -> f64 {
    assert!(
        notional_size > 0.0,
        "le rayon moyen notional_size doit être strictement positif"
    );
    assert!(
        age_at_drying_days >= 0.0,
        "l'âge au début du séchage age_at_drying_days doit être positif ou nul"
    );
    assert!(
        time_days >= age_at_drying_days,
        "l'âge time_days doit être supérieur ou égal à age_at_drying_days"
    );
    let elapsed = time_days - age_at_drying_days;
    elapsed / (elapsed + 0.04 * (notional_size.powi(3)).sqrt())
}

/// Rayon moyen (« notional size ») de la section `h0 = 2 · Ac / u`, avec
/// `cross_section_area` l'aire `Ac` de la section (mm²) et `exposed_perimeter`
/// le périmètre `u` exposé au séchage (mm). Le résultat est en mm.
///
/// `h0 = 2 · Ac / u`
///
/// Panique si `cross_section_area <= 0` ou si `exposed_perimeter <= 0`.
pub fn shrink_notional_size(cross_section_area: f64, exposed_perimeter: f64) -> f64 {
    assert!(
        cross_section_area > 0.0,
        "l'aire cross_section_area doit être strictement positive"
    );
    assert!(
        exposed_perimeter > 0.0,
        "le périmètre exposé exposed_perimeter doit être strictement positif"
    );
    2.0 * cross_section_area / exposed_perimeter
}

/// Contrainte engendrée par un retrait gêné `σ = kr · εcs · Ec,eff`, avec
/// `shrinkage_strain` la déformation de retrait `εcs` (sans dimension),
/// `effective_modulus` le module effectif `Ec,eff` du béton (Pa ou MPa) et
/// `restraint_factor` le facteur de bridage `kr` (sans dimension). La contrainte
/// est restituée dans la même unité que `effective_modulus`.
///
/// `σ = kr · εcs · Ec,eff`
///
/// Panique si `effective_modulus <= 0` ou si `restraint_factor < 0`.
pub fn shrink_restrained_stress(
    shrinkage_strain: f64,
    effective_modulus: f64,
    restraint_factor: f64,
) -> f64 {
    assert!(
        effective_modulus > 0.0,
        "le module effectif effective_modulus doit être strictement positif"
    );
    assert!(
        restraint_factor >= 0.0,
        "le facteur de bridage restraint_factor doit être positif ou nul"
    );
    restraint_factor * shrinkage_strain * effective_modulus
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Additivité : le retrait total est bien la somme de ses deux composantes,
    /// et l'opération est commutative.
    #[test]
    fn total_strain_additivity_and_commutativity() {
        let cd = 3.0e-4;
        let ca = 5.0e-5;
        assert_relative_eq!(shrink_total_strain(cd, ca), cd + ca, epsilon = 1e-12);
        assert_relative_eq!(
            shrink_total_strain(cd, ca),
            shrink_total_strain(ca, cd),
            epsilon = 1e-12
        );
    }

    /// Cas chiffré du retrait endogène final pour `fck = 30 MPa` :
    /// `2,5 · (30 − 10) · 1e−6 = 2,5 · 20 · 1e−6 = 5,0e−5`.
    #[test]
    fn autogenous_final_computed_case() {
        // Recalcul indépendant : 2.5 * 20 = 50 ; 50 * 1e-6 = 5.0e-5.
        assert_relative_eq!(shrink_autogenous_final(30.0), 5.0e-5, epsilon = 1e-12);
    }

    /// Limites de la fonction temporelle : nulle à `t = ts`, et croissante vers
    /// 1 aux grands âges.
    #[test]
    fn drying_time_function_limits() {
        let h0 = 200.0;
        assert_relative_eq!(
            shrink_drying_time_function(28.0, 28.0, h0),
            0.0,
            epsilon = 1e-12
        );
        // Aux très grands âges, βds tend vers 1.
        let far = shrink_drying_time_function(1.0e9, 28.0, h0);
        assert!(far > 0.999 && far <= 1.0);
    }

    /// Cas chiffré de la fonction temporelle : `t = 365 j`, `ts = 28 j`,
    /// `h0 = 200 mm`. `t − ts = 337` ; `0,04 · √(200³) = 0,04 · √(8 000 000) =
    /// 0,04 · 2828,42712… = 113,137085…` ; `βds = 337 / (337 + 113,137085…) =
    /// 337 / 450,137085… = 0,748772…`.
    #[test]
    fn drying_time_function_computed_case() {
        // Recalcul indépendant :
        // sqrt(8_000_000) = 2828.4271247... ; * 0.04 = 113.1370850...
        // 337 / (337 + 113.1370850) = 337 / 450.1370850 = 0.74877185...
        let value = shrink_drying_time_function(365.0, 28.0, 200.0);
        assert_relative_eq!(value, 0.748_771_85, epsilon = 1e-3);
    }

    /// Cohérence du rayon moyen `h0 = 2·Ac/u` : section carrée pleine 300×300 mm
    /// séchant sur tout son périmètre. `Ac = 90 000 mm²`, `u = 1 200 mm` ⇒
    /// `h0 = 2 · 90 000 / 1 200 = 150 mm`.
    #[test]
    fn notional_size_computed_case() {
        // Recalcul indépendant : 2 * 90000 = 180000 ; 180000 / 1200 = 150.
        assert_relative_eq!(
            shrink_notional_size(90_000.0, 1_200.0),
            150.0,
            epsilon = 1e-9
        );
    }

    /// Proportionnalité de la contrainte de retrait gêné : elle est linéaire vis
    /// à vis du facteur de bridage et de la déformation, et un cas chiffré
    /// `kr = 0,5`, `εcs = 3,0e−4`, `Ec,eff = 30 000 MPa` donne
    /// `0,5 · 3,0e−4 · 30 000 = 4,5 MPa`.
    #[test]
    fn restrained_stress_proportionality_and_case() {
        let eps = 3.0e-4;
        let e = 30_000.0;
        // Doublement du bridage ⇒ contrainte doublée.
        let s1 = shrink_restrained_stress(eps, e, 0.5);
        let s2 = shrink_restrained_stress(eps, e, 1.0);
        assert_relative_eq!(s2, 2.0 * s1, epsilon = 1e-9);
        // Recalcul indépendant : 3.0e-4 * 30000 = 9.0 ; 9.0 * 0.5 = 4.5.
        assert_relative_eq!(s1, 4.5, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(
        expected = "le périmètre exposé exposed_perimeter doit être strictement positif"
    )]
    fn notional_size_rejects_zero_perimeter() {
        let _ = shrink_notional_size(90_000.0, 0.0);
    }
}
