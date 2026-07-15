//! AMDEC — indice de priorité de risque (RPN) issu du triplet de notes
//! ordinales gravité, occurrence et détection.
//!
//! ```text
//! indice de priorité   RPN = S · O · D
//! criticité            C   = S · O
//! dépassement de seuil  RPN > seuil ?  (booléen)
//! RPN normalisé        RPNn = RPN / 1000
//! ```
//!
//! `S` gravité (severity), `O` occurrence, `D` détection : notes ORDINALES sans
//! dimension, chacune dans l'intervalle entier `[1, 10]`. `RPN` indice de
//! priorité de risque (sans dimension, dans `[1, 1000]`), `C` criticité
//! gravité×occurrence (sans dimension, dans `[1, 100]`), `RPNn` RPN normalisé
//! sur son maximum théorique `10 · 10 · 10 = 1000` (sans dimension, dans
//! `[0.001, 1]`), `seuil` seuil d'action (sans dimension, même échelle que le
//! RPN).
//!
//! **Limite honnête** : la notation est ORDINALE — les notes `S`, `O`, `D`
//! sont FOURNIES par l'analyse AMDEC selon ses grilles de cotation, jamais
//! inventées ici. Le RPN est un OUTIL DE PRIORISATION RELATIF : le produit de
//! notes ordinales n'a pas de sens métrique absolu (un RPN de 200 n'est pas
//! « deux fois pire » qu'un RPN de 100), il sert seulement à ordonner les modes
//! de défaillance. Le seuil d'action est FOURNI par la politique qualité. Cet
//! indice ne remplace pas une analyse de criticité pondérée ni une évaluation
//! quantitative des risques.

/// Vérifie qu'une note AMDEC est une note ordinale valide dans `[1, 10]`.
fn assert_rating(rating: f64, name: &str) {
    assert!(
        rating.is_finite(),
        "la note {name} doit être un nombre fini"
    );
    assert!(
        (1.0..=10.0).contains(&rating),
        "la note {name} doit être dans l'intervalle [1, 10]"
    );
}

/// Indice de priorité de risque `RPN = S · O · D` (AMDEC).
///
/// Produit des trois notes ordinales gravité, occurrence et détection ; sert à
/// PRIORISER les modes de défaillance (valeur relative, sans sens métrique
/// absolu).
///
/// Panique si `severity`, `occurrence` ou `detection` sort de l'intervalle
/// `[1, 10]` (ou n'est pas fini).
pub fn fmea_risk_priority_number(severity: f64, occurrence: f64, detection: f64) -> f64 {
    assert_rating(severity, "gravité");
    assert_rating(occurrence, "occurrence");
    assert_rating(detection, "détection");
    severity * occurrence * detection
}

/// Criticité `C = S · O` (AMDEC).
///
/// Produit gravité×occurrence, indépendant de la détectabilité ; mesure
/// relative de l'importance d'un mode de défaillance avant prise en compte des
/// moyens de détection.
///
/// Panique si `severity` ou `occurrence` sort de l'intervalle `[1, 10]` (ou
/// n'est pas fini).
pub fn fmea_criticality(severity: f64, occurrence: f64) -> f64 {
    assert_rating(severity, "gravité");
    assert_rating(occurrence, "occurrence");
    severity * occurrence
}

/// Dépassement d'un seuil d'action `RPN > seuil` (AMDEC).
///
/// Renvoie `true` si l'indice de priorité de risque dépasse STRICTEMENT le
/// seuil d'action FOURNI par la politique qualité (déclenchement d'une action
/// corrective).
///
/// Panique si `risk_priority_number` ou `threshold` n'est pas fini, ou si
/// l'un des deux est strictement négatif.
pub fn fmea_exceeds_threshold(risk_priority_number: f64, threshold: f64) -> bool {
    assert!(
        risk_priority_number.is_finite(),
        "le RPN doit être un nombre fini"
    );
    assert!(threshold.is_finite(), "le seuil doit être un nombre fini");
    assert!(
        risk_priority_number >= 0.0,
        "le RPN doit être positif ou nul"
    );
    assert!(threshold >= 0.0, "le seuil doit être positif ou nul");
    risk_priority_number > threshold
}

/// RPN normalisé `RPNn = RPN / 1000` (AMDEC).
///
/// Rapporte l'indice de priorité de risque à son maximum théorique
/// `10 · 10 · 10 = 1000`, produisant une valeur dans `[0.001, 1]` pour un RPN
/// valide.
///
/// Panique si `risk_priority_number` n'est pas fini ou sort de l'intervalle
/// `[1, 1000]`.
pub fn fmea_normalized_rpn(risk_priority_number: f64) -> f64 {
    assert!(
        risk_priority_number.is_finite(),
        "le RPN doit être un nombre fini"
    );
    assert!(
        (1.0..=1000.0).contains(&risk_priority_number),
        "le RPN doit être dans l'intervalle [1, 1000]"
    );
    risk_priority_number / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn realistic_fmea_case() {
        // Mode de défaillance coté S = 8, O = 6, D = 4.
        // RPN = 8 · 6 · 4 = 192 ; criticité C = 8 · 6 = 48.
        let (severity, occurrence, detection) = (8.0, 6.0, 4.0);
        assert_relative_eq!(
            fmea_risk_priority_number(severity, occurrence, detection),
            192.0,
            epsilon = 1e-9
        );
        assert_relative_eq!(fmea_criticality(severity, occurrence), 48.0, epsilon = 1e-9);
    }

    #[test]
    fn rpn_is_criticality_times_detection() {
        // Identité : RPN = C · D = (S · O) · D.
        let (severity, occurrence, detection) = (5.0, 7.0, 3.0);
        let rpn = fmea_risk_priority_number(severity, occurrence, detection);
        let criticality = fmea_criticality(severity, occurrence);
        assert_relative_eq!(rpn, criticality * detection, epsilon = 1e-12);
    }

    #[test]
    fn normalized_rpn_of_maximum_is_one() {
        // Cas limite : le RPN maximal 10·10·10 = 1000 se normalise à 1.
        let rpn = fmea_risk_priority_number(10.0, 10.0, 10.0);
        assert_relative_eq!(rpn, 1000.0, epsilon = 1e-9);
        assert_relative_eq!(fmea_normalized_rpn(rpn), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn normalized_rpn_of_minimum_is_one_thousandth() {
        // Cas limite : le RPN minimal 1·1·1 = 1 se normalise à 0,001.
        let rpn = fmea_risk_priority_number(1.0, 1.0, 1.0);
        assert_relative_eq!(rpn, 1.0, epsilon = 1e-12);
        assert_relative_eq!(fmea_normalized_rpn(rpn), 0.001, epsilon = 1e-12);
    }

    #[test]
    fn rpn_proportional_to_detection() {
        // RPN ∝ D à (S, O) constants : doubler la note de détection double le RPN.
        let (severity, occurrence) = (6.0, 5.0);
        let low = fmea_risk_priority_number(severity, occurrence, 2.0);
        let high = fmea_risk_priority_number(severity, occurrence, 4.0);
        assert_relative_eq!(high, 2.0 * low, epsilon = 1e-12);
    }

    #[test]
    fn threshold_comparison_is_strict() {
        // RPN = 125 (S=O=D=5). Il dépasse 100 mais pas 125 ni 150.
        let rpn = fmea_risk_priority_number(5.0, 5.0, 5.0);
        assert_relative_eq!(rpn, 125.0, epsilon = 1e-9);
        assert!(fmea_exceeds_threshold(rpn, 100.0));
        assert!(!fmea_exceeds_threshold(rpn, 125.0));
        assert!(!fmea_exceeds_threshold(rpn, 150.0));
    }

    #[test]
    #[should_panic(expected = "la note gravité doit être dans l'intervalle [1, 10]")]
    fn out_of_range_severity_panics() {
        fmea_risk_priority_number(11.0, 5.0, 5.0);
    }
}
