//! Modèle des paramètres de risque ISO 25119-2 (*Tractors and machinery
//! for agriculture and forestry — Safety-related parts of control
//! systems — Part 2: Concept phase design*) : Sévérité, Exposition,
//! Contrôlabilité — les trois paramètres qui, combinés dans le graphe de
//! risque de la Figure 1 du §6.3.7, déterminent l'AgPL requis (`a` à
//! `e`, ou `QM` si la fonction n'est pas liée à la sécurité).
//!
//! **Vérifié contre le texte normatif** (aperçus gratuits publiés par
//! iTeh Standards des éditions 2010 et 2019, incluant les tableaux 1-3
//! du §6) : les définitions ci-dessous — noms de niveaux, nombre de
//! niveaux, formulation — sont fiables.
//!
//! **PAS implémenté, délibérément** : la fonction `S × E × C → AgPL`
//! elle-même. Le graphe de risque complet (Figure 1) n'apparaît dans
//! aucune source ouverte ou vérifiable trouvée — les aperçus gratuits du
//! standard s'arrêtent juste avant la figure, et le seul résumé
//! secondaire trouvé (Mitka 2018) contredit le texte normatif vérifié
//! (invente un niveau "S4", réduit la sortie à 3 catégories) et n'est pas
//! fiable. Coder une topologie de graphe de risque de sécurité
//! fonctionnelle *devinée* serait pire que ne rien coder — une
//! détermination d'AgPL erronée est un bug de sécurité, pas un détail.
//! Ce module expose donc le modèle de données des paramètres (correct et
//! citable), pas la fonction de décision : consultez le texte acheté de
//! la norme pour la Figure 1 exacte, ou une reproduction secondaire
//! fiable, avant de dériver un AgPL réel.
//!
//! De même, la notion de catégorie d'architecture SRP/CS (Annexe A,
//! catégories B/1/2/3/4 — partagées avec ISO 13849) et le niveau SRL
//! (Software Requirement Level, B/1/2/3) existent dans la norme mais
//! leurs tables de correspondance n'ont pas pu être vérifiées : non
//! représentées ici.

use serde::{Deserialize, Serialize};

/// Sévérité de blessure (ISO 25119-2 Tableau 1, 4 niveaux). `S0` sort de
/// l'évaluation de risque sans AgPL requis (dommage limité aux biens).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    /// Aucune blessure, dommage limité aux biens.
    S0,
    /// Blessures légères à modérées, soins médicaux requis, récupération totale.
    S1,
    /// Blessures graves ou engageant le pronostic vital (survie probable),
    /// perte permanente partielle de capacité de travail.
    S2,
    /// Blessures engageant le pronostic vital (survie incertaine), invalidité sévère.
    S3,
}

/// Fréquence/durée d'exposition au danger (ISO 25119-2 Tableau 2, 5
/// niveaux). Définie à la fois par fréquence et par durée relative
/// (`t_exp / t_avop`) ; la norme précise que si les deux critères
/// donnent des catégories différentes, la plus élevée s'applique — voir
/// [`Exposure::combine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Exposure {
    /// Improbable : une fois dans la vie de la machine, ou < 0.01%.
    E0,
    /// Rare : moins d'une fois par an, ou 0.01%–<0.1%.
    E1,
    /// Occasionnelle : plus d'une fois par an, ou 0.1%–<1%.
    E2,
    /// Fréquente : plus d'une fois par mois, ou 1%–<10%.
    E3,
    /// Quasi systématique : presque à chaque opération, ou ≥10%.
    E4,
}

impl Exposure {
    /// Applique la règle normative : quand fréquence et durée donnent des
    /// catégories différentes, retient la plus élevée.
    pub fn combine(by_frequency: Exposure, by_duration: Exposure) -> Exposure {
        by_frequency.max(by_duration)
    }
}

/// Possibilité d'éviter le danger / contrôlabilité (ISO 25119-2 Tableau
/// 3, 4 niveaux).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Controllability {
    /// Facilement contrôlable : l'opérateur/tiers maîtrise la situation, dommage évité.
    C0,
    /// Simplement contrôlable : >99% des personnes la maîtrisent ; >99% des
    /// occurrences n'entraînent pas de dommage.
    C1,
    /// Majoritairement contrôlable : >90% / >90%.
    C2,
    /// Non contrôlable : un opérateur/tiers formé typique ne peut généralement pas l'éviter.
    C3,
}

/// Les trois paramètres de risque d'un danger donné, avant détermination
/// de l'AgPL (non implémentée — voir la doc de tête du module).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiskParameters {
    pub severity: Severity,
    pub exposure: Exposure,
    pub controllability: Controllability,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_orders_from_least_to_most_severe() {
        assert!(Severity::S0 < Severity::S1);
        assert!(Severity::S1 < Severity::S2);
        assert!(Severity::S2 < Severity::S3);
    }

    #[test]
    fn exposure_combine_takes_the_higher_category() {
        assert_eq!(Exposure::combine(Exposure::E1, Exposure::E3), Exposure::E3);
        assert_eq!(Exposure::combine(Exposure::E4, Exposure::E0), Exposure::E4);
        assert_eq!(Exposure::combine(Exposure::E2, Exposure::E2), Exposure::E2);
    }

    #[test]
    fn controllability_orders_from_easiest_to_uncontrollable() {
        assert!(Controllability::C0 < Controllability::C1);
        assert!(Controllability::C1 < Controllability::C2);
        assert!(Controllability::C2 < Controllability::C3);
    }

    #[test]
    fn risk_parameters_are_constructible_and_comparable() {
        let a = RiskParameters {
            severity: Severity::S2,
            exposure: Exposure::E3,
            controllability: Controllability::C2,
        };
        let b = a;
        assert_eq!(a, b);
    }
}
