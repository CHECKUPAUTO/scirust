//! Localization for the features this preview actually ships.
//!
//! English and French are both complete for everything visible. A key with
//! no French translation fails a test rather than silently falling back —
//! silent fallback is how a "bilingual" application ends up half-English in
//! front of a user.
//!
//! Machine-readable output (scenario files, stored results, manifests) is
//! never localized: those are data, and a decimal comma in a JSON number
//! would be a corruption, not a translation.

use serde::{Deserialize, Serialize};

/// A supported interface language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Language {
    /// English — also the fallback.
    English,
    /// French.
    French,
}

impl Language {
    /// The BCP 47 tag, for the document's `lang` attribute.
    pub fn tag(self) -> &'static str {
        match self
        {
            Language::English => "en",
            Language::French => "fr",
        }
    }

    /// The language's own name, for the picker.
    pub fn endonym(self) -> &'static str {
        match self
        {
            Language::English => "English",
            Language::French => "Français",
        }
    }

    /// Every supported language.
    pub fn all() -> [Language; 2] {
        [Language::English, Language::French]
    }

    /// Match an operating-system locale, falling back to English.
    ///
    /// Prefix matching on purpose: `fr-CA`, `fr_FR` and `fr` are all French
    /// as far as this interface is concerned, and refusing them because they
    /// carry a region would leave a French user reading English.
    pub fn from_locale(locale: &str) -> Language {
        let lower = locale.trim().to_ascii_lowercase();
        if lower == "fr" || lower.starts_with("fr-") || lower.starts_with("fr_")
        {
            Language::French
        }
        else
        {
            Language::English
        }
    }
}

/// Every user-visible string, keyed.
///
/// A flat table rather than a file loaded at runtime: the strings are
/// compiled in, so a shipped build cannot be missing its own translations,
/// and the completeness test below is a compile-time-adjacent guarantee
/// rather than a deployment concern.
const STRINGS: &[(&str, &str, &str)] = &[
    // key, English, French
    ("app.name", "SciRust Studio", "SciRust Studio"),
    ("nav.home", "Home", "Accueil"),
    ("nav.catalogue", "Catalogue", "Catalogue"),
    ("nav.experiment", "Experiment", "Expérience"),
    ("nav.runs", "Runs", "Exécutions"),
    ("nav.help", "Help", "Aide"),
    ("action.run", "Run", "Exécuter"),
    ("action.cancel", "Cancel", "Annuler"),
    ("action.validate", "Validate", "Valider"),
    ("action.open", "Open scenario", "Ouvrir un scénario"),
    ("action.save", "Save scenario", "Enregistrer le scénario"),
    (
        "action.save_as",
        "Save scenario as",
        "Enregistrer le scénario sous",
    ),
    ("action.new", "New experiment", "Nouvelle expérience"),
    (
        "action.open_catalogue",
        "Open catalogue",
        "Ouvrir le catalogue",
    ),
    (
        "action.open_runs",
        "Open run history",
        "Ouvrir l'historique",
    ),
    (
        "action.verify_run",
        "Verify current run",
        "Vérifier l'exécution",
    ),
    ("action.open_help", "Open help", "Ouvrir l'aide"),
    (
        "action.toggle_activity",
        "Toggle activity panel",
        "Afficher/masquer le journal",
    ),
    (
        "action.toggle_inspector",
        "Toggle inspector",
        "Afficher/masquer l'inspecteur",
    ),
    (
        "action.change_language",
        "Change language",
        "Changer de langue",
    ),
    ("action.change_theme", "Change theme", "Changer de thème"),
    (
        "action.show_diagnostics",
        "Show diagnostics",
        "Afficher les diagnostics",
    ),
    (
        "action.restart_worker",
        "Restart calculation engine",
        "Redémarrer le moteur de calcul",
    ),
    ("palette.title", "Command palette", "Palette de commandes"),
    (
        "palette.placeholder",
        "Type a command…",
        "Tapez une commande…",
    ),
    (
        "palette.no_match",
        "No matching command",
        "Aucune commande correspondante",
    ),
    (
        "palette.unavailable",
        "Unavailable here",
        "Indisponible ici",
    ),
    (
        "home.welcome",
        "Deterministic scientific simulation",
        "Simulation scientifique déterministe",
    ),
    ("home.recent_runs", "Recent runs", "Exécutions récentes"),
    (
        "home.no_runs",
        "No runs recorded yet",
        "Aucune exécution enregistrée",
    ),
    ("home.tutorials", "Tested examples", "Exemples testés"),
    (
        "home.storage_notice",
        "Run history is stored locally on this device.",
        "L'historique des exécutions est stocké localement sur cet appareil.",
    ),
    (
        "home.reveal_store",
        "Show location",
        "Afficher l'emplacement",
    ),
    (
        "catalogue.executable",
        "executable capabilities",
        "capacités exécutables",
    ),
    (
        "catalogue.search",
        "Search capabilities",
        "Rechercher une capacité",
    ),
    (
        "catalogue.all_categories",
        "All categories",
        "Toutes les catégories",
    ),
    ("catalogue.open_example", "Open example", "Ouvrir l'exemple"),
    ("catalogue.parameters", "Parameters", "Paramètres"),
    ("catalogue.initial_state", "Initial state", "État initial"),
    ("catalogue.outputs", "Outputs", "Sorties"),
    (
        "catalogue.checks",
        "Scientific checks",
        "Vérifications scientifiques",
    ),
    ("catalogue.solvers", "Solvers", "Solveurs"),
    (
        "catalogue.not_connected",
        "Present in the repository but not yet connected to Studio",
        "Présent dans le dépôt mais pas encore relié à Studio",
    ),
    ("editor.title", "Scenario", "Scénario"),
    (
        "editor.unsaved",
        "Unsaved changes",
        "Modifications non enregistrées",
    ),
    ("editor.problems", "Problems", "Problèmes"),
    (
        "editor.no_problems",
        "No problems found",
        "Aucun problème détecté",
    ),
    ("editor.line", "line", "ligne"),
    ("editor.column", "column", "colonne"),
    ("run.state.queued", "Queued", "En attente"),
    ("run.state.running", "Running", "En cours"),
    ("run.state.cancelling", "Cancelling", "Annulation"),
    ("run.state.cancelled", "Cancelled", "Annulée"),
    ("run.state.completed", "Completed", "Terminée"),
    ("run.state.failed", "Failed", "Échec"),
    ("run.state.interrupted", "Interrupted", "Interrompue"),
    (
        "run.indeterminate",
        "Running — this solver cannot report progress",
        "En cours — ce solveur ne peut pas indiquer la progression",
    ),
    ("run.elapsed", "Elapsed", "Durée"),
    ("run.metrics", "Metrics", "Mesures"),
    (
        "run.verification",
        "Scientific verification",
        "Vérification scientifique",
    ),
    ("run.integrity", "Store integrity", "Intégrité du stockage"),
    (
        "run.integrity.ok",
        "Stored bytes match their hashes",
        "Les octets stockés correspondent à leurs empreintes",
    ),
    (
        "run.integrity.failed",
        "Stored bytes have changed",
        "Les octets stockés ont changé",
    ),
    ("run.provenance", "Provenance", "Provenance"),
    ("run.status.passed", "Passed", "Réussi"),
    ("run.status.warning", "Warning", "Avertissement"),
    ("run.status.failed", "Failed", "Échoué"),
    ("run.status.not_applicable", "Not applicable", "Sans objet"),
    ("runs.title", "Run history", "Historique des exécutions"),
    ("runs.verify", "Verify", "Vérifier"),
    ("runs.open", "Open", "Ouvrir"),
    ("runs.copy_id", "Copy run ID", "Copier l'identifiant"),
    (
        "runs.interrupted",
        "Interrupted runs",
        "Exécutions interrompues",
    ),
    (
        "runs.interrupted.explain",
        "These runs were never finished; their partial evidence is kept.",
        "Ces exécutions n'ont jamais abouti ; leurs traces partielles sont conservées.",
    ),
    ("chart.title", "Trajectory", "Trajectoire"),
    ("chart.showing", "Showing", "Affichage de"),
    ("field.reduced_to", "reduced to", "réduit à"),
    (
        "field.unreadable",
        "This field could not be drawn: its shape or its values are not usable.",
        "Ce champ n'a pas pu être tracé : sa forme ou ses valeurs sont inutilisables.",
    ),
    (
        "distribution.all-outside",
        "Every sample fell outside the binned range, so there is nothing to draw.",
        "Tous les échantillons sont tombés hors de la plage des classes : il n'y a rien à tracer.",
    ),
    ("chart.of", "of", "sur"),
    ("chart.points", "points", "points"),
    ("chart.table_view", "Show as table", "Afficher en tableau"),
    (
        "chart.legacy_warning",
        "This run predates stored coordinates: the horizontal axis is the sample index, not time.",
        "Cette exécution est antérieure aux coordonnées stockées : l'axe horizontal est l'indice d'échantillon, pas le temps.",
    ),
    (
        "chart.empty.no_series",
        "This result has no series to plot.",
        "Ce résultat ne contient aucune série à tracer.",
    ),
    (
        "chart.empty.all_hidden",
        "Every series is hidden. Enable one in the legend.",
        "Toutes les séries sont masquées. Activez-en une dans la légende.",
    ),
    (
        "chart.empty.no_finite_data",
        "This result contains no finite values to plot.",
        "Ce résultat ne contient aucune valeur finie à tracer.",
    ),
    (
        "chart.empty.length_mismatch",
        "A series does not match the axis length, so it cannot be plotted.",
        "Une série ne correspond pas à la longueur de l'axe et ne peut pas être tracée.",
    ),
    ("panel.activity", "Activity", "Activité"),
    ("panel.events", "Events", "Événements"),
    ("panel.problems", "Problems", "Problèmes"),
    ("panel.metrics", "Metrics", "Mesures"),
    ("panel.clear", "Clear", "Effacer"),
    ("status.worker", "Engine", "Moteur"),
    ("status.worker.running", "running", "actif"),
    ("status.worker.stopped", "not running", "arrêté"),
    ("status.backend", "Backend", "Backend"),
    ("status.precision", "Precision", "Précision"),
    ("status.job", "Job", "Tâche"),
    ("status.store", "Store", "Stockage"),
    ("error.details", "Technical details", "Détails techniques"),
    (
        "error.copy_diagnostics",
        "Copy diagnostics",
        "Copier les diagnostics",
    ),
    ("theme.system", "System", "Système"),
    ("theme.light", "Light", "Clair"),
    ("theme.dark", "Dark", "Sombre"),
    ("theme.high_contrast", "High contrast", "Contraste élevé"),
    (
        "help.title",
        "Help — preview features",
        "Aide — fonctionnalités en préversion",
    ),
    (
        "help.preview_notice",
        "This covers the features in this preview, not the complete SciRust Studio help.",
        "Ceci couvre les fonctionnalités de cette préversion, pas l'aide complète de SciRust Studio.",
    ),
    (
        "app.tagline",
        "Deterministic scientific simulation, recorded and verifiable.",
        "Simulation scientifique déterministe, enregistrée et vérifiable.",
    ),
    (
        "mock.banner",
        "This build serves fabricated data and must not be used for any scientific purpose.",
        "Cette version fournit des données fictives et ne doit servir à aucun usage scientifique.",
    ),
    (
        "chart.axis_legacy",
        "Sample ordinals. This result predates stored coordinates, so the spacing shown is not \
         the spacing the solver used.",
        "Numéros d'échantillon. Ce résultat est antérieur à l'enregistrement des coordonnées ; \
         l'espacement affiché n'est donc pas celui utilisé par le solveur.",
    ),
    ("run.warnings", "Warnings", "Avertissements"),
    (
        "run.none",
        "No run is displayed yet.",
        "Aucune exécution affichée pour l'instant.",
    ),
    ("run.scenario", "Scenario", "Scénario"),
    ("home.open_tutorial", "Open tutorial", "Ouvrir le tutoriel"),
    (
        "catalogue.no_match",
        "No capability matches.",
        "Aucune capacité ne correspond.",
    ),
    (
        "editor.validated",
        "The scenario is valid.",
        "Le scénario est valide.",
    ),
    (
        "runs.none",
        "Nothing has been run yet.",
        "Rien n'a encore été exécuté.",
    ),
    ("runs.schema", "Result schema", "Schéma de résultat"),
    ("runs.legacy", "Legacy result", "Résultat hérité"),
    ("help.shortcuts", "Keyboard shortcuts", "Raccourcis clavier"),
    (
        "help.workflow",
        "The workflow this preview supports",
        "Le flux pris en charge par cette préversion",
    ),
    ("error.dismiss", "Dismiss", "Fermer"),
    ("status.mock", "MOCK DATA", "DONNÉES FICTIVES"),
    ("chart.legend", "Series", "Séries"),
    ("common.loading", "Loading…", "Chargement…"),
];

/// Look up a key in a language, falling back to English.
///
/// A missing key returns the key itself rather than an empty string: an
/// untranslated label is a visible bug, and a blank one is an invisible
/// one.
pub fn t(language: Language, key: &str) -> &'static str {
    for (k, en, fr) in STRINGS
    {
        if *k == key
        {
            return match language
            {
                Language::English => en,
                Language::French => fr,
            };
        }
    }
    // Leak-free: the key came from a `&str` the caller owns, so return a
    // stable marker instead of pretending to have a translation.
    "‹missing translation›"
}

/// Whether a key exists.
pub fn has_key(key: &str) -> bool {
    STRINGS.iter().any(|(k, _, _)| *k == key)
}

/// Every key, for the completeness tests.
pub fn keys() -> Vec<&'static str> {
    STRINGS.iter().map(|(k, _, _)| *k).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property that makes "bilingual" mean something: no key may be
    /// left in English inside the French table.
    #[test]
    fn every_key_has_a_real_french_translation() {
        // Terms that are legitimately identical in both languages — proper
        // nouns and words French has borrowed unchanged. Listed explicitly
        // so an *accidental* untranslated string still fails.
        const IDENTICAL_BY_DESIGN: &[&str] = &[
            "app.name",
            "nav.catalogue",
            "status.backend",
            "chart.points",
            // French for the origin of a thing is the same word English
            // borrowed for it.
            "run.provenance",
        ];

        for (key, en, fr) in STRINGS
        {
            assert!(!fr.trim().is_empty(), "{key} has no French text");
            if IDENTICAL_BY_DESIGN.contains(key)
            {
                continue;
            }
            assert_ne!(
                en, fr,
                "{key} is identical in both languages — if that is deliberate, \
                 add it to IDENTICAL_BY_DESIGN so the check stays meaningful"
            );
        }
    }

    #[test]
    fn no_key_is_defined_twice() {
        let mut seen = std::collections::BTreeSet::new();
        for (key, _, _) in STRINGS
        {
            assert!(seen.insert(*key), "{key} is defined more than once");
        }
    }

    #[test]
    fn every_string_is_non_empty_in_english() {
        for (key, en, _) in STRINGS
        {
            assert!(!en.trim().is_empty(), "{key} has no English text");
        }
    }

    #[test]
    fn lookup_returns_the_requested_language() {
        assert_eq!(t(Language::English, "action.run"), "Run");
        assert_eq!(t(Language::French, "action.run"), "Exécuter");
    }

    /// A missing key must be conspicuous, not blank.
    #[test]
    fn a_missing_key_is_visibly_marked() {
        let missing = t(Language::English, "no.such.key");
        assert!(!missing.is_empty());
        assert!(missing.contains("missing"), "{missing}");
        assert!(!has_key("no.such.key"));
    }

    #[test]
    fn a_locale_with_a_region_still_selects_french() {
        for locale in ["fr", "fr-FR", "fr_CA", "FR-fr", " fr-BE "]
        {
            assert_eq!(
                Language::from_locale(locale),
                Language::French,
                "{locale} must be French"
            );
        }
    }

    #[test]
    fn an_unsupported_locale_falls_back_to_english() {
        for locale in ["de-DE", "ja", "", "french", "en-GB"]
        {
            assert_eq!(Language::from_locale(locale), Language::English, "{locale}");
        }
    }

    #[test]
    fn each_language_has_a_tag_and_an_endonym() {
        for language in Language::all()
        {
            assert!(!language.tag().is_empty());
            assert!(!language.endonym().is_empty());
        }
        assert_eq!(Language::French.endonym(), "Français");
    }

    /// Every empty-chart reason must have a message, or the chart would show
    /// a blank box with no explanation.
    #[test]
    fn every_chart_empty_reason_has_a_translation() {
        use crate::chart::EmptyReason;
        for reason in [
            EmptyReason::NoSeries,
            EmptyReason::AllSeriesHidden,
            EmptyReason::NoFiniteData,
            EmptyReason::LengthMismatch,
        ]
        {
            assert!(
                has_key(reason.message_key()),
                "{} has no translation",
                reason.message_key()
            );
        }
    }
}
