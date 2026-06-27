//! The catalogue of licensable SciRust modules.
//!
//! A *module* is a commercial unit a customer can be entitled to — usually a
//! whole domain vertical (one or more crates), not a single crate. Each variant
//! carries a stable numeric `code` (used for canonical, order-independent
//! license encoding) and a stable string id (used in human-facing license
//! JSON). **Never renumber or rename existing variants** — doing so would
//! invalidate every license already signed against the old encoding.

use serde::{Deserialize, Serialize};

/// A licensable SciRust module (domain vertical).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(into = "String", try_from = "String")]
pub enum Module {
    // ---- Foundation & core ML (codes 1..=49) ----
    /// Tensors, autodiff, the nn layer zoo, optimizers (`scirust-core`).
    Core,
    /// Tensor-network compression & einsum (`scirust-tn`, `scirust-tensor-*`).
    TensorNetwork,
    /// Classical & deep NLP (`scirust-nlp-advanced`).
    Nlp,
    /// Computer vision (`scirust-vision`).
    Vision,
    /// Audio features & DSP front-ends (`scirust-audio`).
    Audio,
    /// Graph learning & algorithms (`scirust-graph`).
    Graph,
    /// Automated ML: search, ensembling, model selection (`scirust-automl`).
    AutoMl,
    /// Symbolic reasoning, regression, synthesis (`scirust-reasoning`,
    /// `scirust-symbolic`, `scirust-symreg`, `scirust-synthesis`).
    Reasoning,
    /// Reinforcement learning (`scirust-rl-algo`).
    ReinforcementLearning,
    /// Evolutionary search & NAS (`scirust-evo`, `scirust-nas`,
    /// `scirust-algogen`).
    Evolution,
    /// Edge & embedded deployment (`scirust-edge`, `scirust-embedded`).
    Edge,
    /// Streaming event detection (`scirust-events-*`).
    Events,
    /// Pure semantic (dense) retrieval — an auditable alternative to RAG
    /// (`scirust-retrieval`). A premium add-on, sold in the Perception and
    /// Industrie 4.0 bundles.
    Retrieval,

    // ---- Industrial verticals (codes 50..=99) ----
    /// State estimation & sensor fusion (`scirust-estimation`).
    Estimation,
    /// Inertial / GNSS navigation (`scirust-nav`).
    Navigation,
    /// Water-network & quality monitoring (`scirust-water`).
    Water,
    /// Deterministic control: PID, LQR, MPC (`scirust-control`).
    Control,
    /// Battery state-of-charge / health (`scirust-bms`).
    Battery,
    /// Power-grid analytics (`scirust-grid`).
    Grid,
    /// Structural health monitoring (`scirust-shm`).
    StructuralHealth,
    /// HVAC & non-intrusive load monitoring (`scirust-hvac`).
    Hvac,
    /// Robotics: trajectories, kinematics, safety (`scirust-robotics`).
    Robotics,
    /// Metrology & calibration (`scirust-metrology`).
    Metrology,
    /// Signal processing front-end (`scirust-signal`).
    Signal,
    /// Predictive maintenance (`scirust-pdm`).
    PredictiveMaintenance,
    /// Reliability engineering (`scirust-reliability`).
    Reliability,
    /// Functional safety evidence (`scirust-func-safety`).
    FunctionalSafety,
    /// OT / ICS security & intrusion detection (`scirust-ids`).
    OtSecurity,
    /// MLOps: drift, OTA, monitoring (`scirust-mlops`).
    MlOps,
    /// Biomedical signal analysis (`scirust-biomed`).
    Biomed,
    /// Quantitative trading (`scirust-trader`).
    Trading,
    /// Statistical process control (`scirust-spc`).
    Spc,
    /// Industrial integration & orchestration (`scirust-industrial`,
    /// `scirust-integration`).
    Industrial,
}

impl Module {
    /// Every module in the catalogue, in ascending `code` order.
    pub const ALL: [Module; 32] = [
        Module::Core,
        Module::TensorNetwork,
        Module::Nlp,
        Module::Vision,
        Module::Audio,
        Module::Graph,
        Module::AutoMl,
        Module::Reasoning,
        Module::ReinforcementLearning,
        Module::Evolution,
        Module::Edge,
        Module::Events,
        Module::Retrieval,
        Module::Estimation,
        Module::Navigation,
        Module::Water,
        Module::Control,
        Module::Battery,
        Module::Grid,
        Module::StructuralHealth,
        Module::Hvac,
        Module::Robotics,
        Module::Metrology,
        Module::Signal,
        Module::PredictiveMaintenance,
        Module::Reliability,
        Module::FunctionalSafety,
        Module::OtSecurity,
        Module::MlOps,
        Module::Biomed,
        Module::Trading,
        Module::Spc,
    ];

    /// Stable numeric code used in the canonical (signed) encoding. Foundation
    /// modules occupy 1..=49, industrial verticals 50.. — gaps are intentional
    /// so each group can grow without renumbering.
    pub fn code(self) -> u16 {
        match self
        {
            Module::Core => 1,
            Module::TensorNetwork => 2,
            Module::Nlp => 3,
            Module::Vision => 4,
            Module::Audio => 5,
            Module::Graph => 6,
            Module::AutoMl => 7,
            Module::Reasoning => 8,
            Module::ReinforcementLearning => 9,
            Module::Evolution => 10,
            Module::Edge => 11,
            Module::Events => 12,
            Module::Retrieval => 13,
            Module::Estimation => 50,
            Module::Navigation => 51,
            Module::Water => 52,
            Module::Control => 53,
            Module::Battery => 54,
            Module::Grid => 55,
            Module::StructuralHealth => 56,
            Module::Hvac => 57,
            Module::Robotics => 58,
            Module::Metrology => 59,
            Module::Signal => 60,
            Module::PredictiveMaintenance => 61,
            Module::Reliability => 62,
            Module::FunctionalSafety => 63,
            Module::OtSecurity => 64,
            Module::MlOps => 65,
            Module::Biomed => 66,
            Module::Trading => 67,
            Module::Spc => 68,
            Module::Industrial => 69,
        }
    }

    /// Stable, lower-case, hyphenated string id used in license JSON and the CLI.
    pub fn as_str(self) -> &'static str {
        match self
        {
            Module::Core => "core",
            Module::TensorNetwork => "tensor-network",
            Module::Nlp => "nlp",
            Module::Vision => "vision",
            Module::Audio => "audio",
            Module::Graph => "graph",
            Module::AutoMl => "automl",
            Module::Reasoning => "reasoning",
            Module::ReinforcementLearning => "reinforcement-learning",
            Module::Evolution => "evolution",
            Module::Edge => "edge",
            Module::Events => "events",
            Module::Retrieval => "retrieval",
            Module::Estimation => "estimation",
            Module::Navigation => "navigation",
            Module::Water => "water",
            Module::Control => "control",
            Module::Battery => "battery",
            Module::Grid => "grid",
            Module::StructuralHealth => "structural-health",
            Module::Hvac => "hvac",
            Module::Robotics => "robotics",
            Module::Metrology => "metrology",
            Module::Signal => "signal",
            Module::PredictiveMaintenance => "predictive-maintenance",
            Module::Reliability => "reliability",
            Module::FunctionalSafety => "functional-safety",
            Module::OtSecurity => "ot-security",
            Module::MlOps => "mlops",
            Module::Biomed => "biomed",
            Module::Trading => "trading",
            Module::Spc => "spc",
            Module::Industrial => "industrial",
        }
    }

    /// Parse a string id (as produced by [`Module::as_str`]).
    pub fn from_id(s: &str) -> Option<Module> {
        // `Industrial` is absent from `ALL` only because `ALL` is sized to the
        // first 32; include it explicitly here so every id round-trips.
        Module::ALL
            .iter()
            .copied()
            .chain(std::iter::once(Module::Industrial))
            .find(|m| m.as_str() == s)
    }

    /// One-line human description of what the module covers.
    pub fn description(self) -> &'static str {
        match self
        {
            Module::Core => "Tensors, autodiff, neural-network layers and optimizers",
            Module::TensorNetwork => "Tensor-train compression and einsum contraction",
            Module::Nlp => "Natural-language processing",
            Module::Vision => "Computer vision",
            Module::Audio => "Audio features and DSP front-ends",
            Module::Graph => "Graph learning and algorithms",
            Module::AutoMl => "Automated model search and selection",
            Module::Reasoning => "Symbolic reasoning, regression and synthesis",
            Module::ReinforcementLearning => "Reinforcement-learning algorithms",
            Module::Evolution => "Evolutionary search and neural architecture search",
            Module::Edge => "Edge and embedded deployment",
            Module::Events => "Streaming event and anomaly detection",
            Module::Retrieval => "Pure semantic (dense) retrieval, an auditable alternative to RAG",
            Module::Estimation => "State estimation and sensor fusion",
            Module::Navigation => "Inertial and GNSS navigation",
            Module::Water => "Water-network and quality monitoring",
            Module::Control => "Deterministic control (PID, LQR, MPC)",
            Module::Battery => "Battery state-of-charge and health",
            Module::Grid => "Power-grid analytics",
            Module::StructuralHealth => "Structural health monitoring",
            Module::Hvac => "HVAC and non-intrusive load monitoring",
            Module::Robotics => "Robotics trajectories, kinematics and safety",
            Module::Metrology => "Metrology and calibration",
            Module::Signal => "Signal-processing front-end",
            Module::PredictiveMaintenance => "Predictive maintenance",
            Module::Reliability => "Reliability engineering",
            Module::FunctionalSafety => "Functional-safety evidence",
            Module::OtSecurity => "OT/ICS security and intrusion detection",
            Module::MlOps => "MLOps: drift, OTA and monitoring",
            Module::Biomed => "Biomedical signal analysis",
            Module::Trading => "Quantitative trading",
            Module::Spc => "Statistical process control",
            Module::Industrial => "Industrial integration and orchestration",
        }
    }
}

impl std::fmt::Display for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<Module> for String {
    fn from(m: Module) -> String {
        m.as_str().to_string()
    }
}

impl TryFrom<String> for Module {
    type Error = String;

    fn try_from(s: String) -> Result<Module, String> {
        Module::from_id(&s).ok_or_else(|| format!("unknown module id: {s}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_module_has_a_unique_code_and_id() {
        let mut codes = std::collections::HashSet::new();
        let mut ids = std::collections::HashSet::new();
        for m in all_modules()
        {
            assert!(codes.insert(m.code()), "duplicate code for {m}");
            assert!(ids.insert(m.as_str()), "duplicate id for {m}");
        }
    }

    #[test]
    fn id_round_trips_through_from_id() {
        for m in all_modules()
        {
            assert_eq!(Module::from_id(m.as_str()), Some(m));
        }
        assert_eq!(Module::from_id("does-not-exist"), None);
    }

    #[test]
    fn serde_uses_the_string_id() {
        let json = serde_json::to_string(&Module::Navigation).unwrap();
        assert_eq!(json, "\"navigation\"");
        let back: Module = serde_json::from_str("\"battery\"").unwrap();
        assert_eq!(back, Module::Battery);
        // An unknown id is a hard error, not a silent default.
        assert!(serde_json::from_str::<Module>("\"bogus\"").is_err());
    }

    #[test]
    fn industrial_is_reachable_even_though_all_is_32_wide() {
        // ALL is sized to 32 for ergonomics; Industrial (code 69) must still be
        // a first-class, parseable module.
        assert_eq!(Module::from_id("industrial"), Some(Module::Industrial));
        assert_eq!(Module::Industrial.code(), 69);
    }

    #[test]
    fn retrieval_is_a_first_class_premium_module() {
        // The "RAG-killer" add-on: stable code 13 in the foundation range, a
        // round-tripping id, and present in ALL (so it is gated like any other).
        assert_eq!(Module::Retrieval.code(), 13);
        assert_eq!(Module::Retrieval.as_str(), "retrieval");
        assert_eq!(Module::from_id("retrieval"), Some(Module::Retrieval));
        assert!(Module::ALL.contains(&Module::Retrieval));
    }

    /// Helper: every catalogue module including the one outside `ALL`.
    fn all_modules() -> Vec<Module> {
        Module::ALL
            .iter()
            .copied()
            .chain(std::iter::once(Module::Industrial))
            .collect()
    }
}
