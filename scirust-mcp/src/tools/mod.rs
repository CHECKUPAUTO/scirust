//! Ensembles d'outils MCP fournis par défaut. Un domaine SciRust
//! supplémentaire (ex. un futur `scirust-pdm` exposé) ajoute simplement un
//! nouveau sous-module ici et l'enregistre dans [`crate::default_registry`].

pub mod cli_passthrough;
pub mod dev;
pub mod discovery;
pub mod linalg;
