//! `scirust-mcp` — serveur Model Context Protocol pour SciRust.
//!
//! Expose les capacités de SciRust (solveurs numériques, outils de
//! développement du SLM `scirust-sciagent`, découverte d'actifs OT/IT de
//! `scirust-discovery`, vérification SIL/IEC 61511 de `scirust-sis`, et un
//! outil par domaine industriel — protection réseau, contrôle médical,
//! collision maritime, contrôle run-to-run semi-conducteur, nettoyage de
//! carte de rendement agricole, dommage de fatigue) comme des outils MCP
//! standard : n'importe quel agent — le SLM embarqué, Claude,
//! ChatGPT, un script — peut les découvrir
//! (`tools/list`) et les appeler (`tools/call`) sans glue code spécifique,
//! avec un schéma JSON explicite par outil et un journal d'audit
//! hash-chaîné (SHA-256) de chaque appel.
//!
//! Voir <https://modelcontextprotocol.io> pour la spécification du
//! protocole, et `README.md` de cette crate pour l'architecture détaillée,
//! les alternatives considérées, et les sources citées.

pub mod audit;
pub mod protocol;
pub mod registry;
pub mod server;
pub mod tools;

use registry::ToolRegistry;

/// Construit le registre d'outils par défaut du serveur `scirust-mcp` :
/// outils de développement (hérités de `scirust-sciagent`), algèbre
/// linéaire (`scirust-solvers`), découverte OT/IT (`scirust-discovery`),
/// sûreté fonctionnelle (`scirust-sis`), et un outil par domaine industriel
/// ajouté depuis (grid, biomed, maritime, fab, agtech, fatigue), plus
/// l'échappatoire générique vers le CLI `scirust`.
pub fn default_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    for tool in tools::dev::dev_tools()
    {
        registry.register(tool);
    }
    for tool in tools::linalg::linalg_tools()
    {
        registry.register(tool);
    }
    for tool in tools::discovery::discovery_tools()
    {
        registry.register(tool);
    }
    for tool in tools::sis::sis_tools()
    {
        registry.register(tool);
    }
    for tool in tools::grid::grid_tools()
    {
        registry.register(tool);
    }
    for tool in tools::biomed::biomed_tools()
    {
        registry.register(tool);
    }
    for tool in tools::maritime::maritime_tools()
    {
        registry.register(tool);
    }
    for tool in tools::fab::fab_tools()
    {
        registry.register(tool);
    }
    for tool in tools::agtech::agtech_tools()
    {
        registry.register(tool);
    }
    for tool in tools::fatigue::fatigue_tools()
    {
        registry.register(tool);
    }
    registry.register(tools::cli_passthrough::cli_tool());
    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_has_no_name_collisions_and_is_non_empty() {
        // `ToolRegistry::register` panics on a duplicate name, so simply
        // building the registry is itself the collision check.
        let registry = default_registry();
        assert!(!registry.is_empty());
        assert!(registry.names().contains(&"linalg_svd"));
        assert!(registry.names().contains(&"dev_search"));
        assert!(registry.names().contains(&"scirust_cli"));
        assert!(registry.names().contains(&"discovery_scan"));
        assert!(registry.names().contains(&"sis_verify_sif_loop"));
        assert!(registry.names().contains(&"sis_reactor_trip_bypass"));
        assert!(registry.names().contains(&"grid_state_estimate"));
        assert!(registry.names().contains(&"biomed_cbf_safe_dose"));
        assert!(registry.names().contains(&"maritime_collision_risk"));
        assert!(registry.names().contains(&"fab_r2r_update"));
        assert!(registry.names().contains(&"agtech_clean_yield_map"));
        assert!(registry.names().contains(&"fatigue_rainflow_damage"));
    }
}
