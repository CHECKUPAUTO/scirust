//! `scirust-mcp` — serveur Model Context Protocol pour SciRust.
//!
//! Expose les capacités de SciRust (solveurs numériques, outils de
//! développement du SLM `scirust-sciagent`, découverte d'actifs OT/IT de
//! `scirust-discovery`) comme des outils MCP standard : n'importe quel
//! agent — le SLM embarqué, Claude, ChatGPT, un script — peut les découvrir
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
/// linéaire (`scirust-solvers`), découverte OT/IT (`scirust-discovery`), et
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
    }
}
