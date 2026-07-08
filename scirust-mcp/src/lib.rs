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
    for tool in tools::trader::trader_tools()
    {
        registry.register(tool);
    }
    for tool in tools::wallet::wallet_tools()
    {
        registry.register(tool);
    }
    for tool in tools::tolerance::tolerance_tools()
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
        assert!(registry.names().contains(&"trader_scan_opportunities"));
        assert!(registry.names().contains(&"trader_backtest"));
        assert!(registry.names().contains(&"trader_execution_plan"));
        assert!(registry.names().contains(&"trader_market_making_quotes"));
        assert!(registry.names().contains(&"trader_portfolio"));
        assert!(registry.names().contains(&"trader_rebalance"));
        assert!(registry.names().contains(&"trader_dashboard"));
        assert!(registry.names().contains(&"trader_walkforward"));
        assert!(registry.names().contains(&"trader_monte_carlo"));
        assert!(registry.names().contains(&"trader_portfolio_construct"));
        assert!(registry.names().contains(&"trader_regime"));
        assert!(registry.names().contains(&"trader_optimize"));
        assert!(registry.names().contains(&"trader_pair_analyze"));
        assert!(registry.names().contains(&"trader_pair_scan"));
        assert!(registry.names().contains(&"trader_option_price"));
        assert!(registry.names().contains(&"trader_option_book"));
        assert!(registry.names().contains(&"wallet_validate_address"));
        assert!(registry.names().contains(&"wallet_build_evm_transaction"));
        assert!(registry.names().contains(&"wallet_authorization_status"));
        assert!(registry.names().contains(&"tolerance_inertial_capability"));
        assert!(registry.names().contains(&"tolerance_chain_allocate"));
        assert!(registry.names().contains(&"tolerance_acceptance_plan"));
        assert!(registry.names().contains(&"tolerance_form_modal"));
        assert!(registry.names().contains(&"tolerance_3d_surface_inertia"));
        assert!(registry.names().contains(&"tolerance_optimize_cost"));
        assert!(registry.names().contains(&"tolerance_nonnormal_capability"));
        assert!(registry.names().contains(&"tolerance_position"));
        assert!(registry.names().contains(&"tolerance_monte_carlo"));
        assert!(registry.names().contains(&"tolerance_geometry"));
        assert!(registry.names().contains(&"tolerance_sensitivity"));
        assert!(registry.names().contains(&"tolerance_discrete_allocate"));
        assert!(registry.names().contains(&"tolerance_drift"));
        assert!(registry.names().contains(&"tolerance_correlated"));
        assert!(registry.names().contains(&"tolerance_gage_rr"));
        assert!(registry.names().contains(&"tolerance_statistical_interval"));
        assert!(registry.names().contains(&"tolerance_dual_sensitivity"));
        assert!(registry.names().contains(&"tolerance_distribution_fit"));
        assert!(registry.names().contains(&"tolerance_gdt"));
        assert!(registry.names().contains(&"tolerance_capability_ci"));
        assert!(registry.names().contains(&"tolerance_variables_plan"));
        assert!(registry.names().contains(&"tolerance_six_sigma"));
        assert!(registry.names().contains(&"tolerance_attribution"));
        assert!(registry.names().contains(&"tolerance_attributes_plan"));
        assert!(registry.names().contains(&"tolerance_interference"));
        assert!(registry.names().contains(&"tolerance_subgroup_capability"));
        assert!(registry.names().contains(&"tolerance_fits"));
        assert!(registry.names().contains(&"tolerance_sequential"));
        assert!(registry.names().contains(&"tolerance_taguchi"));
    }
}
