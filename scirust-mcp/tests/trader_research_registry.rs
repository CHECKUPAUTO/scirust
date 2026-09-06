use scirust_mcp::default_registry;

#[test]
fn final_trader_research_tools_are_exposed_by_production_registry() {
    let registry = default_registry();
    let names = registry.names();

    for expected in [
        "trader_research_purged_cv",
        "trader_research_dsr",
        "trader_research_pbo",
        "trader_research_cost_stress",
        "trader_research_rl_plan",
        "trader_research_manifest",
        "trader_research_compare",
    ]
    {
        assert!(
            names.contains(&expected),
            "production MCP registry is missing {expected}"
        );
    }
}
