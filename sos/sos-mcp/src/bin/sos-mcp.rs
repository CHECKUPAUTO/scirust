//! `sos-mcp` — entry point. Runs the query-only MCP server over stdio.
//!
//! Pass `--full` to also register `sos_propose` (the untrusted-proposer
//! tool) — an explicit, visible opt-in rather than a silent default.

use sos_mcp::server::{McpServer, run_stdio};
use sos_mcp::{RegistryProfile, registry_for_profile};

fn main() -> std::io::Result<()> {
    let full = std::env::args().any(|a| a == "--full");
    let profile = if full
    {
        RegistryProfile::Full
    }
    else
    {
        RegistryProfile::Query
    };
    let server = McpServer::new(registry_for_profile(profile));
    run_stdio(server)
}
