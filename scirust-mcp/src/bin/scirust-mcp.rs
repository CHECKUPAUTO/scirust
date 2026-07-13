//! Point d'entrée du serveur MCP : lit des requêtes JSON-RPC 2.0 sur stdin,
//! écrit les réponses sur stdout — une par ligne, transport « stdio » du
//! Model Context Protocol. Voir `scirust-mcp/README.md` pour la
//! configuration côté client (Claude Desktop, `scirust-sciagent`, etc.).

fn main() -> std::io::Result<()> {
    let profile = match std::env::var("SCIRUST_MCP_PROFILE").as_deref()
    {
        Ok("development") => scirust_mcp::RegistryProfile::Development,
        Ok("production") | Err(_) => scirust_mcp::RegistryProfile::Production,
        Ok(other) =>
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "invalid SCIRUST_MCP_PROFILE `{other}`; expected `production` or `development`"
                ),
            ));
        },
    };
    let registry = scirust_mcp::registry_for_profile(profile);
    let server = scirust_mcp::server::McpServer::new(registry);
    scirust_mcp::server::run_stdio(server)
}
