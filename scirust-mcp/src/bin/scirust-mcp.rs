//! Point d'entrée du serveur MCP : lit des requêtes JSON-RPC 2.0 sur stdin,
//! écrit les réponses sur stdout — une par ligne, transport « stdio » du
//! Model Context Protocol. Voir `scirust-mcp/README.md` pour la
//! configuration côté client (Claude Desktop, `scirust-sciagent`, etc.).

fn main() -> std::io::Result<()> {
    let registry = scirust_mcp::default_registry();
    let server = scirust_mcp::server::McpServer::new(registry);
    scirust_mcp::server::run_stdio(server)
}
