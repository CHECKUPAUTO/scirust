//! # `sos-mcp` — the SOS Model Context Protocol server
//!
//! Exposes the Scientific Operating System's syscalls as MCP tools
//! (RFC-0002 §10.4) over a blocking stdio JSON-RPC 2.0 transport, so any
//! agent — an LLM, `scirust-sciagent`, a script — can discover
//! (`tools/list`) and call (`tools/call`) them without bespoke glue code, each
//! with an explicit JSON schema and every call attested into a tamper-evident
//! chain. See <https://modelcontextprotocol.io> for the protocol.
//!
//! ## Read tools vs. the proposer
//!
//! Every read tool (`sos_log`, `sos_why`, `sos_verify`, `sos_diff`,
//! `sos_know`, `sos_ask`) wraps [`sos_cli`]'s own command functions directly —
//! calling this server runs the exact same code `sos log`/`sos why`/… run
//! from a shell. `sos_plan`/`sos_publish`/`sos_plugins` call their engines
//! directly with **inline** arguments (an MCP client passes structured JSON,
//! not a file path).
//!
//! `sos_propose` is different in kind, not degree: it is the
//! **untrusted-proposer entry point** (Invariant IX — cognition proposes,
//! determinism disposes). Calling it never makes an agent's suggestion true;
//! it seals and stores an [`sos_ccos::Proposal`] that stays untrusted until a
//! deterministic engine later disposes of it. Because that is a materially
//! different trust posture from the read tools, it is **opt-in**: a server
//! built with [`RegistryProfile::Query`] never registers it at all, so an
//! operator who wants a strictly read-only surface for a given deployment
//! gets one by construction, not by hoping every caller behaves.
//!
//! ## Attestation, reused not reinvented
//!
//! Every tool call is attested into a [`sos_ccos::CcosChain`] — the same
//! tamper-evident mechanism `sos-ccos` uses for cognitive acts, because an MCP
//! tool call from an external agent *is* one. See [`server`] for the
//! transport and [`registry`] for the tool-registration surface this crate's
//! own tools plug into ([`tools`]).
//!
//! ## Scope boundary — no stub
//!
//! No async runtime, no third-party JSON-RPC/MCP crate: the stdio transport
//! is a small, blocking, line-oriented loop (mirroring `scirust-mcp`'s own
//! proven pattern in this repository), so this crate adds no new third-party
//! dependency. `sos run` (executing a discovery workflow) is not exposed as a
//! tool for the same reason `sos-cli` does not implement it: it needs a real
//! [`sos_workflow::StageExecutor`] backend that does not exist yet.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod protocol;
pub mod registry;
pub mod server;
pub mod tools;

use registry::ToolRegistry;

/// Which tools a server registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistryProfile {
    /// Read-only: every query tool, no `sos_propose`. Safe to expose to any
    /// caller — nothing it does can introduce an untrusted claim into the
    /// graph.
    Query,
    /// Everything in [`RegistryProfile::Query`], plus `sos_propose` — the
    /// untrusted-proposer entry point. Choose this only when the deployment
    /// genuinely wants agents to submit proposals for later deterministic
    /// review.
    Full,
}

/// Build the default (query-only) tool registry.
#[must_use]
pub fn default_registry() -> ToolRegistry {
    registry_for_profile(RegistryProfile::Query)
}

/// Build the tool registry for a given profile.
#[must_use]
pub fn registry_for_profile(profile: RegistryProfile) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(tools::log_tool());
    registry.register(tools::why_tool());
    registry.register(tools::verify_tool());
    registry.register(tools::diff_tool());
    registry.register(tools::know_tool());
    registry.register(tools::ask_tool());
    registry.register(tools::plan_tool());
    registry.register(tools::publish_tool());
    registry.register(tools::plugins_tool());
    if profile == RegistryProfile::Full
    {
        registry.register(tools::propose_tool());
    }
    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_profile_excludes_propose() {
        let registry = default_registry();
        assert!(!registry.is_empty());
        assert!(registry.names().contains(&"sos_log"));
        assert!(registry.names().contains(&"sos_why"));
        assert!(registry.names().contains(&"sos_verify"));
        assert!(registry.names().contains(&"sos_diff"));
        assert!(registry.names().contains(&"sos_know"));
        assert!(registry.names().contains(&"sos_ask"));
        assert!(registry.names().contains(&"sos_plan"));
        assert!(registry.names().contains(&"sos_publish"));
        assert!(registry.names().contains(&"sos_plugins"));
        assert!(!registry.names().contains(&"sos_propose"));
    }

    #[test]
    fn full_profile_adds_propose() {
        let registry = registry_for_profile(RegistryProfile::Full);
        assert!(registry.names().contains(&"sos_propose"));
    }
}
