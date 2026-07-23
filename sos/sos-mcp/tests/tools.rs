//! End-to-end: call the real SOS tools through the MCP server's `tools/call`
//! dispatch, against a real `FileStore` populated with real engine objects.

use std::fs;
use std::path::PathBuf;

use serde_json::{Value, json};
use sos_core::{Author, HashAlgo, Object, ObjectId};
use sos_mcp::server::McpServer;
use sos_mcp::{RegistryProfile, registry_for_profile};
use sos_reasoning::{Derivation, DerivationStep, Soundness};
use sos_store::{FileStore, TypedStore};

fn temp_root(name: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!("sos-mcp-test-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    dir
}

fn oid(tag: &[u8]) -> ObjectId {
    ObjectId::compute(HashAlgo::default(), b"mcp-test", tag)
}

fn call(server: &mut McpServer, name: &str, arguments: Value) -> Value {
    let resp = server
        .handle(sos_mcp::protocol::RpcRequest {
            jsonrpc: "2.0".to_owned(),
            id: Some(Value::from(1)),
            method: "tools/call".to_owned(),
            params: json!({ "name": name, "arguments": arguments }),
        })
        .unwrap();
    resp.result.unwrap()
}

#[test]
fn sos_log_and_sos_why_reflect_a_real_store() {
    let root = temp_root("log-why");
    let (base, derived);
    {
        let mut s = FileStore::open(&root).unwrap();
        base = Object::builder(Derivation::new(
            "base",
            Vec::new(),
            Vec::new(),
            Soundness::Proof,
        ))
        .author(Author::human("t"))
        .seal();
        s.put_object(&base).unwrap();
        derived = Object::builder(Derivation::new(
            "derived",
            vec![DerivationStep::new("s", vec![base.id], "derived")],
            vec![base.id],
            Soundness::Proof,
        ))
        .author(Author::human("t"))
        .parents(vec![base.id])
        .seal();
        s.put_object(&derived).unwrap();
    }

    let mut server = McpServer::new(registry_for_profile(RegistryProfile::Query));

    let log_result = call(
        &mut server,
        "sos_log",
        json!({ "store": root.to_str().unwrap() }),
    );
    assert_eq!(log_result["isError"], Value::Bool(false));
    let text = log_result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("Derivation"));

    let why_result = call(
        &mut server,
        "sos_why",
        json!({ "store": root.to_str().unwrap(), "object": derived.id.to_string() }),
    );
    let text = why_result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains(&base.id.to_string()));

    // Every call is attested.
    assert_eq!(server.chain().len(), 2);
    server.chain().verify().unwrap();
    fs::remove_dir_all(&root).ok();
}

#[test]
fn sos_verify_reports_missing_argument_as_a_tool_error_not_a_crash() {
    let mut server = McpServer::new(registry_for_profile(RegistryProfile::Query));
    let result = call(
        &mut server,
        "sos_verify",
        json!({ "store": "/nonexistent" }),
    );
    assert_eq!(result["isError"], Value::Bool(true));
    assert!(
        result["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("object")
    );
}

#[test]
fn sos_plan_ranks_inline_candidates() {
    let mut server = McpServer::new(registry_for_profile(RegistryProfile::Query));
    let candidates = json!([
        {
            "experiment": oid(b"weak").to_string(),
            "eig": { "bits_milli": 50, "se_milli": 10, "level": "L3" },
            "cost": { "compute": 1, "time": 1, "samples": 1, "risk": 0 }
        },
        {
            "experiment": oid(b"strong").to_string(),
            "eig": { "bits_milli": 900, "se_milli": 50, "level": "L3" },
            "cost": { "compute": 2, "time": 1, "samples": 1, "risk": 0 }
        }
    ]);
    let result = call(&mut server, "sos_plan", json!({ "candidates": candidates }));
    assert_eq!(result["isError"], Value::Bool(false));
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains(&oid(b"strong").to_string()));
}

#[test]
fn sos_publish_seals_and_renders_an_inline_publication() {
    let mut server = McpServer::new(registry_for_profile(RegistryProfile::Query));
    let evidence = oid(b"evidence");
    let publication = sos_publication::Publication::builder("Inline Paper")
        .declared_root(evidence)
        .claim(sos_publication::Claim::new(
            "C1",
            "x",
            vec![sos_publication::ClaimBinding::new(
                sos_publication::BindingRole::DirectlySupports,
                evidence,
            )],
        ))
        .build();
    let publication_json = serde_json::to_value(&publication).unwrap();

    let result = call(
        &mut server,
        "sos_publish",
        json!({ "publication": publication_json, "format": "md" }),
    );
    assert_eq!(result["isError"], Value::Bool(false));
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("Inline Paper"));
}

#[test]
fn sos_propose_is_absent_from_the_query_profile_but_present_and_working_in_full() {
    let mut query_server = McpServer::new(registry_for_profile(RegistryProfile::Query));
    let result = call(&mut query_server, "sos_propose", json!({}));
    assert_eq!(result["isError"], Value::Bool(true));
    assert!(
        result["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("unknown tool")
    );

    let root = temp_root("propose");
    let mut full_server = McpServer::new(registry_for_profile(RegistryProfile::Full));
    let concern = oid(b"concern");
    let result = call(
        &mut full_server,
        "sos_propose",
        json!({
            "store": root.to_str().unwrap(),
            "kind": "hypothesis",
            "statement": "an untrusted hypothesis",
            "concerns": [concern.to_string()],
            "rationale": "a hunch",
        }),
    );
    assert_eq!(result["isError"], Value::Bool(false));
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("\"trusted\":false"));

    // It really was stored, as a genuine (untrusted) Proposal object.
    let s = FileStore::open(&root).unwrap();
    assert_eq!(s.len(), 1);
    fs::remove_dir_all(&root).ok();
}
