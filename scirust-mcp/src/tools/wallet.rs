//! MCP tools for wallet connectivity — **watch-only / dry-run**, standards
//! compliant.
//!
//! These let an agentic LLM connect to recognized wallets over their real
//! protocols (WalletConnect v2, EVM/EIP-1193, exchange REST) to *read* state and
//! *construct* transactions/requests — while every action that would sign or
//! move funds stays behind the [`scirust_trader::wallet::WalletAuthorization`]
//! gate. The tools here never hold a private key or an exchange secret: signing
//! uses a secret injected by the host process via an environment variable, and
//! the tools that would spend still require an operator authorization.
//!
//! Tools:
//! * `wallet_validate_address`          — EIP-55 checksum validate/format
//! * `wallet_parse_walletconnect_uri`   — parse a WalletConnect v2 pairing URI
//! * `wallet_walletconnect_namespace`   — build an eip155 session proposal
//! * `wallet_build_evm_transaction`     — EIP-1559 tx + signing hash (unsigned)
//! * `wallet_eip712_hash`               — EIP-712 domain separator / digest
//! * `wallet_sign_exchange_request`     — HMAC-sign a REST request (server secret)
//! * `wallet_authorization_status`      — report whether signing is armed

use crate::registry::McpTool;
use serde_json::{Value, json};

use scirust_trader::wallet::{
    Chain, Eip712Domain, Eip1559Tx, EvmAddress, WalletAuthorization, eip155_namespace,
    parse_walletconnect_uri, sign_binance_query, sign_coinbase_request, to_hex,
};

/// All wallet tools.
pub fn wallet_tools() -> Vec<McpTool> {
    vec![
        validate_address_tool(),
        parse_wc_uri_tool(),
        wc_namespace_tool(),
        build_evm_tx_tool(),
        eip712_tool(),
        sign_exchange_tool(),
        authorization_status_tool(),
    ]
}

fn su128(v: &Value, key: &str, default: u128) -> u128 {
    match v.get(key)
    {
        Some(Value::String(s)) => s.parse().unwrap_or(default),
        Some(Value::Number(n)) => n.as_u64().map(|x| x as u128).unwrap_or(default),
        _ => default,
    }
}

fn su64(v: &Value, key: &str, default: u64) -> u64 {
    v.get(key).and_then(|x| x.as_u64()).unwrap_or(default)
}

fn addr20(v: &Value, key: &str) -> Option<[u8; 20]> {
    v.get(key)
        .and_then(|x| x.as_str())
        .and_then(EvmAddress::from_hex)
        .map(|a| a.0)
}

fn validate_address_tool() -> McpTool {
    McpTool {
        name: "wallet_validate_address".to_string(),
        description: "Validate an EVM address and return its EIP-55 mixed-case checksum form. \
            Reports whether the input was already correctly checksummed. Use before showing an \
            address to a user or building a transaction — a bad checksum is a typo guard."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": { "address": { "type": "string", "description": "0x-prefixed 20-byte hex address" } },
            "required": ["address"]
        }),
        handler: Box::new(|args| {
            let a = args
                .get("address")
                .and_then(|x| x.as_str())
                .ok_or("missing `address`")?;
            let addr = EvmAddress::from_hex(a).ok_or("not a 20-byte hex address")?;
            let checksum = addr.to_checksum();
            Ok(json!({
                "valid": true,
                "checksum": checksum,
                // Passes EIP-55 validation (true for all-lowercase input, which
                // claims no checksum; false only for a wrong mixed-case checksum).
                "checksum_valid": EvmAddress::is_valid_checksum(a),
                // Whether the input was already in exact checksum form.
                "already_checksummed": a == checksum,
            }))
        }),
    }
}

fn parse_wc_uri_tool() -> McpTool {
    McpTool {
        name: "wallet_parse_walletconnect_uri".to_string(),
        description: "Parse a WalletConnect v2 pairing URI (wc:{topic}@2?relay-protocol=irn&symKey=…) \
            into its components (topic, version, relay protocol, symKey, expiry). The first step of \
            establishing a WalletConnect session with a non-custodial wallet."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": { "uri": { "type": "string" } },
            "required": ["uri"]
        }),
        handler: Box::new(|args| {
            let uri = args.get("uri").and_then(|x| x.as_str()).ok_or("missing `uri`")?;
            let p = parse_walletconnect_uri(uri).map_err(|e| e.to_string())?;
            Ok(serde_json::to_value(&p).unwrap_or(Value::Null))
        }),
    }
}

fn wc_namespace_tool() -> McpTool {
    McpTool {
        name: "wallet_walletconnect_namespace".to_string(),
        description: "Build the standard eip155 `requiredNamespaces` (chains as CAIP-2, plus the \
            eth_sendTransaction / personal_sign / eth_signTypedData_v4 methods and events) an agent \
            proposes when opening a WalletConnect session for a set of EVM chains."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "chain_ids": { "type": "array", "items": { "type": "integer" }, "description": "EIP-155 chain ids, e.g. [1, 137, 42161]" }
            },
            "required": ["chain_ids"]
        }),
        handler: Box::new(|args| {
            let ids = args.get("chain_ids").and_then(|x| x.as_array()).ok_or("missing `chain_ids`")?;
            let chains: Vec<Chain> = ids
                .iter()
                .filter_map(|v| v.as_u64())
                .map(Chain::Evm)
                .collect();
            let ns = eip155_namespace(&chains);
            Ok(json!({ "eip155": serde_json::to_value(&ns).unwrap_or(Value::Null) }))
        }),
    }
}

fn build_evm_tx_tool() -> McpTool {
    McpTool {
        name: "wallet_build_evm_transaction".to_string(),
        description: "Construct an unsigned EIP-1559 (type-2) transaction and return its exact \
            keccak-256 signing hash — the digest a wallet would sign. This is DRY-RUN: nothing is \
            signed or broadcast. Show the returned hash and fields to the user for confirmation \
            before any signing is authorized. Large wei values may be passed as decimal strings."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "chain_id": { "type": "integer" },
                "nonce": { "type": "integer" },
                "to": { "type": "string", "description": "recipient address (omit for contract creation)" },
                "value_wei": { "type": ["string", "integer"], "description": "amount in wei" },
                "max_fee_per_gas": { "type": ["string", "integer"] },
                "max_priority_fee_per_gas": { "type": ["string", "integer"] },
                "gas_limit": { "type": "integer" },
                "data": { "type": "string", "description": "calldata hex (default empty)" }
            },
            "required": ["chain_id", "nonce", "gas_limit"]
        }),
        handler: Box::new(|args| {
            let data = args
                .get("data")
                .and_then(|x| x.as_str())
                .map(|s| scirust_trader::wallet::from_hex(s).unwrap_or_default())
                .unwrap_or_default();
            let tx = Eip1559Tx {
                chain_id: su64(&args, "chain_id", 1),
                nonce: su64(&args, "nonce", 0),
                max_priority_fee_per_gas: su128(&args, "max_priority_fee_per_gas", 1_000_000_000),
                max_fee_per_gas: su128(&args, "max_fee_per_gas", 20_000_000_000),
                gas_limit: su64(&args, "gas_limit", 21_000),
                to: addr20(&args, "to"),
                value: su128(&args, "value_wei", 0),
                data,
            };
            let hash = tx.signing_hash();
            Ok(json!({
                "signing_hash": format!("0x{}", to_hex(&hash)),
                "tx_type": "eip1559",
                "signed": false,
                "note": "dry-run: unsigned. Signing requires an out-of-band WalletAuthorization.",
                "tx": serde_json::to_value(&tx).unwrap_or(Value::Null),
            }))
        }),
    }
}

fn eip712_tool() -> McpTool {
    McpTool {
        name: "wallet_eip712_hash".to_string(),
        description: "Compute the EIP-712 domain separator for a typed-data domain, and — if a \
            message struct hash is supplied — the final signing digest \
            keccak256(0x1901 ‖ domainSeparator ‖ structHash). Lets an agent show the exact typed-data \
            digest before any signature is authorized."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "version": { "type": "string" },
                "chain_id": { "type": "integer" },
                "verifying_contract": { "type": "string" },
                "struct_hash": { "type": "string", "description": "optional 32-byte hex hashStruct(message)" }
            },
            "required": ["name", "version", "chain_id"]
        }),
        handler: Box::new(|args| {
            let domain = Eip712Domain {
                name: args.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                version: args.get("version").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                chain_id: su64(&args, "chain_id", 1),
                verifying_contract: addr20(&args, "verifying_contract"),
            };
            let sep = domain.separator();
            let mut out = json!({ "domain_separator": format!("0x{}", to_hex(&sep)) });
            if let Some(sh) = args.get("struct_hash").and_then(|x| x.as_str())
            {
                if let Some(bytes) = scirust_trader::wallet::from_hex(sh)
                {
                    if bytes.len() == 32
                    {
                        let mut arr = [0u8; 32];
                        arr.copy_from_slice(&bytes);
                        out["digest"] = json!(format!("0x{}", to_hex(&domain.digest(&arr))));
                    }
                    else
                    {
                        return Err("struct_hash must be 32 bytes".to_string());
                    }
                }
            }
            Ok(out)
        }),
    }
}

fn sign_exchange_tool() -> McpTool {
    McpTool {
        name: "wallet_sign_exchange_request".to_string(),
        description: "HMAC-sign an exchange REST request (Binance or Coinbase style) using a secret \
            the OPERATOR supplies out-of-band via the SCIRUST_EXCHANGE_SECRET environment variable — \
            the secret is never taken from the conversation and never returned. The agent builds the \
            query/prehash; this tool returns only the signature to attach to the request. If the \
            secret env var is unset, signing is refused (the operator has not armed it)."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "style": { "type": "string", "enum": ["binance", "coinbase"] },
                "query": { "type": "string", "description": "binance: the query string to sign" },
                "timestamp": { "type": "string", "description": "coinbase: request timestamp" },
                "method": { "type": "string", "description": "coinbase: HTTP method" },
                "path": { "type": "string", "description": "coinbase: request path" },
                "body": { "type": "string", "description": "coinbase: request body (may be empty)" }
            },
            "required": ["style"]
        }),
        handler: Box::new(|args| {
            let secret = std::env::var("SCIRUST_EXCHANGE_SECRET").map_err(|_| {
                "exchange signing is not armed: the operator has not set SCIRUST_EXCHANGE_SECRET"
                    .to_string()
            })?;
            let style = args.get("style").and_then(|x| x.as_str()).unwrap_or("binance");
            let sig = match style
            {
                "coinbase" =>
                {
                    let ts = args.get("timestamp").and_then(|x| x.as_str()).ok_or("missing `timestamp`")?;
                    let method = args.get("method").and_then(|x| x.as_str()).ok_or("missing `method`")?;
                    let path = args.get("path").and_then(|x| x.as_str()).ok_or("missing `path`")?;
                    let body = args.get("body").and_then(|x| x.as_str()).unwrap_or("");
                    sign_coinbase_request(secret.as_bytes(), ts, method, path, body)
                },
                _ =>
                {
                    let query = args.get("query").and_then(|x| x.as_str()).ok_or("missing `query`")?;
                    sign_binance_query(secret.as_bytes(), query)
                },
            };
            Ok(json!({ "signature": sig, "algo": "hmac-sha256", "style": style }))
        }),
    }
}

fn authorization_status_tool() -> McpTool {
    McpTool {
        name: "wallet_authorization_status".to_string(),
        description: "Report whether fund-moving actions are armed for this server, and validate an \
            optional WalletAuthorization against the operator key (SCIRUST_WALLET_KEY, held \
            server-side). NEVER signs or sends anything — it only tells the agent whether a given \
            authorization would permit a method/chain/value, so the agent knows to ask the operator \
            rather than attempt an action that will be refused."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "authorization": { "type": "object", "description": "a WalletAuthorization to validate (optional)" },
                "chain_id": { "type": "integer" },
                "method": { "type": "string" },
                "value_wei": { "type": ["string", "integer"] },
                "now_unix": { "type": "integer" }
            }
        }),
        handler: Box::new(|args| {
            let key = std::env::var("SCIRUST_WALLET_KEY").ok();
            let armed = key.is_some();
            let mut out = json!({
                "signing_armed": armed,
                "exchange_signing_armed": std::env::var("SCIRUST_EXCHANGE_SECRET").is_ok(),
                "note": "Signing/sending requires a valid WalletAuthorization under the operator's \
                         SCIRUST_WALLET_KEY. The agent cannot mint one itself.",
            });
            if let (Some(k), Some(auth_val)) = (key.as_ref(), args.get("authorization"))
            {
                match serde_json::from_value::<WalletAuthorization>(auth_val.clone())
                {
                    Ok(auth) =>
                    {
                        let sig_ok = auth.verify_signature(k.as_bytes());
                        out["authorization_signature_valid"] = json!(sig_ok);
                        if let (Some(cid), Some(method)) = (
                            args.get("chain_id").and_then(|x| x.as_u64()),
                            args.get("method").and_then(|x| x.as_str()),
                        )
                        {
                            let value = su128(&args, "value_wei", 0);
                            let now = su64(&args, "now_unix", 0);
                            out["would_authorize"] =
                                json!(auth.authorizes(k.as_bytes(), cid, method, value, now));
                        }
                    },
                    Err(e) => out["authorization_parse_error"] = json!(e.to_string()),
                }
            }
            Ok(out)
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str) -> McpTool {
        wallet_tools()
            .into_iter()
            .find(|t| t.name == name)
            .expect("tool exists")
    }

    #[test]
    fn all_wallet_tools_unique() {
        let tools = wallet_tools();
        let mut names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        names.sort();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len());
        assert_eq!(before, 7);
    }

    #[test]
    fn validate_address_checksums() {
        let t = tool("wallet_validate_address");
        let out = (t.handler)(json!({ "address": "0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed" }))
            .unwrap();
        assert_eq!(
            out["checksum"],
            json!("0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed")
        );
        assert_eq!(out["already_checksummed"], json!(false));
        assert_eq!(out["checksum_valid"], json!(true));
    }

    #[test]
    fn parse_wc_uri_works() {
        let t = tool("wallet_parse_walletconnect_uri");
        let out = (t.handler)(json!({
            "uri": "wc:7f6e504bfad60b485450578e05678ed3e8e8c4751d3c6160be17160d63ec90f9@2?relay-protocol=irn&symKey=587d5484ce2a2a6ee3ba1962fdd7e8588e06200c46823bd18fbd67def96ad303"
        })).unwrap();
        assert_eq!(out["version"], json!(2));
        assert_eq!(out["relay_protocol"], json!("irn"));
    }

    #[test]
    fn build_evm_tx_is_dry_run() {
        let t = tool("wallet_build_evm_transaction");
        let out = (t.handler)(json!({
            "chain_id": 1, "nonce": 9, "gas_limit": 21000,
            "to": "0x3535353535353535353535353535353535353535",
            "value_wei": "1000000000000000000"
        }))
        .unwrap();
        assert_eq!(out["signed"], json!(false));
        assert!(out["signing_hash"].as_str().unwrap().starts_with("0x"));
        assert_eq!(out["signing_hash"].as_str().unwrap().len(), 66); // 0x + 64
    }

    #[test]
    fn eip712_domain_hash() {
        let t = tool("wallet_eip712_hash");
        let out = (t.handler)(json!({ "name": "App", "version": "1", "chain_id": 1 })).unwrap();
        assert!(out["domain_separator"].as_str().unwrap().starts_with("0x"));
    }

    #[test]
    fn exchange_signing_refused_without_secret() {
        std::env::remove_var("SCIRUST_EXCHANGE_SECRET");
        let t = tool("wallet_sign_exchange_request");
        let r = (t.handler)(json!({ "style": "binance", "query": "symbol=BTCUSDT" }));
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("SCIRUST_EXCHANGE_SECRET"));
    }

    #[test]
    fn authorization_status_reports_disarmed() {
        std::env::remove_var("SCIRUST_WALLET_KEY");
        let t = tool("wallet_authorization_status");
        let out = (t.handler)(json!({})).unwrap();
        assert_eq!(out["signing_armed"], json!(false));
    }
}
