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
use std::time::{SystemTime, UNIX_EPOCH};

use scirust_trader::wallet::{
    Chain, Eip712Domain, Eip1559Tx, EvmAddress, TxContext, WalletAuthorization, eip155_namespace,
    parse_walletconnect_uri, sign_coinbase_request_base64, to_hex,
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
        description:
            "Parse a WalletConnect v2 pairing URI (wc:{topic}@2?relay-protocol=irn&symKey=…) \
            into its non-secret components (topic, version, relay protocol, expiry). The symKey is \
            validated but never returned. This is the first step of \
            establishing a WalletConnect session with a non-custodial wallet."
                .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": { "uri": { "type": "string" } },
            "required": ["uri"]
        }),
        handler: Box::new(|args| {
            let uri = args
                .get("uri")
                .and_then(|x| x.as_str())
                .ok_or("missing `uri`")?;
            let p = parse_walletconnect_uri(uri).map_err(|e| e.to_string())?;
            let mut value = serde_json::to_value(&p)
                .map_err(|e| format!("failed to encode WalletConnect metadata: {e}"))?;
            value["sym_key_present"] = json!(!p.sym_key.is_empty());
            Ok(value)
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

/// Defense-in-depth scope check for exchange request signing. A hard denylist of
/// fund-exfiltration / key-management endpoint patterns is *always* refused
/// (even if the exchange API key is misconfigured to allow them); an optional
/// operator allowlist (`SCIRUST_EXCHANGE_ALLOWED_PATHS`, comma-separated exact
/// `METHOD /path` routes) restricts what may be signed.
fn exchange_scope_check(
    method: &str,
    path: &str,
    signed_payload: &str,
    allowlist: Option<&str>,
) -> Result<(), String> {
    const DENY: &[&str] = &[
        "withdraw",
        "transfer",
        "apirestrictions",
        "apikey",
        "api-key",
        "apitradingstatus",
    ];
    let normalized_path = normalize_exchange_path(path)?;
    if normalized_path != path
    {
        return Err(format!(
            "exchange path must already be canonical; use `{normalized_path}`"
        ));
    }
    validate_exchange_method(method)?;
    let inspected_payload = percent_decode_for_inspection(signed_payload).to_ascii_lowercase();
    let lc = format!(
        "{} {inspected_payload}",
        normalized_path.to_ascii_lowercase()
    );
    if let Some(bad) = DENY.iter().find(|d| lc.contains(**d))
    {
        return Err(format!(
            "refused: request matches the blocked pattern `{bad}` — withdrawals, transfers and \
             key management are never signed by the agent"
        ));
    }
    let allow = allowlist
        .filter(|value| !value.trim().is_empty())
        .ok_or(
            "exchange signing is not armed: set SCIRUST_EXCHANGE_ALLOWED_PATHS to comma-separated `METHOD /path` routes",
        )?;
    let mut matched = false;
    for allowed_route in allow.split(',').map(str::trim).filter(|p| !p.is_empty())
    {
        let mut fields = allowed_route.split_ascii_whitespace();
        let allowed_method = fields
            .next()
            .ok_or("SCIRUST_EXCHANGE_ALLOWED_PATHS contains an empty route")?;
        let allowed_path = fields.next().ok_or_else(|| {
            format!("invalid exchange route `{allowed_route}`; expected `METHOD /path`")
        })?;
        if fields.next().is_some()
        {
            return Err(format!(
                "invalid exchange route `{allowed_route}`; expected `METHOD /path`"
            ));
        }
        validate_exchange_method(allowed_method)?;
        let normalized_allowed_path = normalize_exchange_path(allowed_path)?;
        if normalized_allowed_path != allowed_path
        {
            return Err(format!(
                "allowlisted exchange path must be canonical; use `{allowed_method} {normalized_allowed_path}`"
            ));
        }
        if allowed_method == method && allowed_path == normalized_path
        {
            matched = true;
        }
    }
    if matched
    {
        Ok(())
    }
    else
    {
        Err("refused: request route does not match SCIRUST_EXCHANGE_ALLOWED_PATHS".to_string())
    }
}

fn validate_exchange_method(method: &str) -> Result<(), String> {
    match method
    {
        "GET" | "POST" | "PUT" | "PATCH" | "DELETE" => Ok(()),
        _ => Err(format!(
            "unsupported exchange HTTP method `{method}`; use an uppercase standard method"
        )),
    }
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte
    {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn percent_decode(input: &str) -> Result<String, String> {
    let bytes = input.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len()
    {
        if bytes[index] == b'%'
        {
            let hi = bytes
                .get(index + 1)
                .and_then(|byte| hex_nibble(*byte))
                .ok_or("exchange path contains an invalid percent escape")?;
            let lo = bytes
                .get(index + 2)
                .and_then(|byte| hex_nibble(*byte))
                .ok_or("exchange path contains an invalid percent escape")?;
            decoded.push((hi << 4) | lo);
            index += 3;
        }
        else
        {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded).map_err(|_| "exchange path is not valid UTF-8".to_string())
}

/// Decode valid percent triplets for denylist inspection without rejecting an
/// otherwise opaque query string or JSON body containing a literal `%`.
fn percent_decode_for_inspection(input: &str) -> String {
    let mut inspected = input.to_string();
    loop
    {
        let bytes = inspected.as_bytes();
        let mut decoded = Vec::with_capacity(bytes.len());
        let mut index = 0;
        while index < bytes.len()
        {
            if bytes[index] == b'%'
            {
                if let (Some(hi), Some(lo)) = (
                    bytes.get(index + 1).and_then(|byte| hex_nibble(*byte)),
                    bytes.get(index + 2).and_then(|byte| hex_nibble(*byte)),
                )
                {
                    decoded.push((hi << 4) | lo);
                    index += 3;
                    continue;
                }
            }
            decoded.push(bytes[index]);
            index += 1;
        }
        let next = String::from_utf8_lossy(&decoded).into_owned();
        if next == inspected
        {
            return inspected;
        }
        inspected = next;
    }
}

/// Canonicalize an exchange request path before applying policy. The path is
/// decoded once and dot segments are resolved; query strings and fragments
/// must be supplied through their dedicated arguments, never smuggled into the
/// path used by the allowlist.
fn normalize_exchange_path(path: &str) -> Result<String, String> {
    if path.is_empty() || path.trim() != path
    {
        return Err("exchange path must be a non-empty, unpadded absolute path".to_string());
    }
    // Decode to a fixed point so `%2577ithdraw` cannot hide `withdraw` behind
    // one extra encoding layer. Every successful pass shortens the string, so
    // the loop is bounded by the input length.
    let mut decoded = path.to_string();
    loop
    {
        let next = percent_decode(&decoded)?;
        if next == decoded
        {
            break;
        }
        decoded = next;
    }
    if !decoded.starts_with('/')
    {
        return Err("exchange path must start with `/`".to_string());
    }
    if decoded
        .bytes()
        .any(|byte| byte.is_ascii_control() || byte == b'\\' || byte == b'?' || byte == b'#')
    {
        return Err("exchange path contains a forbidden character".to_string());
    }

    let mut segments = Vec::new();
    for segment in decoded.split('/').skip(1)
    {
        match segment
        {
            "" | "." =>
            {},
            ".." =>
            {
                segments
                    .pop()
                    .ok_or("exchange path escapes above its root")?;
            },
            _ => segments.push(segment),
        }
    }
    if segments.is_empty()
    {
        Ok("/".to_string())
    }
    else
    {
        Ok(format!("/{}", segments.join("/")))
    }
}

fn sign_exchange_tool() -> McpTool {
    McpTool {
        name: "wallet_sign_exchange_request".to_string(),
        description: "HMAC-sign an endpoint-bound Coinbase Exchange REST request using a secret \
            the OPERATOR supplies out-of-band via the SCIRUST_EXCHANGE_SECRET environment variable — \
            the secret is never taken from the conversation and never returned. The agent builds the \
            query/prehash; this tool returns only the signature to attach to the request. Signing is \
            refused unless both the base64 Coinbase secret and an exact comma-separated \
            `METHOD /path` allowlist in SCIRUST_EXCHANGE_ALLOWED_PATHS are configured. \
            Withdrawal/transfer/API-key routes are always refused. Binance signing is deliberately \
            unavailable here because its HMAC payload does not bind the endpoint, so an agent-facing \
            signer cannot enforce an endpoint allowlist."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "style": { "type": "string", "enum": ["coinbase"] },
                "timestamp": { "type": "string", "description": "coinbase: request timestamp" },
                "method": { "type": "string", "description": "coinbase: HTTP method" },
                "path": { "type": "string", "description": "coinbase: request path" },
                "body": { "type": "string", "description": "coinbase: request body (may be empty)" }
            },
            "required": ["style"]
        }),
        handler: Box::new(|args| {
            let style = args
                .get("style")
                .and_then(|x| x.as_str())
                .ok_or("missing `style`; expected `coinbase`")?;
            if style == "binance"
            {
                return Err(
                    "refused: Binance signatures do not bind the HTTP endpoint, so this agent-facing signer cannot enforce its route allowlist"
                        .to_string(),
                );
            }
            if style != "coinbase"
            {
                return Err(format!(
                    "unsupported exchange signing style `{style}`; expected `coinbase`"
                ));
            }
            let secret = std::env::var("SCIRUST_EXCHANGE_SECRET").map_err(|_| {
                "exchange signing is not armed: the operator has not set SCIRUST_EXCHANGE_SECRET"
                    .to_string()
            })?;
            if secret.trim().is_empty() {
                return Err("exchange signing is not armed: SCIRUST_EXCHANGE_SECRET is empty".to_string());
            }
            let allowlist = std::env::var("SCIRUST_EXCHANGE_ALLOWED_PATHS").ok();
            let ts = args
                .get("timestamp")
                .and_then(|x| x.as_str())
                .ok_or("missing `timestamp`")?;
            validate_coinbase_timestamp(ts)?;
            let method = args
                .get("method")
                .and_then(|x| x.as_str())
                .ok_or("missing `method`")?;
            let path = args
                .get("path")
                .and_then(|x| x.as_str())
                .ok_or("missing `path`")?;
            let body = args.get("body").and_then(|x| x.as_str()).unwrap_or("");
            // Method, canonical path and body all enter the Coinbase prehash, so
            // the allowlist decision is cryptographically bound to this signature.
            exchange_scope_check(method, path, body, allowlist.as_deref())?;
            let signature = sign_coinbase_request_base64(&secret, ts, method, path, body)?;
            Ok(json!({
                "signature": signature,
                "algo": "hmac-sha256",
                "encoding": "base64",
                "style": style,
            }))
        }),
    }
}

fn validate_coinbase_timestamp(timestamp: &str) -> Result<(), String> {
    let mut parts = timestamp.split('.');
    let seconds = parts.next().unwrap_or_default();
    let fraction = parts.next();
    if seconds.is_empty()
        || !seconds.bytes().all(|byte| byte.is_ascii_digit())
        || fraction.is_some_and(|value| {
            value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit())
        })
        || parts.next().is_some()
    {
        return Err("Coinbase timestamp must be Unix seconds using decimal digits".to_string());
    }
    Ok(())
}

fn authorization_status_tool() -> McpTool {
    McpTool {
        name: "wallet_authorization_status".to_string(),
        description: "Report whether fund-moving actions are armed for this server, and preview \
            whether an optional WalletAuthorization would permit a concrete transaction context \
            (chain, method, recipient `to`, calldata `selector`, value, or a bound `tx_hash`), \
            validated against the operator key (SCIRUST_WALLET_KEY, held server-side). NEVER signs \
            or sends — it only tells the agent whether an action would be permitted (and why not), \
            so it can ask the operator rather than attempt something that will be refused. The \
            validity window is checked against the SERVER clock, not any client-supplied time."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "authorization": { "type": "object", "description": "a WalletAuthorization to validate (optional)" },
                "chain_id": { "type": "integer" },
                "method": { "type": "string" },
                "to": { "type": "string", "description": "recipient address (policy-mode allowlist check)" },
                "selector": { "type": "string", "description": "4-byte calldata selector hex (empty = native transfer)" },
                "value_wei": { "type": ["string", "integer"] },
                "tx_hash": { "type": "string", "description": "signing hash (for a bound authorization)" }
            }
        }),
        handler: Box::new(|args| {
            let key = std::env::var("SCIRUST_WALLET_KEY")
                .ok()
                .filter(|value| !value.trim().is_empty());
            let armed = key.is_some();
            let mut out = json!({
                "signing_armed": armed,
                "exchange_signing_armed": std::env::var("SCIRUST_EXCHANGE_SECRET")
                    .map(|value| !value.trim().is_empty())
                    .unwrap_or(false)
                    && std::env::var("SCIRUST_EXCHANGE_ALLOWED_PATHS")
                        .map(|value| !value.trim().is_empty())
                        .unwrap_or(false),
                "note": "Signing/sending requires a valid WalletAuthorization under the operator's \
                         SCIRUST_WALLET_KEY. The agent cannot mint one itself.",
            });
            if let (Some(k), Some(auth_val)) = (key.as_ref(), args.get("authorization"))
            {
                match serde_json::from_value::<WalletAuthorization>(auth_val.clone())
                {
                    Ok(auth) =>
                    {
                        out["authorization_signature_valid"] =
                            json!(auth.verify_signature(k.as_bytes()));
                        if let (Some(cid), Some(method)) = (
                            args.get("chain_id").and_then(|x| x.as_u64()),
                            args.get("method").and_then(|x| x.as_str()),
                        )
                        {
                            let ctx = TxContext {
                                chain_id: cid,
                                method: method.to_string(),
                                to: args.get("to").and_then(|x| x.as_str()).map(str::to_string),
                                selector: args
                                    .get("selector")
                                    .and_then(|x| x.as_str())
                                    .map(str::to_string),
                                value_wei: su128(&args, "value_wei", 0),
                                tx_hash: args
                                    .get("tx_hash")
                                    .and_then(|x| x.as_str())
                                    .map(str::to_string),
                            };
                            // Trusted server clock — never a model-supplied time.
                            let now = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .map(|d| d.as_secs())
                                .unwrap_or(0);
                            // Stateless preview (already_spent = 0); the real spend/replay
                            // accounting lives in the host's SpendLedger.
                            match auth.evaluate(k.as_bytes(), &ctx, now, 0)
                            {
                                Ok(()) => out["would_authorize"] = json!(true),
                                Err(reason) =>
                                {
                                    out["would_authorize"] = json!(false);
                                    out["deny_reason"] = json!(reason);
                                },
                            }
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
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

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
        assert_eq!(out["sym_key_present"], json!(true));
        assert!(out.get("sym_key").is_none());
        assert!(
            !out.to_string()
                .contains("587d5484ce2a2a6ee3ba1962fdd7e8588e06200c46823bd18fbd67def96ad303")
        );
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
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("SCIRUST_EXCHANGE_SECRET");
        let t = tool("wallet_sign_exchange_request");
        let r = (t.handler)(json!({ "style": "coinbase" }));
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("SCIRUST_EXCHANGE_SECRET"));
    }

    #[test]
    fn exchange_signing_refused_with_empty_secret() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("SCIRUST_EXCHANGE_SECRET", "  ");
        let t = tool("wallet_sign_exchange_request");
        let r = (t.handler)(json!({ "style": "coinbase" }));
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("empty"));
        std::env::remove_var("SCIRUST_EXCHANGE_SECRET");
    }

    #[test]
    fn authorization_status_reports_disarmed() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("SCIRUST_WALLET_KEY");
        std::env::remove_var("SCIRUST_EXCHANGE_ALLOWED_PATHS");
        let t = tool("wallet_authorization_status");
        let out = (t.handler)(json!({})).unwrap();
        assert_eq!(out["signing_armed"], json!(false));
    }

    #[test]
    fn authorization_status_treats_empty_keys_as_disarmed() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("SCIRUST_WALLET_KEY", "");
        std::env::set_var("SCIRUST_EXCHANGE_SECRET", "   ");
        std::env::remove_var("SCIRUST_EXCHANGE_ALLOWED_PATHS");
        let out = (tool("wallet_authorization_status").handler)(json!({})).unwrap();
        assert_eq!(out["signing_armed"], json!(false));
        assert_eq!(out["exchange_signing_armed"], json!(false));
        std::env::remove_var("SCIRUST_WALLET_KEY");
        std::env::remove_var("SCIRUST_EXCHANGE_SECRET");
    }

    #[test]
    fn exchange_scope_blocks_dangerous_endpoints() {
        // Withdrawals / transfers / key management are always refused.
        assert!(
            exchange_scope_check(
                "POST",
                "/sapi/v1/capital/withdraw/apply",
                "",
                Some("POST /sapi/v1/capital/withdraw/apply")
            )
            .is_err()
        );
        assert!(
            exchange_scope_check(
                "POST",
                "/sapi/v1/asset/transfer",
                "",
                Some("POST /sapi/v1/asset/transfer")
            )
            .is_err()
        );
        assert!(
            exchange_scope_check(
                "POST",
                "/v2/withdrawals/crypto",
                "",
                Some("POST /v2/withdrawals/crypto")
            )
            .is_err()
        );
        assert!(
            exchange_scope_check(
                "POST",
                "/api/v3/order",
                "symbol=BTCUSDT&type=apiKey",
                Some("POST /api/v3/order")
            )
            .is_err()
        );
        assert!(
            exchange_scope_check(
                "POST",
                "/sapi/v1/capital/%2577ithdraw/apply",
                "",
                Some("POST /sapi/v1/capital/withdraw/apply")
            )
            .is_err()
        );
        // A normal trading request is allowed.
        assert!(
            exchange_scope_check(
                "POST",
                "/api/v3/order",
                "symbol=BTCUSDT&side=BUY&type=LIMIT&quantity=1",
                Some("POST /api/v3/order")
            )
            .is_ok()
        );
        // A secret without an explicit route allowlist is not armed.
        assert!(exchange_scope_check("POST", "/api/v3/order", "", None).is_err());
    }

    #[test]
    fn exchange_scope_respects_operator_allowlist() {
        // Only the normalized path, never query/body content, may satisfy the
        // operator allowlist.
        assert!(
            exchange_scope_check(
                "POST",
                "/api/v3/order",
                "symbol=BTCUSDT&side=BUY",
                Some("POST /api/v3/order")
            )
            .is_ok()
        );
        assert!(
            exchange_scope_check(
                "POST",
                "/api/v3/account",
                "redirect=/api/v3/order",
                Some("POST /api/v3/order")
            )
            .is_err()
        );
        assert!(
            exchange_scope_check("POST", "/api/v3/order-evil", "", Some("POST /api/v3/order"))
                .is_err()
        );
        assert!(
            exchange_scope_check("POST", "/api/v3/%6frder", "", Some("POST /api/v3/order"))
                .is_err()
        );
        assert!(
            exchange_scope_check("DELETE", "/api/v3/order", "", Some("POST /api/v3/order"))
                .is_err()
        );
        // The denylist still wins even if the allowlist would match.
        assert!(
            exchange_scope_check(
                "POST",
                "/sapi/v1/capital/withdraw/apply",
                "coin=BTC",
                Some("POST /sapi/v1/capital/withdraw/apply")
            )
            .is_err()
        );
    }

    #[test]
    fn endpoint_bound_coinbase_signing_matches_oracle() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("SCIRUST_EXCHANGE_SECRET", "c2VjcmV0");
        std::env::set_var("SCIRUST_EXCHANGE_ALLOWED_PATHS", "GET /orders");

        let output = (tool("wallet_sign_exchange_request").handler)(json!({
            "style": "coinbase",
            "timestamp": "1700000000",
            "method": "GET",
            "path": "/orders"
        }))
        .unwrap();
        assert_eq!(
            output["signature"],
            "5pq6uRxdFFvJzbzIJRObr9401649NDsVtb6H2TgxZRY="
        );
        assert_eq!(output["encoding"], "base64");

        std::env::remove_var("SCIRUST_EXCHANGE_SECRET");
        std::env::remove_var("SCIRUST_EXCHANGE_ALLOWED_PATHS");
    }

    #[test]
    fn exchange_signing_rejects_missing_and_unknown_styles() {
        let t = tool("wallet_sign_exchange_request");
        let missing = (t.handler)(json!({})).unwrap_err();
        assert!(missing.contains("missing `style`"));

        let t = tool("wallet_sign_exchange_request");
        let unknown = (t.handler)(json!({ "style": "kraken" })).unwrap_err();
        assert!(unknown.contains("unsupported exchange signing style"));

        let t = tool("wallet_sign_exchange_request");
        let binance = (t.handler)(json!({ "style": "binance" })).unwrap_err();
        assert!(binance.contains("do not bind the HTTP endpoint"));
    }
}
