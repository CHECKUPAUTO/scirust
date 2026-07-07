//! Wallet connectivity — recognized protocols, **watch-only / dry-run by
//! default**.
//!
//! This module gives an agent the standards-compliant plumbing to *connect to*
//! real crypto wallets — WalletConnect v2 pairing, EVM address handling
//! (EIP-55), EIP-1559 transaction construction and EIP-712 typed-data hashing,
//! and HMAC-signed exchange REST requests — without ever letting the LLM move
//! funds on its own.
//!
//! Safety model
//! ------------
//! * **Read/construct is free, sign/send is gated.** Address validation, tx
//!   construction, typed-data hashing and request *building* are pure and
//!   deterministic. Anything that actually **signs or broadcasts** requires a
//!   [`WalletAuthorization`] minted out-of-band by the operator with a
//!   server-side key the agent never sees — the same non-self-authorizing
//!   pattern as [`crate::market`]'s live gate and `scirust-discovery`.
//! * **No key material here.** This module never holds a private key or an
//!   exchange secret in the conversation; a real signer/secret is injected by
//!   the host process (env var) and only ever produces a signature, never
//!   reveals the key.
//! * **No network in the default build.** Balance reads over JSON-RPC / REST are
//!   behind the `live` feature.
//!
//! Everything below is pure Rust with no new dependencies: Keccak-256 and
//! HMAC-SHA256 are implemented here and checked against published test vectors.

use serde::{Deserialize, Serialize};

// ===========================================================================
// Keccak-256 (Ethereum's hash — NOT SHA3-256; the padding domain byte is 0x01).
// ===========================================================================

const KECCAK_RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];
const KECCAK_ROT: [u32; 24] = [
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
];
const KECCAK_PIL: [usize; 24] = [
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
];

fn keccak_f(st: &mut [u64; 25]) {
    for &rc in KECCAK_RC.iter()
    {
        // θ
        let mut bc = [0u64; 5];
        for i in 0..5
        {
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        }
        for i in 0..5
        {
            let t = bc[(i + 4) % 5] ^ bc[(i + 1) % 5].rotate_left(1);
            let mut j = 0;
            while j < 25
            {
                st[j + i] ^= t;
                j += 5;
            }
        }
        // ρ and π
        let mut t = st[1];
        for i in 0..24
        {
            let j = KECCAK_PIL[i];
            let tmp = st[j];
            st[j] = t.rotate_left(KECCAK_ROT[i]);
            t = tmp;
        }
        // χ
        let mut j = 0;
        while j < 25
        {
            let mut row = [0u64; 5];
            row.copy_from_slice(&st[j..j + 5]);
            for i in 0..5
            {
                st[j + i] ^= (!row[(i + 1) % 5]) & row[(i + 2) % 5];
            }
            j += 5;
        }
        // ι
        st[0] ^= rc;
    }
}

/// Keccak-256 (the hash Ethereum uses everywhere: addresses, tx hashes, EIP-712).
pub fn keccak256(input: &[u8]) -> [u8; 32] {
    const RATE: usize = 136; // 1088-bit rate, 512-bit capacity
    let mut st = [0u64; 25];
    let full_blocks = input.len() / RATE;
    for b in 0..full_blocks
    {
        absorb(&mut st, &input[b * RATE..b * RATE + RATE]);
        keccak_f(&mut st);
    }
    // Final block with pad10*1 using Keccak's 0x01 domain byte.
    let rem = &input[full_blocks * RATE..];
    let mut block = [0u8; RATE];
    block[..rem.len()].copy_from_slice(rem);
    block[rem.len()] ^= 0x01;
    block[RATE - 1] ^= 0x80;
    absorb(&mut st, &block);
    keccak_f(&mut st);
    // Squeeze 32 bytes.
    let mut out = [0u8; 32];
    for i in 0..4
    {
        out[i * 8..i * 8 + 8].copy_from_slice(&st[i].to_le_bytes());
    }
    out
}

fn absorb(st: &mut [u64; 25], block: &[u8]) {
    let lanes = block.len() / 8;
    for i in 0..lanes
    {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&block[i * 8..i * 8 + 8]);
        st[i] ^= u64::from_le_bytes(buf);
    }
}

// ===========================================================================
// HMAC-SHA256 (exchange REST request signing), built on the sha2 dependency.
// ===========================================================================

/// HMAC-SHA256 as specified in RFC 2104.
pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    const BLOCK: usize = 64;
    let mut k = [0u8; BLOCK];
    if key.len() > BLOCK
    {
        let h = Sha256::digest(key);
        k[..32].copy_from_slice(&h);
    }
    else
    {
        k[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; BLOCK];
    let mut opad = [0x5cu8; BLOCK];
    for i in 0..BLOCK
    {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }
    let inner = {
        let mut h = Sha256::new();
        h.update(ipad);
        h.update(msg);
        h.finalize()
    };
    let mut h = Sha256::new();
    h.update(opad);
    h.update(inner);
    let mut out = [0u8; 32];
    out.copy_from_slice(&h.finalize());
    out
}

// ===========================================================================
// Hex helpers.
// ===========================================================================

/// Lowercase hex (no `0x`).
pub fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes
    {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Decode a hex string (with or without `0x`). Returns `None` on bad input.
pub fn from_hex(s: &str) -> Option<Vec<u8>> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    if s.len() & 1 != 0
    {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len()
    {
        let hi = (bytes[i] as char).to_digit(16)?;
        let lo = (bytes[i + 1] as char).to_digit(16)?;
        out.push((hi * 16 + lo) as u8);
        i += 2;
    }
    Some(out)
}

// ===========================================================================
// Chains.
// ===========================================================================

/// A supported chain. EVM chains carry their EIP-155 chain id.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Chain {
    Ethereum,
    Polygon,
    Arbitrum,
    Optimism,
    Base,
    BinanceSmartChain,
    Avalanche,
    Solana,
    Bitcoin,
    /// Any other EVM chain by id.
    Evm(u64),
}

impl Chain {
    /// EIP-155 chain id (0 for non-EVM chains).
    pub fn chain_id(&self) -> u64 {
        match self
        {
            Chain::Ethereum => 1,
            Chain::Optimism => 10,
            Chain::BinanceSmartChain => 56,
            Chain::Polygon => 137,
            Chain::Base => 8453,
            Chain::Arbitrum => 42161,
            Chain::Avalanche => 43114,
            Chain::Evm(id) => *id,
            Chain::Solana | Chain::Bitcoin => 0,
        }
    }

    pub fn is_evm(&self) -> bool {
        !matches!(self, Chain::Solana | Chain::Bitcoin)
    }

    /// The CAIP-2 identifier used by WalletConnect namespaces (e.g. `eip155:1`).
    pub fn caip2(&self) -> String {
        match self
        {
            Chain::Solana => "solana:mainnet".to_string(),
            Chain::Bitcoin => "bip122:000000000019d6689c085ae165831e93".to_string(),
            evm => format!("eip155:{}", evm.chain_id()),
        }
    }

    pub fn name(&self) -> &'static str {
        match self
        {
            Chain::Ethereum => "ethereum",
            Chain::Polygon => "polygon",
            Chain::Arbitrum => "arbitrum",
            Chain::Optimism => "optimism",
            Chain::Base => "base",
            Chain::BinanceSmartChain => "bsc",
            Chain::Avalanche => "avalanche",
            Chain::Solana => "solana",
            Chain::Bitcoin => "bitcoin",
            Chain::Evm(_) => "evm",
        }
    }
}

// ===========================================================================
// EVM address (EIP-55 checksum).
// ===========================================================================

/// A 20-byte EVM address.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvmAddress(pub [u8; 20]);

impl EvmAddress {
    /// Parse from a 40-hex-char string (with or without `0x`).
    pub fn from_hex(s: &str) -> Option<Self> {
        let bytes = from_hex(s)?;
        if bytes.len() != 20
        {
            return None;
        }
        let mut a = [0u8; 20];
        a.copy_from_slice(&bytes);
        Some(EvmAddress(a))
    }

    /// EIP-55 mixed-case checksum representation (with `0x`).
    pub fn to_checksum(&self) -> String {
        let lower = to_hex(&self.0); // 40 lowercase hex chars
        let hash = keccak256(lower.as_bytes());
        let mut out = String::with_capacity(42);
        out.push_str("0x");
        for (i, ch) in lower.chars().enumerate()
        {
            // The i-th nibble of the hash decides the case.
            let nibble = if i & 1 == 0
            {
                hash[i / 2] >> 4
            }
            else
            {
                hash[i / 2] & 0x0f
            };
            if ch.is_ascii_alphabetic() && nibble >= 8
            {
                out.push(ch.to_ascii_uppercase());
            }
            else
            {
                out.push(ch);
            }
        }
        out
    }

    /// True if `s` is a correctly EIP-55-checksummed address. An all-lowercase
    /// or all-uppercase address is accepted (no checksum claimed); a mixed-case
    /// address must match exactly.
    pub fn is_valid_checksum(s: &str) -> bool {
        let addr = match EvmAddress::from_hex(s)
        {
            Some(a) => a,
            None => return false,
        };
        let body = s.strip_prefix("0x").unwrap_or(s);
        let has_upper = body.chars().any(|c| c.is_ascii_uppercase());
        let has_lower = body.chars().any(|c| c.is_ascii_lowercase());
        if !(has_upper && has_lower)
        {
            return true; // no case information to check
        }
        addr.to_checksum().strip_prefix("0x") == Some(body)
    }
}

// ===========================================================================
// EIP-1559 transaction construction + signing hash (dry-run; unsigned).
// ===========================================================================

/// An EIP-1559 (type-2) transaction. Values are in wei / gas units.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Eip1559Tx {
    pub chain_id: u64,
    pub nonce: u64,
    pub max_priority_fee_per_gas: u128,
    pub max_fee_per_gas: u128,
    pub gas_limit: u64,
    /// Recipient (20 bytes); `None` for a contract creation.
    pub to: Option<[u8; 20]>,
    pub value: u128,
    pub data: Vec<u8>,
}

impl Eip1559Tx {
    /// The keccak-256 signing hash: `keccak256(0x02 ‖ rlp([...]))` with an empty
    /// access list. This is exactly what a wallet signs — computing it here lets
    /// the agent show the user the precise digest **before** anything is signed.
    pub fn signing_hash(&self) -> [u8; 32] {
        let to_bytes: Vec<u8> = self.to.map(|a| a.to_vec()).unwrap_or_default();
        let fields = [
            rlp_bytes(&uint_bytes(self.chain_id as u128)),
            rlp_bytes(&uint_bytes(self.nonce as u128)),
            rlp_bytes(&uint_bytes(self.max_priority_fee_per_gas)),
            rlp_bytes(&uint_bytes(self.max_fee_per_gas)),
            rlp_bytes(&uint_bytes(self.gas_limit as u128)),
            rlp_bytes(&to_bytes),
            rlp_bytes(&uint_bytes(self.value)),
            rlp_bytes(&self.data),
            rlp_list(&[]), // empty access list
        ];
        let rlp = rlp_list(&fields);
        let mut msg = Vec::with_capacity(rlp.len() + 1);
        msg.push(0x02);
        msg.extend_from_slice(&rlp);
        keccak256(&msg)
    }
}

fn uint_bytes(v: u128) -> Vec<u8> {
    if v == 0
    {
        return Vec::new();
    }
    let b = v.to_be_bytes();
    let start = b.iter().position(|&x| x != 0).unwrap_or(b.len());
    b[start..].to_vec()
}

fn rlp_len_prefix(base: u8, len: usize) -> Vec<u8> {
    if len < 56
    {
        vec![base + len as u8]
    }
    else
    {
        let lb = len.to_be_bytes();
        let start = lb.iter().position(|&x| x != 0).unwrap_or(lb.len());
        let lb = &lb[start..];
        let mut out = vec![base + 55 + lb.len() as u8];
        out.extend_from_slice(lb);
        out
    }
}

fn rlp_bytes(b: &[u8]) -> Vec<u8> {
    if b.len() == 1 && b[0] < 0x80
    {
        return vec![b[0]];
    }
    let mut out = rlp_len_prefix(0x80, b.len());
    out.extend_from_slice(b);
    out
}

fn rlp_list(items: &[Vec<u8>]) -> Vec<u8> {
    let mut payload = Vec::new();
    for it in items
    {
        payload.extend_from_slice(it);
    }
    let mut out = rlp_len_prefix(0xc0, payload.len());
    out.extend_from_slice(&payload);
    out
}

// ===========================================================================
// EIP-712 typed-data hashing.
// ===========================================================================

/// An EIP-712 domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Eip712Domain {
    pub name: String,
    pub version: String,
    pub chain_id: u64,
    /// 20-byte verifying contract (defaults to zero if absent).
    pub verifying_contract: Option<[u8; 20]>,
}

impl Eip712Domain {
    /// `keccak256(abi.encode(typeHash, keccak(name), keccak(version), chainId,
    /// verifyingContract))` — the EIP-712 domain separator.
    pub fn separator(&self) -> [u8; 32] {
        let type_hash = keccak256(
            b"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)",
        );
        let mut enc = Vec::with_capacity(32 * 5);
        enc.extend_from_slice(&type_hash);
        enc.extend_from_slice(&keccak256(self.name.as_bytes()));
        enc.extend_from_slice(&keccak256(self.version.as_bytes()));
        enc.extend_from_slice(&word_u64(self.chain_id));
        enc.extend_from_slice(&word_address(self.verifying_contract.unwrap_or([0u8; 20])));
        keccak256(&enc)
    }

    /// The final EIP-712 digest a wallet signs:
    /// `keccak256(0x19 0x01 ‖ domainSeparator ‖ hashStruct(message))`.
    /// The caller supplies the already-computed `struct_hash` of the message.
    pub fn digest(&self, struct_hash: &[u8; 32]) -> [u8; 32] {
        let sep = self.separator();
        let mut msg = Vec::with_capacity(66);
        msg.push(0x19);
        msg.push(0x01);
        msg.extend_from_slice(&sep);
        msg.extend_from_slice(struct_hash);
        keccak256(&msg)
    }
}

/// Left-pad a u64 into a 32-byte EVM word.
fn word_u64(v: u64) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[24..].copy_from_slice(&v.to_be_bytes());
    w
}

/// Left-pad a 20-byte address into a 32-byte EVM word.
fn word_address(a: [u8; 20]) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[12..].copy_from_slice(&a);
    w
}

// ===========================================================================
// WalletConnect v2 pairing URI + session model.
// ===========================================================================

/// A parsed WalletConnect v2 pairing URI
/// (`wc:{topic}@2?relay-protocol=irn&symKey={hex}`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WalletConnectUri {
    pub topic: String,
    pub version: u8,
    pub relay_protocol: String,
    pub sym_key: String,
    pub expiry_timestamp: Option<u64>,
}

/// Parse a WalletConnect pairing URI. Supports v2 (the current standard).
pub fn parse_walletconnect_uri(uri: &str) -> Result<WalletConnectUri, String> {
    let body = uri
        .strip_prefix("wc:")
        .ok_or("not a WalletConnect URI (missing `wc:`)")?;
    let (topic_version, query) = body.split_once('?').ok_or("missing query parameters")?;
    let (topic, version_str) = topic_version
        .split_once('@')
        .ok_or("missing `@version` in the URI")?;
    if topic.is_empty()
    {
        return Err("empty topic".to_string());
    }
    let version: u8 = version_str
        .parse()
        .map_err(|_| "invalid version".to_string())?;
    let mut relay_protocol = String::new();
    let mut sym_key = String::new();
    let mut expiry_timestamp = None;
    for kv in query.split('&')
    {
        let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
        match k
        {
            "relay-protocol" => relay_protocol = v.to_string(),
            "symKey" => sym_key = v.to_string(),
            "expiryTimestamp" => expiry_timestamp = v.parse().ok(),
            _ =>
            {},
        }
    }
    if version == 2 && sym_key.is_empty()
    {
        return Err("WalletConnect v2 URI missing symKey".to_string());
    }
    Ok(WalletConnectUri {
        topic: topic.to_string(),
        version,
        relay_protocol,
        sym_key,
        expiry_timestamp,
    })
}

/// One namespace in a WalletConnect session (e.g. the `eip155` namespace).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WcNamespace {
    pub chains: Vec<String>,
    pub methods: Vec<String>,
    pub events: Vec<String>,
    pub accounts: Vec<String>,
}

/// Build a standard `eip155` namespace request for the given chains — the
/// `requiredNamespaces` an agent proposes when initiating a session.
pub fn eip155_namespace(chains: &[Chain]) -> WcNamespace {
    WcNamespace {
        chains: chains
            .iter()
            .filter(|c| c.is_evm())
            .map(|c| c.caip2())
            .collect(),
        methods: vec![
            "eth_sendTransaction".to_string(),
            "eth_signTransaction".to_string(),
            "personal_sign".to_string(),
            "eth_signTypedData_v4".to_string(),
        ],
        events: vec!["chainChanged".to_string(), "accountsChanged".to_string()],
        accounts: Vec::new(),
    }
}

// ===========================================================================
// Exchange REST request signing (server-side secret; the agent only builds the
// payload — the secret is injected by the host and never revealed).
// ===========================================================================

/// Sign a Binance-style query string: `hex(HMAC_SHA256(secret, query))`.
/// The signature is appended as `&signature=…` to the request.
pub fn sign_binance_query(secret: &[u8], query: &str) -> String {
    to_hex(&hmac_sha256(secret, query.as_bytes()))
}

/// Sign a Coinbase-style prehash `timestamp+method+path+body` and return the
/// hex HMAC (Coinbase Advanced uses hex for its `CB-ACCESS-SIGN` header).
pub fn sign_coinbase_request(
    secret: &[u8],
    timestamp: &str,
    method: &str,
    path: &str,
    body: &str,
) -> String {
    let prehash = format!("{timestamp}{method}{path}{body}");
    to_hex(&hmac_sha256(secret, prehash.as_bytes()))
}

// ===========================================================================
// Authorization gate — the LLM cannot self-authorize signing/sending.
//
// Hardened model: an authorization is bound to the *transaction context* — the
// recipient, the calldata selector, and the value — not merely a native-value
// cap, and is enforced against a stateful [`SpendLedger`] for one-time use and a
// cumulative budget. This closes the ways a native-value-only cap could be
// side-stepped (an ERC-20 `transfer` moves tokens with `value == 0`) and the
// ways a standing authorization could be replayed.
// ===========================================================================

/// The concrete action being authorized — what would actually be signed/sent.
/// Supplied by the host from the real transaction, never trusted from the model.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TxContext {
    pub chain_id: u64,
    pub method: String,
    /// Recipient address (hex, with or without `0x`); `None` for methods that
    /// carry no recipient (e.g. `personal_sign`).
    pub to: Option<String>,
    /// The 4-byte calldata selector (hex), or `None` when the calldata is empty
    /// (a pure native-value transfer).
    pub selector: Option<String>,
    /// Native value moved (wei).
    pub value_wei: u128,
    /// The exact signing hash of the transaction (hex), when known.
    pub tx_hash: Option<String>,
}

/// Normalize a hex string for comparison: strip `0x`, lowercase.
fn norm_hex(s: &str) -> String {
    s.strip_prefix("0x").unwrap_or(s).to_ascii_lowercase()
}

/// Length-prefixed field append — an unambiguous canonical encoding. Unlike a
/// delimiter-joined string, no field value can be crafted to make two distinct
/// authorizations hash to the same signed bytes.
fn push_bytes(buf: &mut Vec<u8>, b: &[u8]) {
    buf.extend_from_slice(&(b.len() as u64).to_be_bytes());
    buf.extend_from_slice(b);
}
fn push_str_list(buf: &mut Vec<u8>, items: &[String]) {
    buf.extend_from_slice(&(items.len() as u64).to_be_bytes());
    for it in items
    {
        push_bytes(buf, it.as_bytes());
    }
}
fn push_u64_list(buf: &mut Vec<u8>, items: &[u64]) {
    buf.extend_from_slice(&(items.len() as u64).to_be_bytes());
    for it in items
    {
        buf.extend_from_slice(&it.to_be_bytes());
    }
}

/// An out-of-band authorization that permits a *bounded* set of signing/sending
/// actions. Minted by the operator with a server-side key (never in the
/// conversation) and verified here. Without a valid authorization every
/// state-changing wallet action is refused.
///
/// Two modes:
/// * **Bound** (`bound_tx_hash = Some(..)`) — authorizes exactly one transaction
///   whose signing hash matches; one-time-use via the [`SpendLedger`]. The
///   strongest, per-transaction form of approval.
/// * **Policy** (`bound_tx_hash = None`) — a scoped standing authorization,
///   constrained by a recipient allowlist, a calldata-selector allowlist, a
///   per-transaction native cap and a cumulative budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletAuthorization {
    pub operator: String,
    /// Unique id (jti) — the key under which the ledger tracks use and spend.
    pub nonce: String,
    /// EIP-155 chain ids the authorization covers (empty = none).
    pub allowed_chain_ids: Vec<u64>,
    /// RPC/signing methods permitted (e.g. `eth_sendTransaction`, `personal_sign`).
    pub allowed_methods: Vec<String>,
    /// Recipient allowlist (hex). In policy mode a transaction whose `to` is not
    /// listed is refused; an empty list refuses every recipient-bearing tx.
    pub allowed_to: Vec<String>,
    /// Calldata 4-byte selector allowlist (hex). A transaction carrying calldata
    /// whose selector is not listed is refused; an **empty** list means calldata
    /// must be empty (native transfers only) — this is what stops an ERC-20
    /// `transfer`/`approve` from side-stepping the native-value cap.
    pub allowed_selectors: Vec<String>,
    /// Per-transaction native-value cap (wei).
    pub max_value_wei: u128,
    /// Cumulative native-value budget (wei) across every use of this nonce.
    pub cumulative_budget_wei: u128,
    /// If set, the authorization is bound to exactly one transaction whose
    /// signing hash matches — one-time-use.
    pub bound_tx_hash: Option<String>,
    pub valid_from_unix: u64,
    pub valid_until_unix: u64,
    /// Hex HMAC over the canonical (length-prefixed) encoding of every field
    /// above.
    pub signature_hex: String,
}

impl WalletAuthorization {
    /// Canonical bytes signed by the operator (all fields except the signature),
    /// length-prefixed so the encoding is injective.
    fn canonical(&self) -> Vec<u8> {
        let mut b = Vec::new();
        push_bytes(&mut b, b"SCIRUST-WALLET-AUTH-v2");
        push_bytes(&mut b, self.operator.as_bytes());
        push_bytes(&mut b, self.nonce.as_bytes());
        push_u64_list(&mut b, &self.allowed_chain_ids);
        push_str_list(&mut b, &self.allowed_methods);
        push_str_list(&mut b, &self.allowed_to);
        push_str_list(&mut b, &self.allowed_selectors);
        push_bytes(&mut b, &self.max_value_wei.to_be_bytes());
        push_bytes(&mut b, &self.cumulative_budget_wei.to_be_bytes());
        push_bytes(
            &mut b,
            self.bound_tx_hash.as_deref().unwrap_or("").as_bytes(),
        );
        push_bytes(&mut b, &self.valid_from_unix.to_be_bytes());
        push_bytes(&mut b, &self.valid_until_unix.to_be_bytes());
        b
    }

    /// Sign this authorization with the operator's key (HMAC-SHA256).
    pub fn sign(mut self, key: &[u8]) -> Self {
        self.signature_hex = to_hex(&hmac_sha256(key, &self.canonical()));
        self
    }

    /// Verify the signature against the operator key (constant-time comparison).
    pub fn verify_signature(&self, key: &[u8]) -> bool {
        let expected = to_hex(&hmac_sha256(key, &self.canonical()));
        // Length-checked byte comparison; both are 64-char hex.
        expected.len() == self.signature_hex.len()
            && expected
                .bytes()
                .zip(self.signature_hex.bytes())
                .fold(0u8, |acc, (a, b)| acc | (a ^ b))
                == 0
    }

    /// Stateless policy decision for `ctx` at `now_unix`, given how much native
    /// value has *already* been spent under this authorization's nonce.
    /// `Ok(())` ⇒ permitted; `Err(reason)` ⇒ refused. Stateful one-time and
    /// cumulative enforcement lives in [`SpendLedger`]; call that in production
    /// so the `already_spent_wei` and replay accounting cannot be skipped.
    pub fn evaluate(
        &self,
        key: &[u8],
        ctx: &TxContext,
        now_unix: u64,
        already_spent_wei: u128,
    ) -> Result<(), String> {
        if !self.verify_signature(key)
        {
            return Err("authorization signature invalid".to_string());
        }
        if now_unix < self.valid_from_unix || now_unix > self.valid_until_unix
        {
            return Err("outside the authorization validity window".to_string());
        }
        if !self.allowed_chain_ids.contains(&ctx.chain_id)
        {
            return Err(format!("chain {} not authorized", ctx.chain_id));
        }
        if !self.allowed_methods.iter().any(|m| m == &ctx.method)
        {
            return Err(format!("method `{}` not authorized", ctx.method));
        }
        match &self.bound_tx_hash
        {
            Some(h) =>
            {
                // Bound mode: the hash commits to the exact transaction, so the
                // recipient/calldata/value are all fixed by it.
                let want = norm_hex(h);
                let got = ctx.tx_hash.as_deref().map(norm_hex).unwrap_or_default();
                if want != got
                {
                    return Err(
                        "transaction hash does not match the bound authorization".to_string()
                    );
                }
            },
            None =>
            {
                // Policy mode: constrain recipient and calldata explicitly.
                if let Some(to) = &ctx.to
                {
                    let to = norm_hex(to);
                    if !self.allowed_to.iter().any(|a| norm_hex(a) == to)
                    {
                        return Err("recipient not in the authorization allowlist".to_string());
                    }
                }
                if let Some(sel) = &ctx.selector
                {
                    let sel = norm_hex(sel);
                    if !self.allowed_selectors.iter().any(|s| norm_hex(s) == sel)
                    {
                        return Err(
                            "calldata selector not in the authorization allowlist".to_string()
                        );
                    }
                }
            },
        }
        if ctx.value_wei > self.max_value_wei
        {
            return Err("exceeds the per-transaction value cap".to_string());
        }
        if already_spent_wei.saturating_add(ctx.value_wei) > self.cumulative_budget_wei
        {
            return Err("exceeds the cumulative authorization budget".to_string());
        }
        Ok(())
    }
}

/// Stateful ledger enforcing one-time use of bound authorizations and the
/// cumulative spend budget across repeated uses. The host owns one and consults
/// it for every state-changing action; without recording spend, a standing
/// authorization could be replayed up to its per-transaction cap indefinitely.
#[derive(Debug, Default)]
pub struct SpendLedger {
    consumed_nonces: std::collections::BTreeSet<String>,
    spent: std::collections::BTreeMap<String, u128>,
}

impl SpendLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Native value already recorded against `nonce`.
    pub fn spent_for(&self, nonce: &str) -> u128 {
        *self.spent.get(nonce).unwrap_or(&0)
    }

    /// Authorize `ctx` under `auth`; on success record the spend (and consume
    /// the nonce if the authorization is one-time/bound). On `Err` the ledger is
    /// left unchanged.
    pub fn try_consume(
        &mut self,
        auth: &WalletAuthorization,
        key: &[u8],
        ctx: &TxContext,
        now_unix: u64,
    ) -> Result<(), String> {
        if auth.bound_tx_hash.is_some() && self.consumed_nonces.contains(&auth.nonce)
        {
            return Err("authorization already used (one-time)".to_string());
        }
        let already = self.spent_for(&auth.nonce);
        auth.evaluate(key, ctx, now_unix, already)?;
        *self.spent.entry(auth.nonce.clone()).or_insert(0) += ctx.value_wei;
        if auth.bound_tx_hash.is_some()
        {
            self.consumed_nonces.insert(auth.nonce.clone());
        }
        Ok(())
    }
}

// ===========================================================================
// Connector abstraction.
// ===========================================================================

/// How the agent is connected to a wallet.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConnectionMode {
    /// Read balances/positions only; cannot sign.
    WatchOnly,
    /// Connected via a WalletConnect session (signing goes to the remote wallet).
    WalletConnect,
    /// Connected to an exchange via signed REST (custodial).
    ExchangeApi,
}

/// A wallet connection an agent can query. Signing capability is reported via
/// [`WalletConnector::can_sign`], which is `false` for watch-only.
pub trait WalletConnector {
    fn chain(&self) -> Chain;
    fn mode(&self) -> ConnectionMode;
    /// The connected address, if known.
    fn address(&self) -> Option<String>;
    /// Whether this connector can initiate signing (still subject to a
    /// [`WalletAuthorization`] for the actual action).
    fn can_sign(&self) -> bool {
        self.mode() != ConnectionMode::WatchOnly
    }
}

/// A read-only address watcher — the safe default connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchOnlyWallet {
    pub chain: Chain,
    pub address: String,
}

impl WatchOnlyWallet {
    pub fn new(chain: Chain, address: &str) -> Self {
        Self {
            chain,
            address: address.to_string(),
        }
    }
}

impl WalletConnector for WatchOnlyWallet {
    fn chain(&self) -> Chain {
        self.chain
    }
    fn mode(&self) -> ConnectionMode {
        ConnectionMode::WatchOnly
    }
    fn address(&self) -> Option<String> {
        Some(self.address.clone())
    }
}

/// Fetch the native-token balance (wei) of an EVM address over JSON-RPC.
/// Networking is opt-in behind the `live` feature; the default build returns an
/// explanatory error instead of making a request.
#[cfg(not(feature = "live"))]
pub fn evm_native_balance(_rpc_url: &str, _address: &str) -> Result<u128, String> {
    Err(
        "live feature disabled: build scirust-trader with --features live for JSON-RPC reads"
            .to_string(),
    )
}

/// Fetch the native-token balance (wei) of an EVM address via `eth_getBalance`.
#[cfg(feature = "live")]
pub fn evm_native_balance(rpc_url: &str, address: &str) -> Result<u128, String> {
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_getBalance",
        "params": [address, "latest"],
    });
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| format!("client: {e}"))?;
    let resp: serde_json::Value = client
        .post(rpc_url)
        .json(&body)
        .send()
        .map_err(|e| format!("request: {e}"))?
        .json()
        .map_err(|e| format!("decode: {e}"))?;
    let hex = resp
        .get("result")
        .and_then(|v| v.as_str())
        .ok_or("no result in JSON-RPC response")?;
    let bytes = from_hex(hex).ok_or("bad hex balance")?;
    let mut v = 0u128;
    for b in bytes
    {
        v = (v << 8) | b as u128;
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keccak256_known_vectors() {
        // keccak256("") and keccak256("abc") — the canonical Ethereum vectors.
        assert_eq!(
            to_hex(&keccak256(b"")),
            "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
        );
        assert_eq!(
            to_hex(&keccak256(b"abc")),
            "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"
        );
    }

    #[test]
    fn keccak256_long_input_multiblock() {
        // > 136 bytes exercises the multi-block absorb path.
        let input = vec![0xa5u8; 200];
        let h = keccak256(&input);
        assert_eq!(h.len(), 32);
        // Deterministic.
        assert_eq!(h, keccak256(&input));
    }

    #[test]
    fn hmac_sha256_rfc4231_case1() {
        // RFC 4231 test case 1: key = 0x0b*20, data = "Hi There".
        let key = [0x0bu8; 20];
        let mac = hmac_sha256(&key, b"Hi There");
        assert_eq!(
            to_hex(&mac),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn eip55_checksum_spec_examples() {
        // The four canonical EIP-55 examples.
        for a in [
            "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed",
            "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359",
            "0xdbF03B407c01E7cD3CBea99509d93f8DDDC8C6FB",
            "0xD1220A0cf47c7B9Be7A2E6BA89F429762e7b9aDb",
        ]
        {
            let addr = EvmAddress::from_hex(a).unwrap();
            assert_eq!(addr.to_checksum(), a, "checksum mismatch for {a}");
            assert!(EvmAddress::is_valid_checksum(a));
        }
    }

    #[test]
    fn eip55_rejects_bad_checksum() {
        // Flip the case of one letter -> invalid checksum.
        assert!(!EvmAddress::is_valid_checksum(
            "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAeD"
        ));
        // All-lowercase is accepted (no checksum claimed).
        assert!(EvmAddress::is_valid_checksum(
            "0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed"
        ));
        // Wrong length.
        assert!(!EvmAddress::is_valid_checksum("0x1234"));
    }

    #[test]
    fn walletconnect_v2_uri_parses() {
        let uri = "wc:7f6e504bfad60b485450578e05678ed3e8e8c4751d3c6160be17160d63ec90f9@2?relay-protocol=irn&symKey=587d5484ce2a2a6ee3ba1962fdd7e8588e06200c46823bd18fbd67def96ad303";
        let p = parse_walletconnect_uri(uri).unwrap();
        assert_eq!(p.version, 2);
        assert_eq!(p.relay_protocol, "irn");
        assert_eq!(p.topic.len(), 64);
        assert_eq!(p.sym_key.len(), 64);
    }

    #[test]
    fn walletconnect_uri_rejects_garbage() {
        assert!(parse_walletconnect_uri("https://example.com").is_err());
        assert!(parse_walletconnect_uri("wc:topic@2").is_err()); // no query
        assert!(parse_walletconnect_uri("wc:abc@2?relay-protocol=irn").is_err()); // no symKey
    }

    #[test]
    fn eip1559_signing_hash_is_deterministic_32_bytes() {
        let tx = Eip1559Tx {
            chain_id: 1,
            nonce: 9,
            max_priority_fee_per_gas: 2_000_000_000,
            max_fee_per_gas: 30_000_000_000,
            gas_limit: 21_000,
            to: EvmAddress::from_hex("0x3535353535353535353535353535353535353535").map(|a| a.0),
            value: 1_000_000_000_000_000_000, // 1 ETH
            data: Vec::new(),
        };
        let h = tx.signing_hash();
        assert_eq!(h.len(), 32);
        assert_eq!(h, tx.signing_hash());
        // A different nonce yields a different digest.
        let mut tx2 = tx.clone();
        tx2.nonce = 10;
        assert_ne!(tx.signing_hash(), tx2.signing_hash());
    }

    #[test]
    fn eip712_domain_separator_deterministic() {
        let d = Eip712Domain {
            name: "Uniswap V2".to_string(),
            version: "1".to_string(),
            chain_id: 1,
            verifying_contract: EvmAddress::from_hex("0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed")
                .map(|a| a.0),
        };
        let sep = d.separator();
        assert_eq!(sep.len(), 32);
        assert_eq!(sep, d.separator());
        // Different chain id -> different separator.
        let mut d2 = d.clone();
        d2.chain_id = 137;
        assert_ne!(d.separator(), d2.separator());
        // Digest combines the struct hash.
        let sh = keccak256(b"message");
        assert_eq!(d.digest(&sh).len(), 32);
    }

    #[test]
    fn binance_query_signing_matches_hmac() {
        let sig = sign_binance_query(
            b"NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j",
            "symbol=LTCBTC&side=BUY&type=LIMIT",
        );
        assert_eq!(sig.len(), 64); // hex of 32-byte HMAC
    }

    const ADDR_A: &str = "0x3535353535353535353535353535353535353535";
    const ADDR_B: &str = "0x4646464646464646464646464646464646464646";
    const ONE_ETH: u128 = 1_000_000_000_000_000_000;

    fn policy_auth(key: &[u8]) -> WalletAuthorization {
        WalletAuthorization {
            operator: "alice".to_string(),
            nonce: "auth-1".to_string(),
            allowed_chain_ids: vec![1],
            allowed_methods: vec!["eth_sendTransaction".to_string()],
            allowed_to: vec![ADDR_A.to_string()],
            allowed_selectors: vec![], // native transfers only
            max_value_wei: ONE_ETH,
            cumulative_budget_wei: ONE_ETH + ONE_ETH / 2, // 1.5 ETH
            bound_tx_hash: None,
            valid_from_unix: 0,
            valid_until_unix: 4_000_000_000,
            signature_hex: String::new(),
        }
        .sign(key)
    }

    fn native_ctx(to: &str, value: u128) -> TxContext {
        TxContext {
            chain_id: 1,
            method: "eth_sendTransaction".to_string(),
            to: Some(to.to_string()),
            selector: None,
            value_wei: value,
            tx_hash: None,
        }
    }

    #[test]
    fn policy_authorization_enforces_all_bounds() {
        let key = b"operator-server-side-key";
        let auth = policy_auth(key);
        // Allowed: native transfer to allowlisted recipient under the cap.
        assert!(
            auth.evaluate(key, &native_ctx(ADDR_A, ONE_ETH / 2), 1_000_000, 0)
                .is_ok()
        );
        // Recipient not in allowlist.
        assert!(
            auth.evaluate(key, &native_ctx(ADDR_B, 1), 1_000_000, 0)
                .is_err()
        );
        // Wrong chain.
        let mut c = native_ctx(ADDR_A, 1);
        c.chain_id = 137;
        assert!(auth.evaluate(key, &c, 1_000_000, 0).is_err());
        // Method not allowed.
        let mut c = native_ctx(ADDR_A, 1);
        c.method = "personal_sign".to_string();
        assert!(auth.evaluate(key, &c, 1_000_000, 0).is_err());
        // Over the per-tx cap.
        assert!(
            auth.evaluate(key, &native_ctx(ADDR_A, 2 * ONE_ETH), 1_000_000, 0)
                .is_err()
        );
        // Expired.
        assert!(
            auth.evaluate(key, &native_ctx(ADDR_A, 1), 5_000_000_000, 0)
                .is_err()
        );
        // Wrong key.
        assert!(
            auth.evaluate(b"attacker-guess", &native_ctx(ADDR_A, 1), 1_000_000, 0)
                .is_err()
        );
    }

    #[test]
    fn erc20_transfer_cannot_bypass_native_cap() {
        // F1 regression: a value=0 tx carrying an ERC-20 transfer selector must
        // be refused when calldata is not on the allowlist (empty ⇒ native only).
        let key = b"k";
        let auth = policy_auth(key);
        let ctx = TxContext {
            chain_id: 1,
            method: "eth_sendTransaction".to_string(),
            to: Some(ADDR_A.to_string()), // token contract, on the allowlist even
            selector: Some("a9059cbb".to_string()), // transfer(address,uint256)
            value_wei: 0,                 // sneaks under a native-value cap
            tx_hash: None,
        };
        let err = auth.evaluate(key, &ctx, 1_000_000, 0).unwrap_err();
        assert!(err.contains("selector"), "unexpected: {err}");
    }

    #[test]
    fn selector_allowlist_permits_listed_call() {
        let key = b"k";
        let mut auth = policy_auth(key);
        auth.allowed_selectors = vec!["a9059cbb".to_string()];
        let auth = auth.sign(key);
        let ctx = TxContext {
            chain_id: 1,
            method: "eth_sendTransaction".to_string(),
            to: Some(ADDR_A.to_string()),
            selector: Some("0xA9059CBB".to_string()), // case/0x-insensitive
            value_wei: 0,
            tx_hash: None,
        };
        assert!(auth.evaluate(key, &ctx, 1_000_000, 0).is_ok());
    }

    #[test]
    fn cumulative_budget_enforced_by_ledger() {
        // F3: per-tx cap 1 ETH, cumulative budget 1.5 ETH ⇒ second full-cap
        // transfer is refused even though each is individually under the cap.
        let key = b"k";
        let auth = policy_auth(key);
        let mut ledger = SpendLedger::new();
        assert!(
            ledger
                .try_consume(&auth, key, &native_ctx(ADDR_A, ONE_ETH), 1_000)
                .is_ok()
        );
        let err = ledger
            .try_consume(&auth, key, &native_ctx(ADDR_A, ONE_ETH), 1_000)
            .unwrap_err();
        assert!(err.contains("cumulative"), "unexpected: {err}");
        assert_eq!(ledger.spent_for("auth-1"), ONE_ETH);
    }

    #[test]
    fn bound_authorization_is_one_time() {
        // F2: an authorization bound to a specific signing hash authorizes only
        // that transaction, and only once.
        let key = b"k";
        let auth = WalletAuthorization {
            operator: "alice".to_string(),
            nonce: "bound-1".to_string(),
            allowed_chain_ids: vec![1],
            allowed_methods: vec!["eth_sendTransaction".to_string()],
            allowed_to: vec![],
            allowed_selectors: vec![],
            max_value_wei: ONE_ETH,
            cumulative_budget_wei: ONE_ETH,
            bound_tx_hash: Some("0xdeadbeef".to_string()),
            valid_from_unix: 0,
            valid_until_unix: 4_000_000_000,
            signature_hex: String::new(),
        }
        .sign(key);
        let mut ledger = SpendLedger::new();
        let mut ctx = native_ctx(ADDR_B, ONE_ETH / 2);
        ctx.tx_hash = Some("0xDEADBEEF".to_string()); // matches (case-insensitive)
        // A different hash is refused.
        let mut wrong = ctx.clone();
        wrong.tx_hash = Some("0xfeed".to_string());
        assert!(ledger.try_consume(&auth, key, &wrong, 1_000).is_err());
        // The bound tx is authorized once...
        assert!(ledger.try_consume(&auth, key, &ctx, 1_000).is_ok());
        // ...and refused on replay.
        let err = ledger.try_consume(&auth, key, &ctx, 1_000).unwrap_err();
        assert!(err.contains("one-time"), "unexpected: {err}");
    }

    #[test]
    fn tampering_authorization_breaks_signature() {
        let key = b"k";
        let mut auth = policy_auth(key);
        assert!(auth.verify_signature(key));
        auth.max_value_wei = 999; // tamper after signing
        assert!(!auth.verify_signature(key));
    }

    #[test]
    fn canonical_encoding_is_unambiguous() {
        // F6: length-prefixed fields ⇒ shifting a character across a field
        // boundary changes the signature (a `|`-joined string would collide).
        let key = b"k";
        let a = WalletAuthorization {
            operator: "ab".to_string(),
            nonce: "c".to_string(),
            ..policy_auth(key)
        }
        .sign(key);
        let b = WalletAuthorization {
            operator: "a".to_string(),
            nonce: "bc".to_string(),
            ..policy_auth(key)
        }
        .sign(key);
        assert_ne!(a.signature_hex, b.signature_hex);
        // And each still verifies against itself.
        assert!(a.verify_signature(key) && b.verify_signature(key));
    }

    #[test]
    fn watch_only_cannot_sign() {
        let w = WatchOnlyWallet::new(
            Chain::Ethereum,
            "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed",
        );
        assert!(!w.can_sign());
        assert_eq!(w.mode(), ConnectionMode::WatchOnly);
        assert_eq!(w.chain().chain_id(), 1);
    }

    #[test]
    fn chain_caip2_ids() {
        assert_eq!(Chain::Ethereum.caip2(), "eip155:1");
        assert_eq!(Chain::Polygon.caip2(), "eip155:137");
        assert_eq!(Chain::Solana.caip2(), "solana:mainnet");
        assert!(!Chain::Solana.is_evm());
    }
}
