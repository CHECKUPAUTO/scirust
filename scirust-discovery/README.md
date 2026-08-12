# scirust-discovery

**Safe, consented and audited** OT/IT asset discovery: answers the question
"what industrial hardware is actually present on this network" for an agent
(the `scirust-sciagent` SLM, another agent connected via `scirust-mcp`, or a
human operator), without ever becoming a generic port scanner.

## Why not a simple port scan

A generic scan (Nmap-style) is **documented as dangerous** on industrial
controllers:

- Coffey et al. (*Security and Communication Networks*, 2018) document an Nmap
  scan that took a widely deployed PLC into failure, requiring a full power
  cycle — not reproducible by the manufacturer.
- The SQL Slammer incident at Davis-Besse nuclear plant (January 2003):
  "scan"-type UDP traffic (not a targeted attack) crossing an unfirewalled
  corporate→SCADA link disabled the display of safety parameters for nearly
  five hours.
- **NIST SP 800-82** (Guide to OT Security) codifies the resulting doctrine:
  prefer passive monitoring; reserve active probing for a maintenance window
  explicitly authorized by the operator.

This crate therefore takes a **protocol-native** approach: each probe only sends
what a legitimate client of that protocol would send to announce itself or
establish a connection — never an arbitrary packet to an arbitrary port.

## The zones-and-conduits model (ISA/IEC 62443)

`ScopeAuthorization` (`src/scope.rs`) encodes authorization as a **verifiable
datum**, not a convention:

- a whitelist of IPv4 **CIDR** ranges,
- a whitelist of **protocols** (`opcua`, `modbus`, `mdns`),
- a **temporal validity window** (`valid_from_unix`/`valid_until_unix`),
- a **zone** label and its **IEC 62443 security level** (SL0–SL4) — any SL3+
  zone is refused by default, an override must be explicit
  (`allow_high_security_zone: true`),
- an **HMAC-SHA256 signature** (key pre-shared between the operator who
  authorizes and the agent who executes — not a full PKI, see `src/hmac.rs`):
  an unsigned, expired, or post-signature widened scope is rejected before any
  packet is sent.

`DiscoveryEngine::probe_one` (`src/engine.rs`) is the single entry point: it
calls `ScopeAuthorization::authorize` before any network I/O, and logs the
attempt — within scope or refused — in a SHA-256 hash-chained log
(`src/audit.rs`), on the same principle as `scirust-func-safety::audit`.

## Supported protocols

| Protocol | Mechanism | Reference |
|---|---|---|
| **OPC-UA** | UACP `Hello`/`Acknowledge` handshake — the first thing any OPC-UA client exchanges, even before opening a secure channel | OPC UA Part 6 §7.1 |
| **Modbus TCP** | `Read Device Identification` (function code 0x2B, MEI 0x0E) — read-only, provided by the protocol for a device's self-description | Modbus Application Protocol V1.1b3 §6.21 |
| **mDNS/DNS-SD** | Standard DNS service enumeration query (`_services._dns-sd._udp.local`) | RFC 1035, RFC 6762/6763 |
| **BACnet/IP** | Global `Who-Is` broadcast + `I-Am` decoding (device identifier only) | ANSI/ASHRAE 135, Annex J (BVLL), clauses 16.9/16.10 |
| **SNMP v1** | `GET sysDescr.0` (`1.3.6.1.2.1.1.1.0`) — minimal BER encoding/decoding | RFC 1157, RFC 1213 (MIB-II) |
| **EtherNet/IP (CIP)** | `ListIdentity` — high-confidence encapsulation header; the internal layout of the Identity object is to be validated against a real device (see the confidence note in `src/protocols/ethernet_ip.rs`) | ODVA, *CIP Networks Library Vol. 2* |

Purely **passive** discovery (listening to already-present traffic, emitting
nothing) remains a natural extension not yet done — it would require a frame
source (network capture) that this crate does not provide.

## Usage

```rust
use scirust_discovery::{DiscoveryEngine, Protocol, ScopeAuthorization};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

let key = b"shared-secret-negotiated-out-of-band";
let scope = ScopeAuthorization {
    operator: "alice@example.com".to_string(),
    zone: "line3-plc-zone".to_string(),
    zone_security_level: 1,
    allowed_cidrs: vec!["192.168.1.0/24".to_string()],
    allowed_protocols: vec!["opcua".to_string(), "modbus".to_string()],
    valid_from_unix: 0,
    valid_until_unix: u64::MAX,
    allow_high_security_zone: false,
    signature_hex: String::new(),
}
.sign(key);

let mut engine = DiscoveryEngine::new(scope, key.to_vec(), Duration::from_secs(2));
let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
let result = engine.probe_one("192.168.1.10".parse().unwrap(), Protocol::OpcUa, now);
println!("{result:?}");
assert!(engine.audit_log().verify_chain());
```

## Honest limits (documented, not hidden)

- IPv4 only for now (`CIDR`/`ScopeAuthorization`) — IPv6 is a mechanical
  addition but not done.
- `hmac.rs` implements a pre-shared-key signature, not a PKI (no key rotation
  or revocation) — sufficient for a single-operator/single-team use, to be
  hardened before a multi-tenant deployment.
- Purely **passive** discovery (listening to already-present traffic, emitting
  nothing) remains to be implemented — it requires a frame source (network
  capture) that this crate does not provide yet.
- The internal layout of the EtherNet/IP Identity object (`ethernet_ip.rs`)
  follows the generally documented structure but has not been verified against
  a real device in this environment — see the confidence note at the top of
  that module before production use.
- BACnet `I-Am` decoding (`bacnet.rs`) stops at the first parameter (the
  device identifier); the following parameters (max APDU length,
  segmentation, vendor ID) are not decoded.
