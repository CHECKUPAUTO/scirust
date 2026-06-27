# scirust-license

Pure-Rust, deterministic **module-entitlement licensing** for the SciRust
platform. A vendor issues a cryptographically **signed license** listing the
modules a customer may use; the runtime verifies it against an embedded public
key and gates feature access on the result.

## Why hash-based signatures?

SciRust is a pure-`sha2`, no-FFI, bit-deterministic platform (see
`scirust-runtime`'s attestation hash chain). A hash *chain* alone is not a
signature — it has no secret, so anyone can recompute it. Rather than pull in an
elliptic-curve dependency, this crate signs licenses with **Lamport one-time
signatures authenticated by a Merkle tree** (`hashsig`), using SHA-256 only:

* genuinely forgery-resistant — the vendor holds a secret seed; the binary
  embeds only the 32-byte Merkle **root**; forging an unsold entitlement means
  inverting SHA-256;
* deterministic and post-quantum — same seed + leaf + digest → identical
  signature on any platform;
* zero new dependencies beyond `sha2`/`serde`, matching the platform posture.

The trade-off is signature size (~16 KB per license, inherent to Lamport); a
license file is a self-contained `~33 KB` JSON. A Winternitz (WOTS+) variant
would shrink this, at the cost of a more intricate checksum.

## Two sides

| Side    | Holds          | Does                                            |
|---------|----------------|-------------------------------------------------|
| Vendor  | secret seed    | `Vendor::issue_with_leaf` → a `SignedLicense`   |
| Runtime | only the root  | `verify_license` → `Entitlements::require(...)` |

Each one-time leaf signs **at most one** license; the vendor allocates leaves
from a persisted counter.

## Library

```rust
use scirust_license::{Vendor, License, Module, verify_license};

let vendor = Vendor::from_seed(&[42u8; 32], 8); // 2^8 issuable licenses
let root = vendor.root();                        // embed this in the runtime

let license = License::new("Acme", "L-1", [Module::Navigation], 1_000, Some(2_000));
let signed = vendor.issue_with_leaf(license, 0);

let ent = verify_license(&signed, &root, 1_500).unwrap();
assert!(ent.require(Module::Navigation).is_ok());
assert!(ent.require(Module::Water).is_err());     // not licensed
```

A tampered module list, a signature from another vendor, an expired window, or a
malformed signature each fail verification with a distinct `LicenseError`.

## Node-locking (per-machine licenses)

A license can be **bound to a single machine** — the basis for a per-machine
commercial model (e.g. *$1 / machine / month*: monthly via `expires_at`,
per-machine via the node lock).

The crate deliberately does **not** read hardware itself — that would need
platform-specific I/O and break the pure, deterministic, `no_std`-friendly
posture. Instead the **host supplies an opaque machine id** (a provisioned UUID,
`/etc/machine-id`, a TPM value — whatever the deployment trusts as stable). The
license stores only its `node_fingerprint` (a domain-separated SHA-256), so the
file never reveals the raw id, and the lock is part of the signed digest, so it
cannot be edited or removed without breaking the signature.

```rust
use scirust_license::{Vendor, License, Module, verify_license, verify_license_on_node, LicenseError};

let vendor = Vendor::from_seed(&[42u8; 32], 8);
let root = vendor.root();

// Bind the license to one machine at issue time.
let license = License::new("Acme", "L-1", [Module::Navigation], 0, None)
    .with_node_lock("press-line-07");
let signed = vendor.issue_with_leaf(license, 0);

// The runtime presents its own machine id; the right machine is granted…
assert!(verify_license_on_node(&signed, &root, 1, "press-line-07").is_ok());
// …a different machine is refused…
assert_eq!(verify_license_on_node(&signed, &root, 1, "other").err(), Some(LicenseError::NodeMismatch));
// …and the node-blind entry point refuses a locked license outright.
assert_eq!(verify_license(&signed, &root, 1).err(), Some(LicenseError::NodeRequired));
```

A **floating** license (no `with_node_lock`) verifies on any machine, through
either entry point. `verify_license` is for floating licenses; use
`verify_license_on_node` wherever a license might be node-locked.

## CLI

`license-tool` is the vendor + runtime toolchain (uses the bundled demo key when
no `--seed-hex`/`--root-hex` is given):

```
license-tool modules                       # list the licensable catalogue
license-tool keygen --seed-hex <64-hex>    # print a vendor's public root
license-tool issue  --licensee Acme --id L-1 --modules navigation,control \
                    --expires 1893456000 --leaf 0 > acme.license.json
license-tool inspect acme.license.json     # verify + print entitlements
license-tool check   acme.license.json --module navigation   # gate (exit 0/1)

# Per-machine licensing: bind at issue time, present the machine id when verifying.
license-tool issue   --licensee Acme --id L-2 --modules navigation \
                     --node press-line-07 --leaf 1 > node.license.json
license-tool inspect node.license.json                       # INVALID: node-locked
license-tool inspect node.license.json --node press-line-07  # VALID, node: locked
license-tool check   node.license.json --module navigation --node press-line-07
```

## Demo

```
cargo run -p scirust-license --example gate_demo
```

issues a Navigation+Control license, gates two features on it, and shows a
tamper attempt being rejected.

## Catalogue

The licensable units are domain modules (`Module`), each mapping to one or more
SciRust crates — foundation/ML (`core`, `tensor-network`, `nlp`, `vision`,
`audio`, `graph`, `automl`, `reasoning`, `reinforcement-learning`, `evolution`,
`edge`, `events`) and industrial verticals (`estimation`, `navigation`, `water`,
`control`, `battery`, `grid`, `structural-health`, `hvac`, `robotics`,
`metrology`, `signal`, `predictive-maintenance`, `reliability`,
`functional-safety`, `ot-security`, `mlops`, `biomed`, `trading`, `spc`,
`industrial`). Run `license-tool modules` for the live list.

> The bundled demo key is for evaluation only. A production vendor generates a
> random seed offline, keeps it secret, and embeds only the derived root.
