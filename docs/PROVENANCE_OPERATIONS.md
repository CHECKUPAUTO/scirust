# Provenance & licensing operations runbook

This runbook turns the **demo** provenance/licensing scaffolding into a
**production** setup that is actually usable as evidence and as enforcement. It
covers the three operational steps the code cannot do for you:

1. Generate a real vendor master seed (in an HSM / on an air-gapped host) and
   derive its public root.
2. Pin that root in the code and **publish it to a timestamped, immutable venue
   _before_ you distribute anything** — this is what defeats the "the vendor
   planted the signature after the fact" defence.
3. Disclose the watermarking/licensing in the EULA and provide a
   reproducibility-friendly build.

> **Why this matters.** Until these steps are done, every pinned root in the
> tree is the **public demo root**, and the demo seed is public — so a
> demo-signed artifact or license proves *nothing*. The cryptography is only as
> strong as the secrecy of the master seed and the provable-priority of the
> public root.

All tooling referenced here already exists in the workspace:

- `license-tool` (`scirust-license`) — `keygen`, `issue`, `inspect`, `check`.
- `prov` (`scirust-provenance`) — `sign`, `verify` for emitted artifacts.

---

## Step 1 — Generate the master seed and derive the root (offline / HSM)

The master seed is a 32-byte secret. **It must never touch this repository, CI,
a shipped binary, or a networked host.** Generate and store it where signing
happens: an HSM, a hardware token, or at minimum an air-gapped machine with an
encrypted volume and a logged, access-controlled procedure.

### 1a. Create the seed (air-gapped host)

```bash
# 32 cryptographically-random bytes, hex-encoded. Store the OUTPUT in the HSM /
# secrets manager, never in git. Record who generated it, when, and on what host.
openssl rand -hex 32 > vendor-seed.hex   # -> e.g. "9f3a...c1"  (KEEP SECRET)
```

If your HSM generates and holds the key internally, export only the **public
root** (Step 1b) and keep the seed non-exportable — better still.

### 1b. Derive the public root and choose a capacity

The Merkle `height` sets how many one-time licenses/artifacts the key can sign
(`2^height`, clamped to `height <= 20` → ~1M). Pick a height that covers your
expected issuance; rotate the seed when you approach capacity.

```bash
SEED=$(cat vendor-seed.hex)
cargo run -q -p scirust-license --bin license-tool -- keygen --seed "$SEED" --height 20
# -> root (public key): <64-hex>
#    capacity: 1048576 one-time licenses (height 20)
```

Record the 64-hex **root** — this is the only public value; it is safe to embed
and to publish.

### 1c. Chain-of-custody notes (write these down, they are the evidence)

- **Date and host** the seed was generated on, and by whom.
- The seed lives in `<HSM/secret store>`; access is limited to `<people>` and
  logged.
- The **leaf ledger** (Step 4) is authoritative for which leaf signed what.

---

## Step 2 — Pin the production root in the code

Three pinned constants currently default to the public demo root. Replace each
with your production root hex from Step 1b, and keep the drift-guard tests (they
prove the constant matches its documented derivation).

| Constant | File | Purpose |
|---|---|---|
| `EMIT_ROOT_HEX` | `scirust-provenance/src/lib.rs` | verifies signed **emitted artifacts** (`prov`) |
| `GPU_LICENSE_ROOT_HEX` | `scirust-gpu/src/license.rs` | verifies **GPU licenses** (`Module::Gpu`) |
| `PROV_TAG` | `scirust-autodiff/src/lib.rs` (feature `canary`) | keyed seed of the execution-path canary |

For `EMIT_ROOT_HEX` / `GPU_LICENSE_ROOT_HEX`: paste the root hex and update the
matching drift-guard test's expected value (they currently assert equality with
`scirust_license::demo_root()`; point them at your pinned root instead).

For `PROV_TAG` (only if you ship the `canary` feature): re-derive it as the first
8 bytes, little-endian, of
`SHA-256(b"SRL.canary" || <production root bytes> || b"scirust-autodiff")` and
update both the constant and the `prov_tag_matches_offline_derivation` test:

```bash
python3 - <<'PY'
import hashlib, struct
root = bytes.fromhex("<YOUR_PRODUCTION_ROOT_HEX>")
d = hashlib.sha256(b"SRL.canary" + root + b"scirust-autodiff").digest()
print("PROV_TAG = 0x{:016x}".format(struct.unpack("<Q", d[:8])[0]))
PY
```

Run `cargo test -p scirust-provenance -p scirust-gpu --features scirust-gpu/license-gate`
and (if used) `cargo test -p scirust-autodiff --features canary` to confirm the
drift-guards pass against the new root.

Then confirm none of the three constants is still the demo root, and **wire this
guard into your release pipeline** so a demo-signed build can never ship:

```bash
scripts/check-production-root.sh
# ok   EMIT_ROOT_HEX … / FAIL … still the PUBLIC DEMO root
# exit 0 = all production · 1 = a demo root remains · 2 = a constant moved
```

It is deliberately **not** in the default PR CI (which legitimately runs on the
demo roots); add it to the release workflow only.

---

## Step 3 — Publish the root to a timestamped, immutable venue (BEFORE distributing)

The forensic value of a signature is "only the seed holder could have produced
this, **and** the public key provably existed before the suspect's copy." A
root published *after* a dispute is worthless. Publish it now.

Use the helper script, which records the root in a **GPG-signed, annotated git
tag** (timestamped by the tag and by the forge once pushed) and, if the
[OpenTimestamps](https://opentimestamps.org) client `ots` is installed, also
produces a Bitcoin-anchored `.ots` proof:

```bash
scripts/publish-provenance-root.sh --root <YOUR_PRODUCTION_ROOT_HEX> --tag provenance-root-v1
# review the created tag, then push it yourself (outward-facing):
git push origin provenance-root-v1
```

Stronger, independent anchors you can add (any subset — more is better):

- **Certificate-Transparency-style / notarization:** submit the root to a public
  transparency log or a notary service.
- **OpenTimestamps:** `ots stamp provenance-root-v1.txt` → commit the `.ots`
  proof; upgrade it later with `ots upgrade`.
- **Public post** with the root hex and date (a release note, a signed email to
  yourself, a blockchain memo).

Keep every receipt with the chain-of-custody notes from Step 1c.

---

## Step 4 — Operational discipline for signing

### Leaf ledger (mandatory — reuse leaks secrets)

Each one-time `leaf` may sign **at most one distinct digest/license**. Reusing a
leaf for two different payloads leaks Lamport secrets. Keep a persisted,
monotonic ledger `leaf -> (what was signed, when, for whom)` next to the seed.
`license-tool issue` deliberately requires an explicit `--leaf` for this reason.

### Issue a license

```bash
SEED=$(cat vendor-seed.hex)
cargo run -q -p scirust-license --bin license-tool -- issue \
    --seed "$SEED" --height 20 --leaf 1 \
    --licensee "Acme Corp" --id "L-2026-001" --modules gpu \
    --expires 1798761600            # optional; omit for perpetual
    # --node <stable-machine-id>    # optional coarse node-lock (e.g. /etc/machine-id)
# -> signed license JSON; deliver to the customer as their .license.json
```

The customer points `SCIRUST_LICENSE_FILE` (or `SCIRUST_LICENSE`) at it; the GPU
gate (`scirust_gpu::license::activate`) loads and verifies it offline.

### Sign an emitted artifact (offline, on the signing host)

```bash
SEED=$(cat vendor-seed.hex)
cargo run -q -p scirust-provenance --bin prov -- sign \
    --seed "$SEED" --height 20 --leaf 2 generated_module.rs --in-place
```

Anyone can later verify with only the **public** root:

```bash
cargo run -q -p scirust-provenance --bin prov -- verify generated_module.rs
# VERIFIED  generated_module.rs  (root <hex8>, leaf 2)
```

---

## Step 5 — Disclosure & reproducibility (see LICENSING.md)

- Add the **provenance/watermarking/licensing disclosure** clause (in
  `LICENSING.md`) to your EULA, reviewed by counsel.
- Offer a **watermark-free build** for reproducibility-sensitive users: build
  the `canary`-bearing crates with the `canary` feature **off** (the default),
  and document the ordering/label caveats in your reproducibility note. The
  `scirust-simd` reproducibility anchor guarantees the bit-exact numeric paths
  are never silently perturbed either way.

---

## What this does and does not buy you (be honest with counsel)

- **Strong** against verbatim redistribution of your *emitted artifacts* and for
  tracing *which* licensed build leaked (the signature + leaf serial), once the
  root is HSM-backed and timestamp-published.
- **Real but bounded** as licensing enforcement: a graceful gate deters honest
  unlicensed use; a source-level cloner can remove it.
- **Not** a defence against a competitor who *reimplements* the engine from
  scratch — that fight is fought on **source similarity** (kernel schedules,
  packing, constants, structure) and **contract**, not on an in-binary mark.
  Preserve your authorship trail (signed commits, dated design docs) accordingly.
