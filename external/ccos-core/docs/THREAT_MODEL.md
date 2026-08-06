# Threat Model

## Assets

CCOS protects the live source tree, trusted tests and benchmarks, audit/event
logs, persisted memory, license metadata, machine-binding hashes, local model
prompts, host credentials, and release artifacts.

## Adversaries

Inputs may include malicious generated Rust/CUDA, build scripts, procedural
macros, patches, MCP JSON, persisted files, URLs, DNS answers, license tokens,
and documents processed by explicitly installed addons. Pull-request code is
untrusted. A normal local user able to pre-create files in shared temporary
locations is also considered.

The host kernel, host root, repository maintainer credentials, production key
ceremony, and independently trusted compiler/toolchain are outside the
application boundary. Compromise of any of them requires operational
containment rather than an application-level claim of safety.

## Trust Boundaries

The deterministic community core performs no telemetry and has network access
disabled by default. Optional HTTP clients must authorize, resolve once,
validate every address, disable proxies and redirects, and connect using the
pinned answer set. Public remote Claude access additionally requires exact
`CCOS_EGRESS_ALLOW` consent and rejects private, loopback, link-local, mapped,
and multicast DNS answers.

DGM and Forge generated-code execution crosses the `ccos-sandbox` boundary.
Bubblewrap is mandatory; root, missing isolation, invalid mounts, unsupported
limits, mutable trusted harnesses, and malformed candidate paths are
infrastructure refusals, not low fitness scores.

Papers and hardware diagnostic binaries are operator-installed external tools,
not candidate execution. The Papers path is canonicalized from fixed system
directories or an absolute path, must be owned by root/current UID and not
group/world writable, receives a cleared environment, bounded output and a
dedicated process group. Its results remain untrusted input to DGM policy.

## Security Properties

- Candidate source and trusted harnesses are read-only; build output is
  isolated and disposable.
- Network, host home, credentials, sockets, and GPU devices are absent from
  generated-code evaluation.
- Kernel limits bound memory, files, processes, CPU, descriptors, wall time,
  and captured output.
- DGM promotion is policy-gated, hash-attested, backup-preserving, and atomic.
- Forge candidates cannot use native capabilities that would let them exit the
  harness early or forge benchmark files.
- Sandbox failure aborts/quarantines evaluation and cannot be selected as an
  ordinary candidate result.

## Non-Goals

The sandbox does not prove semantic correctness, prevent every timing side
channel, defend against a compromised kernel/toolchain, or make generated code
safe to execute as root. CUDA candidates are not evaluated because granting
GPU device access would widen the boundary beyond this profile.
