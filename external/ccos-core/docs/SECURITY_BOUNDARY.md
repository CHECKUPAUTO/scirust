# CCOS Core — Security Boundary

## Trusted computing base

- the hash-chained journal and CCPS persistence layer;
- the policy fold (scoring weights, thresholds, eviction);
- the offline license verifier (ed25519 / SLH-DSA public keys baked at build
  time; no network, no telemetry);
- the input-hardening path: `guard.rs`, `sanitizer.rs`,
  `injection_classifier.rs`, `adversarial.rs`.

## What Core never does (§4.1)

- modify its own source code;
- generate and automatically apply patches;
- compile generated code (`cargo`/`rustc` on candidates);
- execute arbitrary commands or interpreters on untrusted input;
- launch Docker or access the Docker socket;
- download and execute binaries;
- expose shell execution or repository mutation through MCP;
- depend on RSI, Forge, or the Research Lab sandbox.

Enforcement: `scripts/check-no-research-components.sh` (CI step),
`security/forbidden-core-dependencies.toml`,
`security/process-execution-allowlist.toml` (versioned, minimal).

## Egress

Default posture: **no egress**. LLM/eval calls are feature-gated (`llm`) and
validated against an explicit allowlist (`src/egress.rs`); proxy/redirect
hardening imported from the security series (c721053). The one sanctioned
network path is a user-configured model endpoint.

## MCP exposure (§31)

Exposed: `memory.*`, `context.*`, `policy.*`, `audit.*`, `system.health`-class
read tools under the `ccos.*` namespace. Never exposed: `rsi.*`, `forge.*`,
`patch.apply`, `code.execute`, `shell.execute`, `repository.modify`,
`self.modify`.

## Storage hygiene

Durable files use atomic writes (PID-tmp + rename), `create_new(true)` where
applicable, and 0o600 permissions; persistent envelopes are size-bounded and
fuzzed (`fuzz/fuzz_targets/`).

See also: THREAT_MODEL.md, PRIVACY.md, SUBPROCESS_AUDIT.md.
