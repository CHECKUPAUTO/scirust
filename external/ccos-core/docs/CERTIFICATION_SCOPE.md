# CCOS Core — Certification Scope

## Inside the prepared certifiable boundary

CCOS Core is *designed* for certification readiness (§4.1):

- deterministic builds and replay (`replay == live`);
- no RSI, no Forge, no self-modification, no generated-code execution
  (guardrail: `scripts/check-no-research-components.sh`, CI-enforced);
- no runtime process execution on untrusted input
  (`security/process-execution-allowlist.toml`);
- versioned schemas (CCPS envelopes), stable MCP namespace (`ccos.*`);
- offline, deterministic test suite + canonical replay vectors;
- supply-chain controls (deny.toml, SBOM, pinned toolchains, reproducible-build
  comparison workflow);
- single-maintainer governance with enforced author policy.

## Outside the boundary

| Component | Where it lives |
|---|---|
| RSI / self-improvement / DGM | CCOS Research Lab |
| Forge (candidates, mutation, promotion) | CCOS Research Lab |
| experimental sandbox (OS-level, bubblewrap) | CCOS Research Lab |
| scirust/SLHAv2 full kernel (SIMD, replay-relax) | CCOS Research Lab |
| octacore cascade / vendored octasoma | CCOS Research Lab |
| multi-tenancy, RBAC, claim/licensing server | CCOS Enterprise |

## Honesty clause

**No certification is claimed.** "Prepared" means the engineering properties
above are present and CI-enforced. Any certification statement requires an
actual external assessment approved by ZEKRITI Tarek.
