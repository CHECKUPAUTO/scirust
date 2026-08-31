# SciRust repository agent instructions

Before making repository changes, fetch and read the persistent off-main ecosystem roadmap:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/SCIRUST_ECOSYSTEM_ROADMAP.yaml
```

Treat `AGENTS.md` at repository root as mandatory bootstrap policy.

Reread the roadmap at every session start, before a new major task, before cross-repository integration or research promotion, after strategy/ownership changes, and before PR or merge decisions affecting shared contracts, tensor representation, evidence semantics, or public behavior.

If the roadmap is unavailable, fail closed for major architecture, shared-contract, representation-format, cross-repository promotion, or merge decisions. Do not substitute guesses for missing roadmap state.

SciRust is the shared scientific/computational substrate, not the owner of product-specific control-plane, verification-dossier, capsule-product, evolutionary-search, generic elastic-runtime, native-NVIDIA-runtime, specialized-attention, or compressed-KV semantics.
