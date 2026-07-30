# Operations

Run CCOS and all DGM/Forge work under a dedicated unprivileged account with an
owner-only home and workspace. Install Bubblewrap and util-linux, install the
Rust 1.89.0 rustup toolchain, and run the sandbox security suite before
enabling generated-code campaigns.

Do not configure `CCOS_UNSAFE_ALLOW_ROOT_SANDBOX` in production. On the Jetson,
do not add NVIDIA or DRM device nodes to the untrusted profile. CUDA evaluation
is intentionally refused. Do not change nvpmodel, clocks, affinity, or thermal
settings automatically; benchmark metadata reports the observed state.

Keep network denied by default. Remote optional clients require explicit exact
egress consent. Never place proxy variables, registry credentials, SSH agent
sockets, cloud credentials, or LLM API keys in a generated-code service unit.

Monitor infrastructure refusals separately from candidate failures. Record the
sandbox enforcement backend, manifest digest, command result digest, timeout,
output truncation, resource-limit event, and DGM promotion before/after hashes.
Source text, license tokens, fingerprints, credentials, and absolute paths are
not normal metrics.

For emergency termination:

1. Stop the DGM/Forge service.
2. Locate its delegated cgroup under `/sys/fs/cgroup`.
3. Write `1` to `cgroup.kill` when the cgroup still contains processes.
4. Confirm `cgroup.procs` is empty.
5. Remove abandoned `.ccos-sandbox-target` and owner-only candidate snapshots.
6. Verify the audit chain and live-tree hashes before resuming promotion.

If no delegated cgroup exists, terminate the service process group and confirm
no descendant remains. A cleanup or evaluation whose isolation status is
uncertain must not be promoted.
