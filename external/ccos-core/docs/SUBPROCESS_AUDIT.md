# Subprocess Audit

This inventory classifies direct Rust subprocess creation after PR #13.
Thread `spawn` calls are not operating-system subprocesses.

| Call site | Classification | Security decision |
| --- | --- | --- |
| `crates/ccos-sandbox/src/lib.rs` | trusted fixed internal supervisor | The sole generated-code process boundary; absolute Bubblewrap/program paths, structural arguments, empty environment, bounded pipes, namespaces, limits, and full-tree termination. |
| `crates/ccos-rsi/src/addons.rs` | external tool with controlled arguments | Operator-installed Papers tool only; canonical trusted executable, fixed search directories or absolute path, no shell, cleared environment, explicit cwd, bounded dual streams, timeout, process-group termination. Results are untrusted data. |
| `crates/ccos-rsi/src/hw_probe.rs` | trusted fixed diagnostic | Optional `nvidia-smi` read-only probe; fixed arguments, bounded output and timeout. It never evaluates candidate code. |
| `src/main.rs` updater probe | trusted fixed internal process | Executes the just-installed artifact only after manifest signature and SHA-256 verification, solely to request its version. |
| `crates/ccos-scirust/build.rs` | trusted fixed build probe | Runs fixed `sysctl` arguments during a trusted first-party build; no candidate input. Candidate builds run inside `ccos-sandbox`. |
| `crates/ccos-scirust/src/bin/slha_audit.rs` | external diagnostic with controlled arguments | Operator-invoked `perf` diagnostic, not candidate evaluation or production runtime. |
| Rust files under `tests/` | trusted test harness | Test binaries invoke known workspace artifacts or availability probes; production code does not call them. |

All DGM Cargo build/test/benchmark paths use `ccos_sandbox::cargo_spec`. All
Forge Rust build/run/benchmark and candidate-binary paths use
`isolation::UntrustedCommand`, which delegates to the same runner. Forge CUDA
evaluation is refused because the untrusted profile does not expose GPU
devices. No production generated-code path contains a direct `Command::new`.
