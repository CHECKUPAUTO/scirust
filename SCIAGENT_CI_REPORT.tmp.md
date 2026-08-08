# TEMP SCIAGENT CI REPORT

MSRV_EXIT=101
CLIPPY_EXIT=101
WGPU_EXIT=0

## MSRV tail
```text
    Checking scirust-tensor-reference v0.1.0 (/home/runner/work/scirust/scirust/scirust-tensor-reference)
    Checking scirust-studio-registry v0.1.0 (/home/runner/work/scirust/scirust/scirust-studio-registry)
    Checking scirust-evo v0.1.0 (/home/runner/work/scirust/scirust/scirust-evo)
    Checking scirust-rsi v0.1.0 (/home/runner/work/scirust/scirust/scirust-rsi)
    Checking toml_edit v0.22.27
    Checking scirust-tensor-runtime v0.1.0 (/home/runner/work/scirust/scirust/scirust-tensor-runtime)
    Checking scirust-modalg v0.1.0 (/home/runner/work/scirust/scirust/scirust-modalg)
    Checking scirust-gpu v0.1.0 (/home/runner/work/scirust/scirust/scirust-gpu)
   Compiling rust_decimal v1.42.1
    Checking scirust-som-visualizer v0.1.0 (/home/runner/work/scirust/scirust/scirust-som/crates/visualizer)
    Checking scirust-pdm v0.1.0 (/home/runner/work/scirust/scirust/scirust-pdm)
    Checking arrayvec v0.7.6
    Checking toml v0.8.23
    Checking base64 v0.22.1
   Compiling thiserror v1.0.69
    Checking scirust-studio-schema v0.1.0 (/home/runner/work/scirust/scirust/scirust-studio-schema)
    Checking scirust-trader v0.1.0 (/home/runner/work/scirust/scirust/scirust-trader)
    Checking scirust-finmigrate v0.1.0 (/home/runner/work/scirust/scirust/scirust-finmigrate)
    Checking scirust-studio-runtime v0.1.0 (/home/runner/work/scirust/scirust/scirust-studio-runtime)
    Checking scirust-som-dataset v0.1.0 (/home/runner/work/scirust/scirust/scirust-som/crates/dataset)
    Checking scirust-som-frontend v0.1.0 (/home/runner/work/scirust/scirust/scirust-som/crates/frontend)
    Checking scirust-som-model v0.1.0 (/home/runner/work/scirust/scirust/scirust-som/crates/model)
   Compiling thiserror-impl v1.0.69
    Checking scirust-som-cli v0.1.0 (/home/runner/work/scirust/scirust/scirust-som/crates/cli)
    Checking scirust-events-runtime v0.1.0 (/home/runner/work/scirust/scirust/scirust-events-runtime)
    Checking scirust-events-models v0.1.0 (/home/runner/work/scirust/scirust/scirust-events-models)
    Checking scirust-tn v0.1.0 (/home/runner/work/scirust/scirust/scirust-tn)
    Checking log v0.4.29
    Checking scirust-variational v0.1.0 (/home/runner/work/scirust/scirust/scirust-variational)
    Checking scirust-tolerance v0.1.0 (/home/runner/work/scirust/scirust/scirust-tolerance)
    Checking scirust-studio-store v0.1.0 (/home/runner/work/scirust/scirust/scirust-studio-store)
    Checking scirust-neuro-symbolic v0.1.0 (/home/runner/work/scirust/scirust/scirust-neuro-symbolic)
    Checking scirust-studio-ipc v0.1.0 (/home/runner/work/scirust/scirust/scirust-studio-ipc)
    Checking scirust-som-trainer v0.1.0 (/home/runner/work/scirust/scirust/scirust-som/crates/trainer)
    Checking scirust-hypercrypto v0.1.0 (/home/runner/work/scirust/scirust/scirust-hypercrypto)
    Checking scirust-mqtt v0.1.0 (/home/runner/work/scirust/scirust/scirust-mqtt)
    Checking scirust-opcua v0.1.0 (/home/runner/work/scirust/scirust/scirust-opcua)
    Checking scirust-func-safety v0.1.0 (/home/runner/work/scirust/scirust/scirust-func-safety)
    Checking scirust-symreg v0.1.0 (/home/runner/work/scirust/scirust/scirust-symreg)
    Checking scirust-reliability v0.1.0 (/home/runner/work/scirust/scirust/scirust-reliability)
   Compiling scirust-cli v0.1.0 (/home/runner/work/scirust/scirust/scirust-cli)
    Checking scirust v0.14.0 (/home/runner/work/scirust/scirust)
    Checking scirust-som-inference v0.1.0 (/home/runner/work/scirust/scirust/scirust-som/crates/inference)
    Checking scirust-retrieval v0.1.0 (/home/runner/work/scirust/scirust/scirust-retrieval)
    Checking scirust-sis v0.1.0 (/home/runner/work/scirust/scirust/scirust-sis)
    Checking scirust-ids v0.1.0 (/home/runner/work/scirust/scirust/scirust-ids)
    Checking scirust-discovery v0.1.0 (/home/runner/work/scirust/scirust/scirust-discovery)
    Checking scirust-grid v0.1.0 (/home/runner/work/scirust/scirust/scirust-grid)
    Checking scirust-biomed v0.1.0 (/home/runner/work/scirust/scirust/scirust-biomed)
    Checking scirust-fab v0.1.0 (/home/runner/work/scirust/scirust/scirust-fab)
    Checking scirust-maritime v0.1.0 (/home/runner/work/scirust/scirust/scirust-maritime)
    Checking scirust-agtech v0.1.0 (/home/runner/work/scirust/scirust/scirust-agtech)
    Checking scirust-fatigue v0.1.0 (/home/runner/work/scirust/scirust/scirust-fatigue)
    Checking scirust-itd v0.1.0 (/home/runner/work/scirust/scirust/scirust-itd)
    Checking scirust-transpiler v0.1.0 (/home/runner/work/scirust/scirust/scirust-transpiler)
    Checking scirust-elliptic-discovery v0.3.0 (/home/runner/work/scirust/scirust/scirust-elliptic-discovery)
    Checking scirust-provenance v0.1.0 (/home/runner/work/scirust/scirust/scirust-provenance)
    Checking scirust-sigma v0.1.0 (/home/runner/work/scirust/scirust/scirust-sigma)
    Checking scirust-machining v0.1.0 (/home/runner/work/scirust/scirust/scirust-machining)
    Checking scirust-mcp v0.1.0 (/home/runner/work/scirust/scirust/scirust-mcp)
    Checking scirust-integration v0.1.0 (/home/runner/work/scirust/scirust/scirust-integration)
    Checking scirust-mlops v0.1.0 (/home/runner/work/scirust/scirust/scirust-mlops)
    Checking scirust-nav v0.1.0 (/home/runner/work/scirust/scirust/scirust-nav)
    Checking scirust-water v0.1.0 (/home/runner/work/scirust/scirust/scirust-water)
    Checking scirust-studio-app-service v0.1.0 (/home/runner/work/scirust/scirust/scirust-studio-app-service)
    Checking scirust-fusion v0.1.0 (/home/runner/work/scirust/scirust/scirust-fusion)
    Checking scirust-arena v0.1.0 (/home/runner/work/scirust/scirust/scirust-arena)
    Checking scirust-industrial v0.1.0 (/home/runner/work/scirust/scirust/scirust-industrial)
    Checking industrial-monitor v0.1.0 (/home/runner/work/scirust/scirust/examples/industrial_monitor)
    Checking ids_demo v0.1.0 (/home/runner/work/scirust/scirust/examples/ids_demo)
    Checking scirust-ccos v0.3.0 (/home/runner/work/scirust/scirust/scirust-ccos)
    Checking scirust-studio-worker v0.1.0 (/home/runner/work/scirust/scirust/scirust-studio-worker)
    Checking scirust-events-examples v0.1.0 (/home/runner/work/scirust/scirust/scirust-events-examples)
    Checking scirust-bms v0.1.0 (/home/runner/work/scirust/scirust/scirust-bms)
    Checking scirust-tensor-examples v0.1.0 (/home/runner/work/scirust/scirust/scirust-tensor-examples)
    Checking scirust-automl v0.1.0 (/home/runner/work/scirust/scirust/scirust-automl)
    Checking scirust-algogen v0.1.0 (/home/runner/work/scirust/scirust/scirust-algogen)
    Checking scirust-synthesis v0.1.0 (/home/runner/work/scirust/scirust/scirust-synthesis)
    Checking scirust-nas v0.1.0 (/home/runner/work/scirust/scirust/scirust-nas)
    Checking scirust-rl-algo v0.1.0 (/home/runner/work/scirust/scirust/scirust-rl-algo)
error[E0432]: unresolved import `scirust_sciagent::cuda_model`
  --> scirust-sciagent/examples/cuda_production_bench.rs:21:23
   |
21 | use scirust_sciagent::cuda_model::{CudaModel, CudaPretrainConfig, CudaTrainer};
   |                       ^^^^^^^^^^ could not find `cuda_model` in `scirust_sciagent`
   |
note: found an item that was configured out
  --> /home/runner/work/scirust/scirust/scirust-sciagent/src/lib.rs:11:9
   |
11 | pub mod cuda_model;
   |         ^^^^^^^^^^
note: the item is gated behind the `cuda` feature
  --> /home/runner/work/scirust/scirust/scirust-sciagent/src/lib.rs:10:7
   |
10 | #[cfg(feature = "cuda")]
   |       ^^^^^^^^^^^^^^^^

For more information about this error, try `rustc --explain E0432`.
error: could not compile `scirust-sciagent` (example "cuda_production_bench") due to 1 previous error
warning: build failed, waiting for other jobs to finish...
```

## Clippy tail
```text
    Checking scirust-som-tokenizer v0.1.0 (/home/runner/work/scirust/scirust/scirust-som/crates/tokenizer)
    Checking scirust-tensor-contraction v0.1.0 (/home/runner/work/scirust/scirust/scirust-tensor-contraction)
    Checking scirust-studio-command v0.1.0 (/home/runner/work/scirust/scirust/scirust-studio-command)
    Checking indexmap v2.14.0
    Checking serde_spanned v0.6.9
    Checking toml_datetime v0.6.11
    Checking winnow v0.7.15
    Checking toml_write v0.1.2
    Checking scirust-tensor-compile v0.1.0 (/home/runner/work/scirust/scirust/scirust-tensor-compile)
    Checking scirust-som-symbolic v0.1.0 (/home/runner/work/scirust/scirust/scirust-som/crates/symbolic)
    Checking scirust-studio-registry v0.1.0 (/home/runner/work/scirust/scirust/scirust-studio-registry)
    Checking scirust-tensor-reference v0.1.0 (/home/runner/work/scirust/scirust/scirust-tensor-reference)
    Checking scirust-evo v0.1.0 (/home/runner/work/scirust/scirust/scirust-evo)
    Checking scirust-rsi v0.1.0 (/home/runner/work/scirust/scirust/scirust-rsi)
    Checking scirust-multivariate v0.1.0 (/home/runner/work/scirust/scirust/scirust-multivariate)
    Checking scirust-signal v0.1.0 (/home/runner/work/scirust/scirust/scirust-signal)
    Checking scirust-runtime v0.1.0 (/home/runner/work/scirust/scirust/scirust-runtime)
    Checking scirust-unsupervised v0.1.0 (/home/runner/work/scirust/scirust/scirust-unsupervised)
    Checking scirust-graph v0.1.0 (/home/runner/work/scirust/scirust/scirust-graph)
    Checking scirust-sciagent v0.1.0 (/home/runner/work/scirust/scirust/scirust-sciagent)
    Checking scirust-learning v0.1.0 (/home/runner/work/scirust/scirust/scirust-learning)
    Checking scirust-srcc v0.1.0 (/home/runner/work/scirust/scirust/scirust-srcc)
    Checking scirust-causal v0.1.0 (/home/runner/work/scirust/scirust/scirust-causal)
    Checking scirust-events-core v0.1.0 (/home/runner/work/scirust/scirust/scirust-events-core)
    Checking toml_edit v0.22.27
    Checking scirust-srcc-bench v0.1.0 (/home/runner/work/scirust/scirust/scirust-srcc-bench)
    Checking scirust-cayley-filter v0.1.0 (/home/runner/work/scirust/scirust/scirust-cayley-filter)
    Checking toml v0.8.23
    Checking scirust-tensor-runtime v0.1.0 (/home/runner/work/scirust/scirust/scirust-tensor-runtime)
    Checking scirust-studio-schema v0.1.0 (/home/runner/work/scirust/scirust/scirust-studio-schema)
    Checking scirust-studio-runtime v0.1.0 (/home/runner/work/scirust/scirust/scirust-studio-runtime)
    Checking scirust-modalg v0.1.0 (/home/runner/work/scirust/scirust/scirust-modalg)
    Checking scirust-gpu v0.1.0 (/home/runner/work/scirust/scirust/scirust-gpu)
   Compiling rust_decimal v1.42.1
    Checking scirust-pdm v0.1.0 (/home/runner/work/scirust/scirust/scirust-pdm)
    Checking scirust-som-visualizer v0.1.0 (/home/runner/work/scirust/scirust/scirust-som/crates/visualizer)
    Checking arrayvec v0.7.6
    Checking scirust-som-frontend v0.1.0 (/home/runner/work/scirust/scirust/scirust-som/crates/frontend)
    Checking base64 v0.22.1
    Checking scirust-som-cli v0.1.0 (/home/runner/work/scirust/scirust/scirust-som/crates/cli)
   Compiling thiserror v1.0.69
    Checking scirust-trader v0.1.0 (/home/runner/work/scirust/scirust/scirust-trader)
    Checking scirust-finmigrate v0.1.0 (/home/runner/work/scirust/scirust/scirust-finmigrate)
    Checking scirust-som-dataset v0.1.0 (/home/runner/work/scirust/scirust/scirust-som/crates/dataset)
    Checking scirust-som-model v0.1.0 (/home/runner/work/scirust/scirust/scirust-som/crates/model)
   Compiling thiserror-impl v1.0.69
    Checking scirust-events-models v0.1.0 (/home/runner/work/scirust/scirust/scirust-events-models)
    Checking scirust-events-runtime v0.1.0 (/home/runner/work/scirust/scirust/scirust-events-runtime)
    Checking scirust-tn v0.1.0 (/home/runner/work/scirust/scirust/scirust-tn)
    Checking log v0.4.29
    Checking scirust-studio-store v0.1.0 (/home/runner/work/scirust/scirust/scirust-studio-store)
    Checking scirust-variational v0.1.0 (/home/runner/work/scirust/scirust/scirust-variational)
    Checking scirust-tolerance v0.1.0 (/home/runner/work/scirust/scirust/scirust-tolerance)
    Checking scirust-studio-ipc v0.1.0 (/home/runner/work/scirust/scirust/scirust-studio-ipc)
    Checking scirust-som-trainer v0.1.0 (/home/runner/work/scirust/scirust/scirust-som/crates/trainer)
    Checking scirust-neuro-symbolic v0.1.0 (/home/runner/work/scirust/scirust/scirust-neuro-symbolic)
    Checking scirust-hypercrypto v0.1.0 (/home/runner/work/scirust/scirust/scirust-hypercrypto)
    Checking scirust-mqtt v0.1.0 (/home/runner/work/scirust/scirust/scirust-mqtt)
    Checking scirust-opcua v0.1.0 (/home/runner/work/scirust/scirust/scirust-opcua)
    Checking scirust-func-safety v0.1.0 (/home/runner/work/scirust/scirust/scirust-func-safety)
    Checking scirust-symreg v0.1.0 (/home/runner/work/scirust/scirust/scirust-symreg)
    Checking scirust-reliability v0.1.0 (/home/runner/work/scirust/scirust/scirust-reliability)
   Compiling scirust-cli v0.1.0 (/home/runner/work/scirust/scirust/scirust-cli)
    Checking scirust v0.14.0 (/home/runner/work/scirust/scirust)
    Checking scirust-som-inference v0.1.0 (/home/runner/work/scirust/scirust/scirust-som/crates/inference)
    Checking scirust-retrieval v0.1.0 (/home/runner/work/scirust/scirust/scirust-retrieval)
    Checking scirust-sis v0.1.0 (/home/runner/work/scirust/scirust/scirust-sis)
    Checking scirust-ids v0.1.0 (/home/runner/work/scirust/scirust/scirust-ids)
    Checking scirust-grid v0.1.0 (/home/runner/work/scirust/scirust/scirust-grid)
    Checking scirust-discovery v0.1.0 (/home/runner/work/scirust/scirust/scirust-discovery)
    Checking scirust-biomed v0.1.0 (/home/runner/work/scirust/scirust/scirust-biomed)
    Checking scirust-fab v0.1.0 (/home/runner/work/scirust/scirust/scirust-fab)
    Checking scirust-maritime v0.1.0 (/home/runner/work/scirust/scirust/scirust-maritime)
    Checking scirust-agtech v0.1.0 (/home/runner/work/scirust/scirust/scirust-agtech)
    Checking scirust-fatigue v0.1.0 (/home/runner/work/scirust/scirust/scirust-fatigue)
    Checking scirust-transpiler v0.1.0 (/home/runner/work/scirust/scirust/scirust-transpiler)
    Checking scirust-itd v0.1.0 (/home/runner/work/scirust/scirust/scirust-itd)
    Checking scirust-elliptic-discovery v0.3.0 (/home/runner/work/scirust/scirust/scirust-elliptic-discovery)
    Checking scirust-provenance v0.1.0 (/home/runner/work/scirust/scirust/scirust-provenance)
    Checking scirust-sigma v0.1.0 (/home/runner/work/scirust/scirust/scirust-sigma)
    Checking scirust-machining v0.1.0 (/home/runner/work/scirust/scirust/scirust-machining)
    Checking scirust-mcp v0.1.0 (/home/runner/work/scirust/scirust/scirust-mcp)
    Checking scirust-integration v0.1.0 (/home/runner/work/scirust/scirust/scirust-integration)
    Checking scirust-mlops v0.1.0 (/home/runner/work/scirust/scirust/scirust-mlops)
    Checking scirust-nav v0.1.0 (/home/runner/work/scirust/scirust/scirust-nav)
    Checking scirust-water v0.1.0 (/home/runner/work/scirust/scirust/scirust-water)
    Checking scirust-studio-app-service v0.1.0 (/home/runner/work/scirust/scirust/scirust-studio-app-service)
    Checking scirust-fusion v0.1.0 (/home/runner/work/scirust/scirust/scirust-fusion)
    Checking scirust-arena v0.1.0 (/home/runner/work/scirust/scirust/scirust-arena)
    Checking scirust-industrial v0.1.0 (/home/runner/work/scirust/scirust/scirust-industrial)
    Checking industrial-monitor v0.1.0 (/home/runner/work/scirust/scirust/examples/industrial_monitor)
    Checking scirust-ccos v0.3.0 (/home/runner/work/scirust/scirust/scirust-ccos)
    Checking ids_demo v0.1.0 (/home/runner/work/scirust/scirust/examples/ids_demo)
    Checking scirust-studio-worker v0.1.0 (/home/runner/work/scirust/scirust/scirust-studio-worker)
    Checking scirust-events-examples v0.1.0 (/home/runner/work/scirust/scirust/scirust-events-examples)
    Checking scirust-bms v0.1.0 (/home/runner/work/scirust/scirust/scirust-bms)
    Checking scirust-tensor-examples v0.1.0 (/home/runner/work/scirust/scirust/scirust-tensor-examples)
    Checking scirust-vision v0.1.0 (/home/runner/work/scirust/scirust/scirust-vision)
    Checking scirust-seasonal v0.1.0 (/home/runner/work/scirust/scirust/scirust-seasonal)
    Checking scirust-shm v0.1.0 (/home/runner/work/scirust/scirust/scirust-shm)
    Checking scirust-rl-algo v0.1.0 (/home/runner/work/scirust/scirust/scirust-rl-algo)
    Checking sentiment_demo v0.1.0 (/home/runner/work/scirust/scirust/examples/sentiment_demo)
    Checking scirust-nlp-advanced v0.1.0 (/home/runner/work/scirust/scirust/scirust-nlp-advanced)
error[E0432]: unresolved import `scirust_sciagent::cuda_model`
  --> scirust-sciagent/examples/cuda_production_bench.rs:21:23
   |
21 | use scirust_sciagent::cuda_model::{CudaModel, CudaPretrainConfig, CudaTrainer};
   |                       ^^^^^^^^^^ could not find `cuda_model` in `scirust_sciagent`
   |
note: found an item that was configured out
  --> scirust-sciagent/src/lib.rs:11:9
   |
10 | #[cfg(feature = "cuda")]
   |       ---------------- the item is gated behind the `cuda` feature
11 | pub mod cuda_model;
   |         ^^^^^^^^^^

For more information about this error, try `rustc --explain E0432`.
error: could not compile `scirust-sciagent` (example "cuda_production_bench") due to 1 previous error
warning: build failed, waiting for other jobs to finish...
```

## WGPU tail
```text
   Compiling indexmap v2.14.0
   Compiling safe_arch v0.7.4
   Compiling codespan-reporting v0.11.1
   Compiling bit-set v0.5.3
   Compiling crossbeam-epoch v0.9.20
   Compiling rand_core v0.6.4
   Compiling wgpu-hal v0.21.1
   Compiling spirv v0.3.0+sdk-1.3.268.0
   Compiling gpu-descriptor-types v0.2.0
   Compiling gpu-alloc-types v0.3.0
   Compiling libloading v0.7.4
   Compiling libloading v0.8.9
   Compiling unicode-xid v0.2.6
   Compiling scirust-tensor-core v0.1.0 (/home/runner/work/scirust/scirust/scirust-tensor-core)
   Compiling hexf-parse v0.2.1
   Compiling arrayvec v0.7.6
   Compiling zerocopy-derive v0.8.48
   Compiling serde_derive v1.0.228
   Compiling thiserror-impl v1.0.69
   Compiling rustc-hash v1.1.0
   Compiling serde_json v1.0.149
   Compiling log v0.4.29
   Compiling scirust-tensor-einsum v0.1.0 (/home/runner/work/scirust/scirust/scirust-tensor-einsum)
   Compiling thiserror v1.0.69
   Compiling naga v0.20.0
   Compiling zerocopy v0.8.48
   Compiling serde v1.0.228
   Compiling gpu-alloc v0.6.0
   Compiling gpu-descriptor v0.3.2
   Compiling crossbeam-deque v0.8.6
   Compiling zmij v1.0.21
   Compiling wide v0.7.33
   Compiling ppv-lite86 v0.2.21
   Compiling rand_chacha v0.3.1
   Compiling parking_lot v0.12.5
   Compiling num-bigint v0.4.6
   Compiling approx v0.5.1
   Compiling wgpu-core v0.21.1
   Compiling wgpu-types v0.20.0
   Compiling glow v0.13.1
   Compiling raw-window-handle v0.6.2
   Compiling scirust-compute v0.1.0 (/home/runner/work/scirust/scirust/scirust-compute)
   Compiling renderdoc-sys v1.1.0
   Compiling itoa v1.0.18
   Compiling profiling v1.0.18
   Compiling litrs v1.0.0
   Compiling utf8parse v0.2.2
   Compiling once_cell v1.19.0
   Compiling memchr v2.8.0
   Compiling anstyle-parse v1.0.0
   Compiling document-features v0.2.12
   Compiling scirust-tensor-ir v0.1.0 (/home/runner/work/scirust/scirust/scirust-tensor-ir)
   Compiling num-rational v0.4.2
   Compiling simba v0.9.1
   Compiling rand v0.8.6
   Compiling rayon-core v1.13.0
   Compiling ndarray v0.16.1
   Compiling scirust-tensor-contraction v0.1.0 (/home/runner/work/scirust/scirust/scirust-tensor-contraction)
   Compiling nalgebra-macros v0.2.2
   Compiling thiserror-impl v2.0.18
   Compiling wgpu v0.20.1
   Compiling anstyle-query v1.1.5
   Compiling typenum v1.20.0
   Compiling is_terminal_polyfill v1.70.2
   Compiling either v1.15.0
   Compiling colorchoice v1.0.5
   Compiling anstyle v1.0.14
   Compiling anstream v1.0.0
   Compiling rayon v1.12.0
   Compiling nalgebra v0.33.3
   Compiling scirust-symbolic v0.1.0 (/home/runner/work/scirust/scirust/scirust-symbolic)
   Compiling thiserror v2.0.18
   Compiling scirust-tensor-compile v0.1.0 (/home/runner/work/scirust/scirust/scirust-tensor-compile)
   Compiling rand_distr v0.4.3
   Compiling half v2.7.1
   Compiling scirust-macros v0.1.0 (/home/runner/work/scirust/scirust/scirust-macros)
   Compiling scirust-simd v0.1.0 (/home/runner/work/scirust/scirust/scirust-simd)
   Compiling strsim v0.11.1
   Compiling scirust-autodiff v0.1.0 (/home/runner/work/scirust/scirust/scirust-autodiff)
   Compiling static_assertions v1.1.0
   Compiling scirust-special v0.1.0 (/home/runner/work/scirust/scirust/scirust-special)
   Compiling clap_lex v1.1.0
   Compiling clap_derive v4.6.1
   Compiling clap_builder v4.6.0
   Compiling scirust-tensor-reference v0.1.0 (/home/runner/work/scirust/scirust/scirust-tensor-reference)
   Compiling scirust-core v0.1.0 (/home/runner/work/scirust/scirust/scirust-core)
   Compiling nix v0.27.1
   Compiling pollster v0.3.0
   Compiling command-group v5.0.1
   Compiling clap v4.6.1
   Compiling scirust-agent-protocol v0.1.0 (/home/runner/work/scirust/scirust/scirust-agent-protocol)
   Compiling scirust-gpu v0.1.0 (/home/runner/work/scirust/scirust/scirust-gpu)
   Compiling scirust-sciagent v0.1.0 (/home/runner/work/scirust/scirust/scirust-sciagent)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 44.22s
     Running tests/gpu_parity.rs (target/debug/deps/gpu_parity-394147b16893dd79)

running 13 tests
resident KV-cache on: llvmpipe (LLVM 20.1.2, 256 bits)
full-model GPU parity on: llvmpipe (LLVM 20.1.2, 256 bits)
resident generate on: llvmpipe (LLVM 20.1.2, 256 bits)
resident DoRA on: llvmpipe (LLVM 20.1.2, 256 bits)
forward rel_err 5.32e-7, worst grad rel_err 3.68e-6 — PASS
test full_model_forward_and_backward_match_cpu_on_gpu ... ok
resident LoRA on: llvmpipe (LLVM 20.1.2, 256 bits)
resident greedy generation matches CPU — PASS ([3, 7, 1, 4, 12, 28, 12, 13, 12])
test resident_generate_matches_cpu_greedy ... ok
resident model on: llvmpipe (LLVM 20.1.2, 256 bits)
resident-model forward rel_err 5.57e-7 — PASS
test resident_model_forward_matches_cpu_model ... ok
resident KV-cache matches whole-sequence generate — PASS ([3, 7, 1, 4, 12, 28, 12, 13, 12, 45, 22, 12])
test resident_kv_cache_matches_greedy ... ok
resident sampling on: llvmpipe (LLVM 20.1.2, 256 bits)
resident sampling: T=0 greedy-parity + seed determinism — PASS
test resident_sampled_generation_is_greedy_at_t0_and_seed_deterministic ... ok
resident speculative decoding on: llvmpipe (LLVM 20.1.2, 256 bits)
  checkpoint → /tmp/scirust_resident_pretrain_ckpt/step_20
resident LoRA fine-tune: loss 5.3142 -> 2.3505
resident LoRA fine-tune + merge — PASS
test resident_lora_finetune_reduces_loss_and_syncs ... ok
resident streaming on: llvmpipe (LLVM 20.1.2, 256 bits)
resident streaming matches sampled (tokens + order) — PASS
test resident_streaming_matches_sampled ... ok
  checkpoint → /tmp/scirust_resident_pretrain_ckpt/step_40
resident pretrain: loss 3.2975 -> 0.0308
resident pretrain: checkpoint at /tmp/scirust_resident_pretrain_ckpt/step_40 reloaded — PASS
test resident_pretrain_schedules_and_checkpoints ... ok
resident DoRA fine-tune: loss 5.3142 -> 0.8408
resident DoRA fine-tune + merge — PASS
test resident_dora_finetune_reduces_loss_and_syncs ... ok
resident speculative decoding matches greedy (self + cross draft) — PASS
test resident_speculative_matches_greedy ... ok
post-sync round-trip rel_err 1.64e-7 — PASS
test resident_sync_roundtrips_into_model ... ok
resident training: loss 5.3142 -> 0.0047
test resident_train_step_reduces_loss ... ok
resident pretraining: loss 2.1317 (first 5) -> 0.4506 (last 5)
test resident_train_tokens_reduces_loss ... ok

test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 42.76s

```
