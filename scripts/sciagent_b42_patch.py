#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "scirust-sciagent/src/cuda_model.rs"
EXAMPLE = ROOT / "scirust-sciagent/examples/cuda_pretrain.rs"
EVAL = ROOT / "scirust-sciagent/examples/cuda_eval.rs"
ROUTE = ROOT / "scirust-sciagent/ROUTE_B.md"


def must_replace(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"expected {count}, found {n}: {old[:180]!r}")
    return text.replace(old, new, count)


SEMANTICS_HELPERS = r'''
/// SCIAGENT model-math compatibility generation. Version 2 is the B33 transition
/// from full-projection-width RoPE to GQA-correct head-local RoPE. A checkpoint
/// without a marker is historical version 1.
pub const SCIAGENT_MODEL_SEMANTICS_VERSION: u32 = 2;
const MODEL_SEMANTICS_FILE: &str = "model_semantics.version";

pub fn read_model_semantics_version(path: &Path) -> Option<u32> {
    std::fs::read_to_string(path.join(MODEL_SEMANTICS_FILE))
        .ok()
        .and_then(|raw| raw.trim().parse::<u32>().ok())
}

fn write_model_semantics_marker(path: &Path) -> std::result::Result<(), String> {
    std::fs::write(
        path.join(MODEL_SEMANTICS_FILE),
        format!("{}\n", SCIAGENT_MODEL_SEMANTICS_VERSION),
    )
    .map_err(|e| format!("cannot write model semantics marker in {}: {e}", path.display()))
}
'''


def patch_model(text: str) -> str:
    if "SCIAGENT_MODEL_SEMANTICS_VERSION" in text:
        raise SystemExit("cuda_model already B42 patched")
    marker = "/// One GQA block's weights mirrored into VRAM (bf16).\n"
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("missing semantics helper insertion point")
    text = text[:pos] + SEMANTICS_HELPERS + "\n" + text[pos:]

    # Both exact recovery and model-only best snapshots must carry the marker. The
    # exact call has indentation inside an impl; the best helper is a free function.
    exact = '''        save_checkpoint(model, meta, &partial)\n            .map_err(|e| format!("cannot save model checkpoint: {e}"))?;\n        self.save_optimizer_state(cfg, &partial)?;\n'''
    exact_new = '''        save_checkpoint(model, meta, &partial)\n            .map_err(|e| format!("cannot save model checkpoint: {e}"))?;\n        write_model_semantics_marker(&partial)?;\n        self.save_optimizer_state(cfg, &partial)?;\n'''
    text = must_replace(text, exact, exact_new, 1)

    best = '''    save_checkpoint(model, meta, &partial)\n        .map_err(|e| format!("cannot save best model: {e}"))?;\n    let selection = serde_json::json!({\n'''
    best_new = '''    save_checkpoint(model, meta, &partial)\n        .map_err(|e| format!("cannot save best model: {e}"))?;\n    write_model_semantics_marker(&partial)?;\n    let selection = serde_json::json!({\n'''
    text = must_replace(text, best, best_new, 1)
    return text


def patch_example(text: str) -> str:
    if "checkpoint model semantics" in text:
        raise SystemExit("cuda_pretrain already B42 patched")
    old_import = "use scirust_sciagent::cuda_model::{CudaPretrainConfig, CudaTrainer};\n"
    new_import = "use scirust_sciagent::cuda_model::{\n    CudaPretrainConfig, CudaTrainer, SCIAGENT_MODEL_SEMANTICS_VERSION,\n    read_model_semantics_version,\n};\n"
    text = must_replace(text, old_import, new_import, 1)

    old = '''    if let Some((path, meta)) = &resume\n    {\n        match load_checkpoint(&mut model, path)\n'''
    new = '''    if let Some((path, meta)) = &resume\n    {\n        let checkpoint_semantics = read_model_semantics_version(path).unwrap_or(1);\n        if checkpoint_semantics != SCIAGENT_MODEL_SEMANTICS_VERSION\n        {\n            let message = format!(\n                "checkpoint model semantics v{checkpoint_semantics} != current v{} (B33 head-local RoPE)",\n                SCIAGENT_MODEL_SEMANTICS_VERSION\n            );\n            if !allow_nonexact_resume()\n            {\n                eprintln!(\n                    "{message}; refusing to train historical weights under different model math. \\\n                     Use a fresh SCIAGENT_CKPT directory for the production run. \\\n                     SCIAGENT_ALLOW_NONEXACT_RESUME=1 is research-only."\n                );\n                std::process::exit(1);\n            }\n            eprintln!("WARNING: {message}; non-exact model-semantics override enabled");\n        }\n        match load_checkpoint(&mut model, path)\n'''
    text = must_replace(text, old, new, 1)
    return text


def patch_eval(text: str) -> str:
    if "historical checkpoint semantics" in text:
        raise SystemExit("cuda_eval already B42 patched")
    old_import = "use scirust_sciagent::cuda_model::CudaModel;\n"
    new_import = "use scirust_sciagent::cuda_model::{\n    CudaModel, SCIAGENT_MODEL_SEMANTICS_VERSION, read_model_semantics_version,\n};\n"
    text = must_replace(text, old_import, new_import, 1)

    idx = text.find("read_meta(&ckpt")
    if idx < 0:
        raise SystemExit("cannot find cuda_eval read_meta(&ckpt)")
    line_start = text.rfind("\n", 0, idx) + 1
    guard = r'''    let checkpoint_semantics = read_model_semantics_version(&ckpt).unwrap_or(1);
    if checkpoint_semantics != SCIAGENT_MODEL_SEMANTICS_VERSION {
        let allow = matches!(
            std::env::var("SCIAGENT_ALLOW_NONEXACT_RESUME").as_deref(),
            Ok("1" | "true")
        );
        if !allow {
            eprintln!(
                "historical checkpoint semantics v{checkpoint_semantics} != current v{}; \
                 refusing an invalid post-B33 quality comparison. Evaluate a fresh v2 checkpoint, \
                 or set SCIAGENT_ALLOW_NONEXACT_RESUME=1 only to inspect the mismatch explicitly.",
                SCIAGENT_MODEL_SEMANTICS_VERSION
            );
            std::process::exit(1);
        }
        eprintln!(
            "WARNING: evaluating historical semantics v{checkpoint_semantics} with current v{} math; \
             metrics are not faithful to the historical model",
            SCIAGENT_MODEL_SEMANTICS_VERSION
        );
    }

'''
    text = text[:line_start] + guard + text[line_start:]
    return text


def patch_route(text: str) -> str:
    if "model_semantics.version" in text:
        return text
    return text.rstrip() + r'''

### Checkpoint semantics guard

Post-B33 CUDA recovery checkpoints and model-only `best/` snapshots carry
`model_semantics.version = 2`. Historical checkpoints have no marker and are treated
as version 1. `cuda_pretrain` refuses to continue a v1 checkpoint under v2 head-local
RoPE by default, and `cuda_eval` refuses to present such a mixed-semantics evaluation
as a valid quality result. A fresh checkpoint directory is required for the final
production run. `SCIAGENT_ALLOW_NONEXACT_RESUME=1` exists only for explicit research
experiments and prints a warning.
''' + "\n"


MODEL.write_text(patch_model(MODEL.read_text()))
EXAMPLE.write_text(patch_example(EXAMPLE.read_text()))
EVAL.write_text(patch_eval(EVAL.read_text()))
ROUTE.write_text(patch_route(ROUTE.read_text()))
print("B42 patched: model-semantics checkpoint guard")
