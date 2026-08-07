#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "scirust-sciagent/src/cuda_model.rs"
EXAMPLE = ROOT / "scirust-sciagent/examples/cuda_pretrain.rs"


def must_replace(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"expected {count}, found {n}: {old[:180]!r}")
    return text.replace(old, new, count)


def patch_model(text: str) -> str:
    if "pub max_grad_norm: Option<f32>" in text:
        raise SystemExit("model already B37 patched")
    text = must_replace(
        text,
        "    pub split_version: Option<u32>,\n}\n",
        "    pub split_version: Option<u32>,\n    pub max_grad_norm: Option<f32>,\n    pub val_frac: Option<f32>,\n    pub shuffle: Option<bool>,\n}\n",
        1,
    )
    text = must_replace(text, '            "version": 3,\n', '            "version": 4,\n', 1)
    text = must_replace(
        text,
        '            "split_version": WINDOW_SPLIT_VERSION,\n        });\n',
        '            "split_version": WINDOW_SPLIT_VERSION,\n            "max_grad_norm": cfg.max_grad_norm,\n            "val_frac": cfg.val_frac,\n            "shuffle": cfg.shuffle,\n        });\n',
        1,
    )
    text = must_replace(
        text,
        "        if !(1..=3).contains(&version)\n",
        "        if !(1..=4).contains(&version)\n",
        1,
    )
    # Add optional scalar/bool readers beside optional u64 readers.
    text = must_replace(
        text,
        '''        let optional_usize = |key: &str| meta[key].as_u64().map(|x| x as usize);\n        let optional_u64 = |key: &str| meta[key].as_u64();\n''',
        '''        let optional_usize = |key: &str| meta[key].as_u64().map(|x| x as usize);\n        let optional_u64 = |key: &str| meta[key].as_u64();\n        let optional_f32 = |key: &str| meta[key].as_f64().map(|x| x as f32);\n        let optional_bool = |key: &str| meta[key].as_bool();\n''',
        1,
    )
    text = must_replace(
        text,
        '''            split_version: if version >= 3\n            {\n                optional_u64("split_version").map(|v| v as u32)\n            }\n            else\n            {\n                None\n            },\n        }))\n''',
        '''            split_version: if version >= 3\n            {\n                optional_u64("split_version").map(|v| v as u32)\n            }\n            else\n            {\n                None\n            },\n            max_grad_norm: if version >= 4\n            {\n                optional_f32("max_grad_norm")\n            }\n            else\n            {\n                None\n            },\n            val_frac: if version >= 4\n            {\n                optional_f32("val_frac")\n            }\n            else\n            {\n                None\n            },\n            shuffle: if version >= 4\n            {\n                optional_bool("shuffle")\n            }\n            else\n            {\n                None\n            },\n        }))\n''',
        1,
    )
    return text


HELPERS = r'''
fn allow_nonexact_resume() -> bool {
    matches!(
        std::env::var("SCIAGENT_ALLOW_NONEXACT_RESUME").as_deref(),
        Ok("1" | "true")
    )
}

fn same_f32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits()
}

fn enforce_exact_resume(mismatches: &[String]) {
    if mismatches.is_empty() {
        return;
    }
    if !allow_nonexact_resume() {
        eprintln!(
            "exact resume refused: {}. Set SCIAGENT_ALLOW_NONEXACT_RESUME=1 only for an intentional branch experiment.",
            mismatches.join(", ")
        );
        std::process::exit(1);
    }
    eprintln!(
        "WARNING: non-exact resume explicitly allowed: {}",
        mismatches.join(", ")
    );
}
'''


def patch_example(text: str) -> str:
    if "fn enforce_exact_resume" in text:
        raise SystemExit("example already B37 patched")
    # Add shared exact-resume helpers after env_usize.
    marker = '''fn main() {\n'''
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("missing main")
    text = text[:pos] + HELPERS + "\n" + text[pos:]

    # Collapse the first B/T/corpus mismatch gate onto shared policy.
    old_gate = '''        if !mismatches.is_empty()\n        {\n            let allow = matches!(\n                std::env::var("SCIAGENT_ALLOW_NONEXACT_RESUME").as_deref(),\n                Ok("1" | "true")\n            );\n            if !allow\n            {\n                eprintln!(\n                    "exact resume refused: {}. Set SCIAGENT_ALLOW_NONEXACT_RESUME=1 only for an intentional branch experiment.",\n                    mismatches.join(", ")\n                );\n                std::process::exit(1);\n            }\n            eprintln!(\n                "WARNING: non-exact resume explicitly allowed: {}",\n                mismatches.join(", ")\n            );\n        }\n'''
    text = must_replace(text, old_gate, '''        enforce_exact_resume(&mismatches);\n''', 1)

    # A checkpoint already at its saved target must not silently grow by the old
    # additive SCIAGENT_STEPS default. Extending the cosine horizon is an intentional
    # branch experiment and therefore requires an explicit target.
    old_total = '''    let steps_env = env_usize("SCIAGENT_STEPS", 300);\n    let total_steps = std::env::var("SCIAGENT_TOTAL_STEPS")\n        .ok()\n        .and_then(|v| v.parse::<usize>().ok())\n        .filter(|&t| t > start_step)\n        .or_else(|| {\n            optimizer_resume\n                .as_ref()\n                .map(|s| s.total_steps)\n                .filter(|&t| t > start_step)\n        })\n        .unwrap_or(start_step + steps_env);\n'''
    new_total = '''    let steps_env = env_usize("SCIAGENT_STEPS", 300);\n    let explicit_total_steps = std::env::var("SCIAGENT_TOTAL_STEPS")\n        .ok()\n        .and_then(|v| v.parse::<usize>().ok());\n    if explicit_total_steps.is_none()\n    {\n        if let Some(saved) = optimizer_resume.as_ref()\n        {\n            if saved.total_steps <= start_step\n            {\n                eprintln!(\n                    "checkpoint step {start_step} already reached saved target {}; set \\\n                     SCIAGENT_TOTAL_STEPS to a larger value plus \\\n                     SCIAGENT_ALLOW_NONEXACT_RESUME=1 to intentionally extend the run",\n                    saved.total_steps\n                );\n                std::process::exit(0);\n            }\n        }\n    }\n    let total_steps = explicit_total_steps\n        .filter(|&t| t > start_step)\n        .or_else(|| optimizer_resume.as_ref().map(|s| s.total_steps).filter(|&t| t > start_step))\n        .unwrap_or(start_step + steps_env);\n'''
    text = must_replace(text, old_total, new_total, 1)

    # Inherit all trajectory-changing knobs. Explicit env overrides are allowed only
    # through the exact-contract mismatch gate inserted below.
    old_clip_val_shuffle = '''    // Global grad-norm clip (default 1.0; SCIAGENT_CLIP overrides, <= 0 disables).\n    let max_grad_norm = std::env::var("SCIAGENT_CLIP")\n        .ok()\n        .and_then(|v| v.parse().ok())\n        .unwrap_or(1.0f32);\n    // AdamW epsilon (default 1e-5, bf16-appropriate; SCIAGENT_EPS overrides).\n    let adam_eps = std::env::var("SCIAGENT_EPS")\n        .ok()\n        .and_then(|v| v.parse().ok())\n        .or_else(|| optimizer_resume.as_ref().map(|s| s.adam_eps))\n        .unwrap_or(1e-5f32);\n    // Held-out validation fraction (tail; default 2%; SCIAGENT_VAL_FRAC overrides, 0 disables).\n    let val_frac = std::env::var("SCIAGENT_VAL_FRAC")\n        .ok()\n        .and_then(|v| v.parse().ok())\n        .unwrap_or(0.02f32);\n    // Checkpoint cadence + retention. Saving every 100 steps and never pruning fills\n    // the disk on a long run (each 350M checkpoint is ~1.2 GB fp32) — so save less\n    // often and keep only the last few (SCIAGENT_KEEP) plus the best-val one.\n    let save_interval = env_usize("SCIAGENT_SAVE", 500);\n    let keep_last = env_usize("SCIAGENT_KEEP", 3);\n    // Shuffle training windows (default on; SCIAGENT_SHUFFLE=0 restores sequential\n    // streaming). Deterministic per (start_step, epoch), so runs stay reproducible.\n    let shuffle = !matches!(\n        std::env::var("SCIAGENT_SHUFFLE").as_deref(),\n        Ok("0" | "false")\n    );\n'''
    new_clip_val_shuffle = '''    // Trajectory-changing settings inherit the saved run contract by default.\n    let explicit_clip = std::env::var("SCIAGENT_CLIP")\n        .ok()\n        .and_then(|v| v.parse::<f32>().ok());\n    let max_grad_norm = explicit_clip\n        .or_else(|| optimizer_resume.as_ref().and_then(|s| s.max_grad_norm))\n        .unwrap_or(1.0f32);\n    let explicit_eps = std::env::var("SCIAGENT_EPS")\n        .ok()\n        .and_then(|v| v.parse::<f32>().ok());\n    let adam_eps = explicit_eps\n        .or_else(|| optimizer_resume.as_ref().map(|s| s.adam_eps))\n        .unwrap_or(1e-5f32);\n    let explicit_val_frac = std::env::var("SCIAGENT_VAL_FRAC")\n        .ok()\n        .and_then(|v| v.parse::<f32>().ok());\n    let val_frac = explicit_val_frac\n        .or_else(|| optimizer_resume.as_ref().and_then(|s| s.val_frac))\n        .unwrap_or(0.02f32);\n    // Checkpoint/telemetry cadence does not alter model math.\n    let save_interval = env_usize("SCIAGENT_SAVE", 500);\n    let keep_last = env_usize("SCIAGENT_KEEP", 3);\n    let explicit_shuffle = std::env::var("SCIAGENT_SHUFFLE").ok().map(|v| {\n        !matches!(v.as_str(), "0" | "false")\n    });\n    let shuffle = explicit_shuffle\n        .or_else(|| optimizer_resume.as_ref().and_then(|s| s.shuffle))\n        .unwrap_or(true);\n'''
    text = must_replace(text, old_clip_val_shuffle, new_clip_val_shuffle, 1)

    # Gate every saved mathematical/sampling setting after final values are known.
    insert_marker = '''    let resume_betas = optimizer_resume.as_ref().map(|s| s.betas);\n'''
    pos2 = text.find(insert_marker)
    if pos2 < 0:
        raise SystemExit("missing config insertion point")
    contract = r'''    if let Some(saved) = optimizer_resume.as_ref() {
        let mut mismatches = Vec::new();
        if total_steps != saved.total_steps {
            mismatches.push(format!(
                "total_steps saved={} current={total_steps}", saved.total_steps
            ));
        }
        if warmup_steps != saved.warmup_steps {
            mismatches.push(format!(
                "warmup_steps saved={} current={warmup_steps}", saved.warmup_steps
            ));
        }
        if !same_f32(base_lr, saved.base_lr) {
            mismatches.push(format!("base_lr saved={} current={base_lr}", saved.base_lr));
        }
        if !same_f32(min_lr, saved.min_lr) {
            mismatches.push(format!("min_lr saved={} current={min_lr}", saved.min_lr));
        }
        if !same_f32(adam_eps, saved.adam_eps) {
            mismatches.push(format!("adam_eps saved={} current={adam_eps}", saved.adam_eps));
        }
        if let Some(v) = saved.max_grad_norm {
            if !same_f32(max_grad_norm, v) {
                mismatches.push(format!("clip saved={v} current={max_grad_norm}"));
            }
        }
        if let Some(v) = saved.val_frac {
            if !same_f32(val_frac, v) {
                mismatches.push(format!("val_frac saved={v} current={val_frac}"));
            }
        }
        if let Some(v) = saved.shuffle {
            if shuffle != v {
                mismatches.push(format!("shuffle saved={v} current={shuffle}"));
            }
        }
        enforce_exact_resume(&mismatches);
    }

'''
    text = text[:pos2] + contract + text[pos2:]
    # Update stale comments.
    text = text.replace(
        "// Exact B32 resumes inherit the saved optimizer/LR trajectory by default. An\n    // explicit environment override still wins.",
        "// Exact resumes inherit the saved optimizer/LR trajectory. Explicit changes\n    // are treated as branch experiments and require SCIAGENT_ALLOW_NONEXACT_RESUME=1.",
    )
    return text


MODEL.write_text(patch_model(MODEL.read_text()))
EXAMPLE.write_text(patch_example(EXAMPLE.read_text()))
print("patched B37 exact run contract")
