#!/usr/bin/env python3
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "scirust-gpu/src/ops.rs"
CARGO = ROOT / "scirust-sciagent/Cargo.toml"
NATIVE = ROOT / ".github/workflows/native-arm64.yml"


def must_replace(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"expected {count}, found {n}: {old[:180]!r}")
    return text.replace(old, new, count)


def apply_prior(script: str, marker_file: Path, marker: str) -> None:
    if marker in marker_file.read_text():
        print(f"{script}: already applied")
        return
    path = ROOT / "scripts" / script
    if not path.exists():
        raise SystemExit(f"missing required staged patch {path}")
    subprocess.check_call([sys.executable, str(path)], cwd=ROOT)


# B40/B41/B42 were staged separately because each had its own validation intent.
# Apply them deterministically now that the stale WGPU oracle has been identified.
apply_prior("sciagent_b40_patch.py", ROOT / "scirust-sciagent/src/cuda_model.rs", "GQA-correct RoPE without per-head slicing")
apply_prior("sciagent_b41_patch.py", ROOT / "scirust-sciagent/src/cuda_model.rs", "save_interval_seconds")
apply_prior("sciagent_b42_patch.py", ROOT / "scirust-sciagent/src/cuda_model.rs", "SCIAGENT_MODEL_SEMANTICS_VERSION")

# B33 intentionally changed the GQA model math to head-local RoPE, but the CPU
# oracle in scirust-gpu still encoded the historical full-projection-width basis.
# The production WGPU forward/backward already calls rope_heads/rope_heads_backward;
# make its independent CPU oracle describe the same v2 model semantics.
ops = OPS.read_text()
ops = must_replace(
    ops,
    """/// math exactly: full-width RoPE on `q` and `k` (each using **its own width** in\n/// the frequency, as the model's `rope_on_tape` does — `d_model` for `q`,\n/// `kv_dim` for `k`), then per head `kv = head / (n_heads/n_kv_heads)`,\n""",
    """/// math exactly: head-local RoPE on `q` and `k`, with the frequency index\n/// restarting inside every `dh` block so a query head and its shared KV head use\n/// the same rotary basis, then per head `kv = head / (n_heads/n_kv_heads)`,\n""",
)
ops = must_replace(
    ops,
    """    let d_model = n_heads * dh;\n    let kv_dim = n_kv_heads * dh;\n    let qr = cpu_rope(q, rows, d_model, seq_len, 0, theta);\n    let kr = cpu_rope(k, rows, kv_dim, seq_len, 0, theta);\n    let repeat = n_heads / n_kv_heads;\n""",
    """    let d_model = n_heads * dh;\n    let kv_dim = n_kv_heads * dh;\n    let mut qr = vec![0.0f32; rows * d_model];\n    for head in 0..n_heads\n    {\n        let raw = cpu_slice_cols(q, rows, d_model, head * dh, dh);\n        let rotated = cpu_rope(&raw, rows, dh, seq_len, 0, theta);\n        for r in 0..rows\n        {\n            qr[r * d_model + head * dh..r * d_model + (head + 1) * dh]\n                .copy_from_slice(&rotated[r * dh..(r + 1) * dh]);\n        }\n    }\n    let mut kr = vec![0.0f32; rows * kv_dim];\n    for head in 0..n_kv_heads\n    {\n        let raw = cpu_slice_cols(k, rows, kv_dim, head * dh, dh);\n        let rotated = cpu_rope(&raw, rows, dh, seq_len, 0, theta);\n        for r in 0..rows\n        {\n            kr[r * kv_dim + head * dh..r * kv_dim + (head + 1) * dh]\n                .copy_from_slice(&rotated[r * dh..(r + 1) * dh]);\n        }\n    }\n    let repeat = n_heads / n_kv_heads;\n""",
)
OPS.write_text(ops)

# Portable workspace builds must not compile a CUDA-only example with default
# features. Keep this idempotent because the direct Git-object fix may already be
# present when this script executes.
cargo = CARGO.read_text()
if 'name = "cuda_production_bench"' not in cargo:
    anchor = '''# Route B generation from a trained checkpoint — CUDA-only.\n[[example]]\nname = "cuda_generate"\nrequired-features = ["cuda"]\n'''
    block = '''# Route B production throughput/ETA benchmark — CUDA-only.\n[[example]]\nname = "cuda_production_bench"\nrequired-features = ["cuda"]\n\n'''
    cargo = must_replace(cargo, anchor, block + anchor)
    CARGO.write_text(cargo)

# Restore the production native workflow immediately; this one-shot step must not
# survive in the commit it creates.
native = NATIVE.read_text()
native = must_replace(native, "permissions:\n  contents: write\n", "permissions:\n  contents: read\n")
start = native.find("      # BEGIN SCIAGENT B43 ONE-SHOT AUTOFIX\n")
end_marker = "      # END SCIAGENT B43 ONE-SHOT AUTOFIX\n"
if start < 0:
    raise SystemExit("missing B43 workflow marker")
end = native.find(end_marker, start)
if end < 0:
    raise SystemExit("missing B43 workflow end marker")
end += len(end_marker)
native = native[:start] + native[end:]
NATIVE.write_text(native)

# Remove every staging/diagnostic artifact. The dedicated Thor production gate is
# intentionally retained; it is a useful permanent focused hardware gate.
for rel in [
    "scripts/sciagent_b40_patch.py",
    "scripts/sciagent_b41_patch.py",
    "scripts/sciagent_b42_patch.py",
    "scripts/sciagent_b43_autofix.py",
    ".github/workflows/sciagent-b40-apply.yml",
    ".github/workflows/sciagent-b41-apply.yml",
    ".github/workflows/sciagent-b42-finalize.yml",
    ".github/workflows/sciagent-final-diagnostics.yml",
    ".github/workflows/sciagent-wgpu-diagnostic.yml",
    ".github/workflows/sciagent-ci-report.yml",
    "SCIAGENT_CI_REPORT.tmp.md",
]:
    p = ROOT / rel
    if p.exists():
        p.unlink()
shutil.rmtree(ROOT / "ci-diagnostics", ignore_errors=True)

print("B43 applied: B40/B41/B42 + v2 WGPU oracle + portable Cargo gate + cleanup")
