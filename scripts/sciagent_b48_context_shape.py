#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "scirust-cuda/src/chain.rs"
MODEL = ROOT / "scirust-sciagent/src/cuda_model.rs"
ROUTE = ROOT / "scirust-sciagent/ROUTE_B.md"


def must_replace(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"expected {count}, found {n}: {old[:180]!r}")
    return text.replace(old, new, count)


def patch_chain(text: str) -> str:
    if "pub fn zeros_bf16" in text:
        return text
    anchor = '''    /// Upload an fp32 vector to VRAM **without** rounding (master-weight / moment\n'''
    method = '''    /// Allocate an all-zero resident bf16 matrix. Used by the exact cached-attention\n    /// parity path to preserve the full-forward cuBLASLt output shape without a\n    /// host allocation or transfer.\n    pub fn zeros_bf16(&self, rows: usize, cols: usize) -> CudaMatrix {\n        let buf = self\n            .stream\n            .alloc_zeros::<bf16>(rows.saturating_mul(cols))\n            .expect("cuda alloc bf16 zeros");\n        CudaMatrix { buf, rows, cols }\n    }\n\n'''
    return must_replace(text, anchor, method + anchor)


def patch_model(text: str) -> str:
    if "shape-stable context GEMM" in text:
        return text
    old = '''            let weights = ch.softmax(&scaled);\n            heads.push(ch.matmul(&weights, &vs));\n'''
    marker = '''    fn incremental_attention(\n'''
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("missing incremental_attention")
    before, after = text[:pos], text[pos:]
    if after.count(old) < 1:
        raise SystemExit("missing incremental context matmul")
    new = '''            let weights = ch.softmax(&scaled);\n            // B47 proved the cached K/V, scores, scaling and softmax are bit-identical\n            // to the same-prefix full forward. The first drift is the context GEMM:\n            // cuBLASLt selects a different reduction for `1×T · T×dh` than for\n            // `T×T · T×dh`. Preserve the full output shape and retain its last row.\n            // Zero earlier rows cannot affect the final row.\n            if weights.rows() == 1 && vs.rows() > 1\n            {\n                let pad = ch.zeros_bf16(vs.rows() - 1, weights.cols());\n                let padded_weights = ch.concat_rows(&[&pad, &weights]);\n                let full_shape_ctx = ch.matmul(&padded_weights, &vs);\n                heads.push(ch.slice_rows(&full_shape_ctx, vs.rows() - 1, 1));\n            }\n            else\n            {\n                heads.push(ch.matmul(&weights, &vs));\n            }\n'''
    after = after.replace(old, new, 1)
    return before + after


def patch_route(text: str) -> str:
    if "B48 — shape-stable cached context" in text:
        return text
    return text.rstrip() + '''\n\n### B48 — shape-stable cached context\n\nThor B47 localized the production KV-cache divergence exactly: cached K/V, QKᵀ\nscores, scaling and softmax were bit-identical to the same-prefix full-forward path,\nwhile the first non-zero error appeared in `weights · V`. The cached correctness\npath therefore preserves the full-forward `T×T · T×dh` cuBLASLt output shape using\nzero padding and retains only the last row. This prioritizes exact model semantics;\nThor benchmarking determines whether a dedicated deterministic O(T) context kernel\nis needed afterward.\n'''


CHAIN.write_text(patch_chain(CHAIN.read_text()))
MODEL.write_text(patch_model(MODEL.read_text()))
ROUTE.write_text(patch_route(ROUTE.read_text()))
print("B48 patched: shape-stable cached attention context GEMM")
