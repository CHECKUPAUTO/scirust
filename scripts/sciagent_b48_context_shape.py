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
    # There are two occurrences: full attention and incremental_attention. Only the
    # incremental one must change. Split at incremental_attention and patch there.
    marker = '''    fn incremental_attention(\n'''
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("missing incremental_attention")
    before, after = text[:pos], text[pos:]
    if after.count(old) < 1:
        raise SystemExit("missing incremental context matmul")
    new = '''            let weights = ch.softmax(&scaled);\n            // cuBLASLt may choose a different reduction algorithm for the cached\n            // `1×T · T×dh` GEMM than for the full-forward `T×T · T×dh` GEMM. B47\n            // proved every value through softmax is bit-identical and the first\n            // numerical drift appears here. Preserve the full-forward matrix shape\n            // and keep only its last row; zero earlier rows cannot affect that row.\n            // This is the shape-stable context GEMM correctness path.\n            if weights.rows() == 1 && vs.rows() > 1\n            {\n                let pad = ch.zeros_bf16(vs.rows() - 1, weights.cols());\n                let padded_weights = ch.concat_rows(&[&pad, &weights]);\n                let full_shape_ctx = ch.matmul(&padded_weights, &vs);\n                heads.push(ch.slice_rows(&full_shape_ctx, vs.rows() - 1, 1));\n            }\n            else\n            {\n                heads.push(ch.matmul(&weights, &vs));\n            }\n'''
    after = after.replace(old, new, 1)
    return before + after


def patch_route(text: str) -> str:
    if "B48 — shape-stable cached context" in text:
        return text
    return text.rstrip() + '''\n\n### B48 — shape-stable cached context\n\nThor B47 localized the production KV-cache divergence exactly: cached K/V, QKᵀ\nscores, scaling and softmax were bit-identical to the same-prefix full-forward path,\nwhile the first non-zero error appeared in `weights · V`. cuBLASLt selected different\nreduction paths for `1×T · T×dh` and `T×T · T×dh`; the ~0.285% context error then\namplified across 24 blocks to ~18% in final logits. The cached correctness path now\nexecutes that context product with the full-forward `T×T · T×dh` output shape using\nzero padding and retains only the last row. This deliberately prioritizes exact model\nsemantics; the Thor benchmark decides whether a later dedicated deterministic GEMV\nkernel is needed to recover the O(T) context-product cost without reintroducing drift.\n'''


CHAIN.write_text(patch_chain(CHAIN.read_text()))
MODEL.write_text(patch_model(MODEL.read_text()))
ROUTE.write_text(patch_route(ROUTE.read_text()))
print("B48 patched: shape-stable cached attention context GEMM")
