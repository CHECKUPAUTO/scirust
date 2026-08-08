#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "scirust-cuda/src/chain.rs"
MODEL = ROOT / "scirust-sciagent/src/cuda_model.rs"
ROUTE = ROOT / "scirust-sciagent/ROUTE_B.md"

def rep(s, old, new, n=1):
    c=s.count(old)
    if c!=n: raise SystemExit(f"expected {n}, found {c}: {old[:160]!r}")
    return s.replace(old,new,n)

def patch_chain(s):
    if "attention_context_kernel" in s:
        return s
    anchor = r'''// Scale a t×t score matrix by `scale`, and (if causal) mask j>i to a large
// negative so softmax drives it to ~0.
'''
    kernel = r'''// Deterministic attention context: out = weights · values.
// One thread owns one output element and accumulates positions strictly left-to-right
// in fp32. Therefore a causal row is invariant to unrelated output rows and to later
// exact-zero masked positions. Shared by full inference and incremental KV decode.
extern "C" __global__ void attention_context_kernel(
    unsigned short* out, const unsigned short* weights, const unsigned short* values,
    const size_t rows, const size_t seq, const size_t dim)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * dim) {
        const size_t r = idx / dim;
        const size_t c = idx % dim;
        float acc = 0.0f;
        for (size_t j = 0; j < seq; ++j)
            acc += b2f(weights[r*seq + j]) * b2f(values[j*dim + c]);
        out[idx] = f2b(acc);
    }
}

'''
    s=rep(s,anchor,kernel+anchor)
    s=rep(s,"    softmax: CudaFunction,\n    scale_mask: CudaFunction,","    softmax: CudaFunction,\n    attention_context: CudaFunction,\n    scale_mask: CudaFunction,")
    s=rep(s,'            softmax: f("softmax_fast_kernel"),\n            scale_mask: f("scale_mask_kernel"),','            softmax: f("softmax_fast_kernel"),\n            attention_context: f("attention_context_kernel"),\n            scale_mask: f("scale_mask_kernel"),')
    marker='''    /// Scale a `t×t` score matrix by `scale` and (optionally) apply the causal\n'''
    method='''    /// Deterministic row-local attention context `weights · values`.\n    /// `weights` is `rows × seq`; `values` is `seq × dim`. Each output element\n    /// accumulates positions in a fixed fp32 order, independent of output row count.\n    pub fn attention_context(&self, weights: &CudaMatrix, values: &CudaMatrix) -> CudaMatrix {\n        assert_eq!(weights.cols, values.rows, "attention_context: seq mismatch");\n        let rows = weights.rows;\n        let seq = weights.cols;\n        let dim = values.cols;\n        let mut out = self.stream.alloc_zeros::<bf16>(rows * dim).expect("cuda alloc attention context");\n        let (rows_a, seq_a, dim_a) = (rows, seq, dim);\n        let mut builder = self.stream.launch_builder(&self.kernels().attention_context);\n        builder.arg(&mut out);\n        builder.arg(&weights.buf);\n        builder.arg(&values.buf);\n        builder.arg(&rows_a);\n        builder.arg(&seq_a);\n        builder.arg(&dim_a);\n        unsafe {\n            builder.launch(LaunchConfig::for_num_elems((rows * dim) as u32))\n                .expect("launch attention_context_kernel");\n        }\n        CudaMatrix { buf: out, rows, cols: dim }\n    }\n\n'''
    s=rep(s,marker,method+marker)
    return s

def patch_model(s):
    full='''            let weights = self.chain.softmax(&scaled);\n            heads.push(self.chain.matmul(&weights, &vs));\n'''
    if full in s:
        s=rep(s,full,'''            let weights = self.chain.softmax(&scaled);\n            heads.push(self.chain.attention_context(&weights, &vs));\n''')
    elif "heads.push(self.chain.attention_context(&weights, &vs));" not in s:
        raise SystemExit("full attention context anchor missing")
    start=s.find("    fn incremental_attention(\n")
    if start<0: raise SystemExit("incremental_attention missing")
    end=s.find("    /// Wide prompt prefill",start)
    block=s[start:end]
    wpos=block.find("            let weights = ch.softmax(&scaled);\n")
    if wpos<0: raise SystemExit("incremental weights anchor missing")
    tail=block.find("        }\n        let refs:",wpos)
    if tail<0: raise SystemExit("incremental tail missing")
    replacement='''            let weights = ch.softmax(&scaled);\n            heads.push(ch.attention_context(&weights, &vs));\n'''
    block=block[:wpos]+replacement+block[tail:]
    return s[:start]+block+s[end:]

def patch_route(s):
    if "B49 — deterministic attention context" in s: return s
    return s.rstrip()+'''\n\n### B49 — deterministic attention context\n\nThor diagnostics localized the first KV-cache divergence to `weights · V`: K/V,\nQKᵀ scores, scaling and softmax were bit-identical. A shared CUDA row-local context\nkernel now accumulates positions left-to-right in fp32 for both full inference and\nincremental decode, making causal rows independent of matrix row count while keeping\ncached decode O(T·d_head).\n'''

CHAIN.write_text(patch_chain(CHAIN.read_text()))
MODEL.write_text(patch_model(MODEL.read_text()))
ROUTE.write_text(patch_route(ROUTE.read_text()))
print("B49 patched: deterministic shared attention context kernel")
