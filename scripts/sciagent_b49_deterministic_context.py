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
// in fp32. Therefore row r is invariant to the total number of other output rows and,
// when future causal weights are exact zero, invariant to appending future positions.
// This is intentionally shared by full inference and incremental KV decode.
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
    # Insert public method before scale_causal_mask.
    marker='''    /// Scale scores and optionally apply the upper-triangular causal mask.\n'''
    method='''    /// Deterministic row-local attention context `weights · values`.\n    /// `weights` is `rows × seq`; `values` is `seq × dim`. Each output element\n    /// accumulates `j=0..seq` in a fixed fp32 order, so a cached one-row query and\n    /// the corresponding row of a full causal forward are numerically identical.\n    pub fn attention_context(&self, weights: &CudaMatrix, values: &CudaMatrix) -> CudaMatrix {\n        assert_eq!(weights.cols, values.rows, "attention_context: seq mismatch");\n        let rows = weights.rows;\n        let seq = weights.cols;\n        let dim = values.cols;\n        let mut out = self\n            .stream\n            .alloc_zeros::<bf16>(rows * dim)\n            .expect("cuda alloc attention context");\n        let (rows_a, seq_a, dim_a) = (rows, seq, dim);\n        let mut builder = self.stream.launch_builder(&self.kernels().attention_context);\n        builder.arg(&mut out);\n        builder.arg(&weights.buf);\n        builder.arg(&values.buf);\n        builder.arg(&rows_a);\n        builder.arg(&seq_a);\n        builder.arg(&dim_a);\n        unsafe {\n            builder\n                .launch(LaunchConfig::for_num_elems((rows * dim) as u32))\n                .expect("launch attention_context_kernel");\n        }\n        CudaMatrix { buf: out, rows, cols: dim }\n    }\n\n'''
    s=rep(s,marker,method+marker)
    return s


def patch_model(s):
    # Full inference path: replace exactly the first context GEMM inside attention().
    full='''            let weights = self.chain.softmax(&scaled);\n            heads.push(self.chain.matmul(&weights, &vs));\n'''
    if full in s:
        s=rep(s,full,'''            let weights = self.chain.softmax(&scaled);\n            heads.push(self.chain.attention_context(&weights, &vs));\n''')
    elif "heads.push(self.chain.attention_context(&weights, &vs));" not in s:
        raise SystemExit("full attention context anchor missing")

    # Incremental path: replace B48 shape-padding block with same deterministic op.
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
    s=s[:start]+block+s[end:]
    return s


def patch_route(s):
    if "B49 — deterministic attention context" in s: return s
    return s.rstrip()+'''\n\n### B49 — deterministic attention context\n\nThor B47 proved cached K/V, QKᵀ scores, scale and softmax were bit-identical and\nlocalized the first divergence to `weights · V`. B48 demonstrated that forcing the\ncuBLASLt full matrix shape can remove the current-row error, but it destroys the KV\ncache complexity advantage and cannot make historical cached deeper-layer rows\nindependent of later full-forward matrix shapes. B49 replaces that operation in CUDA\ninference with one shared row-local kernel: each output element accumulates positions\nleft-to-right in fp32 and rounds once to bf16. Full causal inference and incremental\ndecode therefore use the same accumulation order for the same row, while cached\ndecode remains O(T·d_head) rather than padding to O(T²·d_head).\n'''

CHAIN.write_text(patch_chain(CHAIN.read_text()))
MODEL.write_text(patch_model(MODEL.read_text()))
ROUTE.write_text(patch_route(ROUTE.read_text()))
print("B49 patched: deterministic shared attention context kernel")
