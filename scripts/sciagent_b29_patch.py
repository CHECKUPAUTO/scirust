#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "scirust-cuda/src/chain.rs"
MODEL = ROOT / "scirust-sciagent/src/cuda_model.rs"


def must_replace(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"expected {count}, found {n}: {old[:120]!r}")
    return text.replace(old, new, count)


def replace_fn(text: str, marker: str, new_src: str) -> str:
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"missing {marker!r}")
    brace = text.find("{", start)
    depth = 0
    in_str = False
    esc = False
    for i in range(brace, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[:start] + new_src.rstrip() + text[i + 1 :]
    raise SystemExit(f"unterminated {marker}")


COL_KERNEL = r'''
// Copy a narrow contiguous matrix into a disjoint column range of a preallocated
// output. Unlike place_cols, this does not materialize a full zero-padded d_model
// matrix per head. It is the primitive behind allocation-light head assembly.
extern "C" __global__ void place_cols_into_kernel(
    unsigned short* out, const unsigned short* x,
    const size_t rows, const size_t ncols, const size_t col_start, const size_t dst_cols)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * ncols) {
        const size_t r = idx / ncols, c = idx % ncols;
        out[r * dst_cols + col_start + c] = x[idx];
    }
}
'''


def patch_chain(text: str) -> str:
    if "place_cols_into_kernel" in text:
        raise SystemExit("chain already B29 patched")
    text = must_replace(
        text,
        "// Row slicing/placement for true B×T training. Projection/MLP/head GEMMs operate\n",
        COL_KERNEL + "\n// Row slicing/placement for true B×T training. Projection/MLP/head GEMMs operate\n",
    )
    text = must_replace(
        text,
        "    place_cols: CudaFunction,\n    slice_rows: CudaFunction,\n",
        "    place_cols: CudaFunction,\n    place_cols_into: CudaFunction,\n    slice_rows: CudaFunction,\n",
    )
    text = must_replace(
        text,
        '            place_cols: f("place_cols_kernel"),\n            slice_rows: f("slice_rows_kernel"),\n',
        '            place_cols: f("place_cols_kernel"),\n            place_cols_into: f("place_cols_into_kernel"),\n            slice_rows: f("slice_rows_kernel"),\n',
    )
    marker = "    /// Copy a contiguous row range into a new resident matrix.\n"
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("missing concat-cols insertion point")
    method = r'''    /// Concatenate equal-height matrices along columns into one resident matrix.
    /// Each part writes a disjoint range, avoiding the old place_cols+add chain.
    pub fn concat_cols(&self, parts: &[&CudaMatrix]) -> CudaMatrix {
        assert!(!parts.is_empty(), "concat_cols: empty parts");
        let rows = parts[0].rows;
        assert!(parts.iter().all(|p| p.rows == rows), "concat_cols: row mismatch");
        let cols: usize = parts.iter().map(|p| p.cols).sum();
        let mut out = self
            .stream
            .alloc_zeros::<bf16>(rows * cols)
            .expect("cuda alloc concat cols");
        let (rows_a, dst_cols_a) = (rows, cols);
        let mut col_start = 0usize;
        for p in parts {
            let (ncols_a, start_a) = (p.cols, col_start);
            let mut builder = self.stream.launch_builder(&self.kernels().place_cols_into);
            builder.arg(&mut out);
            builder.arg(&p.buf);
            builder.arg(&rows_a);
            builder.arg(&ncols_a);
            builder.arg(&start_a);
            builder.arg(&dst_cols_a);
            // SAFETY: all parts have the same row count and disjoint destination cols.
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems((rows * p.cols) as u32))
                    .expect("launch place_cols_into_kernel");
            }
            col_start += p.cols;
        }
        CudaMatrix { buf: out, rows, cols }
    }

'''
    return text[:pos] + method + text[pos:]


ATTENTION = r'''    fn attention(&self, q: &CudaMatrix, k: &CudaMatrix, v: &CudaMatrix) -> CudaMatrix {
        let dh = self.d_model / self.n_heads;
        let seq = q.rows();
        let qr = self.chain.rope(q, seq, 0, self.theta);
        let kr = self.chain.rope(k, seq, 0, self.theta);
        let repeat = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut heads = Vec::with_capacity(self.n_heads);
        for head in 0..self.n_heads {
            let kv = head / repeat;
            let qs = self.chain.slice_cols(&qr, head * dh, dh);
            let ks = self.chain.slice_cols(&kr, kv * dh, dh);
            let vs = self.chain.slice_cols(v, kv * dh, dh);
            let scores = self.chain.matmul_bt(&qs, &ks);
            let scaled = self.chain.scale_causal_mask(&scores, scale, self.causal);
            let weights = self.chain.softmax(&scaled);
            heads.push(self.chain.matmul(&weights, &vs));
        }
        let refs: Vec<&CudaMatrix> = heads.iter().collect();
        self.chain.concat_cols(&refs)
    }'''

ATTN_BWD = r'''    fn attention_backward(
        &self,
        q: &CudaMatrix,
        k: &CudaMatrix,
        v: &CudaMatrix,
        dout: &CudaMatrix,
    ) -> (CudaMatrix, CudaMatrix, CudaMatrix) {
        let ch = &self.chain;
        let dh = self.d_model / self.n_heads;
        let seq = q.rows();
        let qr = ch.rope(q, seq, 0, self.theta);
        let kr = ch.rope(k, seq, 0, self.theta);
        let repeat = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut dq_heads = Vec::with_capacity(self.n_heads);
        let mut dk_kv: Vec<Option<CudaMatrix>> = (0..self.n_kv_heads).map(|_| None).collect();
        let mut dv_kv: Vec<Option<CudaMatrix>> = (0..self.n_kv_heads).map(|_| None).collect();
        for head in 0..self.n_heads {
            let kv = head / repeat;
            let qs = ch.slice_cols(&qr, head * dh, dh);
            let ks = ch.slice_cols(&kr, kv * dh, dh);
            let vs = ch.slice_cols(v, kv * dh, dh);
            let scores = ch.matmul_bt(&qs, &ks);
            let scaled = ch.scale_causal_mask(&scores, scale, self.causal);
            let weights = ch.softmax(&scaled);
            let d_ctx = ch.slice_cols(dout, head * dh, dh);
            let dweights = ch.matmul_bt(&d_ctx, &vs);
            let dvs = ch.matmul_at(&weights, &d_ctx);
            let dscaled = ch.softmax_backward(&weights, &dweights);
            let dscores = ch.scale_causal_mask_backward(&dscaled, scale, self.causal);
            let dqs = ch.matmul(&dscores, &ks);
            let dks = ch.matmul_at(&dscores, &qs);
            dq_heads.push(dqs);
            dk_kv[kv] = Some(match dk_kv[kv].take() {
                None => dks,
                Some(acc) => ch.add(&acc, &dks),
            });
            dv_kv[kv] = Some(match dv_kv[kv].take() {
                None => dvs,
                Some(acc) => ch.add(&acc, &dvs),
            });
        }
        let dq_refs: Vec<&CudaMatrix> = dq_heads.iter().collect();
        let dk_refs: Vec<&CudaMatrix> = dk_kv.iter().map(|x| x.as_ref().expect("kv grad")).collect();
        let dv_refs: Vec<&CudaMatrix> = dv_kv.iter().map(|x| x.as_ref().expect("kv grad")).collect();
        let dqr = ch.concat_cols(&dq_refs);
        let dkr = ch.concat_cols(&dk_refs);
        let dv = ch.concat_cols(&dv_refs);
        let dq = ch.rope_backward(&dqr, seq, 0, self.theta);
        let dk = ch.rope_backward(&dkr, seq, 0, self.theta);
        (dq, dk, dv)
    }'''

ATTN_TRAIN_SEQ = r'''    fn attention_train_sequence(
        &self,
        q: &CudaMatrix,
        k: &CudaMatrix,
        v: &CudaMatrix,
    ) -> (CudaMatrix, CudaAttentionSequenceCache) {
        let ch = &self.chain;
        let dh = self.d_model / self.n_heads;
        let seq = q.rows();
        let qr = ch.rope(q, seq, 0, self.theta);
        let kr = ch.rope(k, seq, 0, self.theta);
        let repeat = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut heads = Vec::with_capacity(self.n_heads);
        let mut weights_all = Vec::with_capacity(self.n_heads);
        for head in 0..self.n_heads {
            let kv = head / repeat;
            let qs = ch.slice_cols(&qr, head * dh, dh);
            let ks = ch.slice_cols(&kr, kv * dh, dh);
            let vs = ch.slice_cols(v, kv * dh, dh);
            let scores = ch.matmul_bt(&qs, &ks);
            let scaled = ch.scale_causal_mask(&scores, scale, self.causal);
            let weights = ch.softmax(&scaled);
            heads.push(ch.matmul(&weights, &vs));
            weights_all.push(weights);
        }
        let refs: Vec<&CudaMatrix> = heads.iter().collect();
        (
            ch.concat_cols(&refs),
            CudaAttentionSequenceCache { qr, kr, weights: weights_all },
        )
    }'''

ATTN_CACHED_BWD = r'''    fn attention_backward_sequence_cached(
        &self,
        v: &CudaMatrix,
        dout: &CudaMatrix,
        cache: &CudaAttentionSequenceCache,
    ) -> (CudaMatrix, CudaMatrix, CudaMatrix) {
        let ch = &self.chain;
        let dh = self.d_model / self.n_heads;
        let seq = cache.qr.rows();
        let repeat = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut dq_heads = Vec::with_capacity(self.n_heads);
        let mut dk_kv: Vec<Option<CudaMatrix>> = (0..self.n_kv_heads).map(|_| None).collect();
        let mut dv_kv: Vec<Option<CudaMatrix>> = (0..self.n_kv_heads).map(|_| None).collect();
        for head in 0..self.n_heads {
            let kv = head / repeat;
            let qs = ch.slice_cols(&cache.qr, head * dh, dh);
            let ks = ch.slice_cols(&cache.kr, kv * dh, dh);
            let vs = ch.slice_cols(v, kv * dh, dh);
            let weights = &cache.weights[head];
            let d_ctx = ch.slice_cols(dout, head * dh, dh);
            let dweights = ch.matmul_bt(&d_ctx, &vs);
            let dvs = ch.matmul_at(weights, &d_ctx);
            let dscaled = ch.softmax_backward(weights, &dweights);
            let dscores = ch.scale_causal_mask_backward(&dscaled, scale, self.causal);
            let dqs = ch.matmul(&dscores, &ks);
            let dks = ch.matmul_at(&dscores, &qs);
            dq_heads.push(dqs);
            dk_kv[kv] = Some(match dk_kv[kv].take() {
                None => dks,
                Some(acc) => ch.add(&acc, &dks),
            });
            dv_kv[kv] = Some(match dv_kv[kv].take() {
                None => dvs,
                Some(acc) => ch.add(&acc, &dvs),
            });
        }
        let dq_refs: Vec<&CudaMatrix> = dq_heads.iter().collect();
        let dk_refs: Vec<&CudaMatrix> = dk_kv.iter().map(|x| x.as_ref().expect("kv grad")).collect();
        let dv_refs: Vec<&CudaMatrix> = dv_kv.iter().map(|x| x.as_ref().expect("kv grad")).collect();
        let dqr = ch.concat_cols(&dq_refs);
        let dkr = ch.concat_cols(&dk_refs);
        let dv = ch.concat_cols(&dv_refs);
        let dq = ch.rope_backward(&dqr, seq, 0, self.theta);
        let dk = ch.rope_backward(&dkr, seq, 0, self.theta);
        (dq, dk, dv)
    }'''


def patch_model(text: str) -> str:
    if "allocation-light head assembly" in text:
        raise SystemExit("model already B29 patched")
    text = replace_fn(text, "    fn attention(&self, q: &CudaMatrix", ATTENTION)
    text = replace_fn(text, "    fn attention_backward(\n", ATTN_BWD)
    text = replace_fn(text, "    fn attention_train_sequence(\n", ATTN_TRAIN_SEQ)
    text = replace_fn(text, "    fn attention_backward_sequence_cached(\n", ATTN_CACHED_BWD)
    # q/k are only needed transiently in forward; the cached RoPE forms live in the
    # attention cache and are sufficient for the VJP.
    text = must_replace(text, "    q: CudaMatrix,\n    k: CudaMatrix,\n    v: CudaMatrix,\n", "    v: CudaMatrix,\n", 1)
    text = must_replace(text, "                q,\n                k,\n                v,\n", "                v,\n", 1)
    # Mark the optimization in a nearby cache comment for idempotence/readability.
    text = must_replace(
        text,
        "/// Training cache for one sequence's attention.\n",
        "/// Training cache for one sequence's attention; narrow heads are assembled with\n/// allocation-light head assembly rather than full-width padding and addition.\n",
        1,
    )
    return text


CHAIN.write_text(patch_chain(CHAIN.read_text()))
MODEL.write_text(patch_model(MODEL.read_text()))
print("patched B29 attention assembly")
