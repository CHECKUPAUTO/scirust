#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTN = ROOT / "scirust-sciagent/src/attention.rs"
GPU_CHAIN = ROOT / "scirust-gpu/src/chain.rs"
GPU_MODEL = ROOT / "scirust-sciagent/src/gpu.rs"
CUDA_MODEL = ROOT / "scirust-sciagent/src/cuda_model.rs"
ROUTE = ROOT / "scirust-sciagent/ROUTE_B.md"


def must_replace(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"expected {count}, found {n}: {old[:150]!r}")
    return text.replace(old, new, count)


def replace_fn(text: str, marker: str, new_src: str) -> str:
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"missing marker {marker!r}")
    brace = text.find("{", start)
    if brace < 0:
        raise SystemExit(f"missing opening brace for {marker!r}")
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
    raise SystemExit(f"unterminated {marker!r}")


CPU_ROPE_TAPE = r'''    fn rope_on_tape<'t>(
        tape: &'t Tape,
        x: Var<'t>,
        seq_len: usize,
        offset: usize,
        theta: f32,
        n_heads: usize,
    ) -> Var<'t> {
        let (rows, dim) = x.shape();
        assert!(n_heads > 0 && dim.is_multiple_of(n_heads));
        let d_head = dim / n_heads;
        assert!(d_head.is_multiple_of(2));
        let pairs_per_head = d_head / 2;
        let mut c = vec![0.0f32; rows * dim];
        let mut s = vec![0.0f32; rows * dim];
        for r in 0..rows {
            let pos = ((r % seq_len) + offset) as f32;
            for head in 0..n_heads {
                for j in 0..pairs_per_head {
                    let freq = theta.powf(-2.0 * j as f32 / d_head as f32);
                    let a = pos * freq;
                    let col = head * d_head + 2 * j;
                    c[r * dim + col] = a.cos();
                    c[r * dim + col + 1] = a.cos();
                    s[r * dim + col] = a.sin();
                    s[r * dim + col + 1] = a.sin();
                }
            }
        }
        // Pair-swap-and-negate remains block diagonal: each adjacent pair is wholly
        // inside one head, so no rotation ever crosses a head boundary.
        let mut w = vec![0.0f32; dim * dim];
        for head in 0..n_heads {
            for j in 0..pairs_per_head {
                let col = head * d_head + 2 * j;
                w[(col + 1) * dim + col] = -1.0;
                w[col * dim + col + 1] = 1.0;
            }
        }
        let c_v = tape.input(Tensor::from_vec(c, rows, dim));
        let s_v = tape.input(Tensor::from_vec(s, rows, dim));
        let w_v = tape.input(Tensor::from_vec(w, dim, dim));
        x.hadamard(c_v).add(x.matmul(w_v).hadamard(s_v))
    }'''

CPU_ROPE_APPLY = r'''    fn rope_apply(t: &Tensor, offset: usize, theta: f32, n_heads: usize) -> Tensor {
        let rows = t.rows;
        let dim = t.cols;
        assert!(n_heads > 0 && dim.is_multiple_of(n_heads));
        let d_head = dim / n_heads;
        assert!(d_head.is_multiple_of(2));
        let pairs_per_head = d_head / 2;
        let mut out = vec![0.0f32; rows * dim];
        for r in 0..rows {
            let pos = (r + offset) as f32;
            for head in 0..n_heads {
                for j in 0..pairs_per_head {
                    let freq = theta.powf(-2.0 * j as f32 / d_head as f32);
                    let a = pos * freq;
                    let c = a.cos();
                    let s = a.sin();
                    let col = head * d_head + 2 * j;
                    let e = t.data[r * dim + col];
                    let o = t.data[r * dim + col + 1];
                    out[r * dim + col] = e * c - o * s;
                    out[r * dim + col + 1] = e * s + o * c;
                }
            }
        }
        Tensor::from_vec(out, rows, dim)
    }'''


def patch_attention(text: str) -> str:
    if "pairs_per_head" in text:
        raise SystemExit("attention.rs already B33 patched")
    text = replace_fn(text, "    fn rope_on_tape<'t>(", CPU_ROPE_TAPE)
    text = replace_fn(text, "    fn rope_apply(", CPU_ROPE_APPLY)
    text = must_replace(
        text,
        "        let qr = Self::rope_on_tape(tape, q, seq_len, 0, self.rope_theta);\n        let kr = Self::rope_on_tape(tape, k, seq_len, 0, self.rope_theta);\n",
        "        let qr = Self::rope_on_tape(\n            tape, q, seq_len, 0, self.rope_theta, self.n_heads,\n        );\n        let kr = Self::rope_on_tape(\n            tape, k, seq_len, 0, self.rope_theta, self.n_kv_heads,\n        );\n",
    )
    text = must_replace(
        text,
        "        let qr = tape.input(Self::rope_apply(&qv, pos, self.rope_theta));\n        let kr = tape.input(Self::rope_apply(&kv, 0, self.rope_theta));\n",
        "        let qr = tape.input(Self::rope_apply(&qv, pos, self.rope_theta, self.n_heads));\n        let kr = tape.input(Self::rope_apply(\n            &kv,\n            0,\n            self.rope_theta,\n            self.n_kv_heads,\n        ));\n",
    )
    text = must_replace(
        text,
        "        let on_tape = GQAAttention::rope_on_tape(&tape, x, rows, 3, 10000.0);\n        let got = tape.value(on_tape.idx());\n        let want = GQAAttention::rope_apply(&Tensor::from_vec(data, rows, dim), 3, 10000.0);\n",
        "        let on_tape = GQAAttention::rope_on_tape(&tape, x, rows, 3, 10000.0, 2);\n        let got = tape.value(on_tape.idx());\n        let want =\n            GQAAttention::rope_apply(&Tensor::from_vec(data, rows, dim), 3, 10000.0, 2);\n",
    )
    insert = r'''

    #[test]
    fn rope_frequency_schedule_restarts_for_each_head() {
        let rows = 3usize;
        let d_head = 4usize;
        let mut data = Vec::with_capacity(rows * d_head * 2);
        for r in 0..rows {
            let head = [
                0.3 + r as f32,
                -0.7 + 0.1 * r as f32,
                1.2 - 0.2 * r as f32,
                0.4 + 0.3 * r as f32,
            ];
            data.extend_from_slice(&head);
            data.extend_from_slice(&head);
        }
        let out = GQAAttention::rope_apply(
            &Tensor::from_vec(data, rows, d_head * 2),
            5,
            10_000.0,
            2,
        );
        for r in 0..rows {
            let a = &out.data[r * d_head * 2..r * d_head * 2 + d_head];
            let b = &out.data[r * d_head * 2 + d_head..(r + 1) * d_head * 2];
            assert_eq!(a, b, "identical heads at one position must rotate identically");
        }
    }
'''
    end = text.rfind("\n}")
    if end < 0:
        raise SystemExit("missing tests module end")
    text = text[:end] + insert + text[end:]
    return text


GPU_ROPE_HEADS = r'''
    /// Apply RoPE independently to each logical attention head. Frequency indices
    /// restart at zero for every `d_head` block, so Q and its shared GQA K head use
    /// the same rotary basis even when their full projection widths differ.
    pub fn rope_heads(
        &self,
        x: &GpuMatrix,
        n_heads: usize,
        seq_len: usize,
        offset: usize,
        theta: f32,
    ) -> BackendResult<GpuMatrix> {
        if n_heads == 0 || !x.cols().is_multiple_of(n_heads) {
            return Err(BackendError::ShapeMismatch(format!(
                "rope_heads: cols {} not divisible by heads {n_heads}",
                x.cols()
            )));
        }
        let dh = x.cols() / n_heads;
        if !dh.is_multiple_of(2) {
            return Err(BackendError::ShapeMismatch(format!(
                "rope_heads: head dim {dh} must be even"
            )));
        }
        let mut out: Option<GpuMatrix> = None;
        for head in 0..n_heads {
            let raw = self.slice_cols(x, head * dh, dh)?;
            let rotated = self.rope(&raw, seq_len, offset, theta)?;
            let padded = self.place_cols(&rotated, head * dh, x.cols())?;
            out = Some(match out {
                None => padded,
                Some(acc) => self.add(&acc, &padded)?,
            });
        }
        out.ok_or_else(|| BackendError::ShapeMismatch("rope_heads: zero heads".into()))
    }

    /// Adjoint of [`Self::rope_heads`], independently undoing each head rotation.
    pub fn rope_heads_backward(
        &self,
        dy: &GpuMatrix,
        n_heads: usize,
        seq_len: usize,
        offset: usize,
        theta: f32,
    ) -> BackendResult<GpuMatrix> {
        if n_heads == 0 || !dy.cols().is_multiple_of(n_heads) {
            return Err(BackendError::ShapeMismatch(format!(
                "rope_heads_backward: cols {} not divisible by heads {n_heads}",
                dy.cols()
            )));
        }
        let dh = dy.cols() / n_heads;
        let mut out: Option<GpuMatrix> = None;
        for head in 0..n_heads {
            let raw = self.slice_cols(dy, head * dh, dh)?;
            let rotated = self.rope_backward(&raw, seq_len, offset, theta)?;
            let padded = self.place_cols(&rotated, head * dh, dy.cols())?;
            out = Some(match out {
                None => padded,
                Some(acc) => self.add(&acc, &padded)?,
            });
        }
        out.ok_or_else(|| BackendError::ShapeMismatch("rope_heads_backward: zero heads".into()))
    }

'''


def patch_gpu_chain(text: str) -> str:
    if "pub fn rope_heads(" in text:
        raise SystemExit("gpu chain already B33 patched")
    marker = "    /// Gather columns `[col_start, col_start+ncols)` of a resident matrix into a\n"
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("missing rope-heads insertion point")
    text = text[:pos] + GPU_ROPE_HEADS + text[pos:]
    text = must_replace(
        text,
        "        let qr = self.rope(q, seq_len, 0, theta)?;\n        let kr = self.rope(k, seq_len, 0, theta)?;\n",
        "        let qr = self.rope_heads(q, n_heads, seq_len, 0, theta)?;\n        let kr = self.rope_heads(k, n_kv_heads, seq_len, 0, theta)?;\n",
        2,
    )
    text = must_replace(
        text,
        "        let dq = self.rope_backward(&dqr, seq_len, 0, theta)?;\n        let dk = self.rope_backward(&dkr, seq_len, 0, theta)?;\n",
        "        let dq = self.rope_heads_backward(&dqr, n_heads, seq_len, 0, theta)?;\n        let dk = self.rope_heads_backward(&dkr, n_kv_heads, seq_len, 0, theta)?;\n",
        1,
    )
    text = text.replace(
        "RoPE is\n    /// applied to the *full-width* `q`/`k` exactly as `rope_on_tape` does — each\n    /// uses its own width in the frequency schedule — so the result matches the\n    /// CPU model.",
        "RoPE is\n    /// applied head-locally: the frequency schedule restarts within every `dh`\n    /// slice, so query heads and their shared KV heads use the same rotary basis.",
    )
    text = text.replace(
        "Finally RoPE's adjoint maps `dqr→dq` and `dkr→dk` (each at its\n    /// own width);",
        "Finally head-local RoPE adjoints map `dqr→dq` and `dkr→dk`;",
    )
    return text


def patch_gpu_model(text: str) -> str:
    if "rope_heads(&k, self.n_kv_heads" in text:
        raise SystemExit("gpu model already B33 patched")
    text = must_replace(
        text,
        "            let kr = chain.rope(&k, p, 0, self.theta).expect(\"rope k\");\n",
        "            let kr = chain\n                .rope_heads(&k, self.n_kv_heads, p, 0, self.theta)\n                .expect(\"head-local rope k\");\n",
    )
    text = must_replace(
        text,
        "            let qr = chain.rope(&q, 1, pos, self.theta).expect(\"rope q\");\n            let kr = chain.rope(&k, 1, pos, self.theta).expect(\"rope k\");\n",
        "            let qr = chain\n                .rope_heads(&q, self.n_heads, 1, pos, self.theta)\n                .expect(\"head-local rope q\");\n            let kr = chain\n                .rope_heads(&k, self.n_kv_heads, 1, pos, self.theta)\n                .expect(\"head-local rope k\");\n",
    )
    text = must_replace(
        text,
        "            let qr = chain.rope(&q, m, start_pos, self.theta).expect(\"rope q\");\n            let kr = chain.rope(&k, m, start_pos, self.theta).expect(\"rope k\");\n",
        "            let qr = chain\n                .rope_heads(&q, self.n_heads, m, start_pos, self.theta)\n                .expect(\"head-local rope q\");\n            let kr = chain\n                .rope_heads(&k, self.n_kv_heads, m, start_pos, self.theta)\n                .expect(\"head-local rope k\");\n",
    )
    text = text.replace(
        "Each width feeds its own frequency schedule,\n            // exactly as `gqa_attention` ropes the full-width q/k.",
        "Frequency indices restart in every head,\n            // exactly as the corrected `gqa_attention` path.",
    )
    text = text.replace(
        "same full-width RoPE (positions 0..P) `decode_step` produces",
        "same head-local RoPE (positions 0..P) `decode_step` produces",
    )
    return text


CUDA_HELPERS = r'''
    /// Apply the existing narrow-matrix RoPE kernel independently to each logical
    /// head, then concatenate the disjoint head blocks. This is the GQA-correct
    /// rotary basis: frequency index zero restarts for every d_head slice.
    fn rope_heads(
        &self,
        x: &CudaMatrix,
        n_heads: usize,
        seq_len: usize,
        offset: usize,
    ) -> CudaMatrix {
        assert!(n_heads > 0 && x.cols().is_multiple_of(n_heads));
        let dh = x.cols() / n_heads;
        assert!(dh.is_multiple_of(2));
        let mut heads = Vec::with_capacity(n_heads);
        for head in 0..n_heads {
            let raw = self.chain.slice_cols(x, head * dh, dh);
            heads.push(self.chain.rope(&raw, seq_len, offset, self.theta));
        }
        let refs: Vec<&CudaMatrix> = heads.iter().collect();
        self.chain.concat_cols(&refs)
    }

    fn rope_heads_backward(
        &self,
        dy: &CudaMatrix,
        n_heads: usize,
        seq_len: usize,
        offset: usize,
    ) -> CudaMatrix {
        assert!(n_heads > 0 && dy.cols().is_multiple_of(n_heads));
        let dh = dy.cols() / n_heads;
        let mut heads = Vec::with_capacity(n_heads);
        for head in 0..n_heads {
            let raw = self.chain.slice_cols(dy, head * dh, dh);
            heads.push(self.chain.rope_backward(&raw, seq_len, offset, self.theta));
        }
        let refs: Vec<&CudaMatrix> = heads.iter().collect();
        self.chain.concat_cols(&refs)
    }

'''


def patch_cuda_model(text: str) -> str:
    if "fn rope_heads(" in text:
        raise SystemExit("cuda model already B33 patched")
    marker = "    /// Multi-head grouped-query attention over `q` (`t×d_model`) and `k`/`v`\n"
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("missing CUDA helper insertion point")
    text = text[:pos] + CUDA_HELPERS + text[pos:]
    replacements = [
        ("        let qr = self.chain.rope(q, seq, 0, self.theta);\n        let kr = self.chain.rope(k, seq, 0, self.theta);\n",
         "        let qr = self.rope_heads(q, self.n_heads, seq, 0);\n        let kr = self.rope_heads(k, self.n_kv_heads, seq, 0);\n", 1),
        ("        let qr = ch.rope(q, seq, 0, self.theta);\n        let kr = ch.rope(k, seq, 0, self.theta);\n",
         "        let qr = self.rope_heads(q, self.n_heads, seq, 0);\n        let kr = self.rope_heads(k, self.n_kv_heads, seq, 0);\n", 1),
        ("        let dq = ch.rope_backward(&dqr, seq, 0, self.theta);\n        let dk = ch.rope_backward(&dkr, seq, 0, self.theta);\n",
         "        let dq = self.rope_heads_backward(&dqr, self.n_heads, seq, 0);\n        let dk = self.rope_heads_backward(&dkr, self.n_kv_heads, seq, 0);\n", 1),
        ("        let qr = ch.rope(q, seq, 0, self.theta);\n        let kr = ch.rope(k, seq, 0, self.theta);\n",
         "        let qr = self.rope_heads(q, self.n_heads, seq, 0);\n        let kr = self.rope_heads(k, self.n_kv_heads, seq, 0);\n", 1),
        ("        let dq = ch.rope_backward(&dqr, seq, 0, self.theta);\n        let dk = ch.rope_backward(&dkr, seq, 0, self.theta);\n",
         "        let dq = self.rope_heads_backward(&dqr, self.n_heads, seq, 0);\n        let dk = self.rope_heads_backward(&dkr, self.n_kv_heads, seq, 0);\n", 1),
        ("            kcache[layer] = Some(ch.rope(&k, p, 0, self.theta));\n",
         "            kcache[layer] = Some(self.rope_heads(&k, self.n_kv_heads, p, 0));\n", 1),
        ("            let qr = ch.rope(&q, 1, pos, self.theta);\n            let kr = ch.rope(&k, 1, pos, self.theta);\n",
         "            let qr = self.rope_heads(&q, self.n_heads, 1, pos);\n            let kr = self.rope_heads(&k, self.n_kv_heads, 1, pos);\n", 1),
    ]
    for old, new, count in replacements:
        text = must_replace(text, old, new, count)
    text = text.replace(
        "RoPE the full-width q/k,\n    /// then per head",
        "apply RoPE independently inside every logical head,\n    /// then per head",
    )
    return text


ROUTE_NOTE = r'''

## B33 — GQA-correct head-local RoPE

Audit after the throughput work found a model-level positional-encoding defect shared
by the CPU, WGPU and CUDA paths. Q (`d_model=1024`) and K (`kv_dim=256`) were rotated
*before* head slicing, so RoPE used different full-width denominators for a query head
and its shared KV head. That is not the intended GQA geometry: the rotary frequency
schedule must restart within each `d_head` (64 for the 350M preset).

B33 makes RoPE explicitly head-local everywhere:

- CPU on-tape training and CPU incremental inference;
- WGPU `gqa_attention` / backward and all resident KV-cache paths;
- CUDA forward/backward, cached training, and CUDA KV-cached generation.

A regression test duplicates an identical head vector and proves both heads receive
identical rotations at the same position. This is an intentional model-semantics
correction. Legacy checkpoints still load structurally, but their learned positional
basis is the old one; the final post-B22 production run must therefore be trained with
B33 enabled rather than treating old validation numbers as comparable.
'''


def patch_route(text: str) -> str:
    if "## B33 — GQA-correct head-local RoPE" in text:
        raise SystemExit("ROUTE_B already B33 documented")
    return text.rstrip() + ROUTE_NOTE + "\n"


ATTN.write_text(patch_attention(ATTN.read_text()))
GPU_CHAIN.write_text(patch_gpu_chain(GPU_CHAIN.read_text()))
GPU_MODEL.write_text(patch_gpu_model(GPU_MODEL.read_text()))
CUDA_MODEL.write_text(patch_cuda_model(CUDA_MODEL.read_text()))
ROUTE.write_text(patch_route(ROUTE.read_text()))
print("patched B33 head-local RoPE across CPU/WGPU/CUDA")
