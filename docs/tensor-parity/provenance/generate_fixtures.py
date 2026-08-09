#!/usr/bin/env python3
"""Generate SciRust tensor-parity differential fixtures from the frozen PyTorch
baseline (2.13.0). OFFLINE tool: not run in CI. Output fixtures are data only.

Usage:
    python3 generate_fixtures.py

The committed historical fixture corpus is immutable. The official generator
writes only explicitly registered append-only families. Each append-only family
owns an independent frozen PRNG seed, so adding a future family cannot alter the
inputs or hashes of any pre-existing witness.

Before any write, the generator verifies the frozen PyTorch build and validates
every preserved manifest entry against the bytes already committed on disk.

Provenance rules (LICENSING_AND_PROVENANCE.md):
  - executes the pinned baseline (`torch.__version__` must be 2.13.0),
  - deterministic: fixed PRNG seed, fixed shapes per op, no RNG-derived shapes,
  - numeric data only; no PyTorch source is ever copied,
  - records baseline version + source-anchor commit in output manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
REGISTRY = REPO / "tensor-operators.toml"
OUT = REPO / "tests" / "parity" / "fixtures"
BASELINE_COMMIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"
SEED = 0xC0FFEE
BASELINE_VERSION = "2.13.0+cu130"

# Historical families are retained in source for auditability only.
# They are immutable witnesses and MUST NOT be dispatched by main().
HISTORICAL_FAMILIES = (
    "elementwise",
    "reductions",
    "normalization",
    "shape",
    "linear",
    "loss",
    "norm_affine",
    "reduction_extra",
    "unary_extra",
    "special",
    "shape_extra",
    "indexing",
    "linear_extra",
    "norm_stoch",
    "positional",
    "attention",
    "quantization",
    "conversion",
    "convolution",
    "linalg",
    "sparse",
    "einsum",
)

# Append-only generation registry.
#
# Every family has its own frozen seed. Future families MUST be added with a
# new independent seed. Historical families must never be added here.
APPEND_ONLY_FAMILY_SEEDS = {
    "elementwise_broadcast": 0xC0FFEE01,
}

# Populated only after verify_baseline_identity() succeeds.
VERIFIED_BASELINE_COMMIT: str | None = None

# Exact files written by the current invocation. The manifest is built from
# this set only; pre-existing files on disk are never silently certified.
GENERATED_FILES: set[Path] = set()

import torch  # noqa: E402


def verify_baseline_identity() -> str:
    """Fail closed unless the executing PyTorch build is the frozen baseline."""
    if torch.__version__ != BASELINE_VERSION:
        raise SystemExit(
            f"ERROR: baseline must be {BASELINE_VERSION}, got "
            f"{torch.__version__}. Refusing to generate fixtures."
        )

    actual_commit = getattr(torch.version, "git_version", None)
    if not actual_commit:
        raise SystemExit(
            "ERROR: torch.version.git_version is unavailable; "
            "cannot prove frozen-baseline provenance."
        )

    actual_commit = actual_commit.strip().lower()
    if actual_commit != BASELINE_COMMIT.lower():
        raise SystemExit(
            "ERROR: PyTorch source commit mismatch: "
            f"expected {BASELINE_COMMIT}, got {actual_commit}. "
            "Refusing to generate fixtures."
        )

    return actual_commit


UNARY_OPS = {
    "neg": (lambda x: torch.neg(x), None),
    "reciprocal": (lambda x: torch.reciprocal(x), None),
    "exp": (lambda x: torch.exp(x), None),
    "log": (lambda x: torch.log(x), None),
    "log10": (lambda x: torch.log10(x), None),
    "sqrt": (lambda x: torch.sqrt(x), None),
    "sin": (lambda x: torch.sin(x), None),
    "cos": (lambda x: torch.cos(x), None),
    "tan": (lambda x: torch.tan(x), None),
    "asin": (lambda x: torch.asin(x), None),
    "acos": (lambda x: torch.acos(x), None),
    "atan": (lambda x: torch.atan(x), None),
    "sinh": (lambda x: torch.sinh(x), None),
    "cosh": (lambda x: torch.cosh(x), None),
    "tanh": (lambda x: torch.tanh(x), None),
    "sigmoid": (lambda x: torch.sigmoid(x), None),
    "relu": (lambda x: torch.relu(x), None),
    "silu": (lambda x: torch.nn.functional.silu(x), None),
    "gelu": (lambda x: torch.nn.functional.gelu(x, approximate="tanh"), None),
    "pow": None,  # scalar form, handled specially
}
# input domain per op: (lo, hi) for the seeded uniform draw
DOMAIN = {
    "log": (0.5, 3.0), "log10": (0.5, 3.0), "sqrt": (0.5, 4.0),
    "asin": (-1.0, 1.0), "acos": (-1.0, 1.0),
    "reciprocal": (0.3, 3.0),
}
DEFAULT_DOMAIN = (-3.0, 3.0)

BINARY_OPS = {
    "add": lambda a, b: torch.add(a, b),
    "sub": lambda a, b: torch.sub(a, b),
    "mul": lambda a, b: torch.mul(a, b),
    "div": lambda a, b: torch.div(a, b),
    "atan2": lambda a, b: torch.atan2(a, b),
}
BINARY_DOMAIN = {"div": (0.5, 3.0)}  # second operand lower bound to avoid div-by-0

# APPEND-ONLY Profile 1.0 broadcast witnesses.
#
# These shapes intentionally exercise:
#   - mutual broadcasting on both axes,
#   - the opposite operand order,
#   - row broadcast,
#   - column broadcast,
#   - scalar-like 2-D (1x1) broadcast.
#
# This family owns an independent frozen seed in APPEND_ONLY_FAMILY_SEEDS.
# It never consumes or depends on the RNG stream of historical fixtures.
BINARY_BROADCAST_SHAPES = (
    ((1, 3), (2, 1)),
    ((2, 1), (1, 3)),
    ((1, 4), (3, 4)),
    ((3, 1), (3, 4)),
    ((1, 1), (2, 3)),
)

REDUCTIONS = {
    "sum": (lambda x, axis: torch.sum(x, dim=axis), True),
    "mean": (lambda x, axis: torch.mean(x, dim=axis), True),
    "var": (lambda x, axis: torch.var(x, dim=axis, unbiased=False), True),
}

SHAPES = [(3, 4), (5, 2)]
REDUCTION_AXES = [1, 0]


def flat(t: torch.Tensor) -> list[float]:
    return [float(v) for v in t.flatten().tolist()]


def seed_tensor(gen: torch.Generator, shape, lo, hi) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.float32).uniform_(lo, hi, generator=gen)


def grad_of(y: torch.Tensor, x: torch.Tensor, gout: torch.Tensor) -> torch.Tensor:
    (g,) = torch.autograd.grad(y, x, grad_outputs=gout)
    return g


def build_case(gen, op, kind, shape, domain_a, domain_b=None, scalar=None, axis=None, out_shape=None, fn=None):
    """kind in {unary, unary_scalar, binary, reduction}."""
    x = seed_tensor(gen, shape, *domain_a).requires_grad_(kind != "reduction")
    if kind == "unary":
        y = (fn or UNARY_OPS[op][0])(x)
        gout = seed_tensor(gen, shape, -1.0, 1.0)
        gx = grad_of(y, x, gout)
        return {"kind": "unary", "shape": list(shape), "x": flat(x.detach()),
                "y": flat(y.detach()), "gout": flat(gout), "gx": flat(gx)}
    if kind == "unary_scalar":
        y = torch.pow(x, scalar)
        gout = seed_tensor(gen, shape, -1.0, 1.0)
        gx = grad_of(y, x, gout)
        return {"kind": "unary_scalar", "shape": list(shape), "scalar": float(scalar),
                "x": flat(x.detach()), "y": flat(y.detach()),
                "gout": flat(gout), "gx": flat(gx)}
    if kind == "binary":
        b = seed_tensor(gen, shape, *domain_b).requires_grad_(True)
        y = BINARY_OPS[op](x, b)
        gout = seed_tensor(gen, shape, -1.0, 1.0)
        gx, gb = torch.autograd.grad(y, (x, b), grad_outputs=gout)
        return {"kind": "binary", "shape": list(shape), "x": flat(x.detach()),
                "b": flat(b.detach()), "y": flat(y.detach()),
                "gout": flat(gout), "gx": flat(gx), "gb": flat(gb)}
    if kind == "reduction":
        fn, _ = REDUCTIONS[op]
        x = x.requires_grad_(True)
        y = fn(x, axis)
        out = list(y.shape)
        gout = seed_tensor(gen, out, -1.0, 1.0)
        gx = grad_of(y, x, gout)
        return {"kind": "reduction", "shape": list(shape), "axis": axis,
                "out_shape": list(out), "x": flat(x.detach()), "y": flat(y.detach()),
                "gout": flat(gout), "gx": flat(gx)}


def gen_elementwise(gen, family_dir: Path) -> list[str]:
    for op, fn in UNARY_OPS.items():
        if fn is None:
            continue
        domain = DOMAIN.get(op, DEFAULT_DOMAIN)
        cases = [build_case(gen, op, "unary", s, domain) for s in SHAPES]
        write(op, family_dir, cases, note="torch default; grad via torch.autograd.grad")
    # pow scalar form
    cases = [build_case(gen, "pow", "unary_scalar", s, (0.5, 3.0), scalar=2.0) for s in SHAPES]
    write("pow", family_dir, cases, note="scalar exponent 2.0")
    # binary
    for op, _ in BINARY_OPS.items():
        dom_b = BINARY_DOMAIN.get(op, DEFAULT_DOMAIN)
        cases = [build_case(gen, op, "binary", s, DEFAULT_DOMAIN, dom_b) for s in SHAPES]
        write(op, family_dir, cases, note="both operands require_grad; 2-D same shape")


def gen_elementwise_broadcast(gen, family_dir: Path) -> None:
    """PyTorch 2-D binary broadcasting witnesses.

    APPEND-ONLY family with an independent frozen PRNG seed. Its output is
    intentionally independent of every historical generator function.
    """
    for op in BINARY_OPS:
        domain_b = BINARY_DOMAIN.get(op, DEFAULT_DOMAIN)
        cases = []

        for shape_a, shape_b in BINARY_BROADCAST_SHAPES:
            a = seed_tensor(gen, shape_a, *DEFAULT_DOMAIN).requires_grad_(True)
            b = seed_tensor(gen, shape_b, *domain_b).requires_grad_(True)

            y = BINARY_OPS[op](a, b)
            gout = seed_tensor(gen, list(y.shape), -1.0, 1.0)
            ga, gb = torch.autograd.grad(y, (a, b), grad_outputs=gout)

            cases.append({
                "kind": "binary_broadcast",
                "shape": list(shape_a),
                "b_shape": list(shape_b),
                "out_shape": list(y.shape),
                "x": flat(a.detach()),
                "b": flat(b.detach()),
                "y": flat(y.detach()),
                "gout": flat(gout),
                "gx": flat(ga),
                "gb": flat(gb),
            })

        write(
            op,
            family_dir,
            cases,
            note=(
                "both operands require_grad; deterministic 2-D PyTorch "
                "broadcasting incl. mutual/row/column/(1,1)"
            ),
        )


def gen_reductions(gen, family_dir: Path) -> None:
    for op, _ in REDUCTIONS.items():
        cases = [build_case(gen, op, "reduction", s, (-3.0, 3.0), axis=a)
                 for a in REDUCTION_AXES for s in SHAPES]
        write(op, family_dir, cases, note="axis 0/1; var uses unbiased=False")


def gen_normalization(gen, family_dir: Path) -> None:
    for op in ("softmax", "log_softmax"):
        fn = (lambda x: torch.nn.functional.softmax(x, dim=-1)) if op == "softmax" \
            else (lambda x: torch.nn.functional.log_softmax(x, dim=-1))
        cases = [build_case(gen, op, "unary", s, DEFAULT_DOMAIN, fn=fn) for s in SHAPES]
        write(op, family_dir, cases, note="dim=-1 (dernière dimension)")


def gen_unary_extra(gen, family_dir: Path) -> None:
    """rsqrt — APPEND fin de séquence (positionnel, voir gen_norm_affine)."""
    cases = [build_case(gen, "rsqrt", "unary", s, (0.3, 5.0),
                        fn=lambda x: torch.rsqrt(x)) for s in SHAPES]
    write("rsqrt", family_dir, cases, note="1/sqrt(x), domaine > 0")


def gen_special(gen, family_dir: Path) -> None:
    """lgamma/digamma en f64 — APPEND fin de séquence (positionnel).

    Les rows special du registre sont f64 avec tol 1e-10 : fixtures en
    torch.float64 (domaine (0.1, 5.0), hors pôles). Le harness compare via
    un chemin f64 dédié (hors TensorND, qui est f32).
    """
    for op in ("lgamma", "digamma"):
        cases = []
        for s in SHAPES:
            x = torch.empty(s, dtype=torch.float64).uniform_(0.1, 5.0, generator=gen)
            x = x.requires_grad_(True)
            y = torch.lgamma(x) if op == "lgamma" else torch.digamma(x)
            gout = torch.empty(s, dtype=torch.float64).uniform_(-1.0, 1.0, generator=gen)
            (gx,) = torch.autograd.grad(y, x, grad_outputs=gout)
            cases.append({"kind": "unary", "shape": list(s),
                          "x": flat(x.detach()), "y": flat(y.detach()),
                          "gout": flat(gout), "gx": flat(gx)})
        write(op, family_dir, cases, dtype="f64",
              note="f64, domaine (0.1, 5.0), grad via autograd")


def gen_shape_extra(gen, family_dir: Path) -> None:
    """cat / gather / unfold — APPEND fin de séquence (positionnel)."""
    cases = []
    for dim in (0, 1):
        s = (3, 4)
        a = seed_tensor(gen, s, -3.0, 3.0).requires_grad_(True)
        b = seed_tensor(gen, s, -3.0, 3.0).requires_grad_(True)
        y = torch.cat([a, b], dim=dim)
        gout = seed_tensor(gen, list(y.shape), -1.0, 1.0)
        gx, gb = torch.autograd.grad(y, (a, b), grad_outputs=gout)
        cases.append({"kind": "cat", "shape": list(s), "dims": [dim],
                      "a": flat(a.detach()), "b": flat(b.detach()),
                      "out_shape": list(y.shape),
                      "y": flat(y.detach()), "gout": flat(gout),
                      "gx": flat(gx), "gb": flat(gb)})
    write("cat", family_dir, cases, note="cat 2 tenseurs même shape, dim 0/1")
    cases = []
    for axis in (0, 1):
        s = (3, 4)
        x = seed_tensor(gen, s, -3.0, 3.0).requires_grad_(True)
        idx = torch.randint(0, s[axis], s, generator=gen)
        y = torch.gather(x, axis, idx)
        gout = seed_tensor(gen, s, -1.0, 1.0)
        (gx,) = torch.autograd.grad(y, x, grad_outputs=gout)
        cases.append({"kind": "gather", "shape": list(s), "axis": axis,
                      "x": flat(x.detach()),
                      "indices": [int(v) for v in idx.flatten().tolist()],
                      "y": flat(y.detach()), "gout": flat(gout),
                      "gx": flat(gx)})
    write("gather", family_dir, cases, note="gather index même shape que x, axe 0/1")
    cases = []
    for (axis, size, step) in ((1, 2, 1), (0, 2, 2)):
        s = (3, 4)
        x = seed_tensor(gen, s, -3.0, 3.0).requires_grad_(True)
        y = x.unfold(axis, size, step)
        gout = seed_tensor(gen, list(y.shape), -1.0, 1.0)
        (gx,) = torch.autograd.grad(y, x, grad_outputs=gout)
        cases.append({"kind": "unfold", "shape": list(s), "axis": axis,
                      "x": flat(x.detach()),
                      "size": size, "step": step, "out_shape": list(y.shape),
                      "y": flat(y.detach()), "gout": flat(gout),
                      "gx": flat(gx)})
    write("unfold", family_dir, cases, note="unfold axe 0/1, taille/pas")


def gen_indexing(gen, family_dir: Path) -> None:
    """embedding — APPEND fin de séquence (positionnel)."""
    cases = []
    for idx_shape, V, D in (((2, 3), 8, 4), ((3, 2), 6, 3)):
        w = seed_tensor(gen, (V, D), -1.0, 1.0).requires_grad_(True)
        idx = torch.randint(0, V, idx_shape, generator=gen)
        y = torch.nn.functional.embedding(idx, w)
        gout = seed_tensor(gen, list(y.shape), -1.0, 1.0)
        (gw,) = torch.autograd.grad(y, w, grad_outputs=gout)
        cases.append({"kind": "embedding", "idx_shape": list(idx_shape),
                      "indices": [int(v) for v in idx.flatten().tolist()],
                      "w": flat(w.detach()), "out_shape": list(y.shape),
                      "y": flat(y.detach()), "gout": flat(gout),
                      "gw": flat(gw)})
    write("embedding", family_dir, cases, note="embedding(index, weight) grad vers weight")


def gen_linear_extra(gen, family_dir: Path) -> None:
    """cosine_similarity / normalize — APPEND fin de séquence (positionnel)."""
    cases = []
    for s in SHAPES:
        a = seed_tensor(gen, s, -3.0, 3.0).requires_grad_(True)
        b = seed_tensor(gen, s, -3.0, 3.0).requires_grad_(True)
        y = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        out = list(y.shape)
        gout = seed_tensor(gen, out, -1.0, 1.0)
        gx, gb = torch.autograd.grad(y, (a, b), grad_outputs=gout)
        cases.append({"kind": "cosine", "shape": list(s),
                      "a": flat(a.detach()), "b": flat(b.detach()),
                      "out_shape": out, "y": flat(y.detach()),
                      "gout": flat(gout), "gx": flat(gx), "gb": flat(gb)})
    write("cosine_similarity", family_dir, cases, note="cosine_similarity dim=-1")
    cases = []
    for s in SHAPES:
        x = seed_tensor(gen, s, -3.0, 3.0).requires_grad_(True)
        y = torch.nn.functional.normalize(x, p=2, dim=1)
        gout = seed_tensor(gen, s, -1.0, 1.0)
        (gx,) = torch.autograd.grad(y, x, grad_outputs=gout)
        cases.append({"kind": "normalize", "shape": list(s),
                      "x": flat(x.detach()),
                      "y": flat(y.detach()), "gout": flat(gout),
                      "gx": flat(gx)})
    write("normalize", family_dir, cases, note="normalize p=2 dim=1")


def gen_norm_stoch(gen, family_dir: Path) -> None:
    """dropout (masque fixé par seed) / batch_norm eval — APPEND fin de séquence."""
    cases = []
    for p in (0.5, 0.2):
        s = (3, 4)
        x = seed_tensor(gen, s, -3.0, 3.0).requires_grad_(True)
        keep = torch.bernoulli(torch.full(s, 1.0 - p), generator=gen)
        y = x * keep / (1.0 - p)
        gout = seed_tensor(gen, s, -1.0, 1.0)
        (gx,) = torch.autograd.grad(y, x, grad_outputs=gout)
        cases.append({"kind": "dropout", "shape": list(s), "p": p,
                      "x": flat(x.detach()),
                      "mask": [float(v) for v in keep.flatten().tolist()],
                      "y": flat(y.detach()), "gout": flat(gout),
                      "gx": flat(gx)})
    write("dropout", family_dir, cases,
          note="masque bernoulli seedé commité ; y = x*mask/(1-p)")
    cases = []
    for s in SHAPES:
        C = s[1]
        x = seed_tensor(gen, s, -3.0, 3.0).requires_grad_(True)
        rm = seed_tensor(gen, (C,), -0.5, 0.5)
        rv = torch.empty((C,), dtype=torch.float32).uniform_(0.5, 2.0, generator=gen)
        w = seed_tensor(gen, (C,), 0.5, 1.5).requires_grad_(True)
        b = seed_tensor(gen, (C,), -1.0, 1.0).requires_grad_(True)
        y = torch.nn.functional.batch_norm(x, rm, rv, w, b, training=False,
                                           momentum=0.1, eps=1e-5)
        gout = seed_tensor(gen, s, -1.0, 1.0)
        gx, gw, gb = torch.autograd.grad(y, (x, w, b), grad_outputs=gout)
        cases.append({"kind": "batchnorm", "shape": list(s), "dims": [C],
                      "x": flat(x.detach()),
                      "eps": 1e-5, "w": flat(w.detach()), "b": flat(b.detach()),
                      "rm": flat(rm), "rv": flat(rv), "y": flat(y.detach()),
                      "gout": flat(gout), "gx": flat(gx), "gw": flat(gw),
                      "gb": flat(gb)})
    write("batch_norm", family_dir, cases, note="eval mode, running stats, eps=1e-5")


def gen_positional(gen, family_dir: Path) -> None:
    """rope (référence torch en ops natives, paires, base=10000) — APPEND fin de séquence."""
    cases = []
    for s in ((2, 3, 4), (2, 2, 6)):
        L, H, D = s
        x = seed_tensor(gen, s, -3.0, 3.0).requires_grad_(True)
        pos = torch.arange(L, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(0, D, 2, dtype=torch.float32)
        theta = pos / (10000.0 ** (i / D))
        c = torch.cos(theta).unsqueeze(1)
        sn = torch.sin(theta).unsqueeze(1)
        xh = x.view(L, H, D // 2, 2)
        rot0 = xh[..., 0] * c - xh[..., 1] * sn
        rot1 = xh[..., 0] * sn + xh[..., 1] * c
        y = torch.stack([rot0, rot1], dim=-1).view(L, H, D)
        gout = seed_tensor(gen, s, -1.0, 1.0)
        (gx,) = torch.autograd.grad(y, x, grad_outputs=gout)
        cases.append({"kind": "rope", "shape": list(s), "base": 10000.0,
                      "x": flat(x.detach()),
                      "y": flat(y.detach()), "gout": flat(gout),
                      "gx": flat(gx)})
    write("rope", family_dir, cases, note="rope paires, base=10000, dernières dims paires")


def gen_attention(gen, family_dir: Path) -> None:
    """scaled_dot_product_attention (sans masque, dropout=0) — APPEND fin de séquence."""
    cases = []
    for (B, L, E) in ((1, 3, 4), (1, 4, 2)):
        q = seed_tensor(gen, (B, L, E), -1.0, 1.0).requires_grad_(True)
        k = seed_tensor(gen, (B, L, E), -1.0, 1.0).requires_grad_(True)
        v = seed_tensor(gen, (B, L, E), -1.0, 1.0).requires_grad_(True)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        gout = seed_tensor(gen, list(y.shape), -1.0, 1.0)
        gq, gk, gv = torch.autograd.grad(y, (q, k, v), grad_outputs=gout)
        cases.append({"kind": "attention", "shape": [B, L, E],
                      "a": flat(q.detach()), "b": flat(k.detach()),
                      "c": flat(v.detach()), "out_shape": list(y.shape),
                      "y": flat(y.detach()), "gout": flat(gout),
                      "gx": flat(gq), "gk": flat(gk), "gv": flat(gv)})
    write("scaled_dot_product_attention", family_dir, cases,
          note="sdpa sans masque ni dropout, 1 batch")


def gen_quantization(gen, family_dir: Path) -> None:
    """fake_quantize_per_tensor_affine (STE) — APPEND fin de séquence."""
    cases = []
    for s in SHAPES:
        x = seed_tensor(gen, s, -1.0, 1.0).requires_grad_(True)
        scale, zp, qmin, qmax = 0.1, 0, 0, 255
        y = torch.fake_quantize_per_tensor_affine(x, scale, zp, qmin, qmax)
        gout = seed_tensor(gen, s, -1.0, 1.0)
        (gx,) = torch.autograd.grad(y, x, grad_outputs=gout)
        cases.append({"kind": "fakequant", "shape": list(s), "scale": scale,
                      "zp": zp, "qmin": qmin, "qmax": qmax,
                      "x": flat(x.detach()),
                      "y": flat(y.detach()), "gout": flat(gout),
                      "gx": flat(gx)})
    write("fake_quantize_per_tensor", family_dir, cases,
          note="scale=0.1 zp=0 qmin=0 qmax=255, grad STE")


def gen_conversion(gen, family_dir: Path) -> None:
    """to_bf16 (round f32 -> bf16 -> f32, sans autograd) — APPEND fin de séquence."""
    cases = []
    for s in SHAPES:
        x = seed_tensor(gen, s, -1.0, 1.0)
        y = x.to(torch.bfloat16).float()
        cases.append({"kind": "unary", "shape": list(s),
                      "x": flat(x), "y": flat(y)})
    write("to_bf16", family_dir, cases, note="conversion aller-retour bf16, exacte")


def gen_convolution(gen, family_dir: Path) -> None:
    """conv1d/conv2d (valid, stride 1, bias) + max_pool2d/avg_pool2d (k=s=2, pad=0)
    — APPEND fin de séquence (positionnel)."""
    cases = []
    for (B, Cin, L), (Cout, K) in (((1, 2, 8), (3, 3)), ((1, 3, 10), (2, 2))):
        x = seed_tensor(gen, (B, Cin, L), -3.0, 3.0).requires_grad_(True)
        w = seed_tensor(gen, (Cout, Cin, K), -1.0, 1.0).requires_grad_(True)
        b = seed_tensor(gen, (Cout,), -1.0, 1.0).requires_grad_(True)
        y = torch.nn.functional.conv1d(x, w, b)
        gout = seed_tensor(gen, list(y.shape), -1.0, 1.0)
        gx, gw, gb = torch.autograd.grad(y, (x, w, b), grad_outputs=gout)
        cases.append({"kind": "conv1d", "shape": list(x.shape), "x": flat(x.detach()),
                      "w": flat(w.detach()),
                      "b": flat(b.detach()), "kernel": K, "out_shape": list(y.shape),
                      "y": flat(y.detach()), "gout": flat(gout),
                      "gx": flat(gx), "gw": flat(gw), "gb": flat(gb)})
    write("conv1d", family_dir, cases,
          note="conv1d valid stride=1, bias, autograd x/w/b")
    cases = []
    for (B, Cin, H, W), (Cout, KH, KW) in (((1, 2, 5, 5), (3, 3, 3)),
                                           ((1, 1, 4, 4), (2, 2, 2))):
        x = seed_tensor(gen, (B, Cin, H, W), -3.0, 3.0).requires_grad_(True)
        w = seed_tensor(gen, (Cout, Cin, KH, KW), -1.0, 1.0).requires_grad_(True)
        b = seed_tensor(gen, (Cout,), -1.0, 1.0).requires_grad_(True)
        y = torch.nn.functional.conv2d(x, w, b)
        gout = seed_tensor(gen, list(y.shape), -1.0, 1.0)
        gx, gw, gb = torch.autograd.grad(y, (x, w, b), grad_outputs=gout)
        cases.append({"kind": "conv2d", "shape": list(x.shape), "x": flat(x.detach()),
                      "w": flat(w.detach()),
                      "b": flat(b.detach()), "kh": KH, "kw": KW,
                      "out_shape": list(y.shape), "y": flat(y.detach()),
                      "gout": flat(gout), "gx": flat(gx), "gw": flat(gw),
                      "gb": flat(gb)})
    write("conv2d", family_dir, cases,
          note="conv2d valid stride=1, bias, autograd x/w/b")
    cases = []
    for s, k in (((1, 2, 4, 4), 2), ((1, 3, 6, 6), 2)):
        x = seed_tensor(gen, s, -3.0, 3.0).requires_grad_(True)
        y, idx = torch.nn.functional.max_pool2d(x, k, return_indices=True)
        gout = seed_tensor(gen, list(y.shape), -1.0, 1.0)
        (gx,) = torch.autograd.grad(y, x, grad_outputs=gout)
        cases.append({"kind": "maxpool", "shape": list(s), "x": flat(x.detach()),
                      "kernel": k,
                      "indices": [int(v) for v in idx.flatten().tolist()],
                      "out_shape": list(y.shape), "y": flat(y.detach()),
                      "gout": flat(gout), "gx": flat(gx)})
    write("max_pool2d", family_dir, cases,
          note="k=s=2 pad=0, indices torch (premier max par fenêtre)")
    cases = []
    for s, k in (((1, 2, 4, 4), 2), ((1, 3, 6, 6), 2)):
        x = seed_tensor(gen, s, -3.0, 3.0).requires_grad_(True)
        y = torch.nn.functional.avg_pool2d(x, k)
        gout = seed_tensor(gen, list(y.shape), -1.0, 1.0)
        (gx,) = torch.autograd.grad(y, x, grad_outputs=gout)
        cases.append({"kind": "avgpool", "shape": list(s), "x": flat(x.detach()),
                      "kernel": k,
                      "out_shape": list(y.shape), "y": flat(y.detach()),
                      "gout": flat(gout), "gx": flat(gx)})
    write("avg_pool2d", family_dir, cases,
          note="k=s=2 pad=0, count_include_pad default")


def gen_linalg(gen, family_dir: Path) -> None:
    """cholesky (SPD construit A = M·Mᵀ + λI) — APPEND fin de séquence."""
    cases = []
    for n in (4, 3):
        m = seed_tensor(gen, (n, n), -1.0, 1.0)
        a = m @ m.T + torch.eye(n) * 0.5
        l = torch.linalg.cholesky(a)
        cases.append({"kind": "cholesky", "shape": [n, n],
                      "a": flat(a.detach()), "y": flat(l.detach())})
    write("cholesky", family_dir, cases,
          note="SPD A=M·Mᵀ+0.5I, L inférieur, f32, forward only")


def gen_sparse(gen, family_dir: Path) -> None:
    """spmv/spmm CSR (f64) + solve (f64) — APPEND fin de séquence."""
    cases = []
    for (n, m) in ((5, 4), (3, 6)):
        dens = torch.empty((n, m), dtype=torch.float64).uniform_(-1.0, 1.0, generator=gen)
        mask = torch.bernoulli(torch.full((n, m), 0.4), generator=gen)
        a_sp = (dens * mask).to_sparse_csr()
        x = torch.empty((m,), dtype=torch.float64).uniform_(-1.0, 1.0, generator=gen)
        b = torch.empty((m, 2), dtype=torch.float64).uniform_(-1.0, 1.0, generator=gen)
        yv = torch.sparse.mm(a_sp, x.unsqueeze(1)).squeeze(1)
        ym = torch.sparse.mm(a_sp, b)
        cases.append({"kind": "spmv", "n": n, "m": m,
                      "rowptr": [int(v) for v in a_sp.crow_indices().tolist()],
                      "colidx": [int(v) for v in a_sp.col_indices().tolist()],
                      "values": a_sp.values().tolist(),
                      "x": flat(x.detach()), "yv": flat(yv.detach()),
                      "b": flat(b.detach()), "ym": flat(ym.detach())})
    write("spmv", family_dir, cases, dtype="f64",
          note="CSR ~40% nnz, spmv + spmm_dense, f64, forward only")
    cases = []
    for n in (4, 3):
        a = (torch.empty((n, n), dtype=torch.float64).uniform_(-2.0, 2.0, generator=gen)
             + torch.eye(n) * n)
        b1 = torch.empty((n,), dtype=torch.float64).uniform_(-1.0, 1.0, generator=gen)
        b2 = torch.empty((n, 2), dtype=torch.float64).uniform_(-1.0, 1.0, generator=gen)
        y1 = torch.linalg.solve(a, b1)
        y2 = torch.linalg.solve(a, b2)
        cases.append({"kind": "solve", "n": n, "a": flat(a.detach()),
                      "b1": flat(b1.detach()), "y1": flat(y1.detach()),
                      "b2": flat(b2.detach()), "y2": flat(y2.detach())})
    write("solve", family_dir, cases, dtype="f64",
          note="torch.linalg.solve A·x=b, A diagonale dominante, b 1-col et 2-col, f64")


def gen_einsum(gen, family_dir: Path) -> None:
    """einsum subset testé : ij,jk->ik / ij->ji / ij,jk,kl->il / ii->i / ij->
    — APPEND fin de séquence."""
    cases = []
    specs = [
        ("ij,jk->ik", [(3, 4), (4, 5)]),
        ("ij->ji", [(3, 4)]),
        ("ij,jk,kl->il", [(2, 3), (3, 4), (4, 2)]),
        ("ii->i", [(4, 4)]),
        ("ij->", [(2, 3)]),
    ]
    for spec, shapes in specs:
        tensors = [seed_tensor(gen, s, -2.0, 2.0) for s in shapes]
        y = torch.einsum(spec, tensors)
        case = {"kind": "einsum", "spec": spec,
                "shapes": [list(s) for s in shapes],
                "y": flat(y.detach())}
        for name, t in zip(("x", "a", "b"), tensors):
            case[name] = flat(t.detach())
        cases.append(case)
    write("einsum", family_dir, cases, dtype="f32",
          note="subset 5 specs (2 operands, transpose, 3 operands, diag, scalaire)")


def gen_reduction_extra(gen, family_dir: Path) -> None:
    """max (valeurs + indices) et norm p=2.

    IMPORTANT : étape DERNIÈRE de la séquence de génération (positionnelle,
    voir gen_norm_affine).
    """
    cases = []
    for axis in REDUCTION_AXES:
        for s in SHAPES:
            x = seed_tensor(gen, s, -3.0, 3.0).requires_grad_(True)
            vals, idx = torch.max(x, dim=axis)
            out = list(vals.shape)
            gout = seed_tensor(gen, out, -1.0, 1.0)
            gx = grad_of(vals, x, gout)
            cases.append({"kind": "reduction", "shape": list(s), "axis": axis,
                          "out_shape": list(out), "x": flat(x.detach()),
                          "y": flat(vals.detach()),
                          "indices": [int(v) for v in idx.flatten().tolist()],
                          "gout": flat(gout), "gx": flat(gx)})
    write("max", family_dir, cases, note="torch.max(x, dim) values+indices, premier max")
    cases = []
    for axis in REDUCTION_AXES:
        for s in SHAPES:
            x = seed_tensor(gen, s, -3.0, 3.0).requires_grad_(True)
            y = torch.norm(x, p=2, dim=axis)
            out = list(y.shape)
            gout = seed_tensor(gen, out, -1.0, 1.0)
            gx = grad_of(y, x, gout)
            cases.append({"kind": "reduction", "shape": list(s), "axis": axis,
                          "out_shape": list(out), "x": flat(x.detach()),
                          "y": flat(y.detach()), "gout": flat(gout),
                          "gx": flat(gx)})
    write("norm", family_dir, cases, note="torch.norm p=2 (frob), dim=axis")


def gen_norm_affine(gen, family_dir: Path) -> None:
    """layer_norm/rms_norm (2-D, normalized_shape = dernière dim, affine).

    IMPORTANT : cette étape doit rester la DERNIÈRE de la séquence de
    génération. Le générateur est positionnel (une seule stream RNG seedée) ;
    insérer de nouveaux tirages au milieu décalerait tous les hashs des
    fixtures déjà commités.
    """
    norm_cases(gen, family_dir, "layer_norm")
    norm_cases(gen, family_dir, "rms_norm")


def norm_cases(gen, family_dir: Path, op: str) -> None:
    """layer_norm/rms_norm (2-D, normalized_shape = dernière dim, affine)."""
    eps = 1e-5
    cases = []
    for s in SHAPES:
        w = seed_tensor(gen, (s[-1],), 0.5, 1.5).requires_grad_(True)
        x = seed_tensor(gen, s, -3.0, 3.0).requires_grad_(True)
        gout = seed_tensor(gen, s, -1.0, 1.0)
        if op == "layer_norm":
            b = seed_tensor(gen, (s[-1],), -1.0, 1.0).requires_grad_(True)
            y = torch.nn.functional.layer_norm(x, (s[-1],), w, b, eps=eps)
            gx, gw, gb = torch.autograd.grad(y, (x, w, b), grad_outputs=gout)
            cases.append({"kind": "normalization", "shape": list(s),
                          "dims": [s[-1]], "eps": eps,
                          "x": flat(x.detach()), "w": flat(w.detach()),
                          "b": flat(b.detach()), "y": flat(y.detach()),
                          "gout": flat(gout), "gx": flat(gx),
                          "gw": flat(gw), "gb": flat(gb)})
        else:
            y = torch.nn.functional.rms_norm(x, (s[-1],), w, eps=eps)
            gx, gw = torch.autograd.grad(y, (x, w), grad_outputs=gout)
            cases.append({"kind": "normalization", "shape": list(s),
                          "dims": [s[-1]], "eps": eps,
                          "x": flat(x.detach()), "w": flat(w.detach()),
                          "y": flat(y.detach()), "gout": flat(gout),
                          "gx": flat(gx), "gw": flat(gw)})
    write(op, family_dir, cases,
          note=f"{op}: affine, eps={eps}, normalized_shape = dernière dim")


def shape_forward(op, x, params):
    """Retourne (y, gx) pour les ops de shape ; params selon l'op."""
    if op == "reshape":
        return torch.reshape(x, params["new_shape"])
    if op == "transpose":
        d0, d1 = params["dims"]
        return torch.transpose(x, d0, d1)
    if op == "permute":
        return torch.permute(x, params["dims"])
    if op == "broadcast_to":
        return torch.broadcast_to(x, params["bcast_to"])
    if op == "slice":
        axis, start, end = params["axis"], params["start"], params["end"]
        idx = [slice(None)] * x.dim()
        idx[axis] = slice(start, end)
        return x[tuple(idx)]
    if op == "flatten":
        return torch.flatten(x)
    raise SystemExit(f"unknown shape op {op}")


SHAPE_CASES = {
    "reshape": [{"new_shape": [2, 6]}, {"new_shape": [10]}],
    "transpose": [{"dims": (0, 1)}, {"dims": (0, 1)}],
    "permute": [{"dims": (1, 0)}, {"dims": (1, 0)}],
    "broadcast_to": [{"bcast_to": [3, 4]}, {"bcast_to": [2, 4]}],
    "slice": [{"axis": 1, "start": 1, "end": 3}, {"axis": 0, "start": 1, "end": 4}],
    "flatten": [{}, {}],
}
# broadcast_to exige des shapes broadcastables (sinon RuntimeError torch) :
# on utilise des shapes d'entrée spécifiques (1,4)->(3,4) et (2,1)->(2,4).
BROADCAST_IN_SHAPES = [(1, 4), (2, 1)]


def gen_shape(gen, family_dir: Path) -> None:
    for op, param_list in SHAPE_CASES.items():
        cases = []
        shapes = BROADCAST_IN_SHAPES if op == "broadcast_to" else SHAPES
        for shape, params in zip(shapes, param_list):
            x = seed_tensor(gen, shape, -3.0, 3.0).requires_grad_(True)
            y = shape_forward(op, x, params)
            gout = seed_tensor(gen, list(y.shape), -1.0, 1.0)
            (gx,) = torch.autograd.grad(y, x, grad_outputs=gout)
            case = {"kind": "shape", "shape": list(shape), "x": flat(x.detach()),
                    "out_shape": list(y.shape), "y": flat(y.detach()),
                    "gout": flat(gout), "gx": flat(gx)}
            case.update({k: (list(v) if isinstance(v, (list, tuple)) else v)
                         for k, v in params.items()})
            cases.append(case)
        write(op, family_dir, cases, note="exact reorder/copy ops; grads per torch autograd")


def gen_linear(gen, family_dir: Path) -> None:
    # matmul 2-D
    for op, (s1, s2) in {"matmul": (([2, 3], [3, 4]), ([5, 2], [2, 3]))}.items():
        cases = []
        for (sa, sb) in (s1, s2):
            a = seed_tensor(gen, sa, -2.0, 2.0).requires_grad_(True)
            b = seed_tensor(gen, sb, -2.0, 2.0).requires_grad_(True)
            y = torch.matmul(a, b)
            gout = seed_tensor(gen, list(y.shape), -1.0, 1.0)
            ga, gb = torch.autograd.grad(y, (a, b), grad_outputs=gout)
            cases.append({"kind": "linear", "shape": list(sa), "a": flat(a.detach()),
                          "b": flat(b.detach()), "out_shape": list(y.shape), "y": flat(y.detach()),
                          "gout": flat(gout), "gx": flat(ga), "gb": flat(gb)})
        write(op, family_dir, cases, note="2-D matmul, both operands autograd")
    # bmm 3-D
    cases = []
    for (sa, sb) in (([2, 2, 3], [2, 3, 4]), ([3, 5, 2], [3, 2, 4])):
        a = seed_tensor(gen, sa, -2.0, 2.0).requires_grad_(True)
        b = seed_tensor(gen, sb, -2.0, 2.0).requires_grad_(True)
        y = torch.bmm(a, b)
        gout = seed_tensor(gen, list(y.shape), -1.0, 1.0)
        ga, gb = torch.autograd.grad(y, (a, b), grad_outputs=gout)
        cases.append({"kind": "linear", "shape": list(sa), "a": flat(a.detach()),
                      "b": flat(b.detach()), "out_shape": list(y.shape), "y": flat(y.detach()),
                      "gout": flat(gout), "gx": flat(ga), "gb": flat(gb)})
    write("bmm", family_dir, cases, note="3-D batched matmul, both operands autograd")
    # linear F.linear(x, w, b)
    cases = []
    for (sx, sw) in (([2, 4], [3, 4]), ([5, 3], [2, 3])):
        x = seed_tensor(gen, sx, -2.0, 2.0).requires_grad_(True)
        w = seed_tensor(gen, sw, -1.0, 1.0).requires_grad_(True)
        b = seed_tensor(gen, [sw[0]], -1.0, 1.0).requires_grad_(True)
        y = torch.nn.functional.linear(x, w, b)
        gout = seed_tensor(gen, list(y.shape), -1.0, 1.0)
        gx, gw, gb = torch.autograd.grad(y, (x, w, b), grad_outputs=gout)
        cases.append({"kind": "linear", "shape": list(sx), "x": flat(x.detach()),
                      "w": flat(w.detach()), "b": flat(b.detach()), "out_shape": list(y.shape),
                      "y": flat(y.detach()), "gout": flat(gout), "gx": flat(gx),
                      "gw": flat(gw), "gb": flat(gb)})
    write("linear", family_dir, cases, note="F.linear(x, w, b) with bias, autograd on all")


def gen_loss(gen, family_dir: Path) -> None:
    # mse_loss mean
    cases = []
    for s in SHAPES:
        p = seed_tensor(gen, s, -2.0, 2.0).requires_grad_(True)
        t = seed_tensor(gen, s, -2.0, 2.0)
        y = torch.nn.functional.mse_loss(p, t, reduction="mean")
        gout = seed_tensor(gen, [], 1.0, 1.0)
        (gp,) = torch.autograd.grad(y, p, grad_outputs=gout)
        cases.append({"kind": "loss", "shape": list(s), "x": flat(p.detach()),
                      "target": flat(t.detach()), "out_shape": [], "y": flat(y.detach()),
                      "gout": flat(gout), "gx": flat(gp)})
    write("mse_loss", family_dir, cases, note="reduction='mean', scalar out")
    # cross_entropy mean, targets indices
    cases = []
    for (s, targets) in (([2, 4], [1, 3]), ([3, 5], [0, 2, 4])):
        logits = seed_tensor(gen, s, -2.0, 2.0).requires_grad_(True)
        tgt = torch.tensor(targets, dtype=torch.long)
        y = torch.nn.functional.cross_entropy(logits, tgt, reduction="mean")
        gout = seed_tensor(gen, [], 1.0, 1.0)
        (gl,) = torch.autograd.grad(y, logits, grad_outputs=gout)
        cases.append({"kind": "loss", "shape": list(s), "x": flat(logits.detach()),
                      "indices": targets, "out_shape": [], "y": flat(y.detach()),
                      "gout": flat(gout), "gx": flat(gl)})
    write("cross_entropy", family_dir, cases, note="reduction='mean', scalar out, targets LongTensor")


def write(op: str, family_dir: Path, cases: list[dict], note: str, dtype: str = "f32") -> None:
    if VERIFIED_BASELINE_COMMIT is None:
        raise RuntimeError("baseline identity must be verified before writing fixtures")

    try:
        relative_family = family_dir.relative_to(OUT)
    except ValueError as exc:
        raise RuntimeError(
            f"fixture family is outside the canonical fixture root: {family_dir}"
        ) from exc

    if (
        len(relative_family.parts) != 1
        or relative_family.parts[0] not in APPEND_ONLY_FAMILY_SEEDS
    ):
        raise RuntimeError(
            "historical fixture families are immutable; refusing write to "
            f"{relative_family}"
        )

    path = family_dir / f"{op}.json"
    data = {
        "op": op,
        "kind": cases[0]["kind"],
        "dtype": dtype,
        "pytorch": {
            "version": torch.__version__,
            "commit": VERIFIED_BASELINE_COMMIT,
            "generated": "deterministic",
        },
        "note": note,
        "cases": cases,
    }
    family_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n")
    GENERATED_FILES.add(path)
    print(f"  wrote {path.relative_to(REPO)} ({len(cases)} cases)")


def build_manifest_files(
    generated_files: set[Path],
    allowed_ops: set[str],
) -> dict[str, str]:
    """Hash exactly the fixtures produced by the current invocation."""
    files: dict[str, str] = {}
    for path in sorted(generated_files):
        if not path.is_file():
            raise SystemExit(f"ERROR: generated fixture disappeared: {path}")
        try:
            relative = path.relative_to(OUT)
        except ValueError as exc:
            raise SystemExit(f"ERROR: generated fixture outside OUT: {path}") from exc

        opname = path.stem
        if opname not in allowed_ops:
            raise SystemExit(
                f"ERROR: generated fixture {relative} is not present in tensor-operators.toml"
            )

        files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def load_existing_manifest() -> dict:
    """Load and validate the committed manifest before any fixture write."""
    path = OUT / "manifest.json"
    if not path.is_file():
        raise SystemExit("ERROR: committed fixture manifest is missing")

    data = json.loads(path.read_text())

    provenance = data.get("pytorch", {})
    if provenance.get("version") != BASELINE_VERSION:
        raise SystemExit(
            "ERROR: committed manifest baseline version mismatch: "
            f"{provenance.get('version')!r}"
        )
    if provenance.get("source_commit") != BASELINE_COMMIT:
        raise SystemExit(
            "ERROR: committed manifest source commit mismatch: "
            f"{provenance.get('source_commit')!r}"
        )

    files = data.get("files")
    if not isinstance(files, dict):
        raise SystemExit("ERROR: committed manifest has no valid files map")

    return data


def verify_preserved_manifest_files(data: dict) -> dict[str, str]:
    """Verify every non-managed fixture before append-only generation.

    Managed append-only families may be rewritten by this invocation.
    Everything else is immutable and must already match its committed SHA-256.
    """
    managed = set(APPEND_ONLY_FAMILY_SEEDS)
    preserved: dict[str, str] = {}

    for relative, expected_hash in sorted(data["files"].items()):
        rel = Path(relative)

        if (
            rel.is_absolute()
            or ".." in rel.parts
            or len(rel.parts) != 2
        ):
            raise SystemExit(
                f"ERROR: unsafe fixture path in manifest: {relative!r}"
            )

        if rel.parts[0] in managed:
            continue

        path = OUT / rel
        if not path.is_file():
            raise SystemExit(
                f"ERROR: immutable fixture is missing: {relative}"
            )

        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise SystemExit(
                "ERROR: immutable fixture hash mismatch: "
                f"{relative}: expected {expected_hash}, got {actual_hash}"
            )

        preserved[relative] = expected_hash

    actual_preserved = set()
    for path in OUT.rglob("*.json"):
        if path == OUT / "manifest.json":
            continue

        rel = path.relative_to(OUT)
        if len(rel.parts) != 2:
            raise SystemExit(
                f"ERROR: unexpected fixture layout: {rel}"
            )

        if rel.parts[0] in managed:
            continue

        actual_preserved.add(str(rel))

    if actual_preserved != set(preserved):
        missing = sorted(set(preserved) - actual_preserved)
        unlisted = sorted(actual_preserved - set(preserved))
        raise SystemExit(
            "ERROR: immutable fixture inventory mismatch: "
            f"missing={missing}, unlisted={unlisted}"
        )

    return preserved


def manifest(files: dict[str, str], actual_commit: str) -> None:
    m = {
        "pytorch": {
            "version": torch.__version__,
            "source_commit": actual_commit,
        },
        "generator": "docs/tensor-parity/provenance/generate_fixtures.py",
        "seed": SEED,
        "append_only_seeds": APPEND_ONLY_FAMILY_SEEDS,
        "generation_mode": "immutable-history+append-only",
        "dtype": "f32",
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
    print(
        f"  manifest: {len(files)} total files; "
        f"{len(GENERATED_FILES)} append-only files regenerated"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()

    actual_commit = verify_baseline_identity()

    global VERIFIED_BASELINE_COMMIT
    VERIFIED_BASELINE_COMMIT = actual_commit
    GENERATED_FILES.clear()

    existing_manifest = load_existing_manifest()
    preserved_files = verify_preserved_manifest_files(existing_manifest)

    with open(REGISTRY, "rb") as fh:
        registry = tomllib.load(fh)
    allowed = {op["name"] for op in registry["operator"]}

    # Only append-only families are executable. Historical generator functions
    # remain in this file for provenance/auditability but are unreachable here.
    for family, family_seed in APPEND_ONLY_FAMILY_SEEDS.items():
        gen = torch.Generator().manual_seed(family_seed)

        if family == "elementwise_broadcast":
            gen_elementwise_broadcast(
                gen,
                OUT / "elementwise_broadcast",
            )
        else:
            raise SystemExit(
                f"ERROR: unimplemented append-only family {family!r}"
            )

    generated_files = build_manifest_files(GENERATED_FILES, allowed)

    files = dict(preserved_files)
    files.update(generated_files)

    manifest(files, actual_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())