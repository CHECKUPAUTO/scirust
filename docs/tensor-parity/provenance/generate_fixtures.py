#!/usr/bin/env python3
"""Generate SciRust tensor-parity differential fixtures from the frozen PyTorch
baseline (2.13.0). OFFLINE tool: not run in CI. Output fixtures are data only.

Usage:
    python3 generate_fixtures.py [--torch-bin PATH] [--families elementwise,reductions,normalization,shape,linear,loss]

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
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
REGISTRY = REPO / "tensor-operators.toml"
OUT = REPO / "tests" / "parity" / "fixtures"
BASELINE_COMMIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"
SEED = 0xC0FFEE
MIN_TORCH = (2, 13, 0)

import torch  # noqa: E402  (import after sys.path manipulation if needed)


def import_torch(path: str | None) -> None:
    if path:
        import importlib.util

        # load torch from a specific interpreter is not supported; instead
        # document that the script must run under that interpreter.
        print(f"note: running --torch-bin {path}; ensure this interpreter matches", file=sys.stderr)


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


def write(op: str, family_dir: Path, cases: list[dict], note: str) -> None:
    path = family_dir / f"{op}.json"
    data = {
        "op": op,
        "kind": cases[0]["kind"],
        "dtype": "f32",
        "pytorch": {"version": torch.__version__, "commit": BASELINE_COMMIT,
                    "generated": "deterministic"},
        "note": note,
        "cases": cases,
    }
    family_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"  wrote {path.relative_to(REPO)} ({len(cases)} cases)")


def manifest(files: dict[str, str]) -> None:
    m = {"pytorch": {"version": torch.__version__, "source_commit": BASELINE_COMMIT},
         "generator": "docs/tensor-parity/provenance/generate_fixtures.py",
         "seed": SEED, "dtype": "f32", "files": files}
    (OUT / "manifest.json").write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
    print(f"  manifest: {len(files)} files, sha256 recorded")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch-bin", default=None, help="path to the baseline interpreter (informational)")
    ap.add_argument("--families",
                    default="elementwise,reductions,normalization,shape,linear,loss")
    args = ap.parse_args()
    import_torch(args.torch_bin)

    ver = tuple(int(p) for p in torch.__version__.split("+")[0].split("."))
    if ver[:3] != MIN_TORCH:
        raise SystemExit(
            f"ERROR: baseline must be 2.13.0, got {torch.__version__}. "
            "Refusing to generate non-baseline fixtures.")

    with open(REGISTRY, "rb") as fh:
        registry = tomllib.load(fh)
    allowed = {op["name"] for op in registry["operator"]}

    gen = torch.Generator().manual_seed(SEED)
    files: dict[str, str] = {}
    families = [f for f in args.families.split(",") if f]
    if (OUT / "manifest.json").exists():
        existing = json.loads((OUT / "manifest.json").read_text())
        for key, h in existing.get("files", {}).items():
            if key.split("/", 1)[0] not in families:
                files[key] = h
    for fam in families:
        fam_dir = OUT / fam
        if fam == "elementwise":
            gen_elementwise(gen, fam_dir)
        elif fam == "reductions":
            gen_reductions(gen, fam_dir)
        elif fam == "normalization":
            gen_normalization(gen, fam_dir)
        elif fam == "shape":
            gen_shape(gen, fam_dir)
        elif fam == "linear":
            gen_linear(gen, fam_dir)
        elif fam == "loss":
            gen_loss(gen, fam_dir)
        else:
            raise SystemExit(f"unknown family {fam}")
        for p in sorted(fam_dir.glob("*.json")):
            if p.name in ("manifest.json",):
                continue
            opname = p.stem
            if opname not in allowed:
                print(f"WARN: {opname} not in registry; skipping hash entry")
                continue
            files[f"{fam}/{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest(files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())