# PyTorch Baseline (frozen reference)

The SciRust Tensor Parity Program compares SciRust against a **single, frozen,
publicly reproducible PyTorch build**. All coverage claims, differential tests
and the release oracle refer to this baseline and nothing else.

## The frozen reference

| Field          | Value                                                              |
| -------------- | ------------------------------------------------------------------ |
| Version        | **2.13.0**                                                         |
| Tag            | `v2.13.0`                                                          |
| Commit         | `cf30153c4c131c8164ee7798e5022d810682e2cb`                         |
| Release date   | 2026-07-08                                                         |
| Source         | https://github.com/pytorch/pytorch (git tag `v2.13.0`)             |
| Host Python    | 3.13.5                                                             |
| Collected on   | 2026-08-04 (this repository)                                       |

The commit hash is the authoritative anchor. A version string alone is not
sufficient: PyTorch patches behavior between patch releases of the same minor
version; SciRust parity tests must never depend on a moving target.

## Official PyTorch API inventory

The operator list used to seed the SciRust operator registry comes from the
PyTorch public API documentation and the operator database (`torch/_C/_VariableFunctions.pyi.in`, `torch/_C/_TensorMethods.pyi.in`, `torch/_refs/`, `torch/_decomp/`) at commit `cf30153c4c131c8164ee7798e5022d810682e2cb`. The registry records, per operator:

- the PyTorch reference symbol(s),
- the required `meta`/`fake` kernel availability (used for shape inference),
- the dtype/layout/device support matrix **as shipped in 2.13.0**,
- the aliasing semantics (out-of-place vs in-place vs `out=`).

See `OPERATOR_INVENTORY.md` for the SciRust-side inventory and
`../tensor-parity/EXCLUSIONS.md` for operators intentionally excluded from
parity.

## Local dev oracle (not the reference)

A local Python 3.13.5 environment with `torch 2.10.0+cu128` is available at
development time. It is useful for *exploratory* differential work, but:

- it is **not** the frozen reference,
- its results must never be committed as authoritative oracle output,
- the CI/test path that gates SciRust parity must not depend on Python or
  PyTorch being installed.

If the local oracle diverges from the frozen baseline (e.g. new `torch.compile`
defaults, changed promotion rules), the frozen baseline wins and the divergence
is recorded in `EXCLUSIONS.md` or the test skip registry.

## How the baseline is used

1. `tensor-operators.toml` entries carry a `pytorch = { version = "2.13.0", commit = "cf3015..." }` annotation for every row that claims parity.
2. The coverage matrix generator (`artifacts/tensor-coverage.json`/`.csv`) only
   marks a cell `parity` when the implementing SciRust operator has passed the
   differential harness against a PyTorch 2.13.0 witness.
3. Release notes must quote the baseline commit so a reviewer can reproduce
   the exact comparison.
