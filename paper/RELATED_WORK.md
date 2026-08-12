# Related work

> Section written to be citable as-is in the paper (Lot 3).
> Bibliography deliberately restricted to verified references;
> no invented reference. Since the empirical "dead guards" study
> (`docs/DEAD_GUARDS_STUDY.md`) concluded NO-GO, this section does not contain
> a "measured prevalence of the bug class" subsection.

## 1. Classical floating-point reproducibility

The pitfalls of floating-point arithmetic have been documented for a long time.
Goldberg (1991, *What Every Computer Scientist Should Know About
Floating-Point Arithmetic*) establishes the foundation: floating-point addition is not
associative, rounding depends on the intermediate format, and two
mathematically equivalent evaluations can differ bit by bit. Monniaux (2008,
*The pitfalls of verifying floating-point computations*) shows that these
discrepancies are not only algebraic but **material and
compilatory**: 80-bit x87 registers, FMA contraction, flush-to-zero (FTZ/DAZ)
modes, fast-math options — the *same* source code produces
different results depending on the platform and the compiler. These two
references ground our position: bit-for-bit reproducibility is
never a given by default, it is a property that must be **built then
proven by execution**.

On the construction side, ReproBLAS (Demmel & Nguyen) is the classical foundation:
summations that are **reproducible and independent of the order** of the operands,
which make a reduction's result insensitive to parallel
scheduling. SciRust adopts the other possible route to the same invariant:
rather than making the sum order-insensitive, **freeze the order**
(sequential reduction in fixed worker order, sequential
orthogonalizations, fixed iteration budgets). Both approaches deliver a
bit-identical result in the face of parallelism; ours is simpler to
audit — the price is a sequentialization point, measured rather than denied.

## 2. Determinism for deep learning

Established frameworks treat determinism as a **best-effort
mode**. `torch.use_deterministic_algorithms` (PyTorch) forces
deterministic kernels *at fixed configuration*: same machine, same version,
same number of threads. The guarantee is run-to-run; it is **neither
bit-identical across different thread counts, nor across platforms**,
and some operations simply have no deterministic variant.
EasyScale (arXiv:2208.14228) extends the scope to elastic distributed:
**bit-identical** training under variation of the number of heterogeneous GPUs,
by preserving state per logical worker and freezing the effective order of
reductions — the proof that invariance to the degree of parallelism is
achievable at scale, at the price of dedicated engineering.

RepDL (Microsoft Research, 2025, arXiv:2510.09180) is, to our
knowledge, the strongest work on the portability axis: training and
inference **bit-for-bit reproducible across platforms** (different CPU/GPU),
obtained by (a) correct rounding of operations, in the lineage
of MPFR/RLIBM-style libraries, and (b) order invariance (frozen
sequential summations, fixed graphs). The respective positioning deserves
to be stated without beating around the bush. **On the cross-platform float32 axis, RepDL is
stronger than SciRust's f32 *sanitized* path**, which is only deterministic
intra-architecture. Conversely, RepDL is an **overlay on a PyTorch
runtime** — a C++/Python TCB of several million lines outside the
audit scope — limited to a subset of float32 operations, without
low precision (bf16/int8 explicitly out of scope), and its technical
report contains no evaluation section (neither benchmark nor measured
overhead). SciRust occupies the complementary niche: a **100% Rust
stack auditable end to end, without FFI in the compute path**, where
determinism is not only an execution property but a **piece of
evidence** — 64-bit inference fingerprints, hash-chained journals,
manifest-based reconstruction, every claim backed by a CI test — with a
fully integer int8 pipeline (bit-exact cross-platform by construction,
NEON kernel validated on embedded ARM) that the "correct f32 rounding" approach
does not cover. The two works are thus less competitors
than orthogonal: RepDL strengthens the *numerical kernel* of an existing
ecosystem; SciRust rebuilds the entire *chain of trust* above
a deliberately simpler kernel.

## 3. Cross-vendor GPU divergences: why a "sanitized" path

The systematic NVIDIA/AMD comparison (arXiv:2410.09172) documents
numerical differences between GPUs from different vendors executing the
same computation, aggravated by the fact that **floating-point exceptions are
not signaled** on GPU: underflow, divergent roundings, and subnormal
behaviors pass silently. This is precisely the regime that
SciRust's path 3 neutralizes *within one architecture*:
`sanitize_f32` flushes any subnormal (threshold = `f32::MIN_POSITIVE`,
aligned by test on the σ constant of `scirust-sigma`), removing from the
compute path the class of values whose handling differs the most between
hardware and modes (driver FTZ/DAZ). The mining campaign of
`docs/DEAD_GUARDS_STUDY.md` confirms the realism of this threat model —
9 of the 22 major numerical repositories scanned enable fast-math or control
FTZ in their builds — while honestly noting (NO-GO verdict)
that the "subnormal guard" bug class was not observed there: the
motivation of the sanitized path is the *portability of behaviors*, not
a prevalence of bugs in others. For the regimes where cross-platform
identity is required, SciRust provides the integer and fixed-point
paths, bit-exact cross-platform by construction — and, since 2026-07-10,
a **portable f32 path** (`scirust-core/src/portable_f32.rs`): pure-Rust exp/ln
without libm (argument reduction + series, internal f64 evaluation),
reference softmax and GEMM using only basic IEEE-754 operations
in fixed order, therefore bit-exact cross-platform *by construction*
(faithfully rounded; bit-for-bit goldens and FNV fingerprints committed as
a portability contract). The *proven* correct rounding of these transcendentals
(table-maker's dilemma) remains an explicit future work, in
dialogue with the RepDL approach.

## References

- D. Goldberg, *What Every Computer Scientist Should Know About
  Floating-Point Arithmetic*, ACM Computing Surveys, 1991.
- D. Monniaux, *The pitfalls of verifying floating-point computations*,
  ACM TOPLAS, 2008.
- J. Demmel, H. D. Nguyen, *ReproBLAS — Reproducible BLAS* (order-independent
  reproducible summation).
- PyTorch, `torch.use_deterministic_algorithms` (official documentation) —
  run-to-run determinism at fixed configuration.
- EasyScale, arXiv:2208.14228 — bit-identical elastic training on
  heterogeneous GPUs.
- RepDL (Microsoft Research), arXiv:2510.09180, 2025 —
  github.com/microsoft/RepDL.
- arXiv:2410.09172 — NVIDIA vs AMD numerical differences; unsignaled FP
  exceptions on GPU.
