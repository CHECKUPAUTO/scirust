# SciRust Resonant Consensus Closure — SRCC v0.1

Status: experimental mathematical specification.

## 1. Objective

SRCC generates a structured rejection subspace without relying on:

- Cayley-Dickson multiplication;
- Clifford generators;
- a predefined null space;
- direct covariance truncation alone.

The structure emerges from deterministic agreement between independent
linear transport paths.

## 2. Ambient space

Let:

\[
V = \mathbb{R}^{16}
\]

with the standard inner product:

\[
\langle x,y\rangle
=
\sum_{i=0}^{15}x_i y_i.
\]

The initial seed family is:

\[
S=\{s_1,\ldots,s_p\}\subset V.
\]

The transport family is:

\[
\mathcal{T}
=
\{T_1,\ldots,T_m\},
\qquad
T_i:V\rightarrow V.
\]

The operators are arbitrary deterministic real-linear transformations.
They are not required to satisfy Cayley or Clifford relations.

## 3. Novelty residual

For a current orthonormal basis \(Q\), define:

\[
N_Q(x)
=
x-QQ^\top x.
\]

A proposal is novel when:

\[
\frac{\|N_Q(x)\|}{\max(\|x\|,\varepsilon)}
>
\tau.
\]

Here:

- \(\tau>0\) is the novelty threshold;
- \(\varepsilon>0\) prevents division by zero.

## 4. Directional alignment

For two non-zero residuals \(u\) and \(v\), define:

\[
A(u,v)
=
\frac{|\langle u,v\rangle|}
{\|u\|\|v\|}.
\]

The absolute value makes the structure invariant under sign reversal.

Two proposals resonate when:

\[
A(u,v)\geq\rho,
\]

where:

\[
0<\rho\leq1.
\]

## 5. Consensus support

A resonant direction is admitted only when it is produced by at least
\(c_{\min}\) transport paths.

A direction produced by a single unsupported path is treated as a
possible perturbation and is rejected.

This consensus rule distinguishes SRCC from ordinary Krylov closure.

## 6. Closure round

Let \(K_r\) be the current subspace.

Generate:

\[
G_r
=
\left\{
N_{K_r}(T_iq)
\; ;\;
q\in K_r,\ T_i\in\mathcal{T}
\right\}.
\]

Normalize all non-negligible proposals and group them into deterministic
resonance classes.

For every class with support at least \(c_{\min}\), select a canonical
representative using fixed traversal and accumulation order.

Then:

\[
K_{r+1}
=
K_r
\oplus
\operatorname{span}
\{\text{accepted representatives}\}.
\]

## 7. Termination

Closure terminates when one of the following occurs:

1. no new resonant direction is accepted;
2. the configured maximum dimension is reached;
3. the configured maximum number of rounds is reached.

Since \(V=\mathbb{R}^{16}\), dimension growth terminates after at most
sixteen independent admissions.

## 8. SRCC hard projector

Let \(Q_\star\) be an orthonormal basis of the final closure.

The hard projector is:

\[
P_{\mathrm{SRCC}}
=
I-Q_\star Q_\star^\top.
\]

It rejects every component belonging to the resonant closure.

## 9. Required invariants

The implementation must satisfy:

1. deterministic fixed-order accumulation;
2. finite termination;
3. monotonic closure dimension;
4. orthonormal final basis;
5. sign invariance of transport proposals;
6. scale invariance of normalized proposals;
7. projector symmetry within numerical tolerance;
8. projector idempotence within numerical tolerance;
9. exact preservation of vectors orthogonal to the closure;
10. rejection of unsupported isolated perturbations.

## 10. Planned extensions

Later versions may add:

- ordered transport disagreement:
  \[
  T_iT_jq-T_jT_iq;
  \]
- train/dev selection of transport families;
- soft resonance gains;
- temporal transport operators;
- multiscale closure;
- confidence and stability scores;
- automatic transport discovery.

## 11. Scientific controls

SRCC must be compared against Cayley and Clifford with:

- equal rejected dimension;
- identical loss functions;
- identical train/dev/test partitions;
- contamination sweeps;
- sample-efficiency curves;
- deterministic repeated runs.

## 12. Novelty statement

SRCC is an original SciRust design target combining novelty residuals,
transport-path agreement and consensus-gated finite closure.

This specification does not yet claim that every mathematical component
is absent from prior literature. A dedicated literature and prior-art
review is required before making a formal global novelty claim.
