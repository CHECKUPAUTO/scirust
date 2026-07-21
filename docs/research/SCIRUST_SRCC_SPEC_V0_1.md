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

Normalize all non-negligible proposals, canonicalize their signs using
the coordinate of greatest absolute magnitude, and sort them
deterministically.

Resonance classes use complete-link agreement: a proposal joins a class
only when it resonates with every existing member of that class.

For a resonance class \(C\), let \(C_i\) contain the proposals generated
by transport \(T_i\). Using a canonical anchor \(a\), define one vote per
transport:

\[
v_i
=
\operatorname{normalize}
\left(
\sum_{u\in C_i}
\operatorname{sign}
igl(\langle a,uangleigr)u
ight).
\]

The class representative is:

\[
r_C
=
\operatorname{normalize}
\left(
\sum_{i:C_i
eqarnothing}v_i
ight).
\]

Therefore, repeated proposals from one transport cannot outweigh the
other independent transport paths.

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

## 9. Data-driven transport discovery

For an explicit view \(D_j\) containing source-target pairs
\((x_k,y_k)\), SRCC learns:

\[
T_j
=
rac{1}{|D_j|}
\sum_{(x_k,y_k)\in D_j}
rac{y_kx_k^	op}{\|x_k\|^2}.
\]

Samples inside every view are sorted canonically before accumulation.
The learned operator is therefore invariant to sample order.

Explicit views are the primary scientific interface because their
independence is supplied by the experiment rather than inferred from
array position.

A secondary convenience interface can construct deterministic views
from a flat sample family by canonical grouping.

## 10. Structural selection

Candidate structures may vary:

- resonance threshold;
- number of inferred views;
- other validated SRCC configuration parameters.

Each candidate is scored on separate train and development cases using:

\[
L
=
R_{\mathrm{noise}}
+
\lambda D_{\mathrm{signal}},
\]

where \(R_{\mathrm{noise}}\) is the mean residual-noise ratio and
\(D_{\mathrm{signal}}\) is the mean signal-distortion ratio.

The selected SRCC candidate is activated only when its development loss
is strictly lower than the identity transform.

## 11. Leave-one-out stability

For explicit views, remove each removable sample in turn, refit SRCC,
and compare the reduced projector \(P_{-k}\) with the full projector
\(P\).

The normalized Frobenius distance is:

\[
d_F(P,P_{-k})
=
rac{\|P-P_{-k}\|_F}{\sqrt{16}}.
\]

The stability report contains:

- mean Frobenius distance;
- maximum Frobenius distance;
- rejected dimension after each removal;
- proportion of removals preserving the full rejected dimension.

A stability-gated search accepts a candidate only when both:

\[
\max_k d_F(P,P_{-k})
\leq
\delta_{\max},
\]

and:

\[
rac{
\#\{k:\dim K_{-k}=\dim K\}
}{
\#\{k\}
}
\geq
\gamma_{\min}.
\]

If no candidate passes the stability gate, the identity transform is
selected.

## 12. Required invariants

The implementation must satisfy:

1. deterministic fixed-order accumulation;
2. finite termination;
3. monotonic closure dimension;
4. orthonormal final basis;
5. seed sign and scale invariance;
6. transport sign and scale invariance;
7. seed-family order invariance;
8. transport-family order invariance;
9. explicit-view order invariance of the final projector;
10. sample-order invariance inside explicit views;
11. complete-link resonance agreement;
12. one normalized consensus vote per transport;
13. projector symmetry within numerical tolerance;
14. projector idempotence within numerical tolerance;
15. preservation of vectors orthogonal to the closure;
16. rejection of unsupported isolated perturbations;
17. deterministic train/development selection;
18. deterministic leave-one-out stability evaluation.

## 13. Implemented validation oracles

The current implementation includes deterministic tests for:

- exact single-round consensus;
- certified multi-round closure;
- unsupported transport outlier rejection;
- pairwise resonance enforcement;
- sign, scale and ordering invariance;
- equal transport weighting;
- explicit transport views;
- learned transport fitting;
- train/development structural search;
- identity fallback;
- leave-one-out stability gating;
- dense non-axis-aligned approximate consensus;
- projector symmetry, idempotence, rejection and preservation.

## 14. Planned extensions

Later versions may add:

- ordered transport disagreement:
  \[
  T_iT_jq-T_jT_iq;
  \]
- soft resonance gains;
- temporal transport operators;
- multiscale closure;
- bootstrap and repeated holdout stability;
- sparse and structured transport estimators;
- real-signal benchmark suites;
- formal complexity bounds.

## 15. Scientific controls

SRCC must be compared against Cayley and Clifford with:

- equal rejected dimension;
- identical loss functions;
- identical train/dev/test partitions;
- contamination sweeps;
- sample-efficiency curves;
- deterministic repeated runs;
- leave-one-out and repeated-holdout stability;
- dense non-axis-aligned synthetic oracles;
- explicit reporting of identity fallback frequency.

## 16. Novelty statement

SRCC is an original SciRust design target combining novelty residuals,
transport-path agreement and consensus-gated finite closure.

This specification does not yet claim that every mathematical component
is absent from prior literature. A dedicated literature and prior-art
review is required before making a formal global novelty claim.
