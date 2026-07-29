# Layer 3.6 — BSSN on a periodic two-dimensional grid

Layer 3.5 generalised the grid substrate to `D` dimensions and made the BSSN
system dimension-generic. It validated the two-dimensional **right-hand side**
and showed Minkowski to be exactly stationary — but it never measured that a
two-dimensional **evolution** converges or stays bounded.

That gap is the reason this increment exists, and the precedent is one layer
back: Layer 3.4 shipped a right-hand side that converged cleanly at order 2 and
an evolution that blew up. An accurate right-hand side is not evidence of a
working solver. This increment supplies the missing measurement, together with
the deterministic experiment and benchmarks every prior increment carried and
3.5 did not.

## 1. What is new here, and what is not

**Not new**: the equations. `bssn.rs` takes complete 3x3 derivative tensors and
never assumes how many axes vary, and `bssn_grid` has been generic over the grid
dimension since 3.5. Nothing in the physics changed.

**New**: evidence. A genuinely two-dimensional closed-form evolution oracle,
long-time stability measurements, a deterministic 2D experiment, and 2D
benchmarks — plus one bug fix that the experiment itself exposed.

## 2. The diagonal gauge wave — a genuinely 2D closed form

On flat space with `K = 0`, 1+log slicing linearises to
`d_t^2 alpha = 2 grad^2 alpha`. For a diagonal perturbation,
`grad^2 sin(k(x+y)) = -2 k^2`, so

```text
alpha(t, x, y) = 1 + A cos(2 k t) sin(k (x + y))
```

exactly. Both axes vary, which no one-dimensional configuration can arrange.

Two choices in the measurement are load-bearing, and **both were got wrong on
the first attempt**:

- **Sampling time.** `t_end = 1/8` gives `omega t = pi/2`, where a phase error
  appears at *first* order as an amplitude error. The first attempt used
  `t = 1/4`, which lands on an *extremum* of the cosine, where a phase error only
  shows at second order. It reported a flattering order of ~4 and errors two
  decades smaller than the truth.
- **Timestep.** Fixed, not tied to `dx`. Tying them together refines space and
  time simultaneously and measures neither order cleanly.

Measured, `A = 1e-6`, `k = 2 pi`, `dt = 1/1024`:

| N | `L∞(alpha − exact)` | relative | order |
| --- | --- | --- | --- |
| 16 | `1.007408e-8` | `1.01e-2` | — |
| 32 | `2.522578e-9` | `2.52e-3` | **2.00** |

## 3. The mixed derivatives, with a bug to show for them

The conformal Ricci reconstruction with `d_x d_y` genuinely non-zero:

| N | `L∞(R_bssn − R_generic)` | `‖R‖` | `‖Γ̃^k‖` | order |
| --- | --- | --- | --- | --- |
| 16 | `1.542885e-2` | `3.763e-1` | `6.123e-2` | — |
| 32 | `3.939523e-3` | `3.916e-1` | `6.243e-2` | **1.97** |
| 64 | `9.901010e-4` | `3.955e-1` | `6.273e-2` | **1.99** |

This converges only because Layer 3.5 fixed an index error in
`Gammatilde^k Gammatilde_{(ij)k}`: `Gammatilde^k` contracts the *last* index and
the implementation contracted the first. One dimension could not detect it,
because `Gammatilde^k` is nearly zero there — the `‖Γ̃^k‖` column is reported
here precisely to show that this configuration does exercise the term.

## 4. Stability

The measurement Layer 3.5 omitted. Diagonal lapse perturbation, `A = 1e-3`,
1+log slicing, evolved to `t = 2`:

| N | total points | C = 0.25 | C = 0.5 |
| --- | --- | --- | --- |
| 8 | 64 | bounded (`1.000800`) | bounded (`1.000738`) |
| 16 | 256 | bounded (`1.000988`) | bounded (`1.000985`) |
| 32 | 1024 | bounded (`1.000999`) | bounded (`1.000999`) |

Bounded near unity — the lapse is `O(1)` and every other field is `O(A)` — and
*not* worse under refinement, which was the signature of the Layer 3.4
instability. The determinant constraint falls with resolution
(`3.6e-10 -> 1.5e-11 -> 5.2e-13` at `C = 0.25`).

**Stability here is measured, not proven.** Strong hyperbolicity is an analytic
property of the continuum system, and nothing here establishes it.

## 5. A typed error that nobody could see

The experiment exposed a real defect. An anisotropic grid was refused — but with
the message *"state became non-finite; reduce the step size"*, not the typed
`AnisotropicGrid`. The guard lived only inside the right-hand side, which
`System::derivatives` calls; that signature cannot return an error, so the
failure surfaced as a `NaN` and the caller got advice that was both useless and
wrong for a grid whose cells are not square.

The check now runs up front in `evolve_bssn_grid`. A typed error nobody ever
sees is not a typed error, and only running the experiment revealed it.

## 6. Benchmarks

Machine-dependent wall clock; the computation is deterministic.

| operation | N=8 (64 pts) | N=16 (256 pts) | N=32 (1024 pts) |
| --- | --- | --- | --- |
| 2D BSSN right-hand side | 771 µs | 3.10 ms | 12.67 ms |

Ratios 4.02 and 4.09 under axis doubling — linear in the *total* point count, so
the cost of a second dimension is the `N^2` growth of the grid and nothing else.
About 12.4 µs per grid point, comparable to the one-dimensional figure.

## 7. Determinism

No RNG, no wall clock, no parallelism. Points are visited in ascending linear
index order and reductions accumulate in that order. The experiment is
byte-identical across runs (4397 bytes).

## 8. Known limitations

- **Two dimensions, not three.** The third axis is supported by the grid type
  but has no evolution validation whatsoever.
- **Square cells only.** Anisotropic grids are refused, because the Layer 3.3
  conversion path differences with a single step.
- **Periodic domains only.** No outer boundary, radiative or otherwise.
- **Weak, smooth fields only.** No puncture, no horizon, no strong field — which
  is what the moving-puncture gauge exists for and the only setting in which its
  behaviour is really tested.
- **Stability is measured, not proven**, and only over the times and resolutions
  tabulated above.
- No black holes, no punctures, no excision, no AMR, no waveform extraction, no
  observational validation.
