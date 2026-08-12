# Visualizations — `scirust-tolerance`

Self-contained HTML pages (no network dependency, light/dark theme) that make
the concepts of inertial tolerancing visible.

## `inertia_cone.html` — The inertia cone

Interactive tool for Pillet's **inertia cone**. A batch is judged by its
distance to the target in the `(δ, σ)` plane — the inertia `I = √(δ² + σ²)` —
and not by its margin to the tolerance interval.

- **(δ, σ) plane** — the acceptance map (top view of the cone): the inertia
  half-disc `I ≤ I_max` (Cpi ≥ 1) superimposed on the acceptance triangle
  `Cpk ≥ 1` of the classical method. At a glance one sees that a very precise
  but off-center batch, or a centered but dispersed one, can fall outside the
  cone even though Cpk would tolerate it.
- **3D cone** — the graph `z = I(δ, σ)` is a cone of revolution; accepting
  `I ≤ I_max` amounts to cutting it with a horizontal plane. Drag to rotate.
- **Distribution** — the batch density `N(μ, σ)` with respect to `[LSL, USL]`
  and the target, out-of-spec tails shaded.
- **Direct reading** — `I`, `I_max`, `Cpi`, `Cpm`, `Cp`, `Cpk` and the
  non-conformance in ppm, recalculated while dragging the batch point or the
  sliders (IT, target Cp, μ, σ).

The formulas reproduce exactly those of the crate: `InertiaCone`,
`Inertia`, `capability::cpi` — see `scirust-tolerance/src/`.

Open the file in a browser (`file://…/inertia_cone.html`), no compilation or
server required.
