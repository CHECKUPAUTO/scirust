# SciRust — Industrial domains to open (market roadmap)

Complement to `INDUSTRIAL_ROADMAP.md` (go-to-market) and `INDUSTRIAL_VERTICALS.md`
(implementation of the verticals already under way: PdM, estimation, OT safety). Here:
the result of targeted market research on the **regulated** sectors where
SciRust's bit-exact determinism and total auditability constitute a
*measurable* advantage, not just a marketing argument — and which are
**not** already covered by the existing crates (`scirust-signal`,
`scirust-opcua`, `scirust-mqtt`, `scirust-pdm`, `scirust-mlops`,
`scirust-func-safety`, `scirust-estimation`, `scirust-nav`, `scirust-water`,
`scirust-ids`, `scirust-hvac`, `scirust-bms`, `scirust-biomed`, `scirust-grid`,
`scirust-shm`, `scirust-spc`, `scirust-robotics`, `scirust-metrology`,
`scirust-reliability`).

## Doctrine (identical to `INDUSTRIAL_VERTICALS.md`)

1. Pure Rust, zero FFI, bit-exact determinism (seeded PRNG, fixed reduction order).
2. No claim without a test — an honest oracle, not a stub.
3. The differentiator is always a **guarantee**: reproducibility,
   hash-chained traceability, certified bound, compliance with a named standard.
4. Each new domain must go through the single connector described at the end of
   this document (`scirust-mcp`) — an added domain immediately becomes drivable
   by an agent (the `scirust-sciagent` SLM, an external LLM, or a script), without
   specific glue code.

## Why these sectors: the documented common denominator

The literature of every regulated sector documents the **same** friction
point with the dominant tooling (Python/NumPy/SciPy, MATLAB/Simulink, "black-box"
ML): floating-point non-associativity, non-deterministic BLAS threading and the
lack of traceability break the reproducibility required by
their own standards.

- Intel documents the absence of a bit-exact guarantee even at a fixed thread
  count (non-associativity + FMA + compiler reordering).
- MathWorks documents that the simulation ⇄ generated-code correspondence
  (SIL/PIL) is only guaranteed "within tolerance", hence the ISO 26262
  obligation of redundant MIL/SIL/PIL/HIL tests.
- OpenBLAS threading bugs have produced silently
  wrong results (public issue openblas#1844); scikit-learn itself documents
  not controlling the determinism of the underlying BLAS threading.
- DO-178C (aeronautics) and IEC 62304 Edition 2 (medical devices) run
  explicitly into trained ML: the required traceability "assumes
  deterministic behavior" that a trained network does not guarantee — hence
  the new EASA (AI Concept Paper) and FDA/IMDRF (GMLP, PCCP) frameworks that
  explicitly demand algorithmic transparency.

This is exactly the niche that `docs/GROWTH_PLAN.md` already claims
("certifiable, reproducible and auditable AI") — the domains below
are where this niche has the most documented demand.

## Domains ranked by strength of the evidence found

### D1 · Process functional safety (IEC 61511/61508 — SIS) — ✅ done
- **Customer**: petrochemical, fine chemical, refining — safety-instrumented
  systems (SIS).
- **Why now**: the Triton/Trisis attack (2017) re-flashed the safety
  logic of a Schneider Triconex without being detected until an accidental
  trip — the textbook case for *non-auditable* safety logic.
- **Algorithms**: PFDavg/SIL calculation per voting architecture (1oo1, 1oo2,
  2oo2, 2oo3, 1oo3), proof test intervals (numerical inversion via
  `scirust-solvers::roots::bisection`), cause-and-effect matrices, hash-chained
  log of the voting logic (SHA-256, modeled on
  `scirust-mcp::audit`/`scirust-discovery::audit`).
- **Delivered**: `scirust-reliability` (already present, completed with the
  missing 2oo2/1oo3 architectures) for the quantitative computation, and the
  new `scirust-sis` for the systems layer (full SIF loop,
  voting simulation with fault injection, cause-and-effect matrices,
  test interval sizing, audit log) — exposed as
  MCP tools (`sis_verify_sif_loop`, `sis_size_proof_test_interval`). See
  `scirust-sis/README.md`.
- **Size**: small to medium — the fastest path to a differentiating
  "audit-grade" product.

### D2 · Electric grid protection & state estimation (IEC 61850, NERC CIP, IEEE C37.118) — ✅ done
- **Why now**: the post-mortem report of the 2003 North American
  blackout points to a state estimator whose failure could not be
  reconstructed; the GOOSE/Sampled Values protocols of IEC 61850 are
  demonstrably spoofable (cited academic literature).
  `scirust-grid` already existed (frequency/RoCoF/synchrophasors/THD) but
  without the state-estimation / protection layer.
- **Algorithms**: weighted least squares state estimation (WLS,
  closed-form solution `x̂=(HᵀWH)⁻¹HᵀWz` via `scirust-solvers`), bad-data
  detection (global χ² test + largest normalized residual,
  Abur & Expósito §5.3-5.4), distance relay logic with
  multi-zone mho characteristic (IEEE C37.113 §5.2).
- **Delivered**: `scirust-grid::state_estimation` (`wls_state_estimate`,
  `chi_squared_test`, `largest_normalized_residual_test`, verified against an
  independently computed 3-node example) and
  `scirust-grid::distance_relay` (`DistanceRelay`, mho comparator
  `mho_operates`, zones with configurable reach/delay). Deliberately left out
  of scope: iterative non-linear AC state estimation (Newton-Raphson
  on non-linear `h(x)` — the `scirust-solvers::nonlinear` solver would
  allow it but is not wired here), and the χ² test thresholds remain
  the caller's responsibility (no reimplementation of the incomplete
  gamma function inverse, as in `scirust-multivariate`).
- **Size**: medium — builds on the estimation already present.

### D3 · Closed-loop medical devices (IEC 62304 Ed.2, FDA SaMD/GMLP/PCCP) — ✅ done (techniques; not a device)
- **Why now**: the future edition of IEC 62304 adds a dedicated AI/ML
  lifecycle precisely because an adaptive model does not fit into the
  standard's historical deterministic model.
  `scirust-biomed` already existed (signal processing); the opening here
  was **control** (dosing), not just signal analysis.
- **Algorithms**: PID with conditional anti-windup (the controller class of
  1st-generation hybrid systems such as Medtronic 670G/770G), active-insulin
  tracking (IOB, exponential decay), threshold-based
  supervision (suspension on low glucose + predictive variant, exit from
  automatic mode — the publicly documented "Suspend on Low"/SmartGuard
  principle), and a **Control Barrier Function** safety filter
  (Ames et al., IEEE TAC 2017) solved in closed form (CBF-QP with 1 decision
  variable) — the certifiable alternative to ad-hoc guardrail tuning
  cited in the domain doctrine.
- **Delivered**: `scirust-biomed::control` (`pid`, `iob`, `insulin_safety`,
  `barrier`). **Explicitly not delivered, per the anti-overpromising doctrine**:
  no clinical IOB curve (Walsh/OpenAPS/LoopKit — the module uses a
  generic mono-exponential decay), no validated physiological model
  (Bergman/UVA-Padova — the CBF uses a 1-compartment affine model), no MPC
  (recent systems — Tandem Control-IQ, Omnipod 5, CamAPS FX — use it, out of
  scope), no PCCP traceability nor IBP/CROWN bounds. What is delivered is the
  *technique* of certifiable control (PID + CBF-QP + supervision), not an
  approvable dosing algorithm — each code module carries this warning at its head.
- **Size**: medium to large (heavy regulatory requirements) — a real
  device would remain a clinical/regulatory partnership effort,
  not a solo sprint.

### D4 · Aeronautics — flight control laws & structural fatigue (DO-178C/DO-333) — ✅ done (fatigue counting; flight control laws = partnership)
- **Why now**: DO-178C traceability assumes deterministic
  input→output behavior; it is documented as broken by
  floating-point non-associativity and by any embedded ML component.
- **Algorithms**: rainflow counting (ASTM E1049-85 §5.4.4) for fatigue
  lifetime, Palmgren-Miner rule for damage accumulation.
- **Delivered**: `scirust-fatigue` (`rainflow`, `miner`) — port of the
  standard's stack-based algorithm, verified against the PyPI reference
  library `rainflow` (dedicated ASTM E1049-85 implementation) on two independent
  sequences, value by value (range/mean/count/indices).
  **Explicitly not delivered**: the deterministic fixed-point numerical
  flight control laws and the certified bounds for a learned
  component — that part remains, as documented below, an aeronautical
  certification effort requiring a partnership, not a solo
  sprint.
- **Size**: large — aeronautical certification expertise required
  for the flight-control part; to be treated as a partnership rather
  than a solo sprint.

### D5 · Autonomous maritime & DNV classification (IMO MASS Code 2026, DNV AROS, IACS UR E26/E27) — ✅ done (primitives)
- **Why now**: the new MASS code (mandatory, 2026) requires that
  autonomous decisions remain "explainable and auditable" with no
  industry-consensus verification method yet — a window of
  opportunity to set a reference standard.
- **Algorithms**: COLREG geometric encounter classification
  (head-on/crossing/overtaking from the relative bearing,
  Rules 13-15), collision risk assessment via CPA/TCPA
  (straight-line trajectories), weighted pseudo-inverse thrust
  allocation for dynamic positioning (DP).
- **Delivered**: `scirust-maritime` (`colregs`, `cpa_tcpa`,
  `thrust_allocation`), verified against an independent worked CPA/TCPA
  example (two vessels, TCPA≈54.5 min, CPA≈3.41 nm) and an
  over-actuated 4-thruster DP configuration (compared to the numpy
  Moore-Penrose pseudo-inverse). **Explicitly not delivered**: the full
  regulatory status logic of Rules 11-18 (sail vs
  mechanical propulsion, restricted maneuverability), the complete DP
  control loop (observer, reference model, PID/MPC 3-DOF —
  this crate takes the desired generalized force as input), and
  stability/seakeeping (out of scope, not addressed).
- **Size**: medium.

### D6 · Run-to-run control in semiconductor manufacturing (SEMI E10/E58/E116) — ✅ done
- **Why now**: the R2R controller feeds the output of statistical
  process control (FDC/virtual metrology) directly back into the
  next run's recipe — a silent numerical drift costs
  wafers; the audit standards there are close to the 21 CFR Part 11 already
  handled by `scirust-func-safety::golden_batch`.
- **Algorithms**: EWMA run-to-run control (Sachs, Hu & Ingolfsson 1995),
  multivariate FDC T²/SPE via PCA (Kourti & MacGregor 1995) — reuses the general
  SVD from `scirust-solvers` rather than reduplicating it.
- **Delivered**: `scirust-fab` (`r2r::EwmaR2rController` verified against a
  worked example and a geometric convergence proof;
  `pca::Pca` with complementary `t2`/`spe`, verified against three numpy
  cases — steady-state point, correlation break captured by SPE, excursion
  along the known correlation captured by T²). `scirust_spc` (already
  existing: `EwmaChart`, `HotellingT2`, Western Electric rules)
  already covered basic univariate/multivariate SPC — this crate adds the
  *control* (R2R) and *PCA* layer on top, without duplicating the existing.
- **Size**: large (broad statistical surface) — the base brick is
  delivered; automatic `k`/UCL threshold selection remain the
  caller's responsibility (see the honest limitation in `scirust-fab::pca`).

### D7 · Precision agriculture — compliance & traceability (ISO 25119, ISO 18497, ISOBUS/ISO 11783) — ✅ done (partially, per the anti-overpromising doctrine)
- **Why now**: a documented case shows that the same yield
  data, run through QGIS / Agro-Map / Farm Works, produces *different* yield
  maps — a concrete and published reproducibility break. Phytosanitary
  registers and carbon MRV increasingly require an
  inviolable timestamped trace.
- **Algorithms**: global + local outlier filters (Sudduth & Drummond
  2007, the USDA-ARS "Yield Editor" reference tool), explicit IDW
  interpolation, ISO 25119-2 risk parameter model (Severity/
  Exposure/Controllability).
- **Delivered**: `scirust-agtech` (`outlier_filter`, `idw`) — a deterministic,
  auditable yield-map cleaning pipeline, verified by a
  constructed case where a global filter *structurally* cannot
  distinguish a legitimate point from a same-valued anomaly while the
  local filter can. `agpl` exposes the data model of the three
  ISO 25119-2 risk parameters (verified against the normative text
  — iTeh Standards previews of the 2010/2019 editions, tables 1-3).
  **Explicitly not delivered, per the anti-overpromising doctrine**: the
  `S×E×C → AgPL` decision function itself (the risk graph of Figure
  1, §6.3.7) appears in no verifiable open source found — the
  only secondary reproduction available (Mitka 2018) contradicts the verified
  normative text (invents an "S4" level, reduces the output to 3
  categories) and was judged unreliable. Coding a *guessed* functional safety
  graph topology would be worse than coding nothing — see
  `scirust-agtech::agpl` for the detail. Likewise not delivered: the SRP/CS
  architecture categories (Annex A) and the SRL level, whose
  correspondence tables could not be verified; hash-chained phytosanitary
  treatment log (out of scope for this pass).
- **Size**: medium.

### D8 · Nuclear — reactor protection (IEC 61513/60880/62138) — ✅ done (voting primitive; licensing = partnership)
- **Why now**: the IAEA and the cited academic literature name
  common-cause software failure across redundant channels as an
  unresolved licensing point; no open platform of this level
  exists today.
- **Algorithms**: 2-out-of-4 voting logic with channel bypass
  (IEC 61513 §6.2.3.5) — a channel in maintenance/surveillance reduces `N`
  without changing `M`.
- **Delivered**: `scirust-sis::reactor_trip` (`architecture_with_bypass`,
  `pfd_avg_during_bypass`), built entirely on the already-verified
  primitives of `scirust-sis::voting::Architecture` and
  `scirust_reliability::pfd_moon` (2oo4 included) — no new
  unverified formula. **Explicitly not delivered, per the anti-overpromising
  doctrine**: the ISA-67.04 threshold calculation methodology
  (Analytical Limit → Trip Set Point via SRSS → Nominal Trip Set Point →
  Limiting Trip Setpoint) and the NUREG-0800 BTP 7-19 common-mode
  failure fallback requirements — researched and documented, but
  not ported into code for lack of verification judged sufficient for
  nuclear safety code in this pass.
- **Size**: modest LOC, but very high licensing expertise — to be
  approached only in partnership with a qualified operator/integrator.

*(Rail EN 50128/50716 and mining ISO 17757 were also
studied: the documented pain there is verification/model-checking
complexity, not numerical reproducibility — less specifically aligned
with SciRust's determinism/auditability DNA; to be revisited if a
sector partner shows up.)*

## What makes all these domains executable without code explosion

The common point of these eight domains is not a single algorithm: it is
that they all require (a) a solid numerical brick (least squares,
eigen/SVD, constrained optimization, filtering), already strengthened in
this iteration (`scirust-solvers` — see `CHANGELOG.md`), and (b) a standard
way to plug those bricks into a client's real infrastructure
(sensors, PLCs, historians) and onto an agent that orchestrates it all.
That is the role of the two new crates of this iteration:

- **`scirust-mcp`** — exposes any SciRust capability (solver, PdM, signal,
  discovery) as a standard [Model Context Protocol](https://modelcontextprotocol.io)
  tool, callable by `scirust-sciagent` or by any external
  agent, with JSON schema and a hash-chained audit log per call. A
  new domain only has to register its tools in the existing registry.
- **`scirust-discovery`** — finds, safely and with consent (IEC 62443
  zone/conduit model, protocol-native discovery rather than generic
  scanning), the industrial hardware actually present on a client's
  network, so the agent knows *what* to connect those tools to.

See `scirust-mcp/README.md` and `scirust-discovery/README.md` for the technical
detail and the cited sources.
