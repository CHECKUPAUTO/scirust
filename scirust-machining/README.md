# scirust-machining

**Computation library for mechanical engineering and manufacturing — pure Rust, deterministic, no runtime dependency.**

`scirust-machining` groups **458 modules** (~1800 public functions, 2863 tests) covering the computation chain of mechanical engineering and manufacturing: from cutting kinematics to machine-element design, from fluid mechanics to thermal engineering, including vibrations, metrology, quality and energetics.

The documentation is in **French**, the identifiers (functions, types, parameters) in **English**.

```toml
[dependencies]
scirust-machining = { path = "…" }   # no runtime dependency
```

---

## Philosophy: honest engineering models

Each module follows three strict principles, verifiable in the code:

1. **No invented physical constants.** Material / process / installation quantities — cutting coefficient `kc1.1`, Taylor exponent `n`, Lewis form factor `Y`, friction coefficient `µ`, Bond work index `Wi`, Seebeck coefficient… — are **provided by the caller** from a catalog, a standard or trials. The crate computes their *consequences*; it never introduces an unverifiable "default" value. The only exposed constants are **universal** and named (π via `core::f64::consts`, Wien constant, Betz limit 16/27, μ₀…).

2. **An "honest limit" section per module.** The `//!` header of each file explicitly states the assumptions and validity domain of the model (steady state, linear elasticity, incompressible flow, gray body, small oscillations…) — what the formula does *not* say.

3. **Physical identity tests.** Each module verifies reciprocities, edge cases, proportionalities and one realistic worked example — not magic numbers — plus a panic test on invalid input. The functions guard their inputs with `assert!` and French messages.

```rust
use scirust_machining::{spindle_speed_rpm, mrr_turning_cm3_min};

// Turning of a Ø80 mm steel at Vc = 200 m/min, ap = 3 mm, f = 0.25 mm/rev.
let n = spindle_speed_rpm(200.0, 80.0);          // ≈ 796 rpm
let q = mrr_turning_cm3_min(200.0, 3.0, 0.25);   // 150 cm³/min
```

All identifiers are re-exported flat from the crate root; they are also reachable through their module (`scirust_machining::kinematics::spindle_speed_rpm`).

---

## Covered domains

The families below are **illustrative** (a sample per theme), not exhaustive — see the rustdoc for the complete list.

### Cutting & machining
Cutting kinematics (`Vc↔N`, `Vf`, MRR in turning/milling/drilling), force and power by the **Kienzle** model, tool life by **Taylor**, **Gilbert** machining economics, cutting time, theoretical roughness, **Merchant** shear angle, cutting temperature, drill geometry, broaching, hobbing, boring, taper turning, knurling.

### Forming & fabrication
Deep drawing, bending (developed length, K factor), roll forming, flat rolling, forging (Hollomon), extrusion, wire drawing, flow forming (sine law), blanking/punching, thread rolling, necking, powder compaction.

### Assembly & special processes
Welding (heat input, preheating, carbon equivalent, dilution, cooling, fillet weld, weld group), NDT (ultrasound, eddy current, radiography), casting (Chvorinov, risers, shrinkage), injection molding, EDM, ECM, laser/waterjet cutting, brazing, adhesive bonding, coating (electrolytic, anodizing).

### Machine elements
Gears (spur/helical/bevel/worm, **Lewis**, **ISO 6336**, epicyclic, backlash, scuffing, **Buckingham** wear and dynamic load), bearings (L10 **ISO 281**, static **ISO 76**, Palmgren friction, hydrostatic, PV), springs (helical, Belleville, conical, wave, constant-force, spiral, torsion), belts & chains, clutches and brakes (disc, cone, centrifugal, band, eddy-current), couplings (hydrodynamic, magnetic, torque converter), shafts, keys, pins, screws, bolting (preload, torque-angle tightening, ASME flange).

### Strength of materials & structures
Beams (reactions, deflections, distributed loads, continuous beam by the three-moment theorem, on Winkler elastic foundation), buckling (Euler, Rankine, plate, shell, beam-column), sections (moduli, shear centers, Jourawski shear flow), stresses (Mohr, von Mises, combined, concentration, Hertz), plasticity (Ramberg-Osgood, plastic bending), energy (Castigliano, strain energy).

### Dynamics, vibration & fatigue
Rigid-body kinematics/dynamics, mechanisms (four-bar, crank-slider, cams, Geneva drive, toggle), balancing, critical speeds, vibrations (1 and 2 dof, forced, isolation, Coulomb, tuned damper, unbalance, **ISO 10816**), fatigue (Goodman/Soderberg/Gerber, Coffin-Manson, Paris, Weibull, endurance).

### Fluid mechanics & hydraulics
Bernoulli, pressure losses (Darcy, Colebrook), flow measurement (Venturi, orifice plate, Pitot, rotameter), free surface (weirs, channel, hydraulic jump, underflow gate, Parshall), cavitation/NPSH, water hammer, surge tank, Stokes sedimentation, compressible flow (choked nozzle, isentropic), gas pipeline (Weymouth), siphon, ejector.

### Fluid machinery
Pumps (centrifugal, gear, vane, peristaltic, specific speed, affinity laws, NPSH), fans, compressors (reciprocating, Roots blower), hydraulic turbines, wind turbines (Betz limit), hydraulic/pneumatic cylinders, hydraulic press, hydrostatic bearing.

### Thermal & energetics
Conduction (steady/transient, resistances, fins, critical insulation), convection, radiation (Stefan-Boltzmann, view factors, resistance network, shields), heat exchangers (LMTD, NTU, fouling), two-phase (Nusselt condensation, Rohsenow/Zuber boiling), thermodynamic cycles, refrigeration, heat pump, psychrometry, combustion, boiler, cooling tower, thermoelectricity (Peltier, Seebeck).

### Processes & equipment
Silos (Janssen), grinding (Bond/Rittinger/Kick), cyclones, cake filtration, agitation, pneumatic conveying, conveyors (belt, screw, bucket).

### Metrology & quality
Dimensional inspection (sine bar, gauge blocks, Abbe/cosine error, optical flat, flatness, runout), GD&T, uncertainty, MSA, control charts, sampling, capability, Six Sigma (DPMO), Taguchi, FMEA (RPN).

### Production, economics & ergonomics
Scheduling (Johnson, CPM, PERT), line balancing, Little's law, takt time, EOQ, learning curve, forecasting, SMED, OEE, machine cost, profitability; ergonomics (NIOSH, WBGT, hand-arm and whole-body vibration).

### Instrumentation & mechatronics
Strain gauges and rosettes, thermocouple, actuators (reluctance solenoid, moving coil), piezoelectricity, motors (asynchronous, DC, three-phase, V/f drive, starting), control (PID, first/second order, Bode), vibration analysis (orders, bearing defect frequencies).

---

## Unit conventions

Coherent SI, with the usual datasheet conventions recalled by each function: `Vc` in m/min, lengths/diameters in mm, speeds in rpm, feeds in mm, forces in N, powers in kW or W, torques in N·m, pressures in Pa, temperatures in K (unless stated °C), angles in radians for trigonometric functions.

## Runnable examples

```bash
cargo run -p scirust-machining --example atelier
```

`examples/atelier.rs` chains the modules on a concrete turning case: cutting speed selection → spindle power check → tool life → economic optimum → time and cost → surface finish → tolerance → gear check.

## Tests

```bash
cargo test  -p scirust-machining          # 2863 physical identity tests
cargo clippy -p scirust-machining --all-targets -- -D warnings
```

## Position within SciRust

This crate completes the other mechanical bricks of the ecosystem: `scirust-tolerance` (inertial/statistical tolerancing, ISO 286/1101), `scirust-metrology` (GUM uncertainty), `scirust-fatigue` (rainflow, Palmgren-Miner) and `scirust-fab` (process control). It constitutes its **deterministic computation core**.

## License

See the `scirust` root repository.
