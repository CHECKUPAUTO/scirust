# SciRust Algebra Phase 2

Phase 2 extends the algebra crate merged by PR #973 with algorithms that were explicitly left as follow-up work.

Implemented in this branch:

- deterministic fixed-capacity Todd-Coxeter coset enumeration;
- compact `u16` coset tables with generator/inverse column pairs;
- coincidence handling through deterministic union-find merging;
- subgroup-index enumeration without heap allocation;
- tests for `C2`, the Klein four-group, and a subgroup of index two.

Next on this branch:

- deterministic Schreier-Sims stabilizer chains and sifting;
- stronger presented-group word reduction / completion support;
- Lie and representation-theory extensions.

All code remains subject to the workspace constraints: Rust 1.89 MSRV, `#![forbid(unsafe_code)]`, deterministic execution, and strict formatting/lint gates.
