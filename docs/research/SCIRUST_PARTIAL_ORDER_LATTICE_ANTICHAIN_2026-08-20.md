# SciRust — Partial Orders, Lattices, and Antichains

Date: 2026-08-20

## Motivation

A literature review performed while studying incremental/distributed computation exposed a general algebraic capability that was absent from `scirust-algebra`: explicit mathematical partial orders, joins/meets, product orders, and finite minimal antichains.

The immediate motivating systems were Differential Dataflow and Naiad, but the abstraction is intentionally independent of those systems and of ElasticXxx.

General scientific uses include:

- causal and logical-time orders;
- distributed progress frontiers;
- multi-objective/Pareto boundaries;
- dependency/version orders;
- abstract interpretation and fixed-point methods;
- order-theoretic algorithms;
- algebraic dynamic systems.

## Repository audit

Before this addition, `scirust-algebra` exposed algebraic structures such as:

```text
Magma
Semigroup
Monoid
Group
Ring
Field
Semiring
```

Repository searches did not identify a reusable `PartialOrder`, `Lattice`, `Join`, `Meet`, `Poset`, or `Antichain` abstraction.

Rust's standard `PartialOrd` is not a replacement for the intended abstraction. In particular, tuples use language-defined ordering semantics, whereas many scientific/distributed applications need a **coordinate-wise product order** in which values can be incomparable.

## Added module

```text
scirust-algebra/src/order.rs
```

Public abstractions:

```text
PartiallyOrdered
JoinSemilattice
MeetSemilattice
Lattice
TotalOrder<T>
ProductOrder2<A,B>
Antichain<T>
```

## Mathematical contracts

### `PartiallyOrdered`

Implementors are responsible for a relation `<=` satisfying:

```text
reflexivity
transitivity
antisymmetry
```

Rust's trait system does not prove these laws.

### Join semilattice

For every `a,b`, `join(a,b)` must be their least upper bound.

### Meet semilattice

For every `a,b`, `meet(a,b)` must be their greatest lower bound.

### Lattice

A type implementing both join and meet semilattice operations automatically implements `Lattice`.

## Adapters / concrete constructions

### `TotalOrder<T>`

Wraps any ordinary `Ord` type. Join is `max`; meet is `min`.

### `ProductOrder2<A,B>`

Coordinate-wise product order:

```text
(a1,b1) <= (a2,b2)
iff
a1 <= a2 && b1 <= b2
```

Two values can therefore be incomparable even when each coordinate is totally ordered.

## `Antichain<T>`

Maintains a deterministic finite set of pairwise-incomparable **minimal** elements.

Insertion semantics:

1. if an existing element is `<=` the new element, the new value is dominated and rejected;
2. otherwise, existing values dominated by the new element are removed;
3. the new element is appended.

For identical insertion sequences, surviving element order is deterministic.

The implementation uses a `Vec<T>` because antichains are intrinsically variable-size. This is not intended as a lock-free/high-frequency distributed frontier implementation. Performance-specialized frontiers can be built separately when a concrete workload requires them.

## Scope discipline

This module deliberately does **not** contain:

```text
TimelyTimestamp
DifferentialTrace
ElasticVersion
DistributedProgressTracker
```

Those are domain/runtime concepts.

The algebra crate supplies only reusable mathematical structure.

## Validation status

Unit tests are included for:

- total-order join/meet;
- coordinate-wise incomparability;
- antichain dominance/minimality.

At the time this note was written, GitHub had not reported CI status for the integration commit. The feature should therefore be considered integrated on `master` but not release-qualified solely from this note.

## Possible future work — only if independently justified

Do not automatically add these merely because distributed dataflow uses them:

- bounded lattices / top and bottom traits;
- distributive lattices;
- complete lattices;
- fixed-capacity/no-allocation antichains;
- antichain join/meet algorithms;
- distributive fixed-point solvers;
- law-checking/property-test helpers.

Each requires an independent scientific need and repository/state-of-the-art audit.
