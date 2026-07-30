# Resolution coverage — the call/data-flow resolver across Rust path shapes

> Reproduce: `cargo run --release --example resolution_coverage`

The call graph (`Calls`: fn → fn) and data-flow graph (`DataFlow`: fn → `static`/`const`) grew slice
by slice, each closing one path shape under the same discipline: **resolve-uniquely-or-skip**, so a
wrong edge is never invented. This measurement is the arc's capstone — it enumerates every shape the
resolver now handles (tagged with the slice that added it), asserts the shapes it *deliberately
skips*, and reports the structural yield on CCOS's own `src/`. Two runs are bit-identical.

## What resolves — 14/14 idioms

| shape | example | slice |
|---|---|---|
| crate-rooted call | `crate::m::f()` | call crux |
| imported fn | `use m::f; f()` | Tier A |
| imported module | `use crate::m; m::f()` | Slice 2 |
| **local submodule path, no `use`** | `mod m; m::f()` | **#122** |
| **nested submodule path, no `use`** | `a::b::f()` | **#122** |
| **typed receiver method (incl. `&T`)** | `fn r(x: &T) { x.bar() }` | **#23 + #126** |
| `Self` method | `self.helper()` | #20 |
| bare const (globally unique) | `FOO` | data-flow Slice 1 |
| **import-scoped const** | `use m::MAX; MAX` | **#113** |
| **renamed const import** | `use m::MAX as L; L` | **#124** |
| **field receiver** | `self.db.q()`, `s.db.q()` | **S4** |
| **fn-return receiver** | `make().q()`, `let x = make(); x.q()` | **S4** |
| **method chain (single & multi hop)** | `x.b().q()` | **S4** |
| **assoc `-> Self` chain** | `A::make().t()` | **S4** |

The bolded rows are the additions of this arc and of **Slice 4** (the paper's "richer receiver
inference" future-work item — field receivers and return-type chains). Slice 4 types a compound
receiver from the same *syntactically-certain declarations* Slice 3 trusts, extended to two new
fact families collected per scope: **struct field types** (`struct S { db: Db }` ⇒
`(S, db) → Db`) and **declared return types** (`fn make() -> B`, `impl A { fn b(&self) -> B }`,
with `-> Self` resolved to the impl's concrete type). A receiver expression is then typed
recursively — locals, `self`, field accesses, calls, chains, `&expr`, parens — and anything
uncertain still yields nothing. Two precision rules matter:

- **Evidence beats convention.** `let x = C::new()` infers `C` *by convention* only when `C`'s
  definition is out of scope; if `C::new` is declared in scope with a non-concrete return
  (`-> Option<C>`), the convention is refuted and the receiver is dropped.
- **Wrappers never unwrap.** A field/return typed `Vec<Db>`/`Option<C>`/generic `T` is never a
  concrete receiver (same `NON_RECEIVER_WRAPPERS` + generic-param gates as Slice 3).

Earlier arc rows worth spelling out:

- **Non-`use` module paths (#122).** `submod::f()` with no import is among the commonest idioms in
  real Rust, and it produced *no edge*. The fix resolves the prefix as a submodule of the **caller's
  own crate only** — module-file-must-exist (exact, no ancestor shortening), then symbol-must-exist.
  An adversarial multi-agent review of the first design *empirically confirmed* that also trying an
  external-crate interpretation minted false edges (a symbol-less local `mod sib` fell through to a
  same-named extern crate — an edge Rust's shadow rule forbids) and stole valid type-method edges
  (an extern crate named like a type). The external reading was excluded; both defects have
  regression tests.
- **Reference-typed receivers (#126).** `fn r(x: &T) { x.bar() }` — the pervasive `&self`-taking
  pattern — inferred no receiver type because `&T` is not a `Type::Path`. Peeling leading `&`/`&mut`
  down to the referent reuses the exact #23 gates (`&dyn Tr`, `&Box<T>`, `&[T]`, generic params all
  still skip), so it adds recall with zero new false-edge surface. *Building this measurement is what
  surfaced the gap* — the first run printed `✗ MISSED` on its own fixture.

## What skips — 4/4 precision holds

| shape | why skipping is correct |
|---|---|
| bare `othercrate::f()` (no `use`) | a local `mod` shadows a same-named extern crate — linking into an unrelated crate can contradict Rust name resolution (the #122 review's confirmed false edge); cross-crate calls resolve via an explicit `use` |
| `f()` with two defs, no import | globally ambiguous — either link is a guess |
| `nope::f()` (unknown module) | no module file — never fall back to a same-named free fn |
| `self.v.push()` where `v: Vec<_>` | a wrapper-typed field is never a concrete receiver — the method dispatches to the wrapper (S4) |

## At scale — CCOS's own `src/` (51 files)

```
                    pre-Slice-4                with Slice 4
2801 call references parsed
  Calls edges       1007                       1114   (+107, ≈ +10.6 %)
  87 const/static references
  DataFlow edges    45                         45     (unchanged, as expected)
```

Slice 4 (field receivers + return-type chains) is the **single largest recall gain of the whole
call-graph arc**: +107 edges (≈ +10.6 %) on identical corpus, ahead of the `&T` peel's +60 (≈ +7 %
on the 48-file corpus of its day, 903 → 963). The reason is plain in the code itself: `self.field.m()`
is *the* dominant method-call shape in real Rust (`self.live.recall(…)`, `self.chain.last()`,
`self.graph().edges()` …). The field's declared type is same-file evidence, and the minted
`Type::method` callee still resolves through the graph-wide `(type, method)` unique-or-skip index —
so a field receiver links **across files** (declared `db: Db` here, `impl Db` there) with no new
false-edge surface. The parsed-but-unresolved remainder is dominated by calls into `std`/external
crates, receivers whose types live outside the scope's declarations, and macro paths — all
**correctly** unresolved: the graph never asserts a causal edge it cannot prove, so the structure an
agent walks is always trustworthy. This is the representation a vector RAG index does not have —
relatedness can rank `db.rs` near `api.rs`, but only the resolved edge says *`api::handle` calls
`repo::fetch` calls `db::timeout_ms`*, deterministically and replay-exact.

## Provenance

The arc landed as #113 (import-scoped bare consts) → #122 (non-`use` local module paths; designed and
adversarially reviewed via multi-agent workflows) → #124 (renamed const imports) → #126 (`&T`
receivers + this measurement) → **Slice 4** (field receivers + return-type chains — the paper's
"richer receiver inference" item; trait-object dynamic dispatch remains skipped *by design*, since a
`dyn Trait` receiver has no statically-certain concrete type). Each slice: crux tests first,
full-suite regression (569 lib tests at Slice 4's close), clippy `-D warnings`, and a bit-identical
two-run determinism check.
