# Method-call edges — `x.bar()` receiver-type inference (Slice 3)

> Reproduce: `cargo run --release --example method_crux`

Slice 1 (`docs/MEASUREMENT_call_crux.md`) resolved bare `foo()` calls; Slice 2 added qualified
`a::b::foo()` and `self`/`Self` method calls. **Slice 3 (#23)** closes the last common call form: a
method call `x.bar()` on a local whose type the parser can infer — the `caller → callee` link a flat
symbol index structurally cannot make.

## Why it's hard: a method call doesn't name its type

`x.bar()` names the method `bar` but not the type `x` belongs to. CCOS stores a method as a flat
`sym:<file>:bar` symbol — no type association — so when two types each define `bar`, the name is
**ambiguous** and the resolver (correctly, precision-first) skips it. The structure the agent wants —
"this call goes to *Widget*'s `render`, not *Gadget*'s" — is invisible at the name level.

## How (precision-first, deterministic)

Two halves, both **resolve-uniquely-or-skip**:

- **Infer the receiver type** (`parser.rs`): a local's concrete type is recorded only from four
  high-confidence, syntactically-certain idioms — a typed parameter `fn f(x: Foo)`, a `let`
  annotation `let x: Foo`, a constructor `let x = Foo::new()` / `Foo::default()` / `Foo::with_*()`,
  and a single-segment struct literal `let x = Foo { .. }`. A **PascalCase-head** guard separates a
  type `Foo::new()` from a module function `foo::new()`; generic params and std wrappers (`Box`,
  `Vec`, …) are excluded; and a name bound to two types, re-`let`, or reassigned is **poisoned**
  (dropped). A bare-ident receiver with a known type emits a `Foo::bar` callee — anything else is
  dropped, never guessed (a wrong type would mint a *false* edge, strictly worse than the data-ref
  case where a wrong guess only ever drops one).
- **Resolve `Foo::bar`** (`MemoryGraph::resolve_symbol_calls`): a `(type, method) → symbol` index,
  built from each `impl Foo { fn bar }`, with per-bucket **cardinality** so a type name shared by two
  impls is ambiguous and skipped. A 2-segment `A::b` callee resolves *both* interpretations —
  `A`-as-module (the existing path) and `A`-as-type (the new index) — and links only when they
  **agree or exactly one resolves**; a genuine collision is skipped. (Bonus: explicit `Type::assoc()`
  calls, which skipped before, now resolve too.)

The new edges are `EdgeType::Calls`, **resolution-owned** (pruned and rebuilt by the whole-graph pass
over the final node set), so `replay == live` and eager ≡ batch hold by construction — proven by
`external_memory::tests::method_call_receiver_inference_resolves_cross_file_without_false_edges`.

## The measurement — an adversarial twin

`examples/method_crux.rs` builds the case a flat index cannot handle: `render` defined on **two**
types (`Widget`, `Gadget`) in two files, with cross-file callers reaching each via a different idiom.
A wrong inference is then a concrete, assertable false edge — not a silent miss.

| caller | receiver idiom | resolves to | correct |
|---|---|---|:---:|
| `drive_ctor` | `let w = Widget::new()` | `Widget::render` | ✅ |
| `drive_param` | `fn(g: Gadget)` | `Gadget::render` | ✅ |
| `drive_annot` | `let g: Gadget = Gadget::new()` | `Gadget::render` | ✅ |

**precision 100 % (3/3), zero cross-type false edges** — with the twin present, any mis-inference
would link `drive_ctor → Gadget::render` (or its mirror) and the example asserts it does not. A flat
`sym:<file>:render` index sees an ambiguous name (`render` on two types) and recovers **none** of
these; the `(type, method)` index recovers all three because the inferred receiver type disambiguates.

## Scope & the honest recall holes

Precision is the headline; recall is deliberately bounded. The example lists the forms #23 **skips by
design** — a method-chain receiver `f().bar()` (receiver is a call, not a bare ident), a trait-object
`&dyn Tr` (no single concrete type), a generic `T`, fields `self.field.bar()`, and reassigned or
shadowed bindings. Skipping these is correct-by-policy: a wrong guess would mint a false call edge.
Cross-crate / same-final-name type homonyms also skip (the index keys on the final type segment), and
a method whose impl file is not re-ingested in a partial post-load resolve may drop until a full
replay — both precision-safe (a dropped edge, never a false one). The result is the trade the rest of
the call graph makes: **resolve uniquely or skip**, so the structure the agent follows is always
trustworthy.
