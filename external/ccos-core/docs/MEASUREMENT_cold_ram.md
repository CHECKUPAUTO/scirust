# COLD-tier resident RAM — measure before bounding it (slice 5 input)

> Reproduce: `cargo run --release --example cold_ram`.

Slice 3 spills a demoted node's **content** to disk, but its **metadata** (id, label,
archived edges, spill-hash stub) stays resident — an `O(N)` footprint we documented
but never quantified. Before choosing how to bound it (slice 5), measure it.

The benchmark builds N realistic nodes (files + 3 symbols each + `Contains`/`DependsOn`
edges), demotes the whole graph to COLD, spills every content blob to disk, and reports
the stuck resident bytes vs the offloaded disk bytes, cross-checked against process
`VmRSS`.

## Result

| files | cold nodes | resident/node | disk/node | **res : disk** |
|------:|-----------:|--------------:|----------:|---------------:|
| 2000  |      8000  |      494 B    |   179 B   |   **2.76×**    |
| 8000  |     32000  |      496 B    |   179 B   |   **2.77×**    |
| 30000 |    120000  |      500 B    |   179 B   |   **2.78×**    |

(`VmRSS` delta is ≈1300–1700 B/node — the logical figure plus allocator slack and
the live `nodes`/`edges` vectors. The logical 500 B/node is the part slice 5 controls.)

## What it says — and how it settles the lossless-vs-lossy fork

1. **The stuck metadata (~500 B/node) is ~2.8× larger than the content we spilled to
   disk (179 B/node).** For *code* — where a symbol's content is small — **spilling
   content barely dents resident RAM**; the metadata dominates. (Slice 3 is still a
   real win for large blobs and for *disk*-unboundedness, but on a code graph it is
   not where the RAM is.)
2. **~60 % of that metadata is edges.** Each cold entry archives its incident edges
   (~5 here, ~60 B each ≈ 300 B of the 500 B).
3. Therefore:
   - **Lossy edge-contraction (the discussion's idea) is the *wrong* lever here.**
     Contracting a hub of degree *d* replaces *d* edges with up to *d²* bridge edges —
     it would **inflate** the single biggest component of the resident cost on exactly
     the nodes (hubs) that matter. The data argues against it.
   - **The clean win is lossless *full-entry* spill**: archive the edges and metadata
     to the on-disk store too (extending slice 3 from content to the whole `ColdNode`),
     keeping only a minimal `NodeId → handle` resident. That attacks the dominant cost,
     stays non-destructive (“page don’t drop”), and reuses the content-addressed spill
     machinery already built.

## Is it urgent?

At ~500 B/node the COLD metadata reaches **1 GiB at ~2.1 M nodes** (logical), or
~700 K nodes by `VmRSS`. Typical repos (< ~100 K nodes) sit at tens of MB — tolerable.
So slice 5 is a **scaling** fix for very large monorepos or long-running daemons that
accumulate history, not a present bottleneck — and when built, it should be **lossless
full-entry spill**, not lossy contraction.

## Slice 5 result — deep-spill (built; lossless, off by default)

`set_cold_resident_budget(Some(b))` deep-spills the coldest entries until resident COLD
metadata is within `b`. The first cut (slice 5) moved each entry's `label` + edges to
the store but kept a full `ColdNode` husk — and stalled at **~11 %** on the 120 K-node
fixture: only the 30 K edge-bearing file nodes had enough to shed, and the remainder was
the **per-entry `ColdNode` struct floor** (~`size_of::<ColdNode>()` + id + hash stub),
which a same-shape husk can't touch.

**Slice 5b — compact husk.** A deep-spilled entry is now archived *whole* (node, content
folded inline, edges) to one content-addressed blob and represented in RAM by a compact
husk — body-blob stub + the neighbour **ids** (`adj`, all `cold_neighbours`/region paging
need) — held in its own `cold_deep` map. Because that husk is far smaller than a full
`ColdNode`, *every* entry shrinks when spilled, so the budget is actually **reached**:
on the same fixture, resident metadata halves (**−50 %**, 60 MB → 30 MB at the `b = half`
budget, ~108 K of 120 K entries archived) instead of hitting the slice-5 wall. The whole
node faults back, hash-verified, on `page_in`; deep husks are **terminal** (not re-scored
for further spill/compaction). Still lossless, still off by default (the `cold_deep` map
is `serde`-elided when empty → byte-identical default snapshot/replay), and still
**without** the bridge-edge blow-up that lossy contraction would inflict on hubs.
