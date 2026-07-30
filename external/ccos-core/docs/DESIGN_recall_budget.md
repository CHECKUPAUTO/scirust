# Design — anchor budget balancing (the budget-scaling caveat)

## The measured caveat (see `FIELD_CAMPAIGN_H.md`, syn confirmation)

After the triad (granularity, degree-aware propagation, proximity), `recall around`
returns the symptom + **all** its real dependencies — *if the budget fits the anchor's own
size*. On `syn`'s large files this broke at a fixed budget: `item.rs` (≈88 symbols) fills a
2048-token window with its own (relevant) content before any dependency gets in.

Measured breakdown of `around item.rs` @2048 before the fix:

| node | tokens | note |
| ---- | ------ | ---- |
| `file:item.rs` (header) | **639** | 31 % of budget — lists all 88 signatures |
| `sym:item.rs:FlexibleItemType` | 535 | one big own-symbol |
| `sym:item.rs:Item` | 287 | |
| … item.rs own | … | |
| `file:attr.rs` | 401 | only **1 of 7** deps fit |

Result: **1/7 deps** at budget 2048 (7/7 only at 8192). The window scales with the anchor,
not the budget.

## Two levers, simulated on the real `syn` region (actual scores)

| policy | deps reached @2048 |
| ------ | ------------------ |
| current (greedy by score) | 1/7 |
| per-file cap 40 % | 5/7 |
| header cap K=24 | 5/7 |
| **header cap + per-file cap 40 %** | **7/7** ✅ |

Neither lever alone suffices; together they fit all seven deps at the fixed budget.

## The fix (shipped)

1. **Header cap** (`parser.rs`, `CCOS_HEADER_SYMBOLS`, default 24). A file-header node lists
   at most K signatures, then `// … (+N more)`. A huge file's index no longer eats a third of
   the budget; the capped-out symbols remain their own span nodes (reachable, just not all
   listed up front). No-op for files with ≤ K symbols (CCOS's own src, the toys).

2. **Per-file budget cap** (`external_memory.rs::assemble_window`, `CCOS_RECALL_FILE_CAP`,
   default 0.40). When anchored (`around`/`task`), no single file may fill more than
   `cap × budget`; an over-quota node is skipped and packing continues with smaller ones. The
   anchor's own content is bounded, leaving room for its dependencies. `working_set` (no
   anchor) is unaffected.

## Measured result (fixed budget 2048, `syn`, was 1/7, 0/7, 2/6, 0/5)

| anchor | deps reached @2048 |
| ------ | ------------------ |
| item.rs | **7/7** |
| token.rs | **7/7** |
| ty.rs | **6/6** |
| expr.rs | **5/5** |

All real dependencies now fit the fixed budget regardless of anchor size. No regression on
CCOS's own src (symptom + 2/2 deps) or the toy Campaign H (cause reached 5/5). Two regression
tests added (`file_header_caps_its_symbol_list`,
`around_caps_anchor_footprint_so_cross_file_deps_fit_a_fixed_budget`).

## Trade-off & tuning

The per-file cap means a genuinely single-file recall gives at most `cap` of the anchor and
spends the rest on its nearest neighbours (still relevant context). Raise `CCOS_RECALL_FILE_CAP`
toward 1.0 to recover pure single-file focus, or lower it for more breadth. The defaults
(K=24, cap=0.40) fit the symptom + all direct deps of large real files at a 2 k budget.
