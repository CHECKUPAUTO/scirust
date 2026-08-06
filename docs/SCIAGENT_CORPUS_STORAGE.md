# SCIAGENT generated corpus storage

## Decision

`data/crates_raw` is a generated local training corpus, not SciRust source,
not a Cargo workspace input, and not a tracked dataset. It contains downloaded
crate archives, extracted crate trees, completion markers, and `all/`, a
generated symlink aggregate of Rust files. The directory must live outside the
repository checkout.

This is a physical-storage rule rather than an ignore-file workaround. `rg`
and `fd` normally honour ignore files, but `grep -R` and `find` do not. Keeping
the corpus outside the checkout is the only arrangement that keeps all normal
repository searches focused on SciRust source without per-command exclusions.

## Default locations

`fetch-crates` writes the raw corpus and `collect-data` writes packed shards to
the platform data directory:

| Platform | Raw corpus | Packed shards |
|---|---|---|
| Linux and other XDG systems | `$XDG_DATA_HOME/scirust/sciagent/crates_raw`, or `$HOME/.local/share/scirust/sciagent/crates_raw` when `XDG_DATA_HOME` is unset | Same base, ending in `shards` |
| macOS | `$HOME/Library/Application Support/scirust/sciagent/crates_raw` | Same base, ending in `shards` |
| Windows | `%LOCALAPPDATA%\\scirust\\sciagent\\crates_raw` | Same base, ending in `shards` |

The environment variables `SCIRUST_SCIAGENT_CORPUS_DIR` and
`SCIRUST_SCIAGENT_SHARDS_DIR` override the relevant location. Every selected
output path is validated before it is created. A path lexically inside the
checkout, including a symlink placed inside the checkout, is rejected.

Print the selected storage paths without creating directories:

```bash
cargo run -p scirust-sciagent --bin sciagent-corpus -- location
```

## One-time migration of a legacy corpus

First inspect the migration. This command makes no changes:

```bash
cargo run -p scirust-sciagent --bin sciagent-corpus -- migrate
```

Then apply it explicitly:

```bash
cargo run -p scirust-sciagent --bin sciagent-corpus -- migrate --apply
```

The migration accepts only `data/crates_raw` below the checkout and refuses to
overwrite the external destination. Before moving it, it inventories the tree
and reconstructs `all/` in deterministic lexical order. The old `all/` is
replaced only after proving that it consists solely of generated symlinks; no
downloaded archive or extracted crate source is deleted. The move uses a single
filesystem rename. If the destination is on another filesystem, the operation
fails and leaves the source in place; choose an external destination on the
same filesystem and retry, for example:

```bash
SCIRUST_SCIAGENT_CORPUS_DIR="$HOME/.local/share/scirust/sciagent/crates_raw" \
cargo run -p scirust-sciagent --bin sciagent-corpus -- migrate --apply
```

The legacy aggregate needs reconstruction because older `fetch-crates` code
stored a target such as `data/crates_raw/example/src/a.rs` inside
`data/crates_raw/all/`. That target is resolved relative to `all/`, so it is
broken. The current collector creates `../example/src/a.rs` instead; the
migration repairs existing links rather than relying on a fresh download.

## Normal collection

```bash
cargo run -p scirust-sciagent --features fetch --bin fetch-crates -- --count 50
cargo run -p scirust-sciagent --bin collect-data -- \
  --input "$HOME/.local/share/scirust/sciagent/crates_raw" \
  --tokenizer scirust-sciagent/tokenizer/bpe.json --recursive
```

Use the printed corpus location or `SCIRUST_SCIAGENT_CORPUS_DIR` if it differs
from the Linux fallback shown above. Neither command needs an in-repository
`--output`.

## Enforcement

`scripts/check-sciagent-corpus-location.sh` rejects the legacy generated path
`data/crates_raw`; CI runs the same check. The root `.gitignore` intentionally
does not hide `/data/`, so an accidental in-checkout corpus is visible in
`git status` instead of silently escaping review.
