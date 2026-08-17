# SciRust unified CLI contract

This document defines the user-visible contract of the `scirust` binary.
`docs/REFERENCE.md` remains the exhaustive command reference; this file records
cross-command rules that must stay consistent as new subcommands are added.

## Executable commands vs library capabilities

`scirust help` contains two kinds of entries:

1. **Executable commands** — accepted by the unified dispatcher and runnable as
   `scirust <command> ...`.
2. **Library crates (not subcommands)** — capability crates exposed to Rust
   callers but intentionally not dispatched by the `scirust` binary.

The library-only groups are rendered with the explicit suffix
`LIBRARY CRATES (NOT SUBCOMMANDS)` in interactive help. Their crate names must
not be interpreted as runnable CLI commands.

## Process exit codes

The unified binary uses the following process exit-code space:

| Code | Meaning |
|---:|---|
| `0` | success |
| `1` | domain-level failure, mismatch, or negative result where the command defines that outcome |
| `2` | usage, argument, parsing, or I/O failure |
| `3` | validation failure / invalid Studio model state |
| `5` | numerical failure in Studio execution |
| `6` | Studio execution cancelled |
| `7` | Studio internal, store, or serialization failure |

Not every command can emit every code. Command-specific sections of
`docs/REFERENCE.md` may narrow this set, but they must not redefine a code with
an incompatible meaning.

## Maintenance rule

When a command introduces a new process exit code or a new library-only help
group, update this contract and the corresponding command-specific reference in
the same change. Tests in `scirust-cli` preserve the visual distinction between
executable command groups and library-only capability groups.
