# SciRust — npm installer

[SciRust](https://github.com/CHECKUPAUTO/scirust) is an industrial-grade deep learning framework in pure Rust: autodiff, SIMD, GPU, transformers, symbolic math, and KV-cache compression.

## Quick Start

```bash
# Install globally via npm
npm install -g scirust

# Then install the binary
scirust-install install
```

Or use `npx` without global install:

```bash
npx scirust-install install
```

## Commands

| Command | Description |
|---------|-------------|
| `scirust-install install` | Clone, build, and install SciRust binary |
| `scirust-install update` | Update to the latest version |
| `scirust` | Run SciRust (after install) |

## After Install

```bash
scirust                 # Show capabilities
scirust simd            # SIMD vs scalar benchmark (SAXPY)
scirust autodiff        # Train XOR classifier via autodiff
scirust symbolic        # Symbolic math (parse, derive, simplify)
scirust bench           # Run all benchmarks
```

## Prerequisites

- **Rust toolchain** (cargo): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **git**
- **Node.js ≥ 16**

## Manual Install

If you prefer a shell script directly:

```bash
curl -sSf https://raw.githubusercontent.com/CHECKUPAUTO/scirust/feat/portable-simd-and-views/npm/install.sh | bash
```

## Update

```bash
scirust-install update
# or
curl -sSf https://raw.githubusercontent.com/CHECKUPAUTO/scirust/feat/portable-simd-and-views/npm/update.sh | bash
```
