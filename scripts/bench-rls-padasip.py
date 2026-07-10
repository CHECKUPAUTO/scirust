#!/usr/bin/env python3
"""Benchmark padasip FilterRLS — the cross-library half of the RLS comparison.

Protocol (run BOTH halves on the SAME machine, e.g. the Jetson):

  1. Rust side:   cargo run -p scirust-estimation --bin bench_rls --release
  2. Python side: pip install numpy padasip && python3 scripts/bench-rls-padasip.py

Both halves measure ns per update() on the same dimensions (n = 4, 16, 64)
with deterministic inputs. Quote the two tables together, with the machine
named — never numbers measured on different hosts. Until both halves have
run on one machine, no cross-library claim is made anywhere in this repo
(discipline: claims backed by measurements).
"""

import time

try:
    import numpy as np
    import padasip as pa
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"missing dependency: {exc}\n"
        "install with: pip install numpy padasip"
    )


def bench(n: int, iters: int) -> float:
    """Return ns/update for padasip FilterRLS with n taps."""
    f = pa.filters.FilterRLS(n=n, mu=0.98, w="zeros")
    rng = np.random.default_rng(42)
    inputs = rng.standard_normal((256, n))
    targets = rng.standard_normal(256)
    # Warmup (JIT-free, but populates caches and stabilizes timers).
    for i in range(iters // 10):
        f.adapt(targets[i % 256], inputs[i % 256])
    t0 = time.perf_counter()
    for i in range(iters):
        f.adapt(targets[i % 256], inputs[i % 256])
    return (time.perf_counter() - t0) / iters * 1e9


def main() -> None:
    print("padasip FilterRLS adapt() benchmark — deterministic inputs\n")
    for n in (4, 16, 64):
        iters = max(2_000_000 // (n * 8), 10_000)
        ns = bench(n, iters)
        print(f"padasip FilterRLS  n={n:>3}  {ns:>9.1f} ns/update  ({iters} iters)")
    print(
        "\nCompare against the same machine's:\n"
        "  cargo run -p scirust-estimation --bin bench_rls --release"
    )


if __name__ == "__main__":
    main()
