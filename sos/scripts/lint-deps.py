#!/usr/bin/env python3
"""Enforce the sos/ workspace's dependency invariants (Invariant VIII; RFC-0002
docs/sos/11-workspace-and-crate-graph.md §5) against real `cargo metadata`.

Four rules, each checked over the manifest-declared dependency graph:

1. `sos-core` is the universal sink: it depends on no other SOS crate, and
   every other SOS crate depends on it (directly or transitively).
2. No cyclic dependencies among SOS crates.
3. No engine depends on another engine by type, except the composition edges
   documented in `ALLOWED_ENGINE_EDGES` below.
4. `scirust-*` (the computational backend, including `scirust-sciagent` —
   CCOS, the cognitive backend) is named only by `sos-scirust`/`sos-ccos`.

Cycle- and sink-checks run over normal + build dependencies (a real
compile-time edge); dev-dependencies are excluded there since a dev-only
back-edge (e.g. a crate's tests exercising a crate that depends on it) is a
Cargo-supported pattern, not an architectural cycle. Rule 4 checks every
dependency kind: a test-only leak of a backend crate is still a coupling
Invariant VIII forbids.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "Cargo.toml"

# The 8 engines, per docs/sos/11-workspace-and-crate-graph.md §1's explicit
# count ("5 kernel/substrate, 8 engines, 8 discovery stages, 2 backends, 2
# userland"). sos-provenance/sos-repro/sos-registry are kernel/substrate
# despite "Engine" appearing in their one-line descriptions too.
ENGINES = frozenset(
    {
        "sos-knowledge",
        "sos-reasoning",
        "sos-curiosity",
        "sos-theory",
        "sos-simulation",
        "sos-planner",
        "sos-workflow",
        "sos-publication",
    }
)

# Composition edges sanctioned by §5 rule 3. Listing an edge here means it is
# ALLOWED, not that it is currently exercised — sos-curiosity -> sos-planner
# is designed-for but not yet wired (the sos-scirust plan's gap #6).
ALLOWED_ENGINE_EDGES = frozenset(
    {
        ("sos-reasoning", "sos-knowledge"),
        ("sos-curiosity", "sos-knowledge"),
        ("sos-curiosity", "sos-reasoning"),
        ("sos-curiosity", "sos-planner"),
        ("sos-theory", "sos-reasoning"),
    }
)
# Discovery stages (sde-question .. sde-ranking, not yet landed) are also
# documented to compose with sos-reasoning; any future crate named `sde-*`
# gets this one exception without needing a per-crate list entry.
DISCOVERY_STAGE_PREFIX = "sde-"
DISCOVERY_STAGE_TARGET = "sos-reasoning"

BACKEND_PREFIX = "scirust-"
COMPUTATIONAL_HOME = "sos-scirust"
COGNITIVE_HOME = "sos-ccos"

CargoDep = dict
CargoPkg = dict

REAL_EDGE_KINDS = frozenset({None, "build"})
ALL_EDGE_KINDS = frozenset({None, "dev", "build"})


def load_metadata(manifest_path: Path) -> dict:
    try:
        result = subprocess.run(
            [
                "cargo",
                "metadata",
                "--manifest-path",
                str(manifest_path),
                "--format-version=1",
                "--locked",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"error: `cargo` not found on PATH ({exc})") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"error: `cargo metadata` failed:\n{exc.stderr}") from exc
    return json.loads(result.stdout)


def workspace_packages(metadata: dict) -> dict[str, CargoPkg]:
    members = set(metadata["workspace_members"])
    return {pkg["name"]: pkg for pkg in metadata["packages"] if pkg["id"] in members}


def dep_names(pkg: CargoPkg, kinds: frozenset) -> list[str]:
    return [dep["name"] for dep in pkg["dependencies"] if dep["kind"] in kinds]


def reaches(start: str, target: str, packages: dict[str, CargoPkg]) -> bool:
    seen: set[str] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen or node not in packages:
            continue
        seen.add(node)
        stack.extend(dep_names(packages[node], REAL_EDGE_KINDS))
    return False


def check_core_is_sink(packages: dict[str, CargoPkg]) -> list[str]:
    errors = []
    core = packages.get("sos-core")
    if core is None:
        return ["sos-core is not a workspace member — cannot check the sink invariant"]

    stray = [dep for dep in dep_names(core, REAL_EDGE_KINDS) if dep in packages]
    if stray:
        errors.append(f"sos-core must depend on no other SOS crate; found: {sorted(stray)}")

    for name in packages:
        if name != "sos-core" and not reaches(name, "sos-core", packages):
            errors.append(f"{name} does not depend on sos-core, directly or transitively")
    return errors


def check_no_cycles(packages: dict[str, CargoPkg]) -> list[str]:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(packages, WHITE)
    errors: list[str] = []

    def visit(name: str, path: list[str]) -> None:
        color[name] = GRAY
        for dep in dep_names(packages[name], REAL_EDGE_KINDS):
            if dep not in packages:
                continue
            if color[dep] == GRAY:
                errors.append("dependency cycle: " + " -> ".join(path + [dep]))
            elif color[dep] == WHITE:
                visit(dep, path + [dep])
        color[name] = BLACK

    for name in packages:
        if color[name] == WHITE:
            visit(name, [name])
    return errors


def check_engine_composition(packages: dict[str, CargoPkg]) -> list[str]:
    errors = []
    for name, pkg in packages.items():
        is_stage = name.startswith(DISCOVERY_STAGE_PREFIX)
        if name not in ENGINES and not is_stage:
            continue
        for dep in dep_names(pkg, REAL_EDGE_KINDS):
            if dep not in ENGINES:
                continue
            if is_stage:
                allowed = dep == DISCOVERY_STAGE_TARGET
            else:
                allowed = (name, dep) in ALLOWED_ENGINE_EDGES
            if not allowed:
                errors.append(
                    f"{name} -> {dep} is an undocumented engine-to-engine edge "
                    "(docs/sos/11-workspace-and-crate-graph.md §5 rule 3)"
                )
    return errors


def check_backend_confinement(packages: dict[str, CargoPkg]) -> list[str]:
    errors = []
    for name, pkg in packages.items():
        if name in (COMPUTATIONAL_HOME, COGNITIVE_HOME):
            continue
        for dep in dep_names(pkg, ALL_EDGE_KINDS):
            if dep.startswith(BACKEND_PREFIX):
                errors.append(
                    f"{name} depends on {dep} — {BACKEND_PREFIX}* is confined to "
                    f"{COMPUTATIONAL_HOME}/{COGNITIVE_HOME} (Invariant VIII)"
                )
    return errors


CHECKS = (
    check_core_is_sink,
    check_no_cycles,
    check_engine_composition,
    check_backend_confinement,
)


def run_all_checks(packages: dict[str, CargoPkg]) -> list[str]:
    errors: list[str] = []
    for check in CHECKS:
        errors.extend(check(packages))
    return errors


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    metadata = load_metadata(args.manifest_path)
    packages = workspace_packages(metadata)
    if not packages:
        print("error: no workspace members found — wrong --manifest-path?", file=sys.stderr)
        return 2

    errors = run_all_checks(packages)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        print(f"\n{len(errors)} dependency-invariant violation(s).", file=sys.stderr)
        return 1

    print(f"OK: {len(packages)} sos-* crates satisfy all 4 dependency invariants.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
