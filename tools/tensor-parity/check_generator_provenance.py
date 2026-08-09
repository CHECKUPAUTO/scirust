#!/usr/bin/env python3
"""Static integrity checks for the Tensor Parity fixture generator.

The checker uses only the Python standard library. Historical fixtures are
immutable; the offline PyTorch generator may execute append-only families only.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GENERATOR = REPO / "docs/tensor-parity/provenance/generate_fixtures.py"

EXPECTED_BASELINE_VERSION = "2.13.0+cu130"
EXPECTED_BASELINE_COMMIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"

EXPECTED_HISTORICAL_FAMILIES = (
    "elementwise",
    "reductions",
    "normalization",
    "shape",
    "linear",
    "loss",
    "norm_affine",
    "reduction_extra",
    "unary_extra",
    "special",
    "shape_extra",
    "indexing",
    "linear_extra",
    "norm_stoch",
    "positional",
    "attention",
    "quantization",
    "conversion",
    "convolution",
    "linalg",
    "sparse",
    "einsum",
)

EXPECTED_APPEND_ONLY_FAMILY_SEEDS = {
    "elementwise_broadcast": 0xC0FFEE01,
}

EXPECTED_BROADCAST_SHAPES = (
    ((1, 3), (2, 1)),
    ((2, 1), (1, 3)),
    ((1, 4), (3, 4)),
    ((3, 1), (3, 4)),
    ((1, 1), (2, 3)),
)


def literal_assignment(tree: ast.AST, name: str):
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
            )
            if any(
                isinstance(target, ast.Name)
                and target.id == name
                for target in targets
            ):
                try:
                    return ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    return None
    return None


def function_nodes(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in getattr(tree, "body", [])
        if isinstance(node, ast.FunctionDef)
    }


def named_calls(node: ast.AST) -> set[str]:
    result = set()
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
        ):
            result.add(child.func.id)
    return result


def main() -> int:
    source = GENERATOR.read_text()
    tree = ast.parse(source, filename=str(GENERATOR))
    funcs = function_nodes(tree)
    errors: list[str] = []

    baseline_version = literal_assignment(tree, "BASELINE_VERSION")
    if baseline_version != EXPECTED_BASELINE_VERSION:
        errors.append(
            "BASELINE_VERSION must remain exactly "
            f"{EXPECTED_BASELINE_VERSION!r}; got {baseline_version!r}"
        )

    baseline_commit = literal_assignment(tree, "BASELINE_COMMIT")
    if baseline_commit != EXPECTED_BASELINE_COMMIT:
        errors.append(
            "BASELINE_COMMIT drifted from the frozen PyTorch source commit"
        )

    historical = literal_assignment(tree, "HISTORICAL_FAMILIES")
    if tuple(historical or ()) != EXPECTED_HISTORICAL_FAMILIES:
        errors.append(
            "HISTORICAL_FAMILIES must contain exactly the 22 immutable "
            f"families in canonical order; got {historical!r}"
        )

    append_only = literal_assignment(
        tree,
        "APPEND_ONLY_FAMILY_SEEDS",
    )
    if append_only != EXPECTED_APPEND_ONLY_FAMILY_SEEDS:
        errors.append(
            "APPEND_ONLY_FAMILY_SEEDS drifted from the canonical "
            f"registry; got {append_only!r}"
        )

    broadcast_shapes = literal_assignment(
        tree,
        "BINARY_BROADCAST_SHAPES",
    )
    if tuple(broadcast_shapes or ()) != EXPECTED_BROADCAST_SHAPES:
        errors.append(
            "BINARY_BROADCAST_SHAPES drifted from the canonical set; "
            f"got {broadcast_shapes!r}"
        )

    if "DEFAULT_FAMILIES" in source:
        errors.append(
            "DEFAULT_FAMILIES must not exist: full historical regeneration "
            "is forbidden"
        )

    if '"--families"' in source or "'--families'" in source:
        errors.append(
            "--families must not be exposed by the official generator"
        )

    if '"--torch-bin"' in source or "'--torch-bin'" in source:
        errors.append(
            "--torch-bin must not be exposed by the official generator"
        )

    required_funcs = {
        "verify_baseline_identity",
        "gen_elementwise_broadcast",
        "write",
        "build_manifest_files",
        "load_existing_manifest",
        "verify_preserved_manifest_files",
        "manifest",
        "main",
    }

    missing_funcs = sorted(required_funcs - set(funcs))
    if missing_funcs:
        errors.append(
            f"required provenance functions missing: {missing_funcs}"
        )

    main_node = funcs.get("main")
    if main_node is not None:
        calls = named_calls(main_node)
        generator_calls = sorted(
            name for name in calls if name.startswith("gen_")
        )

        if generator_calls != ["gen_elementwise_broadcast"]:
            errors.append(
                "main() may dispatch only gen_elementwise_broadcast(); "
                f"got {generator_calls}"
            )

        if any(
            isinstance(node, ast.Name)
            and node.id == "HISTORICAL_FAMILIES"
            for node in ast.walk(main_node)
        ):
            errors.append(
                "main() must not iterate or dispatch HISTORICAL_FAMILIES"
            )

    if "torch.version.git_version" not in source:
        errors.append(
            "generator must verify torch.version.git_version"
        )

    if "torch.__version__ != BASELINE_VERSION" not in source:
        errors.append(
            "generator must verify the exact frozen wheel version"
        )

    if ".manual_seed(family_seed)" not in source:
        errors.append(
            "each append-only family must use its independent frozen seed"
        )

    if (
        "relative_family.parts[0] not in APPEND_ONLY_FAMILY_SEEDS"
        not in source
    ):
        errors.append(
            "write() must refuse every non-append-only family"
        )

    if "verify_preserved_manifest_files" not in source:
        errors.append(
            "immutable manifest entries must be verified before writing"
        )

    if "files = dict(preserved_files)" not in source:
        errors.append(
            "new manifest must start from verified preserved files"
        )

    if "files.update(generated_files)" not in source:
        errors.append(
            "new manifest must merge only current append-only outputs"
        )

    if 'OUT.glob("*/*.json")' in source:
        errors.append(
            "generator must not globally certify OUT/*/*.json"
        )

    if errors:
        print("Tensor Parity generator provenance check: FAILED")
        for index, error in enumerate(errors, 1):
            print(f"{index}. {error}")
        return 1

    print("Tensor Parity generator provenance check: OK")
    print(
        "historical immutable families:",
        len(EXPECTED_HISTORICAL_FAMILIES),
    )
    print(
        "append-only families:",
        len(EXPECTED_APPEND_ONLY_FAMILY_SEEDS),
    )
    print("historical dispatches from main: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
