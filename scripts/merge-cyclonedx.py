#!/usr/bin/env python3
"""Merge cargo-cyclonedx member BOMs into one checkout-independent BOM."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote


CANONICAL_WORKSPACE_URI = "file://workspace"
CANONICAL_WORKSPACE_PATH = "$WORKSPACE"


def key(component: dict[str, Any]) -> str:
    """Return the normalized CycloneDX identity used for de-duplication."""
    return component.get("bom-ref") or component.get("purl") or "|".join(
        str(component.get(field, "")) for field in ("type", "name", "version")
    )


class WorkspaceNormalizer:
    """Remove checkout-specific workspace paths from every JSON string.

    cargo-cyclonedx uses absolute ``file://`` URLs in workspace component
    ``bom-ref`` values. The same absolute URL can also occur in PURLs, nested
    target components, dependency refs, properties, or external references.
    Normalizing the complete JSON tree before aggregation keeps references
    internally consistent instead of fixing only the top-level components.
    """

    def __init__(self, workspace_root: str) -> None:
        root = Path(workspace_root).resolve()
        if not root.is_absolute():
            raise ValueError("Cargo metadata workspace_root must be absolute")

        root_uri = root.as_uri().rstrip("/")
        raw_native = str(root).rstrip("/\\")
        raw_posix = root.as_posix().rstrip("/")

        # Replace URI forms before raw paths, otherwise replacing the path
        # substring inside ``file:///root`` would leave a malformed hybrid.
        mappings: list[tuple[str, str, bool]] = [
            (root_uri, CANONICAL_WORKSPACE_URI, True),
            (unquote(root_uri), CANONICAL_WORKSPACE_URI, True),
            (
                quote(root_uri, safe=""),
                quote(CANONICAL_WORKSPACE_URI, safe=""),
                True,
            ),
            (
                quote(root_uri, safe="/:"),
                quote(CANONICAL_WORKSPACE_URI, safe="/:"),
                True,
            ),
            (raw_native, CANONICAL_WORKSPACE_PATH, os.name == "nt"),
            (raw_posix, CANONICAL_WORKSPACE_PATH, os.name == "nt"),
            (
                quote(raw_posix, safe=""),
                quote(CANONICAL_WORKSPACE_PATH, safe=""),
                True,
            ),
        ]

        # De-duplicate variants (URI forms without spaces are often equal)
        # and always replace the longest spelling first.
        unique: dict[str, tuple[str, bool]] = {}
        for source, replacement, ignore_case in mappings:
            if source:
                unique[source] = (replacement, ignore_case)
        self._mappings = sorted(
            (
                (source, replacement, ignore_case)
                for source, (replacement, ignore_case) in unique.items()
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        self._checkout_spellings = tuple(unique)

    def text(self, value: str) -> str:
        normalized = value
        for source, replacement, ignore_case in self._mappings:
            normalized = re.sub(
                re.escape(source),
                lambda _match, replacement=replacement: replacement,
                normalized,
                flags=re.IGNORECASE if ignore_case else 0,
            )
        return normalized

    def value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, dict):
            return {name: self.value(item) for name, item in value.items()}
        return value

    def assert_checkout_independent(self, value: Any) -> None:
        strings: list[str] = []

        def visit(item: Any) -> None:
            if isinstance(item, str):
                strings.append(item)
            elif isinstance(item, list):
                for child in item:
                    visit(child)
            elif isinstance(item, dict):
                for child in item.values():
                    visit(child)

        visit(value)
        for text in strings:
            for spelling in self._checkout_spellings:
                if spelling and spelling.casefold() in text.casefold():
                    raise ValueError(f"checkout-specific path remains in SBOM: {text!r}")


def build_aggregate(
    documents: list[dict[str, Any]],
    cargo_metadata: dict[str, Any],
    epoch: int,
) -> dict[str, Any]:
    """Build an aggregate whose canonical JSON is independent of checkout."""
    if not documents:
        raise ValueError("no member BOMs were generated")

    normalizer = WorkspaceNormalizer(cargo_metadata["workspace_root"])
    normalized_documents = [normalizer.value(copy.deepcopy(document)) for document in documents]
    root = next(
        (
            document
            for document in normalized_documents
            if document.get("metadata", {}).get("component", {}).get("name") == "scirust"
        ),
        normalized_documents[0],
    )

    components: dict[str, dict[str, Any]] = {}
    dependencies: dict[str, set[str]] = {}
    for document in normalized_documents:
        metadata_component = document.get("metadata", {}).get("component")
        if metadata_component:
            components[key(metadata_component)] = metadata_component
        for component in document.get("components", []):
            components[key(component)] = component
        for dependency in document.get("dependencies", []):
            reference = dependency.get("ref")
            if reference:
                dependencies.setdefault(reference, set()).update(dependency.get("dependsOn", []))

    represented = {
        (component.get("name"), component.get("version")) for component in components.values()
    }
    missing = sorted(
        (package["name"], package["version"])
        for package in cargo_metadata["packages"]
        if (package["name"], package["version"]) not in represented
    )
    if missing:
        formatted = ", ".join(f"{name}@{version}" for name, version in missing)
        raise ValueError(f"workspace packages missing from aggregate SBOM: {formatted}")

    timestamp = dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    metadata = dict(root.get("metadata", {}))
    metadata["timestamp"] = timestamp

    aggregate = {
        "bomFormat": "CycloneDX",
        "specVersion": root.get("specVersion", "1.5"),
        "version": 1,
        "metadata": metadata,
        "components": [components[reference] for reference in sorted(components)],
        "dependencies": [
            {"ref": reference, "dependsOn": sorted(depends_on)}
            for reference, depends_on in sorted(dependencies.items())
        ],
    }
    normalizer.assert_checkout_independent(aggregate)
    return aggregate


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cargo-metadata", required=True, type=Path)
    parser.add_argument("--input-list", required=True, type=Path)
    args = parser.parse_args()

    inputs = [Path(line) for line in args.input_list.read_text().splitlines() if line]
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    cargo_metadata = json.loads(args.cargo_metadata.read_text(encoding="utf-8"))
    aggregate = build_aggregate(documents, cargo_metadata, int(os.environ["SOURCE_DATE_EPOCH"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(aggregate), encoding="utf-8")


if __name__ == "__main__":
    main()
