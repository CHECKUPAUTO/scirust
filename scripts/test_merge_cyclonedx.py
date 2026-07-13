#!/usr/bin/env python3
"""Regression tests for checkout-independent CycloneDX aggregation."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote


SCRIPT = Path(__file__).with_name("merge-cyclonedx.py")
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("merge_cyclonedx", SCRIPT)
assert SPEC and SPEC.loader
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)


def fixture(checkout: Path) -> tuple[list[dict], dict]:
    checkout = checkout.resolve()
    root_uri = checkout.as_uri()
    member_uri = (checkout / "crates" / "member").as_uri()
    root_ref = f"path+{root_uri}#1.0.0"
    member_ref = f"path+{member_uri}#0.1.0"
    target_ref = f"{member_ref} bin-target-0"
    encoded_member_uri = quote(member_uri, safe="")

    documents = [
        {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "serialNumber": "urn:uuid:checkout-specific",
            "metadata": {
                "timestamp": "2099-01-01T00:00:00Z",
                "component": {
                    "type": "application",
                    "bom-ref": root_ref,
                    "name": "scirust",
                    "version": "1.0.0",
                    "purl": f"pkg:cargo/scirust@1.0.0?download_url={root_uri}",
                    "properties": [{"name": "manifest", "value": str(checkout / "Cargo.toml")}],
                },
            },
            "components": [
                {
                    "type": "library",
                    "bom-ref": member_ref,
                    "name": "member",
                    "version": "0.1.0",
                    "purl": f"pkg:cargo/member@0.1.0?download_url={encoded_member_uri}",
                    "components": [
                        {
                            "type": "library",
                            "bom-ref": target_ref,
                            "name": "member-target",
                            "version": "0.1.0",
                            "purl": f"pkg:cargo/member@0.1.0?download_url={member_uri}#src/lib.rs",
                        }
                    ],
                }
            ],
            "dependencies": [
                {"ref": root_ref, "dependsOn": [member_ref]},
                {"ref": member_ref, "dependsOn": [target_ref]},
            ],
        }
    ]
    metadata = {
        "workspace_root": str(checkout),
        "packages": [
            {"name": "scirust", "version": "1.0.0"},
            {"name": "member", "version": "0.1.0"},
        ],
    }
    return documents, metadata


class MergeCycloneDxTests(unittest.TestCase):
    def test_identical_output_from_two_checkout_locations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            checkout_a = base / "short" / "scirust"
            checkout_b = base / "a path with spaces" / "different" / "scirust"
            checkout_a.mkdir(parents=True)
            checkout_b.mkdir(parents=True)
            documents_a, metadata_a = fixture(checkout_a)
            documents_b, metadata_b = fixture(checkout_b)

            output_a = merge.build_aggregate(documents_a, metadata_a, epoch=1_700_000_000)
            output_b = merge.build_aggregate(documents_b, metadata_b, epoch=1_700_000_000)

            self.assertEqual(merge.canonical_json(output_a), merge.canonical_json(output_b))
            serialized = merge.canonical_json(output_a)
            self.assertNotIn(str(checkout_a), serialized)
            self.assertIn("file://workspace", serialized)
            component_refs = {component["bom-ref"] for component in output_a["components"]}
            dependency_refs = {dependency["ref"] for dependency in output_a["dependencies"]}
            self.assertTrue(dependency_refs.issubset(component_refs))

    def test_missing_workspace_package_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / "scirust"
            checkout.mkdir()
            documents, metadata = fixture(checkout)
            metadata["packages"].append({"name": "missing", "version": "9.9.9"})
            with self.assertRaisesRegex(ValueError, "missing@9.9.9"):
                merge.build_aggregate(documents, metadata, epoch=1_700_000_000)


if __name__ == "__main__":
    unittest.main()
