#!/usr/bin/env python3
"""Tests for lint-deps.py's four dependency-invariant checks."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("lint-deps.py")
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("lint_deps", SCRIPT)
assert SPEC and SPEC.loader
lint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lint)


def pkg(*deps: tuple[str, str | None]) -> dict:
    """Build a minimal synthetic cargo-metadata package dict."""
    return {"dependencies": [{"name": name, "kind": kind} for name, kind in deps]}


class CoreIsSinkTests(unittest.TestCase):
    def test_clean_graph_passes(self) -> None:
        packages = {
            "sos-core": pkg(),
            "sos-store": pkg(("sos-core", None)),
            "sos-knowledge": pkg(("sos-core", None), ("sos-store", None)),
        }
        self.assertEqual(lint.check_core_is_sink(packages), [])

    def test_core_depending_on_sos_crate_fails(self) -> None:
        packages = {
            "sos-core": pkg(("sos-store", None)),
            "sos-store": pkg(("sos-core", None)),
        }
        errors = lint.check_core_is_sink(packages)
        self.assertEqual(len(errors), 1)
        self.assertIn("sos-core must depend on no other SOS crate", errors[0])

    def test_crate_not_reaching_core_fails(self) -> None:
        packages = {
            "sos-core": pkg(),
            "sos-orphan": pkg(),
        }
        errors = lint.check_core_is_sink(packages)
        self.assertEqual(len(errors), 1)
        self.assertIn("sos-orphan does not depend on sos-core", errors[0])

    def test_transitive_reach_through_another_sos_crate_is_sufficient(self) -> None:
        packages = {
            "sos-core": pkg(),
            "sos-store": pkg(("sos-core", None)),
            "sos-provenance": pkg(("sos-store", None)),
        }
        self.assertEqual(lint.check_core_is_sink(packages), [])

    def test_dev_dependency_does_not_count_toward_reaching_core(self) -> None:
        packages = {
            "sos-core": pkg(),
            "sos-isolated": pkg(("sos-core", "dev")),
        }
        errors = lint.check_core_is_sink(packages)
        self.assertEqual(len(errors), 1)
        self.assertIn("sos-isolated does not depend on sos-core", errors[0])


class NoCyclesTests(unittest.TestCase):
    def test_dag_passes(self) -> None:
        packages = {
            "sos-core": pkg(),
            "sos-reasoning": pkg(("sos-core", None), ("sos-knowledge", None)),
            "sos-knowledge": pkg(("sos-core", None)),
        }
        self.assertEqual(lint.check_no_cycles(packages), [])

    def test_direct_two_cycle_fails(self) -> None:
        packages = {
            "sos-planner": pkg(("sos-workflow", None)),
            "sos-workflow": pkg(("sos-planner", None)),
        }
        errors = lint.check_no_cycles(packages)
        self.assertEqual(len(errors), 1)
        self.assertIn("cycle", errors[0])

    def test_longer_cycle_fails(self) -> None:
        packages = {
            "a": pkg(("b", None)),
            "b": pkg(("c", None)),
            "c": pkg(("a", None)),
        }
        errors = lint.check_no_cycles(packages)
        self.assertEqual(len(errors), 1)

    def test_dev_dependency_back_edge_is_not_a_cycle(self) -> None:
        # sos-workflow's tests depend on sos-store; nothing architectural
        # depends back on sos-workflow from sos-store, so this must pass.
        packages = {
            "sos-core": pkg(),
            "sos-store": pkg(("sos-core", None)),
            "sos-workflow": pkg(("sos-core", None), ("sos-store", "dev")),
        }
        self.assertEqual(lint.check_no_cycles(packages), [])


class EngineCompositionTests(unittest.TestCase):
    def test_documented_edge_passes(self) -> None:
        packages = {
            "sos-reasoning": pkg(("sos-knowledge", None)),
            "sos-knowledge": pkg(),
        }
        self.assertEqual(lint.check_engine_composition(packages), [])

    def test_undocumented_engine_to_engine_edge_fails(self) -> None:
        packages = {
            "sos-simulation": pkg(("sos-planner", None)),
            "sos-planner": pkg(),
        }
        errors = lint.check_engine_composition(packages)
        self.assertEqual(len(errors), 1)
        self.assertIn("sos-simulation -> sos-planner", errors[0])

    def test_engine_depending_on_non_engine_is_unrestricted(self) -> None:
        packages = {
            "sos-theory": pkg(("sos-core", None), ("sos-store", None)),
            "sos-core": pkg(),
            "sos-store": pkg(),
        }
        self.assertEqual(lint.check_engine_composition(packages), [])

    def test_discovery_stage_may_depend_on_reasoning(self) -> None:
        packages = {
            "sde-hypothesis": pkg(("sos-reasoning", None)),
            "sos-reasoning": pkg(),
        }
        self.assertEqual(lint.check_engine_composition(packages), [])

    def test_discovery_stage_depending_on_other_engine_fails(self) -> None:
        packages = {
            "sde-hypothesis": pkg(("sos-planner", None)),
            "sos-planner": pkg(),
        }
        errors = lint.check_engine_composition(packages)
        self.assertEqual(len(errors), 1)
        self.assertIn("sde-hypothesis -> sos-planner", errors[0])


class BackendConfinementTests(unittest.TestCase):
    def test_sos_scirust_may_depend_on_scirust_star(self) -> None:
        packages = {"sos-scirust": pkg(("scirust-gp", None))}
        self.assertEqual(lint.check_backend_confinement(packages), [])

    def test_sos_ccos_may_depend_on_scirust_sciagent(self) -> None:
        packages = {"sos-ccos": pkg(("scirust-sciagent", None))}
        self.assertEqual(lint.check_backend_confinement(packages), [])

    def test_other_crate_depending_on_scirust_star_fails(self) -> None:
        packages = {"sos-cli": pkg(("scirust-gp", None))}
        errors = lint.check_backend_confinement(packages)
        self.assertEqual(len(errors), 1)
        self.assertIn("sos-cli depends on scirust-gp", errors[0])

    def test_dev_dependency_leak_still_fails(self) -> None:
        packages = {"sos-mcp": pkg(("scirust-sciagent", "dev"))}
        errors = lint.check_backend_confinement(packages)
        self.assertEqual(len(errors), 1)
        self.assertIn("sos-mcp depends on scirust-sciagent", errors[0])


class RealWorkspaceIntegrationTest(unittest.TestCase):
    """Runs the real `cargo metadata` against this repo's actual sos/Cargo.toml."""

    def test_current_workspace_satisfies_all_invariants(self) -> None:
        metadata = lint.load_metadata(lint.DEFAULT_MANIFEST)
        packages = lint.workspace_packages(metadata)
        self.assertGreaterEqual(len(packages), 16)
        self.assertEqual(lint.run_all_checks(packages), [])


if __name__ == "__main__":
    unittest.main()
