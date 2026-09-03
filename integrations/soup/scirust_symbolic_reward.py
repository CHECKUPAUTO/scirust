"""SOUP custom reward bridge backed by the deterministic SciRust symbolic prover.

Configure SOUP to load this file as its custom ``reward_fn``. The bridge starts
one ``scirust-reward`` subprocess per reward batch, not per completion.

Scope: mathematical/symbolic answers that can be parsed by ``scirust-symbolic``.
A score of 0 means "not proven equivalent" or "candidate did not parse"; it is
not a proof that two expressions are unequal.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Any

MAX_BATCH = 4096
MAX_EXPRESSION_CHARS = 65536
REWARD_TIMEOUT_SECONDS = 30

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def _completion_text(completion: Any) -> str:
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion or "")


def _extract_expression(text: str) -> str:
    """Prefer standard final-answer markers, then fall back to the full output."""
    if "####" in text:
        return text.rsplit("####", 1)[1].strip()
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    return text.strip()


def _reward_binary() -> str:
    configured = os.environ.get("SCIRUST_REWARD_BIN")
    if configured:
        return configured
    resolved = shutil.which("scirust-reward")
    if resolved is None:
        raise RuntimeError(
            "scirust-reward was not found in PATH; install the SciRust CLI or set SCIRUST_REWARD_BIN"
        )
    return resolved


def reward_fn(completions: list[list[dict]], **kwargs: Any) -> list[float]:
    """Return 1.0 only when SciRust proves the candidate equivalent to ``answer``.

    SOUP supplies the dataset column ``answer`` through ``**kwargs`` for GRPO
    custom reward functions. Reference rows are treated as trusted input: if a
    reference expression is missing or invalid, ``scirust-reward`` fails closed
    and this bridge raises instead of silently assigning rewards.
    """
    answers = kwargs.get("answer")
    if not isinstance(answers, (list, tuple)):
        raise ValueError("SciRust symbolic reward requires an `answer` sequence")
    if len(completions) != len(answers):
        raise ValueError("completion/answer batch lengths differ")
    if len(completions) > MAX_BATCH:
        raise ValueError(f"reward batch exceeds {MAX_BATCH} records")

    records: list[str] = []
    for index, (completion, reference) in enumerate(zip(completions, answers)):
        candidate = _extract_expression(_completion_text(completion))
        reference_text = "" if reference is None else str(reference).strip()
        if not reference_text:
            raise ValueError(f"answer[{index}] is empty")
        if len(candidate) > MAX_EXPRESSION_CHARS or len(reference_text) > MAX_EXPRESSION_CHARS:
            raise ValueError(f"symbolic expression at index {index} exceeds size limit")
        records.append(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "symbolic_equivalence",
                    "id": index,
                    "candidate": candidate,
                    "reference": reference_text,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )

    if not records:
        return []
    completed = subprocess.run(
        [_reward_binary()],
        input="\n".join(records) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        check=False,
        timeout=REWARD_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()[:4096]
        raise RuntimeError(f"scirust-reward failed with exit {completed.returncode}: {detail}")

    rows = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != len(records):
        raise RuntimeError(
            f"scirust-reward returned {len(rows)} records for a batch of {len(records)}"
        )
    rewards = [0.0] * len(rows)
    for expected_id, line in enumerate(rows):
        result = json.loads(line)
        if result.get("schema_version") != 1 or result.get("kind") != "symbolic_equivalence":
            raise RuntimeError("scirust-reward returned an unsupported result schema")
        if result.get("id") != expected_id:
            raise RuntimeError("scirust-reward result order/id mismatch")
        score = result.get("score")
        if score not in (0, 0.0, 1, 1.0):
            raise RuntimeError("scirust-reward returned an invalid score")
        rewards[expected_id] = float(score)
    return rewards
