from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path(__file__).with_name("scirust_symbolic_reward.py")
SPEC = importlib.util.spec_from_file_location("scirust_symbolic_reward", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
reward = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reward)


class SciRustSoupRewardTests(unittest.TestCase):
    def _fake_reward(self, directory: Path) -> Path:
        if os.name == "nt":
            self.skipTest("fake executable fixture uses a POSIX shebang")
        binary = directory / "scirust-reward"
        binary.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "for line in sys.stdin:\n"
            "    row = json.loads(line)\n"
            "    score = 1.0 if row['candidate'].replace(' ', '') in {'x+x','2*x'} and row['reference'].replace(' ', '') == '2*x' else 0.0\n"
            "    print(json.dumps({'schema_version':1,'kind':'symbolic_equivalence','id':row['id'],'score':score}))\n",
            encoding="utf-8",
        )
        binary.chmod(0o755)
        return binary

    def test_batch_bridge_extracts_final_answers_and_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            old = os.environ.get("SCIRUST_REWARD_BIN")
            os.environ["SCIRUST_REWARD_BIN"] = str(self._fake_reward(tmp))
            try:
                scores = reward.reward_fn(
                    [
                        [{"role": "assistant", "content": "reasoning #### x + x"}],
                        [{"role": "assistant", "content": "\\boxed{x+1}"}],
                    ],
                    answer=["2*x", "2*x"],
                )
            finally:
                if old is None:
                    os.environ.pop("SCIRUST_REWARD_BIN", None)
                else:
                    os.environ["SCIRUST_REWARD_BIN"] = old
            self.assertEqual(scores, [1.0, 0.0])

    def test_empty_reference_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            reward.reward_fn([[{"content": "x"}]], answer=[""])

    def test_length_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            reward.reward_fn([], answer=["x"])


if __name__ == "__main__":
    unittest.main()
