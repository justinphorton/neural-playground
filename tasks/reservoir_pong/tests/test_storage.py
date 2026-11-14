from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tasks.reservoir_pong.storage import save_json_atomic


class AtomicStorageTests(unittest.TestCase):
    def test_json_checkpoint_replaces_atomically_and_leaves_no_temporary_files(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "checkpoint.json"
            save_json_atomic(destination, {"stage": 1})
            save_json_atomic(destination, {"stage": 2})
            self.assertEqual(json.loads(destination.read_text()), {"stage": 2})
            self.assertEqual([path.name for path in destination.parent.iterdir()],
                             ["checkpoint.json"])


if __name__ == "__main__":
    unittest.main()
