import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from full_data_supervisor import (  # noqa: E402
    acquire_singleton_lock,
    build_smoke_resume_command,
)


@unittest.skipUnless(os.name == "posix", "the supervisor is Linux/WSL-only")
class FullDataSupervisorTests(unittest.TestCase):
    def test_smoke_resume_preserves_identity_limit_and_removes_only_stop_gate(self):
        command = [
            "/opt/minimind-flash-venv/bin/python", "train_full_data_stage.py",
            "--stage", "pretrain", "--smoke-records", "1024",
            "--stop-after-updates", "2",
        ]

        resumed = build_smoke_resume_command(command, "/tmp/smoke/checkpoint.pt")

        self.assertEqual(resumed, [
            "/opt/minimind-flash-venv/bin/python", "train_full_data_stage.py",
            "--stage", "pretrain", "--smoke-records", "1024",
            "--resume", "/tmp/smoke/checkpoint.pt",
        ])

    def test_singleton_flock_rejects_second_process_without_removing_lock_file(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("MINIMIND_LOCK_TEST_DIR")) as directory:
            root = Path(directory)
            lock = root / ".pipeline.lock"
            first = acquire_singleton_lock(root)
            contender = (
                "import sys; "
                f"sys.path.insert(0, {str(SCRIPTS)!r}); "
                "from full_data_supervisor import acquire_singleton_lock; "
                "acquire_singleton_lock(sys.argv[1])"
            )

            rejected = subprocess.run([sys.executable, "-c", contender, str(root)],
                                      check=False, capture_output=True, text=True)

            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("another supervisor holds", rejected.stderr)
            self.assertTrue(lock.is_file())
            first.close()

            second = acquire_singleton_lock(root)
            try:
                self.assertTrue(lock.is_file())
            finally:
                second.close()


if __name__ == "__main__":
    unittest.main()
