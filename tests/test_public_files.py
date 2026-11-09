"""Exercise the public-file guard, including Git's staged bytes."""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if ROOT.name == "tests":
    ROOT = ROOT.parent
SPEC = importlib.util.spec_from_file_location("public_guard", ROOT / "tools/check_public_files.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class PublicFileTests(unittest.TestCase):
    def test_data_and_private_config_are_rejected(self):
        for name in [
            "data/source/measurements.py",
            "results/table.csv",
            "private/input.txt",
            "config/project.json",
            "model.joblib",
            "photo.jpg",
            "tests/expected/labels.txt",
        ]:
            with self.subTest(name=name):
                self.assertIsNotNone(guard.problem(Path(name), b"example"))

    def test_code_and_placeholder_configuration_are_allowed(self):
        for name in ["src/hedgeseg/io.py", "README.md", "config/project.example.json"]:
            self.assertIsNone(guard.problem(Path(name), b"example"))

    def test_binary_and_embedded_payload_are_rejected(self):
        self.assertIsNotNone(guard.problem(Path("notes.md"), b"a\x00b"))
        content = b"data:image/png;base64," + b"A" * 100
        self.assertIsNotNone(guard.problem(Path("page.html"), content))

    def test_index_bytes_are_checked_instead_of_working_copy(self):
        original = guard.ROOT
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            try:
                guard.ROOT = root
                subprocess.run(["git", "init", "-q"], cwd=root, check=True)
                (root / ".gitignore").write_text("private/\n")
                (root / "README.md").write_text("public")
                (root / "private").mkdir()
                (root / "private/secret.txt").write_text("do not publish")
                subprocess.run(["git", "add", ".gitignore", "README.md"], cwd=root, check=True)
                subprocess.run(["git", "add", "-f", "private/secret.txt"], cwd=root, check=True)
                # Removing the working file does not remove its staged bytes.
                (root / "private/secret.txt").unlink()
                staged = dict(guard.files_to_check())
                self.assertIn(Path("private/secret.txt"), staged)
                self.assertIsNotNone(
                    guard.problem(Path("private/secret.txt"), staged[Path("private/secret.txt")])
                )
            finally:
                guard.ROOT = original


if __name__ == "__main__":
    unittest.main()
