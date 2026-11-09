"""Reject common private-file mistakes before staging or publishing an export.

Within Git, inspect the INDEX (the bytes that would be committed), not just the
working files. Outside Git, inspect this directory. Never print file contents.
This does not audit earlier commits or establish ownership of research data.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_DIRS = {
    "private",
    "data",
    "raw",
    "results",
    "outputs",
    "viewers",
    "preparation_work",
    "report",
}
IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
}
ALLOWED_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".css", ".js", ".html"}
ALLOWED_NAMES = {".gitignore", ".gitattributes", "LICENSE"}
PATTERNS = [
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?:ghp|github_pat)_[A-Za-z0-9_]{30,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"data:(?:image|application)/[^;]+;base64,[A-Za-z0-9+/]{80,}"),
]


def problem(path, content):
    parts = path.parts
    # Source templates live under src/.../viewers and are not generated outputs.
    if parts[0] in BLOCKED_DIRS or "expected" in parts:
        return "private/generated directory or field-derived fixture"
    if path.name in {"project.json", "preparation.json"} or path.name.startswith(".env"):
        return "local configuration or credentials"
    if not (
        path.name in ALLOWED_NAMES
        or path.suffix in ALLOWED_SUFFIXES
        or path.name.endswith(".example.json")
    ):
        return "file type is not on the public-source allowlist"
    if any(pattern.search(content) for pattern in PATTERNS):
        return "possible credential or embedded data payload"
    if b"\x00" in content:
        return "binary content"
    return None


def files_to_check():
    git = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True
    )
    if git.returncode == 0:
        if Path(git.stdout.strip()).resolve() != ROOT:
            raise ValueError("Run in a standalone repository, not a nested parent Git tree.")
        entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=ROOT)
        for entry in entries.split(b"\x00"):
            if not entry:
                continue
            meta, raw_path = entry.split(b"\t", 1)
            mode, _, stage = meta.split()
            path = Path(raw_path.decode("utf-8"))
            if mode == b"120000" or stage != b"0":
                raise ValueError("Symlink or unresolved merge entry in the index.")
            content = subprocess.check_output(["git", "show", ":" + path.as_posix()], cwd=ROOT)
            yield path, content
    else:
        for path in sorted(ROOT.rglob("*")):
            relative = path.relative_to(ROOT)
            if any(part in IGNORE_DIRS or part.endswith(".egg-info") for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError("Symlinks are not allowed in the public export.")
            if path.is_file():
                yield relative, path.read_bytes()


def main():
    bad, count = [], 0
    try:
        for path, content in files_to_check():
            count += 1
            reason = problem(path, content)
            if reason:
                bad.append((path, reason))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print("Public-file check failed:", error, file=sys.stderr)
        return 1
    if not count:
        print("No staged files: stage the public source before checking.", file=sys.stderr)
        return 1
    for path, reason in bad:
        print(f"REJECT {path}: {reason}", file=sys.stderr)
    if bad:
        return 1
    print(f"Public-file check passed for {count} source files; review the staged diff too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
