#!/usr/bin/env python3
"""Pre-commit quality checks for HaoLine.

Usage:
    python scripts/check.py          # Format + lint (default)
    python scripts/check.py --all    # Format + lint + mypy + tests
    python scripts/check.py --mypy   # Just mypy
"""

import subprocess
import sys


def run(cmd: list[str], check: bool = True) -> int:
    """Run a command and return exit code."""
    print(f"\n{'='*60}\n$ {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        print(f"FAILED: {' '.join(cmd)}")
    return result.returncode


def main() -> int:
    """Run quality checks."""
    args = set(sys.argv[1:])
    failed = False

    # Always run format + lint
    if run(["python", "-m", "ruff", "format", "src/haoline/"]) != 0:
        failed = True
    if run(["python", "-m", "ruff", "check", "src/haoline/", "--fix"]) != 0:
        failed = True

    # Optional: mypy
    if "--all" in args or "--mypy" in args:
        if run(["python", "-m", "mypy", "src/haoline/", "--ignore-missing-imports"]) != 0:
            failed = True

    # Optional: tests
    if "--all" in args or "--test" in args:
        if run(["python", "-m", "pytest", "src/haoline/tests/", "-v", "--tb=short"]) != 0:
            failed = True

    if failed:
        print("\n[FAILED] Some checks failed")
        return 1
    print("\n[OK] All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

