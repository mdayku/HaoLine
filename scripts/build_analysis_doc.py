#!/usr/bin/env python3
"""Build consolidated analysis document for deep research agents.

Collects all .py and .md files in the repository into a single markdown file
for feeding to AI agents for comprehensive codebase analysis.

Usage:
    python scripts/build_analysis_doc.py
    python scripts/build_analysis_doc.py --output custom_name.md
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


# Directories to skip
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".github",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    ".venv-dev",
    "venv",
    "env",
    ".env",
    "build",
    "dist",
    "*.egg-info",
    "runs",  # YOLO training outputs
    "demo_outputs",  # Demo output files
    "integration_output",  # Integration test outputs
    "site-packages",  # Installed packages (in case of partial match)
    "Lib",  # Windows venv lib directory
    "lib",  # Linux venv lib directory
}

# Files to skip
SKIP_FILES = {
    "HAOLINE_ANALYSIS.md",  # Don't include ourselves
}

# File patterns to skip
SKIP_PATTERNS = {
    "*.pyc",
    "*.pyo",
}


def should_skip_dir(path: Path) -> bool:
    """Check if directory should be skipped."""
    return path.name in SKIP_DIRS or path.name.endswith(".egg-info")


def should_skip_file(path: Path) -> bool:
    """Check if file should be skipped."""
    return path.name in SKIP_FILES


def collect_files(repo_root: Path) -> tuple[list[Path], list[Path]]:
    """Collect all .py and .md files in the repository.

    Returns:
        Tuple of (python_files, markdown_files), each sorted by path.
    """
    py_files: list[Path] = []
    md_files: list[Path] = []

    for item in repo_root.rglob("*"):
        # Skip directories in SKIP_DIRS
        if any(skip in item.parts for skip in SKIP_DIRS):
            continue

        if item.is_file() and not should_skip_file(item):
            if item.suffix == ".py":
                py_files.append(item)
            elif item.suffix == ".md":
                md_files.append(item)

    # Sort by path for consistent ordering
    py_files.sort()
    md_files.sort()

    return py_files, md_files


def get_relative_path(file_path: Path, repo_root: Path) -> str:
    """Get relative path from repo root."""
    try:
        return str(file_path.relative_to(repo_root))
    except ValueError:
        return str(file_path)


def build_toc(md_files: list[Path], py_files: list[Path], repo_root: Path) -> str:
    """Build table of contents."""
    lines = ["## Table of Contents\n"]

    lines.append("### Documentation Files\n")
    for f in md_files:
        rel_path = get_relative_path(f, repo_root)
        anchor = rel_path.replace("/", "").replace("\\", "").replace(".", "").lower()
        lines.append(f"- [{rel_path}](#{anchor})")

    lines.append("\n### Python Files\n")

    # Group by directory
    current_dir = ""
    for f in py_files:
        rel_path = get_relative_path(f, repo_root)
        parent = str(f.parent.relative_to(repo_root)) if f.parent != repo_root else ""

        if parent != current_dir:
            current_dir = parent
            if parent:
                lines.append(f"\n**{parent}/**\n")

        anchor = rel_path.replace("/", "").replace("\\", "").replace(".", "").lower()
        lines.append(f"- [{f.name}](#{anchor})")

    return "\n".join(lines)


def read_file_content(file_path: Path) -> str:
    """Read file content, handling encoding issues."""
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return file_path.read_text(encoding="latin-1")
        except Exception as e:
            return f"# Error reading file: {e}"


def build_analysis_doc(repo_root: Path, output_path: Path) -> None:
    """Build the consolidated analysis document."""
    py_files, md_files = collect_files(repo_root)

    total_files = len(py_files) + len(md_files)
    print(f"Found {len(md_files)} markdown files and {len(py_files)} Python files")

    lines: list[str] = []

    # Header
    lines.append("# HaoLine Complete Codebase Analysis Document")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Files:** {total_files} ({len(md_files)} .md, {len(py_files)} .py)")
    lines.append("")
    lines.append("> This document contains all documentation and Python source code")
    lines.append("> from the HaoLine repository for comprehensive AI analysis.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Table of contents
    lines.append(build_toc(md_files, py_files, repo_root))
    lines.append("")
    lines.append("---")
    lines.append("")

    # Documentation files first
    lines.append("# Documentation Files")
    lines.append("")

    for md_file in md_files:
        rel_path = get_relative_path(md_file, repo_root)
        print(f"  Adding: {rel_path}")

        lines.append(f"## {rel_path}")
        lines.append("")
        content = read_file_content(md_file)
        lines.append(content)
        lines.append("")
        lines.append("---")
        lines.append("")

    # Python files
    lines.append("# Python Source Files")
    lines.append("")

    for py_file in py_files:
        rel_path = get_relative_path(py_file, repo_root)
        print(f"  Adding: {rel_path}")

        lines.append(f"## {rel_path}")
        lines.append("")
        lines.append("```python")
        content = read_file_content(py_file)
        lines.append(content)
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Write output
    output_content = "\n".join(lines)
    output_path.write_text(output_content, encoding="utf-8")

    # Stats
    line_count = output_content.count("\n")
    size_kb = len(output_content.encode("utf-8")) / 1024
    size_mb = size_kb / 1024

    print("")
    print(f"Generated: {output_path}")
    print(f"  Lines: {line_count:,}")
    print(f"  Size: {size_mb:.2f} MB ({size_kb:.0f} KB)")
    print(f"  Estimated tokens: ~{line_count * 4:,} (rough estimate)")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build consolidated analysis document for deep research agents."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="HAOLINE_ANALYSIS.md",
        help="Output filename (default: HAOLINE_ANALYSIS.md)",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Repository root (default: auto-detect)",
    )

    args = parser.parse_args()

    # Find repo root
    if args.repo_root:
        repo_root = Path(args.repo_root)
    else:
        # Auto-detect: find directory containing pyproject.toml
        current = Path(__file__).resolve().parent
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                repo_root = current
                break
            current = current.parent
        else:
            repo_root = Path.cwd()

    output_path = repo_root / args.output

    print(f"Repository root: {repo_root}")
    print(f"Output file: {output_path}")
    print("")

    build_analysis_doc(repo_root, output_path)


if __name__ == "__main__":
    main()

