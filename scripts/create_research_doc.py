#!/usr/bin/env python3
"""
Create a single markdown document with the entire codebase for LLM ingestion.

Includes:
- Project overview from README
- File tree of the repo
- All Python source code with file headers
- Key documentation (PRD, BACKLOG, Architecture)
"""

import subprocess
from pathlib import Path


# Files to include in order (documentation first, then code)
DOC_FILES = [
    "README.md",
    "PRD.md",
    "BACKLOG.md",
    "Architecture.md",
    "memory_bank/GLOSSARY.md",
    "memory_bank/DECISIONS.md",
]

# Directories containing Python code to include
CODE_DIRS = [
    "src/haoline",
]

# Files/dirs to exclude from tree and code
EXCLUDE_PATTERNS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "*.egg-info",
    "demos",
    "demo_outputs",
    "integration_output",
    ".pytest_cache",
    "assets",
}

# Skip test files to reduce size (optional)
SKIP_TESTS = False


def get_file_tree() -> str:
    """Generate a file tree of the repo."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    files = sorted(result.stdout.strip().split("\n"))
    
    # Filter out excluded patterns
    filtered = []
    for f in files:
        skip = False
        for pattern in EXCLUDE_PATTERNS:
            if pattern in f:
                skip = True
                break
        if not skip:
            filtered.append(f)
    
    # Build tree structure
    tree_lines = ["```"]
    tree_lines.append("haoline/")
    
    for f in filtered:
        # Simple indentation based on depth
        parts = f.split("/")
        indent = "  " * len(parts)
        tree_lines.append(f"{indent}{parts[-1]}")
    
    tree_lines.append("```")
    return "\n".join(tree_lines)


def get_python_files() -> list[Path]:
    """Get all Python files from code directories."""
    py_files = []
    
    for code_dir in CODE_DIRS:
        code_path = Path.cwd() / code_dir
        if code_path.exists():
            for py_file in sorted(code_path.rglob("*.py")):
                # Skip tests if configured
                if SKIP_TESTS and "/tests/" in str(py_file):
                    continue
                # Skip excluded patterns
                skip = False
                for pattern in EXCLUDE_PATTERNS:
                    if pattern in str(py_file):
                        skip = True
                        break
                if not skip:
                    py_files.append(py_file)
    
    return py_files


def read_file_safe(path: Path) -> str:
    """Read file with fallback encoding."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def create_research_doc(output_name: str = "HAOLINE_CODEBASE.md") -> None:
    """Create the consolidated research document."""
    sections = []
    
    # Header
    sections.append("# HaoLine Complete Codebase")
    sections.append("")
    sections.append("> This document contains the complete HaoLine codebase for LLM analysis.")
    sections.append("> Generated for deep research ingestion.")
    sections.append("")
    sections.append("---")
    sections.append("")
    
    # Table of Contents
    sections.append("## Table of Contents")
    sections.append("")
    sections.append("1. [File Tree](#file-tree)")
    sections.append("2. [Documentation](#documentation)")
    sections.append("3. [Source Code](#source-code)")
    sections.append("")
    sections.append("---")
    sections.append("")
    
    # File Tree
    sections.append("## File Tree")
    sections.append("")
    sections.append(get_file_tree())
    sections.append("")
    sections.append("---")
    sections.append("")
    
    # Documentation
    sections.append("## Documentation")
    sections.append("")
    
    for doc_file in DOC_FILES:
        doc_path = Path.cwd() / doc_file
        if doc_path.exists():
            sections.append(f"### {doc_file}")
            sections.append("")
            content = read_file_safe(doc_path)
            # Indent content to avoid markdown conflicts
            sections.append("```markdown")
            sections.append(content)
            sections.append("```")
            sections.append("")
    
    sections.append("---")
    sections.append("")
    
    # Source Code
    sections.append("## Source Code")
    sections.append("")
    
    py_files = get_python_files()
    print(f"Including {len(py_files)} Python files...")
    
    for py_file in py_files:
        relative_path = py_file.relative_to(Path.cwd())
        sections.append(f"### `{relative_path}`")
        sections.append("")
        sections.append("```python")
        sections.append(read_file_safe(py_file))
        sections.append("```")
        sections.append("")
    
    # Write output
    output_path = Path.cwd() / output_name
    output_path.write_text("\n".join(sections), encoding="utf-8")
    
    # Report
    size_kb = output_path.stat().st_size / 1024
    size_mb = size_kb / 1024
    
    print()
    print(f"Created: {output_path}")
    print(f"Size: {size_mb:.2f} MB ({size_kb:.0f} KB)")
    print(f"Python files: {len(py_files)}")
    print(f"Doc files: {len([d for d in DOC_FILES if (Path.cwd() / d).exists()])}")


if __name__ == "__main__":
    create_research_doc()

