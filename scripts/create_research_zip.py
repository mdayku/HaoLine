#!/usr/bin/env python3
"""
Create a minimal .zip of the codebase for deep research ingestion.

Uses git ls-files to respect .gitignore, then applies additional filters
to keep only code, config, and documentation files.
"""

import subprocess
import zipfile
from pathlib import Path


# File extensions to INCLUDE (research-relevant)
INCLUDE_EXTENSIONS = {
    # Source code
    ".py",
    # Config
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".cfg",
    ".ini",
    # Documentation
    ".md",
    ".rst",
    ".txt",
    # Web assets (for streamlit)
    ".css",
    ".js",
    ".html",  # Only if small/template
    # Shell scripts
    ".sh",
    ".ps1",
    ".bat",
}

# Specific files to always include (even without extension match)
INCLUDE_FILES = {
    "Makefile",
    "Dockerfile",
    ".gitignore",
    ".cursorrules",
    "LICENSE",
    "requirements.txt",
}

# Directories to EXCLUDE even if tracked
EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    "demos",  # Large demo outputs
    "assets",  # Generated images
    "integration_output",
}

# Max file size to include (skip large generated files)
MAX_FILE_SIZE_KB = 500  # 500KB max per file


def get_tracked_files() -> list[str]:
    """Get list of files tracked by git (respects .gitignore)."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().split("\n")


def should_include(file_path: str) -> bool:
    """Determine if a file should be included in the research zip."""
    path = Path(file_path)
    
    # Check if in excluded directory
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return False
    
    # Check if it's a specifically included file
    if path.name in INCLUDE_FILES:
        return True
    
    # Check extension
    if path.suffix.lower() not in INCLUDE_EXTENSIONS:
        return False
    
    # Check file size (if file exists)
    full_path = Path.cwd() / path
    if full_path.exists():
        size_kb = full_path.stat().st_size / 1024
        if size_kb > MAX_FILE_SIZE_KB:
            print(f"  Skipping (too large: {size_kb:.0f}KB): {file_path}")
            return False
    
    return True


def create_research_zip(output_name: str = "haoline_research.zip") -> None:
    """Create the research zip file."""
    tracked_files = get_tracked_files()
    included_files = [f for f in tracked_files if should_include(f)]
    
    print(f"Found {len(tracked_files)} tracked files")
    print(f"Including {len(included_files)} research-relevant files")
    print()
    
    output_path = Path.cwd() / output_name
    
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(included_files):
            full_path = Path.cwd() / file_path
            if full_path.exists():
                zf.write(full_path, file_path)
                print(f"  Added: {file_path}")
    
    # Report final size
    size_kb = output_path.stat().st_size / 1024
    size_mb = size_kb / 1024
    
    print()
    print(f"Created: {output_path}")
    print(f"Size: {size_mb:.2f} MB ({size_kb:.0f} KB)")
    print(f"Files: {len(included_files)}")


if __name__ == "__main__":
    create_research_zip()

