"""Enable running haoline as a module: python -m haoline.

This provides an alternative to the installed CLI entry points,
which may not be on PATH for user-level pip installs.

Usage:
    python -m haoline model.onnx --out-html report.html
    python -m haoline --help
    python -m haoline check-install

Subcommands:
    python -m haoline web          # Launch Streamlit UI
    python -m haoline compare ...  # Compare multiple models
    python -m haoline list-hardware  # List hardware profiles
    python -m haoline list-formats   # List supported formats
"""

import sys
from pathlib import Path


def _maybe_insert_inspect() -> None:
    """If first arg looks like a model file, insert 'inspect' subcommand.

    This allows `haoline model.onnx` to work without explicitly typing
    `haoline inspect model.onnx`.
    """
    if len(sys.argv) < 2:
        return

    first_arg = sys.argv[1]

    # Skip if it's already a known subcommand or option
    known_commands = {
        "inspect",
        "web",
        "compare",
        "list-hardware",
        "list-formats",
        "check-install",
        "check-deps",
        "--help",
        "-h",
        "--version",
        "-V",
        "--list-hardware",
        "--list-formats",
    }
    if first_arg in known_commands or first_arg.startswith("-"):
        return

    # Check if it looks like a file path (has extension or exists)
    path = Path(first_arg)
    model_extensions = {
        ".onnx",
        ".pt",
        ".pth",
        ".safetensors",
        ".gguf",
        ".tflite",
        ".mlmodel",
        ".mlpackage",
        ".pb",
        ".h5",
        ".keras",
        ".engine",
        ".plan",
        ".xml",
    }
    if path.suffix.lower() in model_extensions or path.exists():
        # Insert 'inspect' as the subcommand
        sys.argv.insert(1, "inspect")


def main() -> None:
    """Entry point for `python -m haoline`."""
    _maybe_insert_inspect()

    from haoline.cli_typer import app

    app()


if __name__ == "__main__":
    main()
