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

# Check Python version early, before importing Typer (which fails on 3.13+)
MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 12)

if sys.version_info < MIN_PYTHON or sys.version_info >= (MAX_PYTHON[0], MAX_PYTHON[1] + 1):
    print(
        f"\n[ERROR] HaoLine requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}-"
        f"{MAX_PYTHON[0]}.{MAX_PYTHON[1]}, but you're running {sys.version_info[0]}.{sys.version_info[1]}.\n"
        "\nMost ML packages (PyTorch, TensorFlow, ONNX) don't support Python 3.13+ yet."
        "\n\nTo fix this, create a new environment with a supported Python version:"
        "\n  conda create -n haoline python=3.11"
        "\n  conda activate haoline"
        "\n  pip install haoline"
        "\n"
    )
    sys.exit(1)


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
