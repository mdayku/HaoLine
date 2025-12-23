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

from haoline.cli_typer import main

if __name__ == "__main__":
    main()
