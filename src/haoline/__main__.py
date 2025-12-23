"""Enable running haoline as a module: python -m haoline.

This provides an alternative to the installed CLI entry points,
which may not be on PATH for user-level pip installs.

Usage:
    python -m haoline model.onnx --out-html report.html
    python -m haoline --help

For subcommands (if haoline-compare/haoline-web aren't on PATH):
    python -c "from haoline.compare import main; main()"
    python -c "from haoline.web import main; main()"
"""

from haoline.cli import run_inspect

if __name__ == "__main__":
    run_inspect()
