"""Enable running haoline as a module: python -m haoline.

This provides an alternative to the installed CLI entry points,
which may not be on PATH for user-level pip installs.

Usage:
    python -m haoline model.onnx --out-html report.html
    python -m haoline --help
    python -m haoline --check-install

Subcommands:
    python -m haoline web          # Launch Streamlit UI
    python -m haoline compare ...  # Compare multiple models
    python -m haoline import-eval  # Import evaluation results
"""

from __future__ import annotations

import sys


def main() -> None:
    """Main entry point with subcommand routing."""
    # Check for subcommands
    if len(sys.argv) > 1:
        subcommand = sys.argv[1]

        if subcommand == "web":
            # Remove 'web' from argv so haoline-web sees correct args
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            from haoline.web import main as web_main

            sys.exit(web_main())

        elif subcommand == "compare":
            # Remove 'compare' from argv
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            from haoline.compare import main as compare_main

            compare_main()
            return

        elif subcommand == "import-eval":
            # Remove 'import-eval' from argv
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            from haoline.eval.cli import main as eval_main

            eval_main()
            return

        elif subcommand == "--check-install":
            check_install()
            return

    # Default: run main CLI
    from haoline.cli import run_inspect

    run_inspect()


def check_install() -> None:
    """Check installation status and report issues."""
    import shutil

    print("HaoLine Installation Check")
    print("=" * 40)

    # Version
    try:
        from haoline import __version__

        print(f"Version: {__version__}")
    except ImportError:
        print("Version: ERROR - haoline not properly installed")
        return

    # Check if CLI scripts are on PATH
    print("\nCLI Commands:")
    cli_commands = {
        "haoline": "python -m haoline",
        "haoline-compare": "python -m haoline compare",
        "haoline-web": "python -m haoline web",
        "haoline-import-eval": "python -m haoline import-eval",
    }
    for cmd, alt in cli_commands.items():
        path = shutil.which(cmd)
        if path:
            print(f"  {cmd}: {path}")
        else:
            print(f"  {cmd}: NOT ON PATH (use: {alt})")

    # Check optional dependencies
    print("\nOptional Dependencies:")
    extras = {
        "streamlit": "web UI (haoline[web])",
        "torch": "PyTorch conversion (haoline[pytorch])",
        "tensorflow": "TensorFlow conversion (haoline[tensorflow])",
        "openai": "LLM summaries (haoline[llm])",
        "playwright": "PDF export (haoline[pdf])",
        "pynvml": "GPU metrics (haoline[gpu])",
        "onnxruntime": "benchmarking (haoline[runtime])",
        "safetensors": "SafeTensors format (haoline[safetensors])",
        "coremltools": "CoreML format (haoline[coreml])",
        "tensorrt": "TensorRT format (haoline[tensorrt])",
    }

    for module, desc in extras.items():
        try:
            __import__(module)
            print(f"  {module}: installed")
        except ImportError:
            print(f"  {module}: not installed - {desc}")

    # Python info
    print(f"\nPython: {sys.version}")
    print(f"Executable: {sys.executable}")

    # Suggest fixes if CLI not on PATH
    if not shutil.which("haoline"):
        print("\n" + "=" * 40)
        print("TIP: CLI commands not on PATH.")
        print("Use module invocation instead:")
        print("  python -m haoline model.onnx --out-html report.html")
        print("  python -m haoline web")
        print("  python -m haoline compare ...")


if __name__ == "__main__":
    main()
