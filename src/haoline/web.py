#!/usr/bin/env python
# Copyright (c) 2025 HaoLine Contributors
# SPDX-License-Identifier: MIT

"""
HaoLine Web UI launcher.

Launches the Streamlit web interface for model analysis.

Usage:
    haoline-web           # Launch the web UI
    haoline-web --port 8080  # Custom port
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_streamlit_app_path() -> Path:
    """Get the path to the bundled streamlit app."""
    # The app is bundled within the package
    return Path(__file__).parent / "streamlit_app.py"


def get_config_dir() -> Path:
    """Get the path to the bundled .streamlit config directory."""
    return Path(__file__).parent / ".streamlit"


def main(argv: list[str] | None = None) -> int:
    """Launch the HaoLine Streamlit web interface."""
    parser = argparse.ArgumentParser(
        prog="haoline-web",
        description="Launch the HaoLine web interface for model analysis.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run the web server on (default: 8501)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to (default: localhost)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )

    args = parser.parse_args(argv)

    # Check if streamlit is installed
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Error: Streamlit is not installed.")
        print("Install it with: pip install haoline[web]")
        return 1

    app_path = get_streamlit_app_path()

    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}")
        print("This may indicate a broken installation. Try reinstalling:")
        print("  pip install --force-reinstall haoline[web]")
        return 1

    # Build streamlit command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]

    if args.no_browser:
        cmd.extend(["--server.headless", "true"])

    print(f"Starting HaoLine Web UI at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.\n")

    # Set up environment with config directory for dark theme
    env = os.environ.copy()
    config_dir = get_config_dir()
    if config_dir.exists():
        env["STREAMLIT_CONFIG_DIR"] = str(config_dir)

    try:
        return subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0


if __name__ == "__main__":
    sys.exit(main())
