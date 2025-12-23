"""Unit tests for the Typer CLI.

Tests the new Typer-based CLI to ensure commands work correctly.
"""

import re

from typer.testing import CliRunner

from haoline.cli_typer import app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_pattern.sub("", text)


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_help(self):
        """Test --help flag."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "HaoLine" in result.output
        assert "Universal Model Inspector" in result.output

    def test_version(self):
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "HaoLine" in result.output
        assert "version" in result.output

    def test_no_args_shows_help(self):
        """Test that no args shows help (exit code 2 is expected for missing args)."""
        result = runner.invoke(app, [])
        # Typer returns exit code 2 for missing required args/no command
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output


class TestInspectCommand:
    """Test the inspect command."""

    def test_inspect_help(self):
        """Test inspect --help."""
        result = runner.invoke(app, ["inspect", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "Analyze" in output
        assert "--out-json" in output
        assert "--out-html" in output
        assert "--hardware" in output

    def test_inspect_no_model(self):
        """Test inspect with no model path gives error."""
        result = runner.invoke(app, ["inspect"])
        assert result.exit_code == 1
        assert "Error" in result.output or "No model" in result.output


class TestListCommands:
    """Test the list commands."""

    def test_list_hardware(self):
        """Test list-hardware command."""
        result = runner.invoke(app, ["list-hardware"])
        assert result.exit_code == 0
        assert "Hardware Profiles" in result.output
        assert "RTX" in result.output or "rtx" in result.output

    def test_list_formats(self):
        """Test list-formats command."""
        result = runner.invoke(app, ["list-formats"])
        assert result.exit_code == 0
        assert "Model Formats" in result.output
        assert "ONNX" in result.output
        assert "PyTorch" in result.output
        assert "Built-in" in result.output


class TestCheckInstall:
    """Test the check-install command."""

    def test_check_install(self):
        """Test check-install command."""
        result = runner.invoke(app, ["check-install"])
        assert result.exit_code == 0
        assert "Installation Check" in result.output
        assert "Version" in result.output
        assert "CLI Commands" in result.output
        assert "Optional Features" in result.output

    def test_check_deps(self):
        """Test check-deps command."""
        result = runner.invoke(app, ["check-deps"])
        assert result.exit_code == 0
        assert "Dependency Check" in result.output
        assert "Format Converters" in result.output
        assert "Format Readers" in result.output
        assert "Features" in result.output
        assert "Summary" in result.output


class TestSubcommands:
    """Test subcommand routing."""

    def test_web_help(self):
        """Test web --help."""
        result = runner.invoke(app, ["web", "--help"])
        assert result.exit_code == 0
        assert "Launch" in result.output or "Streamlit" in result.output

    def test_compare_help(self):
        """Test compare --help."""
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "Compare" in result.output or "model" in result.output.lower()
