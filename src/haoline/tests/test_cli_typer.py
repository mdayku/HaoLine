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

    def test_list_hardware_subcommand(self):
        """Test list-hardware subcommand."""
        result = runner.invoke(app, ["list-hardware"])
        assert result.exit_code == 0
        assert "Hardware Profiles" in result.output
        assert "RTX" in result.output or "rtx" in result.output

    def test_list_hardware_flag(self):
        """Test --list-hardware flag (backwards compatible)."""
        result = runner.invoke(app, ["--list-hardware"])
        assert result.exit_code == 0
        assert "Hardware Profiles" in result.output

    def test_list_formats_subcommand(self):
        """Test list-formats subcommand."""
        result = runner.invoke(app, ["list-formats"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "Supported" in output or "Format" in output
        assert "ONNX" in output
        assert "PyTorch" in output

    def test_list_formats_flag(self):
        """Test --list-formats flag (backwards compatible)."""
        result = runner.invoke(app, ["--list-formats"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "Supported Formats" in output or "Format" in output


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


class TestDirectModelPath:
    """Test the 'haoline model.onnx' shortcut (inserts 'inspect' automatically)."""

    def test_maybe_insert_inspect_with_onnx(self):
        """Test that .onnx files trigger inspect insertion."""
        import sys

        from haoline.__main__ import _maybe_insert_inspect

        # Save original argv
        original_argv = sys.argv.copy()
        try:
            sys.argv = ["haoline", "model.onnx"]
            _maybe_insert_inspect()
            assert sys.argv == ["haoline", "inspect", "model.onnx"]
        finally:
            sys.argv = original_argv

    def test_maybe_insert_inspect_with_subcommand(self):
        """Test that known subcommands are not modified."""
        import sys

        from haoline.__main__ import _maybe_insert_inspect

        original_argv = sys.argv.copy()
        try:
            sys.argv = ["haoline", "web"]
            _maybe_insert_inspect()
            assert sys.argv == ["haoline", "web"]  # Unchanged
        finally:
            sys.argv = original_argv

    def test_maybe_insert_inspect_with_flag(self):
        """Test that flags are not treated as model files."""
        import sys

        from haoline.__main__ import _maybe_insert_inspect

        original_argv = sys.argv.copy()
        try:
            sys.argv = ["haoline", "--help"]
            _maybe_insert_inspect()
            assert sys.argv == ["haoline", "--help"]  # Unchanged
        finally:
            sys.argv = original_argv


class TestFailOnThresholds:
    """Test the --fail-on threshold functionality."""

    def test_parse_threshold_percentage(self):
        """Test parsing percentage thresholds."""
        from haoline.cli_typer import _parse_threshold

        metric, value, is_pct = _parse_threshold("latency_increase=10%")
        assert metric == "latency_increase"
        assert value == 10.0
        assert is_pct is True

    def test_parse_threshold_absolute(self):
        """Test parsing absolute value thresholds."""
        from haoline.cli_typer import _parse_threshold

        metric, value, is_pct = _parse_threshold("memory_increase=1000")
        assert metric == "memory_increase"
        assert value == 1000.0
        assert is_pct is False

    def test_parse_threshold_boolean(self):
        """Test parsing boolean thresholds."""
        from haoline.cli_typer import _parse_threshold

        metric, value, is_pct = _parse_threshold("new_risk_signals")
        assert metric == "new_risk_signals"
        assert value is None
        assert is_pct is False

    def test_check_thresholds_pass(self):
        """Test threshold checking when all pass."""
        from haoline.cli_typer import _check_thresholds

        compare_json = {
            "variants": [
                {
                    "precision": "fp32",
                    "total_params": 1000000,
                    "memory_bytes": 4000000,
                    "deltas_vs_baseline": None,  # Baseline
                    "hardware_estimates": {"theoretical_latency_ms": 10.0},
                },
                {
                    "precision": "fp16",
                    "total_params": 1000000,
                    "memory_bytes": 2000000,
                    "deltas_vs_baseline": {
                        "total_params": 0,
                        "memory_bytes": -2000000,
                        "latency_ms": 0.5,  # 5% increase
                    },
                    "new_risk_signals": [],
                },
            ]
        }

        results = _check_thresholds(compare_json, ["latency_increase=10%"])
        assert len(results) >= 1
        # 5% < 10% threshold, should pass
        assert all(passed for _, _, passed in results)

    def test_check_thresholds_fail(self):
        """Test threshold checking when threshold is violated."""
        from haoline.cli_typer import _check_thresholds

        compare_json = {
            "variants": [
                {
                    "precision": "fp32",
                    "total_params": 1000000,
                    "memory_bytes": 4000000,
                    "deltas_vs_baseline": None,  # Baseline
                    "hardware_estimates": {"theoretical_latency_ms": 10.0},
                },
                {
                    "precision": "int8",
                    "total_params": 1000000,
                    "memory_bytes": 5000000,
                    "deltas_vs_baseline": {
                        "total_params": 0,
                        "memory_bytes": 1000000,  # 25% increase
                    },
                },
            ]
        }

        results = _check_thresholds(compare_json, ["memory_increase=20%"])
        assert len(results) >= 1
        # 25% > 20% threshold, should fail
        failed = [r for r in results if not r[2]]
        assert len(failed) >= 1

    def test_compare_help_shows_fail_on(self):
        """Test that compare --help mentions --fail-on."""
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--fail-on" in output

    def test_compare_help_shows_decision_report(self):
        """Test that compare --help mentions --decision-report."""
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--decision-report" in output


class TestDecisionReport:
    """Test decision report generation."""

    def test_build_decision_report_structure(self):
        """Test that decision report has correct structure."""
        from pathlib import Path

        from haoline.cli_typer import _build_decision_report

        compare_json = {
            "baseline_precision": "fp32",
            "architecture_compatible": True,
            "compatibility_warnings": [],
            "variants": [
                {
                    "precision": "fp32",
                    "total_params": 1000000,
                    "memory_bytes": 4000000,
                    "deltas_vs_baseline": None,
                },
                {
                    "precision": "int8",
                    "total_params": 1000000,
                    "memory_bytes": 2000000,
                    "deltas_vs_baseline": {"memory_bytes": -2000000},
                },
            ],
        }

        threshold_results = [
            ("memory_increase", "Memory decreased by 50%", True),
        ]

        # Create mock paths
        models = [Path("baseline.onnx"), Path("candidate.onnx")]

        report = _build_decision_report(
            compare_json,
            ["memory_increase=20%"],
            threshold_results,
            models,
        )

        assert "decision_report" in report
        dr = report["decision_report"]
        assert "timestamp" in dr
        assert "haoline_version" in dr
        assert "models_compared" in dr
        assert "constraints" in dr
        assert "decision" in dr
        assert dr["decision"] == "APPROVED"

    def test_build_decision_report_rejected(self):
        """Test that decision report shows REJECTED when thresholds fail."""
        from pathlib import Path

        from haoline.cli_typer import _build_decision_report

        compare_json = {
            "baseline_precision": "fp32",
            "architecture_compatible": True,
            "variants": [],
        }

        threshold_results = [
            ("latency_increase", "Latency increased by 15%", False),  # Failed
        ]

        models = [Path("model.onnx")]

        report = _build_decision_report(
            compare_json,
            ["latency_increase=10%"],
            threshold_results,
            models,
        )

        assert report["decision_report"]["decision"] == "REJECTED"

    def test_decision_report_to_markdown(self):
        """Test markdown conversion of decision report."""
        from haoline.cli_typer import _decision_report_to_markdown

        report = {
            "decision_report": {
                "timestamp": "2024-01-01T00:00:00Z",
                "haoline_version": "0.9.7",
                "models_compared": [
                    {"path": "baseline.onnx", "hash_md5": "abc123", "size_bytes": 1000000},
                    {"path": "candidate.onnx", "hash_md5": "def456", "size_bytes": 500000},
                ],
                "baseline": "fp32",
                "constraints": {
                    "memory_increase": {
                        "threshold": "20%",
                        "results": [{"message": "Memory decreased", "passed": True}],
                    }
                },
                "decision": "APPROVED",
                "architecture_compatible": True,
                "compatibility_warnings": [],
                "recommendations": ["Consider INT8 quantization"],
            }
        }

        md = _decision_report_to_markdown(report)

        assert "# Model Decision Report" in md
        assert "0.9.7" in md
        assert "baseline.onnx" in md
        assert "candidate.onnx" in md
        assert "APPROVED" in md
        assert "memory_increase" in md
        assert "Consider INT8 quantization" in md


class TestErrorSuggestions:
    """Test CLI error message suggestions (Task 52.3.5)."""

    def test_error_suggestion_for_missing_tensorrt(self):
        """Task 52.3.5: CLI should suggest installation for missing tensorrt."""
        from pathlib import Path

        from haoline.cli_typer import _get_error_suggestion

        # Simulate ModuleNotFoundError for tensorrt
        error = ModuleNotFoundError("No module named 'tensorrt'")
        suggestion = _get_error_suggestion(
            error, model_path=Path("model.engine"), from_pytorch=False
        )

        # Should suggest installing the missing module
        assert suggestion is not None
        assert "tensorrt" in suggestion.lower()

    def test_error_suggestion_for_engine_compatibility(self):
        """Task 52.3.5: CLI should warn about engine compatibility issues."""
        from pathlib import Path

        from haoline.cli_typer import _get_error_suggestion

        # Simulate TensorRT compatibility error
        error = RuntimeError("tensorrt deserialization failed")
        suggestion = _get_error_suggestion(
            error, model_path=Path("model.engine"), from_pytorch=False
        )

        # Should mention that engines are GPU-specific
        assert suggestion is not None
        assert "GPU-specific" in suggestion or "incompatible" in suggestion.lower()

    def test_error_suggestion_for_file_not_found(self):
        """CLI should provide helpful message for missing files."""
        from haoline.cli_typer import _get_error_suggestion

        error = FileNotFoundError("No such file or directory: 'model.onnx'")
        suggestion = _get_error_suggestion(error, model_path=None, from_pytorch=False)

        assert suggestion is not None
        assert "path" in suggestion.lower() or "exists" in suggestion.lower()
