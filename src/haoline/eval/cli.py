"""
HaoLine Eval Import CLI.

Import evaluation results from external tools and combine with architecture analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .schemas import (
    CombinedReport,
    EvalResult,
    create_combined_report,
    link_eval_to_model,
    validate_eval_result,
)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for import-eval command."""
    parser = argparse.ArgumentParser(
        prog="haoline-import-eval",
        description="Import evaluation results from external tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import Ultralytics YOLO validation results
  haoline-import-eval --from-ultralytics results.json --model yolo.onnx

  # Import HuggingFace evaluate results (classification)
  haoline-import-eval --from-hf-evaluate eval_results.json --task classification

  # Import HuggingFace evaluate results (NLP task like QA)
  haoline-import-eval --from-hf-evaluate qa_results.json --task nlp

  # Import lm-eval-harness LLM benchmark results
  haoline-import-eval --from-lm-eval lm_eval_output.json --model llama.onnx

  # Import timm benchmark results
  haoline-import-eval --from-timm validate_results.json --model resnet50.onnx

  # Auto-detect format and import
  haoline-import-eval --auto results.json --out-json standardized.json

  # Import generic CSV with custom column mapping
  haoline-import-eval --from-csv results.csv --model model.onnx \\
      --map-column accuracy=top1_acc --map-column f1=f1_score

  # Validate an eval results file
  haoline-import-eval --validate results.json

  # Combine eval results with model architecture analysis
  haoline-import-eval --from-ultralytics results.json --model yolo.onnx --combine

  # Combine and output to file
  haoline-import-eval --auto results.json --model model.onnx --combine --out-json combined.json
""",
    )

    # Input sources
    input_group = parser.add_argument_group("Input Sources")
    input_group.add_argument(
        "--from-ultralytics",
        type=Path,
        metavar="PATH",
        help="Import from Ultralytics YOLO validation output (JSON).",
    )
    input_group.add_argument(
        "--from-hf-evaluate",
        type=Path,
        metavar="PATH",
        help="Import from HuggingFace evaluate output (JSON).",
    )
    input_group.add_argument(
        "--from-lm-eval",
        type=Path,
        metavar="PATH",
        help="Import from lm-eval-harness output (JSON).",
    )
    input_group.add_argument(
        "--from-csv",
        type=Path,
        metavar="PATH",
        help="Import from generic CSV file.",
    )
    input_group.add_argument(
        "--from-timm",
        type=Path,
        metavar="PATH",
        help="Import from timm benchmark output (JSON).",
    )
    input_group.add_argument(
        "--from-json",
        type=Path,
        metavar="PATH",
        help="Import from generic JSON file (auto-detected or schema-compliant).",
    )
    input_group.add_argument(
        "--auto",
        type=Path,
        metavar="PATH",
        help="Auto-detect format and import (tries all adapters).",
    )

    # Model linking
    link_group = parser.add_argument_group("Model Linking")
    link_group.add_argument(
        "--model",
        type=Path,
        metavar="PATH",
        help="Path to the model file to link eval results to.",
    )
    link_group.add_argument(
        "--combine",
        action="store_true",
        help="Combine eval results with model architecture analysis (requires --model).",
    )
    link_group.add_argument(
        "--use-hash",
        action="store_true",
        help="Use file hash instead of filename as model identifier.",
    )

    # Task type
    parser.add_argument(
        "--task",
        choices=["detection", "classification", "nlp", "llm", "segmentation"],
        default=None,
        help="Override task type (auto-detected from adapter if not specified).",
    )

    # Output
    parser.add_argument(
        "--out-json",
        type=Path,
        metavar="PATH",
        help="Output path for standardized eval results JSON.",
    )

    # CSV column mapping
    parser.add_argument(
        "--map-column",
        action="append",
        metavar="METRIC=COLUMN",
        help="Map a metric name to a CSV column (can be repeated).",
    )

    # Validation
    parser.add_argument(
        "--validate",
        type=Path,
        metavar="PATH",
        help="Validate an eval results file against the schema.",
    )

    # Verbosity
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output except errors.",
    )

    return parser


def import_from_ultralytics(path: Path) -> EvalResult | None:
    """Import eval results from Ultralytics YOLO validation output."""
    try:
        from .adapters import load_ultralytics_json

        return load_ultralytics_json(path)
    except Exception as e:
        print(f"Error parsing Ultralytics results from {path}: {e}")
        return None


def import_from_hf_evaluate(path: Path, task_type: str = "classification") -> EvalResult | None:
    """Import eval results from HuggingFace evaluate output."""
    try:
        from .adapters import load_hf_evaluate

        return load_hf_evaluate(path, task_type=task_type)
    except Exception as e:
        print(f"Error parsing HuggingFace evaluate results from {path}: {e}")
        return None


def import_from_lm_eval(path: Path) -> EvalResult | None:
    """Import eval results from lm-eval-harness output."""
    try:
        from .adapters import load_lm_eval

        return load_lm_eval(path)
    except Exception as e:
        print(f"Error parsing lm-eval-harness results from {path}: {e}")
        return None


def import_from_json(path: Path) -> EvalResult | None:
    """Import eval results from generic JSON (auto-detect or schema-compliant)."""
    try:
        from .adapters import detect_and_parse

        # Try auto-detect first
        result = detect_and_parse(path)
        if result:
            return result

        # Fall back to schema-compliant parsing
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not validate_eval_result(data):
            print(f"Error: Invalid eval result schema in {path}")
            return None

        validated: EvalResult = EvalResult.model_validate(data)
        return validated
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def validate_file(path: Path) -> bool:
    """Validate an eval results file against the schema."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if validate_eval_result(data):
            print(f"Valid eval result: {path}")
            print(f"  model_id: {data.get('model_id')}")
            print(f"  task_type: {data.get('task_type')}")
            print(f"  metrics: {len(data.get('metrics', []))} metrics")
            return True
        else:
            print(f"Invalid eval result: {path}")
            print("  Missing required fields: model_id and/or task_type")
            return False
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {path}: {e}")
        return False
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return False


def main() -> int:
    """Main entry point for haoline-import-eval command."""
    parser = create_parser()
    args = parser.parse_args()

    # Validation mode
    if args.validate:
        return 0 if validate_file(args.validate) else 1

    # Check for input source
    input_sources = [
        args.from_ultralytics,
        args.from_hf_evaluate,
        args.from_lm_eval,
        args.from_timm,
        args.from_csv,
        args.from_json,
        args.auto,
    ]
    active_sources = [s for s in input_sources if s is not None]

    if len(active_sources) == 0:
        parser.print_help()
        print("\nError: No input source specified.")
        return 1

    if len(active_sources) > 1:
        print("Error: Only one input source can be specified at a time.")
        return 1

    # Import based on source
    result: EvalResult | None = None

    if args.from_ultralytics:
        result = import_from_ultralytics(args.from_ultralytics)
    elif args.from_hf_evaluate:
        task = args.task if args.task in ("classification", "nlp") else "classification"
        result = import_from_hf_evaluate(args.from_hf_evaluate, task_type=task)
    elif args.from_lm_eval:
        result = import_from_lm_eval(args.from_lm_eval)
    elif args.from_timm:
        try:
            from .adapters import load_timm_benchmark

            result = load_timm_benchmark(args.from_timm)
        except Exception as e:
            print(f"Error parsing timm results: {e}")
            return 1
    elif args.auto:
        result = import_from_json(args.auto)  # Uses detect_and_parse
    elif args.from_json:
        result = import_from_json(args.from_json)
    elif args.from_csv:
        try:
            from .adapters import load_generic_csv

            # Parse column mappings if provided
            column_mapping: dict[str, str] = {}
            if args.map_column:
                for mapping in args.map_column:
                    if "=" in mapping:
                        metric, column = mapping.split("=", 1)
                        column_mapping[column] = metric

            results = load_generic_csv(args.from_csv)
            if results:
                result = results[0]  # Return first row for single result
                if not args.quiet and len(results) > 1:
                    print(f"Note: CSV contains {len(results)} rows, returning first.")
            else:
                print(f"No valid rows found in {args.from_csv}")
                return 1
        except Exception as e:
            print(f"Error parsing CSV: {e}")
            return 1

    if result is None:
        print("Failed to import eval results.")
        return 1

    # Link to model if specified
    if args.model:
        result = link_eval_to_model(
            str(args.model),
            result,
            use_hash=args.use_hash,
        )
        if not args.quiet:
            print(f"Linked eval to model: {result.model_id}")

    # Combine with architecture analysis if requested
    output_data: EvalResult | CombinedReport = result
    if args.combine:
        if not args.model:
            print("Error: --combine requires --model to be specified.")
            return 1

        if not args.model.exists():
            print(f"Error: Model file not found: {args.model}")
            return 1

        if not args.quiet:
            print(f"Running architecture analysis on {args.model}...")

        combined = create_combined_report(
            str(args.model),
            eval_results=[result],
            run_inspection=True,
        )

        if not args.quiet:
            arch = combined.architecture
            print(f"  Parameters: {arch.get('params_total', 0):,}")
            print(f"  FLOPs: {arch.get('flops_total', 0):,}")
            print(f"  Memory: {arch.get('model_size_bytes', 0) / 1024 / 1024:.1f} MB")
            if combined.latency_ms > 0:
                print(f"  Latency: {combined.latency_ms:.2f} ms")
            if combined.throughput_fps > 0:
                print(f"  Throughput: {combined.throughput_fps:.1f} fps")

        output_data = combined

    # Output
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(output_data.to_json(), encoding="utf-8")
        if not args.quiet:
            print(
                f"{'Combined report' if args.combine else 'Eval results'} written to: {args.out_json}"
            )

    if not args.quiet and not args.out_json:
        print(output_data.to_json())

    return 0


if __name__ == "__main__":
    sys.exit(main())
