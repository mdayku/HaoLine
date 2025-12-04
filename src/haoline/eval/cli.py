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
    EvalResult,
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

  # Import HuggingFace evaluate results
  haoline-import-eval --from-hf-evaluate eval_results.json --model bert.onnx

  # Import generic CSV with custom column mapping
  haoline-import-eval --from-csv results.csv --model model.onnx \\
      --map-column accuracy=top1_acc --map-column f1=f1_score

  # Validate an eval results file
  haoline-import-eval --validate results.json
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
        "--from-json",
        type=Path,
        metavar="PATH",
        help="Import from generic JSON file (must match HaoLine schema).",
    )

    # Model linking
    parser.add_argument(
        "--model",
        type=Path,
        metavar="PATH",
        help="Path to the model file to link eval results to.",
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
    # TODO: Implement Ultralytics adapter (Task 12.3.1)
    print(f"[TODO] Ultralytics adapter not yet implemented: {path}")
    print("This will parse YOLO val results and return DetectionEvalResult.")
    return None


def import_from_hf_evaluate(path: Path) -> EvalResult | None:
    """Import eval results from HuggingFace evaluate output."""
    # TODO: Implement HF evaluate adapter (Task 12.3.2)
    print(f"[TODO] HuggingFace evaluate adapter not yet implemented: {path}")
    return None


def import_from_lm_eval(path: Path) -> EvalResult | None:
    """Import eval results from lm-eval-harness output."""
    # TODO: Implement lm-eval adapter (Task 12.3.3)
    print(f"[TODO] lm-eval-harness adapter not yet implemented: {path}")
    return None


def import_from_json(path: Path) -> EvalResult | None:
    """Import eval results from generic JSON (must match schema)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not validate_eval_result(data):
            print(f"Error: Invalid eval result schema in {path}")
            return None

        return EvalResult.from_dict(data)
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
        args.from_csv,
        args.from_json,
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
        result = import_from_hf_evaluate(args.from_hf_evaluate)
    elif args.from_lm_eval:
        result = import_from_lm_eval(args.from_lm_eval)
    elif args.from_json:
        result = import_from_json(args.from_json)
    elif args.from_csv:
        # TODO: Implement CSV adapter (Task 12.3.5)
        print(f"[TODO] CSV adapter not yet implemented: {args.from_csv}")
        return 1

    if result is None:
        print("Failed to import eval results.")
        return 1

    # Output
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(result.to_json(), encoding="utf-8")
        if not args.quiet:
            print(f"Eval results written to: {args.out_json}")

    if not args.quiet and not args.out_json:
        print(result.to_json())

    return 0


if __name__ == "__main__":
    sys.exit(main())
