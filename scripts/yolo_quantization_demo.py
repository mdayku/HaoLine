#!/usr/bin/env python
"""
YOLO Quantization Demo - Task 12.7

Train YOLOv8n on roof damage dataset, export to fp32/fp16/int8 ONNX,
validate each, and import results into HaoLine for comparison.

Usage:
    python scripts/yolo_quantization_demo.py

Requirements:
    pip install ultralytics
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Configuration
DATASET_PATH = Path(r"C:\Users\marcu\tiptop\data\combined_v3")
OUTPUT_DIR = Path("demo_outputs/yolo_quantization")
MODEL_SIZE = "n"  # n=nano, s=small, m=medium
EPOCHS = 20
IMGSZ = 640
BATCH_SIZE = 16  # Adjust based on GPU memory


def setup_data_yaml() -> Path:
    """Create a local data.yaml with correct paths."""
    data_yaml = OUTPUT_DIR / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)

    content = f"""# YOLO Data Configuration
path: {DATASET_PATH.as_posix()}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['damage']
"""
    data_yaml.write_text(content)
    print(f"Created data.yaml at {data_yaml}")
    return data_yaml


def train_model(data_yaml: Path) -> Path:
    """Train YOLOv8n for EPOCHS epochs."""
    from ultralytics import YOLO

    print(f"\n{'=' * 60}")
    print(f"Training YOLOv8{MODEL_SIZE} for {EPOCHS} epochs...")
    print(f"{'=' * 60}\n")

    model = YOLO(f"yolov8{MODEL_SIZE}.pt")

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        project=str(OUTPUT_DIR),
        name="train",
        exist_ok=True,
        verbose=True,
    )

    # Best weights path
    best_pt = OUTPUT_DIR / "train" / "weights" / "best.pt"
    print(f"\nTraining complete! Best weights: {best_pt}")
    return best_pt


def export_variants(model_pt: Path) -> dict[str, Path]:
    """Export model to fp32, fp16, and int8 ONNX."""
    from ultralytics import YOLO

    exports = {}
    model = YOLO(model_pt)

    print(f"\n{'=' * 60}")
    print("Exporting ONNX variants...")
    print(f"{'=' * 60}\n")

    # FP32 (default)
    print("Exporting FP32...")
    model.export(format="onnx", imgsz=IMGSZ, simplify=True)
    fp32_path = model_pt.with_suffix(".onnx")
    fp32_dest = OUTPUT_DIR / "models" / "yolov8n_fp32.onnx"
    fp32_dest.parent.mkdir(parents=True, exist_ok=True)
    fp32_path.rename(fp32_dest)
    exports["fp32"] = fp32_dest
    print(f"  -> {fp32_dest}")

    # FP16
    print("Exporting FP16...")
    model.export(format="onnx", imgsz=IMGSZ, simplify=True, half=True)
    fp16_path = model_pt.with_suffix(".onnx")
    fp16_dest = OUTPUT_DIR / "models" / "yolov8n_fp16.onnx"
    fp16_path.rename(fp16_dest)
    exports["fp16"] = fp16_dest
    print(f"  -> {fp16_dest}")

    # INT8 (requires calibration data)
    print("Exporting INT8 (with calibration)...")
    try:
        model.export(
            format="onnx",
            imgsz=IMGSZ,
            simplify=True,
            int8=True,
            data=str(OUTPUT_DIR / "data.yaml"),  # For calibration
        )
        int8_path = model_pt.with_suffix(".onnx")
        int8_dest = OUTPUT_DIR / "models" / "yolov8n_int8.onnx"
        int8_path.rename(int8_dest)
        exports["int8"] = int8_dest
        print(f"  -> {int8_dest}")
    except Exception as e:
        print(f"  INT8 export failed: {e}")
        print("  (INT8 ONNX requires onnxruntime-gpu with TensorRT)")

    return exports


def validate_model(model_path: Path, data_yaml: Path, precision: str) -> dict:
    """Run validation and return metrics."""
    from ultralytics import YOLO

    print(f"\nValidating {precision.upper()} model...")

    # For ONNX models, use the ONNX file directly
    if model_path.suffix == ".onnx":
        model = YOLO(model_path)
    else:
        model = YOLO(model_path)

    results = model.val(
        data=str(data_yaml),
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        verbose=False,
    )

    # Extract metrics
    metrics = {
        "model_id": f"yolov8{MODEL_SIZE}_{precision}",
        "precision": precision,
        "task_type": "detection",
        "dataset": "roof_damage_combined_v3",
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        },
        "speed": {
            "preprocess_ms": float(results.speed.get("preprocess", 0)),
            "inference_ms": float(results.speed.get("inference", 0)),
            "postprocess_ms": float(results.speed.get("postprocess", 0)),
        },
    }

    print(f"  mAP@50: {metrics['metrics']['mAP50']:.4f}")
    print(f"  mAP@50-95: {metrics['metrics']['mAP50-95']:.4f}")
    print(f"  Precision: {metrics['metrics']['precision']:.4f}")
    print(f"  Recall: {metrics['metrics']['recall']:.4f}")

    return metrics


def save_eval_results(metrics: dict, precision: str) -> Path:
    """Save eval results in HaoLine-compatible format."""
    eval_dir = OUTPUT_DIR / "eval_results"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Convert to HaoLine eval format
    haoline_format = {
        "model_id": metrics["model_id"],
        "task_type": "detection",
        "dataset": metrics["dataset"],
        "timestamp": metrics["timestamp"],
        "metrics": [
            {
                "name": "mAP@50",
                "value": metrics["metrics"]["mAP50"],
                "unit": "",
                "higher_is_better": True,
                "category": "accuracy",
            },
            {
                "name": "mAP@50-95",
                "value": metrics["metrics"]["mAP50-95"],
                "unit": "",
                "higher_is_better": True,
                "category": "accuracy",
            },
            {
                "name": "precision",
                "value": metrics["metrics"]["precision"],
                "unit": "",
                "higher_is_better": True,
                "category": "accuracy",
            },
            {
                "name": "recall",
                "value": metrics["metrics"]["recall"],
                "unit": "",
                "higher_is_better": True,
                "category": "accuracy",
            },
            {
                "name": "inference_ms",
                "value": metrics["speed"]["inference_ms"],
                "unit": "ms",
                "higher_is_better": False,
                "category": "speed",
            },
        ],
        "metadata": {
            "precision": precision,
            "imgsz": IMGSZ,
            "epochs": EPOCHS,
        },
    }

    output_path = eval_dir / f"eval_{precision}.json"
    output_path.write_text(json.dumps(haoline_format, indent=2))
    print(f"  Saved eval results: {output_path}")

    return output_path


def run_haoline_comparison(model_paths: dict[str, Path], eval_paths: dict[str, Path]):
    """Run HaoLine comparison on all variants."""
    print(f"\n{'=' * 60}")
    print("Running HaoLine Comparison...")
    print(f"{'=' * 60}\n")

    # Import and compare
    try:
        from haoline.eval import (
            DeploymentScenario,
            EvalResult,
            compare_models,
            create_combined_report,
        )

        reports = []

        for precision in ["fp32", "fp16", "int8"]:
            if precision not in model_paths:
                continue

            model_path = model_paths[precision]
            eval_path = eval_paths.get(precision)

            print(f"Analyzing {precision.upper()} variant...")

            # Load eval results
            eval_results = []
            if eval_path and eval_path.exists():
                with open(eval_path) as f:
                    eval_data = json.load(f)
                eval_results = [EvalResult.model_validate(eval_data)]

            # Create combined report
            report = create_combined_report(
                str(model_path),
                eval_results=eval_results,
                run_inspection=True,
            )
            reports.append(report)

        # Compare all variants
        if reports:
            scenario = DeploymentScenario.realtime_video(fps=30)
            table = compare_models(reports, scenario, title="YOLO Quantization Comparison")

            # Output
            comparison_dir = OUTPUT_DIR / "comparison"
            comparison_dir.mkdir(parents=True, exist_ok=True)

            # Save outputs
            table.save_json(comparison_dir / "comparison.json")
            table.save_csv(comparison_dir / "comparison.csv")
            (comparison_dir / "comparison.md").write_text(table.to_markdown())

            print("\nComparison Results:")
            print(table.to_console())

            print(f"\nOutputs saved to {comparison_dir}/")

    except ImportError as e:
        print(f"HaoLine import failed: {e}")
        print("Run: pip install -e .")


def main():
    """Main entry point."""
    print("=" * 60)
    print("YOLO Quantization Demo - HaoLine Task 12.7")
    print("=" * 60)
    print(f"\nDataset: {DATASET_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Model: YOLOv8{MODEL_SIZE}")
    print(f"Epochs: {EPOCHS}")

    # Check ultralytics
    try:
        import ultralytics

        print(f"Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Setup
    data_yaml = setup_data_yaml()

    # Train
    model_pt = train_model(data_yaml)

    # Export variants
    model_paths = export_variants(model_pt)

    # Validate each variant
    eval_paths = {}
    for precision, model_path in model_paths.items():
        try:
            metrics = validate_model(model_path, data_yaml, precision)
            eval_paths[precision] = save_eval_results(metrics, precision)
        except Exception as e:
            print(f"  Validation failed for {precision}: {e}")

    # Run HaoLine comparison
    run_haoline_comparison(model_paths, eval_paths)

    print(f"\n{'=' * 60}")
    print("Demo complete!")
    print(f"{'=' * 60}")
    print(f"\nOutputs in: {OUTPUT_DIR.absolute()}")
    print("\nNext steps:")
    print("  1. View comparison: cat demo_outputs/yolo_quantization/comparison/comparison.md")
    print("  2. Open in Streamlit: haoline-web")
    print("     Upload the ONNX models from demo_outputs/yolo_quantization/models/")


if __name__ == "__main__":
    main()
