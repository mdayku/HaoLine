#!/usr/bin/env python
"""
YOLO Quantization Demo - Task 12.7

Train YOLOv8n on roof damage dataset, export to fp32 ONNX,
then properly quantize to fp16 and int8 using ONNX Runtime.

Usage:
    python scripts/yolo_quantization_demo.py

Requirements:
    pip install ultralytics onnx onnxruntime
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

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


def export_fp32(model_pt: Path) -> Path:
    """Export model to FP32 ONNX."""
    from ultralytics import YOLO

    print("\nExporting FP32 ONNX...")
    model = YOLO(model_pt)
    model.export(format="onnx", imgsz=IMGSZ, simplify=True)

    # Move to models dir
    src = model_pt.with_suffix(".onnx")
    dest = OUTPUT_DIR / "models" / "yolov8n_fp32.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        dest.unlink()
    src.rename(dest)

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  -> {dest} ({size_mb:.2f} MB)")
    return dest


def quantize_to_fp16(fp32_path: Path) -> Path:
    """Convert FP32 ONNX to FP16 using onnx library."""
    import onnx
    from onnx import numpy_helper

    print("\nQuantizing to FP16...")

    model = onnx.load(str(fp32_path))

    # Convert initializers (weights) to float16
    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT:
            # Convert to numpy, then to float16
            arr = numpy_helper.to_array(initializer)
            arr_fp16 = arr.astype(np.float16)
            new_init = numpy_helper.from_array(arr_fp16, initializer.name)
            initializer.CopyFrom(new_init)

    # Note: We keep IO as float32 for compatibility
    # The weights are now float16, inference will use mixed precision

    fp16_path = OUTPUT_DIR / "models" / "yolov8n_fp16.onnx"
    onnx.save(model, str(fp16_path))

    size_mb = fp16_path.stat().st_size / (1024 * 1024)
    print(f"  -> {fp16_path} ({size_mb:.2f} MB)")
    return fp16_path


def quantize_to_int8(fp32_path: Path, data_yaml: Path) -> Path | None:
    """Quantize FP32 ONNX to INT8 using ONNX Runtime dynamic quantization."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        print("\nQuantizing to INT8 (dynamic quantization)...")

        int8_path = OUTPUT_DIR / "models" / "yolov8n_int8.onnx"

        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
        )

        size_mb = int8_path.stat().st_size / (1024 * 1024)
        print(f"  -> {int8_path} ({size_mb:.2f} MB)")
        return int8_path

    except ImportError:
        print("  INT8 quantization requires: pip install onnxruntime")
        return None
    except Exception as e:
        print(f"  INT8 quantization failed: {e}")
        return None


def validate_model(model_path: Path, data_yaml: Path, precision: str) -> dict:
    """Run validation on TEST set and return metrics."""
    from ultralytics import YOLO

    print(f"\nValidating {precision.upper()} model on TEST set...")

    model = YOLO(model_path)

    # Run on TEST split (not val)
    results = model.val(
        data=str(data_yaml),
        split="test",  # Use test set for final evaluation
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
        "model_size_mb": model_path.stat().st_size / (1024 * 1024),
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

    print(f"  Size: {metrics['model_size_mb']:.2f} MB")
    print(f"  mAP@50: {metrics['metrics']['mAP50']:.4f}")
    print(f"  mAP@50-95: {metrics['metrics']['mAP50-95']:.4f}")
    print(f"  Precision: {metrics['metrics']['precision']:.4f}")
    print(f"  Recall: {metrics['metrics']['recall']:.4f}")
    print(f"  Inference: {metrics['speed']['inference_ms']:.1f} ms")

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
            {
                "name": "model_size_mb",
                "value": metrics["model_size_mb"],
                "unit": "MB",
                "higher_is_better": False,
                "category": "efficiency",
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


def print_comparison_table(all_metrics: list[dict]):
    """Print a nice comparison table."""
    print(f"\n{'=' * 80}")
    print("QUANTIZATION COMPARISON RESULTS")
    print(f"{'=' * 80}")
    print(
        f"{'Precision':<10} {'Size (MB)':<12} {'mAP@50':<10} {'mAP@50-95':<12} "
        f"{'Precision':<12} {'Recall':<10} {'Inference':<12}"
    )
    print("-" * 80)

    fp32_size = None
    for m in all_metrics:
        if m["precision"] == "fp32":
            fp32_size = m["model_size_mb"]
            break

    for m in all_metrics:
        size = m["model_size_mb"]
        size_str = f"{size:.2f}"
        if fp32_size and m["precision"] != "fp32":
            reduction = (1 - size / fp32_size) * 100
            size_str += f" (-{reduction:.0f}%)"

        print(
            f"{m['precision'].upper():<10} {size_str:<12} "
            f"{m['metrics']['mAP50']:.4f}     {m['metrics']['mAP50-95']:.4f}       "
            f"{m['metrics']['precision']:.4f}       {m['metrics']['recall']:.4f}     "
            f"{m['speed']['inference_ms']:.1f} ms"
        )

    print(f"{'=' * 80}")


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

    # Check if we already have a trained model
    best_pt = OUTPUT_DIR / "train" / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\nFound existing trained model: {best_pt}")
        print("Skipping training, using existing weights...")
    else:
        # Setup and train
        data_yaml = setup_data_yaml()
        best_pt = train_model(data_yaml)

    # Setup data yaml (needed for validation)
    data_yaml = setup_data_yaml()

    print(f"\n{'=' * 60}")
    print("STEP 1: Export and Quantize Models")
    print(f"{'=' * 60}")

    # Export FP32
    fp32_path = export_fp32(best_pt)

    # Quantize to FP16
    fp16_path = quantize_to_fp16(fp32_path)

    # Quantize to INT8
    int8_path = quantize_to_int8(fp32_path, data_yaml)

    model_paths = {"fp32": fp32_path, "fp16": fp16_path}
    if int8_path:
        model_paths["int8"] = int8_path

    print(f"\n{'=' * 60}")
    print("STEP 2: Validate All Variants on TEST Set")
    print(f"{'=' * 60}")

    # Validate each variant
    all_metrics = []
    eval_paths = {}

    for precision, model_path in model_paths.items():
        try:
            metrics = validate_model(model_path, data_yaml, precision)
            all_metrics.append(metrics)
            eval_paths[precision] = save_eval_results(metrics, precision)
        except Exception as e:
            print(f"  Validation failed for {precision}: {e}")

    # Print comparison
    if all_metrics:
        print_comparison_table(all_metrics)

    # Run HaoLine comparison
    run_haoline_comparison(model_paths, eval_paths)

    print(f"\n{'=' * 60}")
    print("Demo complete!")
    print(f"{'=' * 60}")
    print(f"\nOutputs in: {OUTPUT_DIR.absolute()}")
    print("\nModel files:")
    for precision, path in model_paths.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {precision.upper()}: {path} ({size_mb:.2f} MB)")
    print("\nNext steps:")
    print("  1. View comparison: type demo_outputs\\yolo_quantization\\comparison\\comparison.md")
    print("  2. Open in Streamlit: haoline-web")
    print("     Upload the ONNX models from demo_outputs/yolo_quantization/models/")


if __name__ == "__main__":
    main()
