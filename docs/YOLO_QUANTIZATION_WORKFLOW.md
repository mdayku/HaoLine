# YOLO Quantization Workflow

This guide walks through quantizing a YOLO model from FP32 → FP16 → INT8 and comparing all three variants using HaoLine.

---

## Overview

```
YOLO FP32 (.pt)
    ↓ Export
ONNX FP32 (.onnx)
    ↓ Quantize
┌────────────┬────────────┬────────────┐
│ ONNX FP32  │ ONNX FP16  │ ONNX INT8  │
└────────────┴────────────┴────────────┘
    ↓ Evaluate (batchtestv1.py or YOLO val)
┌────────────┬────────────┬────────────┐
│ Eval FP32  │ Eval FP16  │ Eval INT8  │
└────────────┴────────────┴────────────┘
    ↓ Import to HaoLine
┌─────────────────────────────────────────┐
│ Unified Comparison Report               │
│ - Accuracy (mAP, P/R/F1)               │
│ - Speed (latency, throughput)          │
│ - Cost ($/day for edge deployment)     │
└─────────────────────────────────────────┘
```

---

## Step 1: Export YOLO to ONNX (FP32)

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

# Export to ONNX (FP32 by default)
model.export(format="onnx", opset=17, simplify=True)
# Output: best.onnx
```

---

## Step 2: Create FP16 and INT8 Variants

### FP16 (Half Precision)

```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="onnx", opset=17, simplify=True, half=True)
# Output: best_fp16.onnx (rename manually)
```

Or convert existing ONNX:

```python
import onnx
from onnxconverter_common import float16

model = onnx.load("best.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "best_fp16.onnx")
```

### INT8 (Quantized)

Using ONNX Runtime quantization:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="best.onnx",
    model_output="best_int8.onnx",
    weight_type=QuantType.QInt8,
)
```

For better accuracy, use static quantization with calibration data:

```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader

class YOLOCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_images):
        self.images = calibration_images
        self.index = 0

    def get_next(self):
        if self.index >= len(self.images):
            return None
        # Preprocess image to model input format
        input_data = preprocess(self.images[self.index])
        self.index += 1
        return {"images": input_data}

calibration_reader = YOLOCalibrationDataReader(calibration_images)
quantize_static(
    model_input="best.onnx",
    model_output="best_int8_static.onnx",
    calibration_data_reader=calibration_reader,
)
```

---

## Step 3: Evaluate All Variants

### Option A: Using Ultralytics Val

```bash
# FP32
yolo val model=best.onnx data=coco.yaml > eval_fp32.txt

# FP16
yolo val model=best_fp16.onnx data=coco.yaml > eval_fp16.txt

# INT8
yolo val model=best_int8.onnx data=coco.yaml > eval_int8.txt
```

### Option B: Using Custom Batch Eval (batchtestv1.py)

If you have a custom evaluation script:

```bash
python batchtestv1.py
# Outputs: model_metrics_*.xlsx with P/R/F1 per model
```

---

## Step 4: Analyze with HaoLine

### Architecture Analysis

```bash
# Analyze each variant
haoline best.onnx --hardware auto --out-json analysis_fp32.json
haoline best_fp16.onnx --hardware auto --out-json analysis_fp16.json
haoline best_int8.onnx --hardware auto --out-json analysis_int8.json
```

### Import Eval Results (Coming Soon)

```bash
# Import eval results from Ultralytics
haoline-import-eval --from-ultralytics eval_fp32.json --model best.onnx
haoline-import-eval --from-ultralytics eval_fp16.json --model best_fp16.onnx
haoline-import-eval --from-ultralytics eval_int8.json --model best_int8.onnx
```

### Compare Models

```bash
haoline-compare \
    --models best.onnx best_fp16.onnx best_int8.onnx \
    --hardware auto \
    --out-html comparison.html
```

---

## Step 5: Deployment Cost Analysis

For edge deployment (e.g., 3 fps continuous on Jetson):

```bash
haoline best_int8.onnx \
    --hardware jetson-orin-nano \
    --deployment-fps 3 \
    --deployment-hours 24 \
    --out-html deployment_analysis.html
```

This will estimate:
- Required hardware tier
- Latency at target FPS
- Power consumption
- Monthly cost (for cloud) or power cost (for edge)

---

## Expected Results

| Variant | Size | mAP@50 | Latency | $/month (cloud) |
|---------|------|--------|---------|-----------------|
| FP32    | 45MB | 0.92   | 33ms    | $144            |
| FP16    | 22MB | 0.91   | 18ms    | $79             |
| INT8    | 11MB | 0.88   | 8ms     | $36             |

**Recommendation:** INT8 provides 75% cost savings with only 4% accuracy drop.

---

## Troubleshooting

### INT8 Quantization Fails

- Ensure your model doesn't have unsupported ops for INT8
- Use `haoline --lint-quantization` (coming soon) to check compatibility

### Accuracy Drop Too Large

- Use static quantization with representative calibration data
- Keep sensitive layers (final conv, classifier) at FP16

### ONNX Export Issues

- Try different opset versions (13-17)
- Use `simplify=True` to clean up the graph
- Check for dynamic axes issues

---

## See Also

- [HaoLine Compare Mode](../README.md#compare-model-variants)
- [Hardware Profiles](../README.md#hardware-options)
- [Deployment Guide](../DEPLOYMENT.md)

