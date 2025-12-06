#!/usr/bin/env python3
"""
Test format readers with real models.

This script downloads small test models and validates the format readers.

IMPORTANT: When adding a new format reader to src/haoline/formats/:
  1. Add a test_<format>() function to this file
  2. Add it to the TESTS dict in main()
  3. Don't create a separate test script!

Supported formats:
  - SafeTensors (.safetensors) - HuggingFace weights
  - GGUF (.gguf) - llama.cpp LLMs
  - TFLite (.tflite) - TensorFlow Lite mobile
  - CoreML (.mlmodel) - Apple devices
  - OpenVINO (.xml/.bin) - Intel inference
  - [Add new formats here]
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_safetensors():
    """Test SafeTensors reader with a real HuggingFace model."""
    print("\n" + "=" * 60)
    print("Testing SafeTensors Reader")
    print("=" * 60)

    from huggingface_hub import hf_hub_download

    # Download a tiny safetensors file (sentence-transformers model weights)
    # Using a very small model: ~17MB
    print("\nDownloading small SafeTensors model...")
    try:
        model_path = hf_hub_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            filename="model.safetensors",
            cache_dir="./test_models/cache",
        )
        print(f"Downloaded: {model_path}")
    except Exception as e:
        print(f"Failed to download: {e}")
        return False

    # Test the reader
    from haoline.formats.safetensors import SafeTensorsReader, is_safetensors_file

    print(f"\nIs SafeTensors file: {is_safetensors_file(model_path)}")

    reader = SafeTensorsReader(model_path)

    # Test header-only read (fast)
    print("\n--- Header-only read ---")
    info = reader.read_header_only()
    print(f"Total params: {info.total_params:,}")
    print(f"Total size: {info.total_size_bytes / 1e6:.2f} MB")
    print(f"Num tensors: {len(info.tensors)}")
    print(f"Dtype breakdown: {info.dtype_breakdown}")

    # Test full read
    print("\n--- Full read ---")
    info = reader.read()
    print(f"Total params: {info.total_params:,}")
    print(f"Metadata: {info.metadata}")

    # Show first 5 tensors
    print("\nFirst 5 tensors:")
    for t in info.tensors[:5]:
        print(f"  {t.name}: {t.shape} ({t.dtype})")

    print("\n[SUCCESS] SafeTensors reader works correctly!")
    return True


def test_gguf():
    """Test GGUF reader with a real llama.cpp model."""
    print("\n" + "=" * 60)
    print("Testing GGUF Reader")
    print("=" * 60)

    from huggingface_hub import hf_hub_download

    # Download a tiny GGUF model (~43MB)
    print("\nDownloading small GGUF model (TinyLlama Q4_K_M)...")
    try:
        model_path = hf_hub_download(
            repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            filename="tinyllama-1.1b-chat-v1.0.Q2_K.gguf",  # Smallest quant ~400MB
            cache_dir="./test_models/cache",
        )
        print(f"Downloaded: {model_path}")
    except Exception as e:
        print(f"Failed to download (this is a larger file, may take time): {e}")
        # Try an even smaller model
        print("\nTrying alternative: smollm 135M...")
        try:
            model_path = hf_hub_download(
                repo_id="MaziyarPanahi/smollm-135M-instruct-GGUF",
                filename="smollm-135M-instruct.Q4_K_M.gguf",
                cache_dir="./test_models/cache",
            )
            print(f"Downloaded: {model_path}")
        except Exception as e2:
            print(f"Failed to download alternative: {e2}")
            return False

    # Test the reader
    from haoline.formats.gguf import GGUFReader, format_size, is_gguf_file

    print(f"\nIs GGUF file: {is_gguf_file(model_path)}")

    reader = GGUFReader(model_path)
    info = reader.read()

    print("\n--- GGUF Model Info ---")
    print(f"Version: {info.version}")
    print(f"Architecture: {info.architecture}")
    print(f"Model name: {info.model_name}")
    print(f"Total params: {info.total_params:,}")
    print(f"Total size: {format_size(info.total_size_bytes)}")
    print(f"Tensor count: {info.tensor_count}")

    # Architecture details
    print("\n--- Architecture Details ---")
    print(f"Context length: {info.context_length}")
    print(f"Embedding dim: {info.embedding_length}")
    print(f"Layers: {info.block_count}")
    print(f"Attention heads: {info.head_count}")
    print(f"KV heads: {info.head_count_kv}")

    # Quantization breakdown
    print("\n--- Quantization Breakdown ---")
    for qtype, count in sorted(info.quantization_breakdown.items()):
        print(f"  {qtype}: {count} tensors")

    # VRAM estimate
    print("\n--- VRAM Estimate (2K context) ---")
    vram = info.estimate_vram(2048)
    print(f"  Weights: {format_size(vram['weights'])}")
    print(f"  KV Cache: {format_size(vram['kv_cache'])}")
    print(f"  Total: {format_size(vram['total'])}")

    print("\n[SUCCESS] GGUF reader works correctly!")
    return True


def test_tflite():
    """Test TFLite reader (requires tflite-runtime or tensorflow for full parsing)."""
    print("\n" + "=" * 60)
    print("Testing TFLite Reader")
    print("=" * 60)

    from haoline.formats.tflite import is_available

    # Check if we have the runtime
    if not is_available():
        print("[SKIP] tflite-runtime not installed (not available on Windows)")
        print("The pure Python fallback only validates file format, doesn't extract tensors.")
        print("Full parsing requires: pip install tensorflow (or tflite-runtime on Linux/Mac)")
        return None

    import urllib.request
    from pathlib import Path

    # Download MobileNetV2 TFLite from TensorFlow Hub
    model_dir = Path("./test_models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "mobilenet_v2.tflite"

    if not model_path.exists():
        print("\nDownloading TFLite model from TensorFlow Hub...")
        # Try multiple sources in order of preference
        urls = [
            # TF Hub hosted model (more reliable)
            "https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224/1/default/1?lite-format=tflite",
            # Kaggle mirror
            "https://storage.googleapis.com/tfhub-lite-models/tensorflow/lite-model/mobilenet_v2_1.0_224/1/default/1.tflite",
        ]
        downloaded = False
        for url in urls:
            try:
                print(f"  Trying: {url[:60]}...")
                req = urllib.request.Request(url, headers={"User-Agent": "HaoLine/1.0"})
                with urllib.request.urlopen(req, timeout=30) as response:
                    with open(model_path, "wb") as f:
                        f.write(response.read())
                print(f"Downloaded: {model_path}")
                downloaded = True
                break
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        if not downloaded:
            print("[SKIP] Could not download TFLite model from any source")
            return None
    else:
        print(f"\nUsing cached: {model_path}")

    from haoline.formats.tflite import TFLiteReader, is_tflite_file

    print(f"\nIs TFLite file: {is_tflite_file(model_path)}")

    reader = TFLiteReader(model_path)
    info = reader.read()

    print("\n--- TFLite Model Info ---")
    print(f"Description: {info.description}")
    print(f"Total params: {info.total_params:,}")
    print(f"Total size: {info.total_size_bytes / 1e6:.2f} MB")
    print(f"Num tensors: {len(info.tensors)}")
    print(f"Op types: {info.op_type_counts}")

    print("\n[SUCCESS] TFLite reader works correctly!")
    return True


def test_coreml():
    """Test CoreML reader (coremltools works on Linux for parsing)."""
    print("\n" + "=" * 60)
    print("Testing CoreML Reader")
    print("=" * 60)

    from haoline.formats.coreml import is_available

    if not is_available():
        print("[SKIP] coremltools not installed")
        print("Install with: pip install coremltools")
        return None

    from pathlib import Path

    import coremltools as ct
    import numpy as np

    # Create a simple CoreML model for testing (no download needed)
    print("\nCreating test CoreML model...")
    model_dir = Path("./test_models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "test_model.mlmodel"

    if not model_path.exists():
        try:
            # Create a simple MLP model using coremltools
            from coremltools.models.neural_network import NeuralNetworkBuilder

            input_features = [("input", ct.models.datatypes.Array(10))]
            output_features = [("output", ct.models.datatypes.Array(5))]

            builder = NeuralNetworkBuilder(input_features, output_features)
            builder.add_inner_product(
                name="fc1",
                input_name="input",
                output_name="fc1_out",
                input_channels=10,
                output_channels=20,
                W=np.random.randn(10, 20).astype(np.float32),
                b=np.zeros(20, dtype=np.float32),
                has_bias=True,
            )
            builder.add_activation(
                name="relu1", non_linearity="RELU", input_name="fc1_out", output_name="relu1_out"
            )
            builder.add_inner_product(
                name="fc2",
                input_name="relu1_out",
                output_name="output",
                input_channels=20,
                output_channels=5,
                W=np.random.randn(20, 5).astype(np.float32),
                b=np.zeros(5, dtype=np.float32),
                has_bias=True,
            )

            spec = builder.spec
            model = ct.models.MLModel(spec)
            model.save(str(model_path))
            print(f"Created: {model_path}")
        except Exception as e:
            print(f"Failed to create CoreML model: {e}")
            print("[SKIP] Could not create CoreML test model")
            return None
    else:
        print(f"Using cached: {model_path}")

    from haoline.formats.coreml import CoreMLReader, is_coreml_file

    print(f"\nIs CoreML file: {is_coreml_file(model_path)}")

    try:
        reader = CoreMLReader(model_path)
        info = reader.read()

        print("\n--- CoreML Model Info ---")
        print(f"Spec version: {info.spec_version}")
        print(f"Description: {info.description}")
        print(f"Model type: {info.model_type}")
        print(f"Layer count: {info.layer_count}")
        print(f"Layer types: {info.layer_type_counts}")

        print("\n[SUCCESS] CoreML reader works correctly!")
        return True
    except Exception as e:
        print(f"Error reading model: {e}")
        return False


def test_openvino():
    """Test OpenVINO reader."""
    print("\n" + "=" * 60)
    print("Testing OpenVINO Reader")
    print("=" * 60)

    from haoline.formats.openvino import is_available

    if not is_available():
        print("[SKIP] openvino not installed")
        print("Install with: pip install openvino")
        return None

    print("[INFO] OpenVINO test requires .xml + .bin files")
    return None


def test_tensorrt():
    """Test TensorRT engine reader."""
    print("\n" + "=" * 60)
    print("Testing TensorRT Reader")
    print("=" * 60)

    from haoline.formats.tensorrt import is_available

    if not is_available():
        print("[SKIP] tensorrt not installed")
        print("Install with: pip install haoline[tensorrt]")
        print("Requires: NVIDIA GPU + CUDA 12.x")
        return None

    from pathlib import Path

    # Check for existing engine or build one
    engine_path = Path("test_models/resnet18.engine")

    if not engine_path.exists():
        print("Building TensorRT engine from ONNX (this may take a minute)...")
        onnx_path = Path("test_models/resnet18.onnx")

        # Download ONNX if needed
        if not onnx_path.exists():
            import urllib.request

            print("Downloading ResNet18 ONNX...")
            url = "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx"
            onnx_path.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(url, onnx_path)

        # Build TRT engine
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print("Failed to parse ONNX")
                return False

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        # Add optimization profile for dynamic batch
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        profile.set_shape(input_name, (1, 3, 224, 224), (1, 3, 224, 224), (8, 3, 224, 224))
        config.add_optimization_profile(profile)

        # Enable FP16 if available
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        serialized = builder.build_serialized_network(network, config)
        if serialized:
            with open(engine_path, "wb") as f:
                f.write(bytes(serialized))
            print(f"Built: {engine_path}")
        else:
            print("Failed to build engine")
            return False
    else:
        print(f"Using cached: {engine_path}")

    # Test the reader
    from haoline.formats.tensorrt import TRTEngineReader, format_bytes, is_tensorrt_file

    print(f"\nIs TensorRT file: {is_tensorrt_file(engine_path)}")

    reader = TRTEngineReader(engine_path)
    info = reader.read()

    print("\n--- TensorRT Engine Info ---")
    print(f"TRT Version: {info.trt_version}")
    print(f"Device: {info.device_name}")
    print(f"Compute Capability: SM {info.compute_capability[0]}.{info.compute_capability[1]}")
    print(f"Device Memory: {format_bytes(info.device_memory_bytes)}")
    print(f"Layer Count: {info.layer_count}")

    print("\n--- Bindings ---")
    for b in info.input_bindings:
        print(f"  Input: {b.name} {b.shape} ({b.dtype})")
    for b in info.output_bindings:
        print(f"  Output: {b.name} {b.shape} ({b.dtype})")

    print("\n--- Layer Type Distribution ---")
    for ltype, count in sorted(info.layer_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ltype}: {count}")

    # Count fusions
    fused = len([layer for layer in info.layers if "+" in layer.name])
    print("\n--- Optimization Summary ---")
    print(f"  Fused layers: {fused}/{info.layer_count} ({100 * fused / info.layer_count:.0f}%)")

    print("\n[SUCCESS] TensorRT reader works correctly!")
    return True


def main():
    """Run all format reader tests.

    To add a new format:
      1. Create test_<format>() function above
      2. Add entry to TESTS dict below
    """
    print("=" * 60)
    print("HaoLine Format Reader Tests")
    print("=" * 60)

    # Register all format tests here
    # Format: "DisplayName": test_function
    TESTS = {
        "SafeTensors": test_safetensors,
        "GGUF": test_gguf,
        "TFLite": test_tflite,
        "CoreML": test_coreml,
        "OpenVINO": test_openvino,
        "TensorRT": test_tensorrt,
        # Add new formats here:
        # "NewFormat": test_newformat,
    }

    results = {}
    for name, test_fn in TESTS.items():
        results[name] = test_fn()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, result in results.items():
        if result is True:
            status = "[PASS]"
        elif result is False:
            status = "[FAIL]"
        else:
            status = "[SKIP]"
        print(f"  {status} {name}")

    return all(r is not False for r in results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
