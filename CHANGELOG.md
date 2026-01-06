# Changelog

All notable changes to HaoLine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-01-06

### GGUF LLM Details Tab (Epic 24.2)

New "LLM Details" tab for GGUF models in Streamlit UI with comprehensive LLM analysis:

- **Architecture display**: Model name, architecture, layers, hidden size, attention heads, KV heads (GQA), vocab size, context length
- **Quantization breakdown**: Bar charts showing tensor count and size by quantization type (Q4_K_M, Q5_K_S, etc.)
- **VRAM calculator**: Interactive context length slider with dynamic VRAM estimation (weights + KV cache)
- **GPU compatibility table**: Shows which GPUs fit the model with headroom
- **Tensor explorer**: Searchable table of all tensors with name, shape, type, and size; CSV export

### Export Enhancements

All export formats now include GGUF-specific details when analyzing LLM models:

- **Markdown**: LLM Model Details section with architecture table, VRAM requirements, quantization breakdown
- **HTML**: Styled LLM Details section with metric cards and tables
- **JSON**: Full `gguf_info` object with all metadata, tensors, and quantization breakdown
- **Version tracking**: Report `autodoc_version` now dynamically references package version

---

## [1.1.1] - 2025-12-29

### Python Version Check

- Added early Python version check (3.10-3.12) with friendly error message
- Clear instructions for creating compatible conda environment
- Updated README Quick Start with version requirement
- Fixed `LayerRiskScore.name` and `QuantWarning.message` attribute errors in Streamlit

---

## [1.1.0] - 2025-12-29

### CLI Parity Complete

All 42 legacy CLI flags have been ported to the new Typer CLI for 100% feature parity.

### Added
- **Global list commands**: `--list-cloud` (show cloud instances), `--list-conversions` (show format conversion matrix)
- **Conversion flags**: `--from-frozen-graph`, `--tf-inputs`, `--tf-outputs`, `--jax-apply-fn`, `--pytorch-weights`
- **Deployment flags**: `--deployment-target`, `--target-latency-ms`, `--target-throughput-fps`
- **Profiling flags**: `--no-gpu-metrics`, `--no-bottleneck-analysis`
- **Quantization flags**: `--quant-bottlenecks`, `--quant-advice-report`

### Fixed
- `--list-cloud` now correctly iterates over cloud instance dictionary

---

## [1.0.0] - 2025-12-29

### 🎉 First Stable Release

HaoLine 1.0 is the universal model inspector that helps ML teams make informed deployment decisions.

### Added
- **10 Format Support**: ONNX, PyTorch, TensorFlow, TensorRT, CoreML, TFLite, OpenVINO, GGUF, SafeTensors
- **CI/CD Integration**: `--fail-on` thresholds for automated model validation in pipelines
- **Decision Reports**: `--decision-report` for audit trails (JSON/Markdown)
- **GitHub Actions Template**: Ready-to-use workflow for PR model checks
- **Streamlit Demo Model Parity**: Demo models now use identical code path as uploads
- **uv Installation Docs**: Added `uv tool install haoline` and `uvx` examples

### Changed
- **Slimmed `[full]` extra**: Now includes practical defaults (TF, PDF, SafeTensors, PyTorch/YOLO) without exotic formats
- **New `[all]` extra**: For power users who need JAX, CoreML, OpenVINO converters
- **README examples updated**: Fixed `compare` command syntax, model download URLs

### Fixed
- `RiskSignal.message` → `RiskSignal.description` attribute bug
- `has_quantization_lint` → `has_quantization_info` mypy error
- Redundant `Path` import causing `UnboundLocalError` in Streamlit export tab
- `QuantizationAdvice.recommendations` attribute error (use `strategy` + `qat_workflow` fields)
- HuggingFace Spaces Docker cache issue (factory reboot required for updates)

---

# 0.9.3 - 2025-12-12

### Fixed
- **CI test fix:** Made `streamlit_tabs.py` importable without streamlit installed
- Lazy imports for streamlit in render functions (data prep functions don't require it)

# 0.9.2 - 2025-12-12

### Changed
- **Refactored Streamlit tabs:** Extracted tab rendering into `streamlit_tabs.py` module
- **Single source of truth:** Both demo and upload flows now share the same render functions
- **Added 39 unit tests** for tab data preparation logic (format_number, format_bytes, prepare_* functions)

# 0.8.10 - 2025-12-11

### Fixed
- **Full UI restoration from v0.8.0:** Restored complete Streamlit app with all features
- **Demo/upload parity:** Demo models now use same 6-tab interface as uploaded models
- **Fixed graph_info loading:** Demo path now properly loads graph for Layer Details/Quantization
- **All ~1000 missing lines restored:** System Requirements, Deployment Cost Calculator, full Quantization analysis, Privacy Controls, and more

# 0.8.9 - 2025-12-11

### Fixed
- **Layer Details tab:** Fixed `LayerSummaryBuilder` API - use `build(graph_info)` not constructor
- **Quantization tab:** Fixed `QuantWarning` attribute - use `node_name` not `layer_name`
- **Overview tab restored:** Re-added missing charts and analysis:
  - Parameter Distribution by Op Type (bar chart)
  - FLOPs Distribution by Op Type (bar chart)
  - Precision Breakdown table
  - Bottleneck Analysis (compute/memory/VRAM bound indicators)

# 0.8.8 - 2025-12-11

### Fixed
- **Streamlit UI restored:** Re-added 6 tabs (Overview, Interactive Graph, Details, Layer Details, Quantization, Export)
- **Format Capabilities table:** Added collapsible table on homepage showing format support matrix
- Fixed `use_container_width` deprecation (replaced with `width="stretch"`)
- Root streamlit_app.py remains thin wrapper (single source of truth pattern)

# 0.8.7 - 2025-12-11

### Changed
- Root `streamlit_app.py` now imports from package (`from haoline.streamlit_app import main`)
- Single source of truth: all UI code in `src/haoline/streamlit_app.py`
- Eliminates sync issues between root and src versions
- Added E402 ignore for root streamlit_app.py in pyproject.toml

# 0.8.5 - 2025-12-11

### Fixed
- Restored Streamlit UI with 6 tabs (Overview, Interactive Graph, Layer Details, Quantization, Details, Export)
- Restored Format Capabilities table on homepage (was accidentally moved to sidebar)
- Fixed `use_container_width` deprecation warnings
- Synced root and src streamlit_app.py files

# 0.8.4 - 2025-12-11

### Fixed
- **Critical:** Fixed Pydantic validation errors in `QuantizationAdvice` when LLM returns nested structures
  - `sensitive_layers` and `safe_layers` fields now properly handle `{"layer_names": [...]}` format
  - `runtime_recommendations` now properly handles deeply nested dicts like `{"recommendation": {"settings": "...", "description": "..."}}`
- Added robust normalization functions: `_extract_string_from_nested`, `_normalize_str_list`, `_normalize_runtime_recs`
- Added 35 unit tests covering LLM response edge cases and production failure patterns

# 0.8.1 - 2025-12-11

### Added
- Streamlit auto-conversion to ONNX for PyTorch (input shape prompt), TFLite (tflite2onnx), and CoreML (coremltools) with graceful fallbacks.
- Documentation updates for auto-convert and format fidelity (README/PRD).

### Changed
- BACKLOG trimmed to tasks-only (removed commentary).

### Fixed
- Preserve original suffix when not converted to avoid mis-reading non-ONNX uploads.

# 0.8.2 - 2025-12-11

### Fixed
- Streamlit: load graph_info from current tmp_path (fixes UnboundLocalError on HF Spaces after conversions).
- Doc updates aligned with auto-convert behavior (PRD/PRDBacklogArchive).

# 0.8.3 - 2025-12-11

### Fixed
- Streamlit: guard quantization advisor outputs to avoid pydantic validation crashes; align graph loader imports; remove `use_container_width` deprecation warnings.
- .gitignore: ignore zips and HAOLINE_CODEBASE.md.

# 0.8.0 - 2025-12-11

### Added
- Streamlit Layer Details tab (search/filter, CSV/JSON download)
- Streamlit Quantization tab (readiness score, warnings, recommendations, layer sensitivity)
- Streamlit uploader now accepts TFLite/CoreML/OpenVINO/GGUF (plus existing ONNX/PT/TRT/SafeTensors)
- CLI: Added `--lint-quant` alias to `--lint-quantization`

### Changed
- Format capabilities matrix clarified (tiers, CLI vs in-app)

### Fixed
- Mypy no-any-return for format readers; generator fixture typing
- Ruff formatting fixes

## [0.4.0] - 2025-12-06

### Added
- Epic 33 complete: QAT & Quantization Linters (41/41 tasks)
- Quantization readiness scoring (0-100)
- QAT graph validation (fake-quant detection, scale consistency)
- LLM-powered quantization recommendations
- `--lint-quantization`, `--quant-report`, `--quant-llm-advice` CLI flags
- Quantization analysis in Streamlit UI

### Changed
- Documentation archival system: PRDBacklogArchive.md for completed epics
- Cursor rules updated with mypy frequency (every 3 commits)
- BACKLOG.md slimmed from 1848 to ~800 lines (57% reduction)
- Merged overlapping Deep Research epics (45→11, 46→18, 48→33)

### Fixed
- 62 mypy type errors across codebase
- Ruff lint error (unused import)

## [0.3.0] - 2025-12-06

### Added
- Epic 41 complete: Full CLI-Streamlit parity (44/44 tasks)
- System Requirements section (Steam-style min/rec/optimal)
- Deployment Cost Calculator ($/month estimates)
- Batch/Resolution Sweep views
- Per-Layer Timing breakdown
- Memory Overview chart
- Run Benchmark button
- Privacy Controls (redact names, summary only)
- Universal IR export (JSON + DOT graph)
- Cloud Instance selector (AWS/Azure/GCP)

### Changed
- LLM prompts now include all analysis data (KV cache, precision, memory breakdown)

## [0.2.3] - 2025-12-04

### Added
- CLI Reference section in README with all flags documented
- Privacy controls: `--offline`, `--redact-names`, `--summary-only` flags
- Privacy documentation (PRIVACY.md)
- Eval import framework: `haoline-import-eval` CLI command
- Evaluation schemas for detection, classification, NLP, LLM, segmentation tasks
- `GenericEvalResult` for user-defined metrics
- `CombinedReport` dataclass for architecture + eval data
- Ultralytics YOLO adapter for importing validation results
- Generic CSV/JSON adapter for eval import
- Deployment cost CLI flags: `--deployment-fps`, `--deployment-hours`
- YOLO quantization workflow guide

### Changed
- Moved Docker distribution to Epic 38 (deferred)

## [0.2.2] - 2025-12-04

### Added
- Format readers for GGUF, SafeTensors, TFLite, CoreML, OpenVINO
- Model comparison mode in Streamlit UI (side-by-side metrics)
- Session history in Streamlit UI (last 10 analyses)
- FLOPs-based node sizing in interactive graph (log scale)
- Collapsible sidebar in D3.js graph visualization
- PDF export in Streamlit UI

### Changed
- Modern dark theme with emerald accents in Streamlit UI
- Improved hardware dropdown with search and categorization

### Fixed
- Various mypy type errors
- Black/ruff formatting compliance

## [0.2.1] - 2025-12-04

### Added
- `haoline-web` CLI command to launch Streamlit app
- `haoline-compare` CLI command for model comparison
- Hardware selection dropdown in Streamlit (50+ GPUs)
- Interactive D3.js graph visualization in Streamlit

### Fixed
- `.env` file auto-loading for API keys
- LLM summary prompts when API key missing

## [0.2.0] - 2025-12-04

### Added
- Streamlit Web UI (`streamlit_app.py`)
- Model file upload and analysis
- HTML/JSON/Markdown/PDF export options
- LLM summary integration with secure API key input

### Changed
- Moved `matplotlib` from optional to core dependency
- Bumped version for PyPI release

## [0.1.0] - 2025-12-03

### Added
- Initial PyPI release
- Core analysis engine (params, FLOPs, memory estimation)
- Pattern detection (Conv-BN-ReLU, Transformer blocks, residual connections)
- Risk signal detection (deep networks, oversized layers, dynamic shapes)
- Hardware estimation for 50+ GPU profiles
- Runtime profiling with ONNX Runtime
- Visualization module (matplotlib charts)
- Interactive D3.js graph export
- LLM summarization (OpenAI, Anthropic, Google, xAI)
- PDF generation via Playwright
- Model conversion (PyTorch, TensorFlow, Keras, JAX → ONNX)
- Compare mode for quantization analysis
- CLI: `haoline` command with comprehensive flags

[1.1.1]: https://github.com/mdayku/HaoLine/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/mdayku/HaoLine/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/mdayku/HaoLine/compare/v0.9.11...v1.0.0
[0.4.0]: https://github.com/mdayku/HaoLine/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/mdayku/HaoLine/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/mdayku/HaoLine/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/mdayku/HaoLine/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/mdayku/HaoLine/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/mdayku/HaoLine/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mdayku/HaoLine/releases/tag/v0.1.0

