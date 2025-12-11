# Changelog

All notable changes to HaoLine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.4.0]: https://github.com/mdayku/HaoLine/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/mdayku/HaoLine/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/mdayku/HaoLine/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/mdayku/HaoLine/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/mdayku/HaoLine/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/mdayku/HaoLine/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mdayku/HaoLine/releases/tag/v0.1.0

