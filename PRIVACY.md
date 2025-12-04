# HaoLine Privacy & Security

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           YOUR MACHINE (Local)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌────────────────────────────────────────────────┐    │
│  │  Your Model  │───▶│              HaoLine CLI                       │    │
│  │  (.onnx, .pt │    │                                                │    │
│  │  .safetensors)    │  ┌─────────────┐  ┌─────────────┐            │    │
│  └──────────────┘    │  │  Analyzer   │  │  Estimator  │            │    │
│                      │  │  (params,   │  │  (FLOPs,    │            │    │
│                      │  │   layers,   │  │   memory,   │            │    │
│                      │  │   shapes)   │  │   latency)  │            │    │
│                      │  └─────────────┘  └─────────────┘            │    │
│                      │                                                │    │
│                      │  ┌─────────────┐  ┌─────────────┐            │    │
│                      │  │  Patterns   │  │  Risk       │            │    │
│                      │  │  (blocks,   │  │  Signals    │            │    │
│                      │  │   arch)     │  │  (warnings) │            │    │
│                      │  └─────────────┘  └─────────────┘            │    │
│                      └────────────────────────────────────────────────┘    │
│                                       │                                     │
│                                       ▼                                     │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                         Local Output Files                         │    │
│  │  report.json  │  report.html  │  report.md  │  graph.html         │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │ OPTIONAL (user-initiated)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INTERNET (Optional)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐     Only if --llm-summary is used:                   │
│  │  OpenAI API      │◀────Text summaries sent (NOT model weights)          │
│  │  (LLM Summaries) │     Use --offline to block this                      │
│  └──────────────────┘                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

| Step | Data | Stays Local? |
|------|------|--------------|
| 1. Load model | Model file (weights, graph) | YES |
| 2. Analyze | Parameters, shapes, ops | YES |
| 3. Estimate | FLOPs, memory, latency | YES |
| 4. Generate report | JSON, HTML, Markdown | YES |
| 5. LLM summary (optional) | Text description only | NO (sent to API) |

**Key Guarantee:** Your model weights and architecture graph NEVER leave your machine. The optional LLM summary only sends a text description of the model (e.g., "45M param ResNet-50 with 4 bottleneck blocks").

---

## Core Guarantee

**Your models never leave your machine.**

HaoLine is designed as a local-first tool. All analysis happens on your hardware — we don't upload, transmit, or store your model files anywhere.

---

## What HaoLine Does

| Action | Data Involved | Where It Happens |
|--------|---------------|------------------|
| Load model | Your `.onnx`, `.pt`, `.gguf`, etc. | Your machine |
| Analyze architecture | Graph structure, weights | Your machine |
| Generate reports | JSON, HTML, PDF | Your machine |
| Hardware estimation | Model metrics vs GPU specs | Your machine |

---

## What HaoLine Does NOT Do

- ❌ Upload models to any server
- ❌ Send telemetry or usage analytics
- ❌ Phone home for license checks
- ❌ Require internet connection for core features

---

## Optional Network Features

These features require network access **only when explicitly enabled**:

| Feature | When It Connects | What It Sends |
|---------|------------------|---------------|
| `--llm-summary` | When you request AI summary | Model metadata only (params, FLOPs, op counts) — **NOT weights** |
| `pip install` | During installation | Standard PyPI package download |
| HF Spaces / Streamlit Cloud | If you deploy the web UI | Whatever you upload to that platform |

### LLM Summary Details

When you use `--llm-summary`, HaoLine sends a **text summary** to the configured LLM provider (OpenAI, Anthropic, etc.):

```
Sent: "Model has 25M params, 4.1B FLOPs, 152 Conv layers, ResNet-like architecture..."
NOT sent: Actual weight values, tensor data, or model files
```

---

## Offline Mode

For air-gapped environments or maximum privacy:

```bash
haoline model.onnx --offline
```

This flag will fail if any network call is attempted, ensuring complete isolation.

*(Note: `--offline` flag is planned but not yet implemented)*

---

## Self-Hosting the Web UI

For teams who want web access without public cloud:

```bash
# Run locally
haoline-web

# Or deploy to your own infrastructure
docker run -p 8501:8501 haoline-web
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker and self-hosted options.

---

## Source Code Audit

HaoLine is open source. You can audit the code yourself:

- **Repository:** [github.com/mdayku/HaoLine](https://github.com/mdayku/HaoLine)
- **Network calls:** Search for `requests`, `urllib`, `httpx`, `socket` in the codebase
- **All dependencies:** Listed in `pyproject.toml`

---

## Enterprise Considerations

For enterprise deployments with strict compliance requirements:

1. **Run offline** — no network features needed for core analysis
2. **Self-host** — deploy the web UI on your infrastructure
3. **Audit the code** — it's all open source
4. **Air-gap friendly** — install from local wheel, no PyPI needed

---

## Data Handling by Output Format

| Format | Contains | Reveals |
|--------|----------|---------|
| **JSON** | Full analysis | Architecture, layer names, shapes, metrics |
| **Markdown** | Summary | Model type, total params/FLOPs, risks |
| **HTML** | Full report | Everything in JSON + visualizations |
| **PDF** | Full report | Same as HTML |
| **Graph HTML** | Interactive viz | Full graph structure, op types, connections |

### Redacting Sensitive Information

For sharing reports without revealing proprietary details:

```bash
# Coming soon
haoline model.onnx --redact-names --out-html report.html
```

This will replace layer/tensor names with generic identifiers (e.g., `layer_001`, `tensor_042`).

---

## Open Source Audit

HaoLine is fully open source under the MIT license.

### Audit Checklist

1. **Clone the repo:**
   ```bash
   git clone https://github.com/mdayku/HaoLine.git
   ```

2. **Search for network calls:**
   ```bash
   grep -r "requests\|urllib\|httpx\|socket" src/haoline/
   ```
   
   Expected results:
   - `llm_summarizer.py` — OpenAI/Anthropic API (optional, user-initiated)
   - `tests/` — downloading test models (test code only)

3. **Check dependencies:**
   ```bash
   cat pyproject.toml | grep -A 50 "dependencies"
   ```

4. **Review CLI entry points:**
   ```bash
   grep -r "def main\|def run_" src/haoline/*.py
   ```

### No Telemetry Guarantee

HaoLine does NOT:
- Send usage analytics
- Phone home for license checks
- Track which models you analyze
- Upload any data without explicit user action (like `--llm-summary`)

---

## Compliance Notes

### GDPR / Data Residency

Since HaoLine runs entirely locally, your model data never crosses network boundaries (unless you explicitly use LLM features). This makes it suitable for:
- Air-gapped environments
- Regulated industries (finance, healthcare)
- Government/defense use cases

### SOC 2 / ISO 27001

For organizations with compliance requirements:
- All processing is local
- No cloud dependencies for core features
- Audit trail via open source code

---

## Contact

Questions about privacy or security? Open an issue on GitHub or contact the maintainers.

