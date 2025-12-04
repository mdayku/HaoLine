# HaoLine Privacy & Security

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

## Contact

Questions about privacy or security? Open an issue on GitHub or contact the maintainers.

