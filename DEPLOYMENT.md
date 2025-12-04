# HaoLine Deployment Guide

This guide covers deploying the HaoLine web interface to various platforms.

---

## Option 1: Local Development

Run the Streamlit app locally:

```bash
# Install with web dependencies
pip install haoline[web]

# Launch the dashboard
haoline-web
```

Opens at `http://localhost:8501`

---

## Option 2: Hugging Face Spaces (Recommended)

Free hosting with optional GPU support. Best for public demos.

### Quick Deploy

1. **Create a new Space** at [huggingface.co/new-space](https://huggingface.co/new-space)
   - Choose **Streamlit** as the SDK
   - Select **CPU basic** (free) or **GPU** if needed

2. **Create these files in your Space:**

**`app.py`**
```python
import subprocess
import sys

# Install haoline
subprocess.check_call([sys.executable, "-m", "pip", "install", "haoline[web]"])

# Import and run
from haoline.streamlit_app import main
main()
```

**`requirements.txt`**
```
haoline[web]>=0.2.3
```

3. **Push and deploy** — HF Spaces auto-builds on push

### Environment Variables (Optional)

For LLM summaries, add secrets in Space Settings:
- `OPENAI_API_KEY` — enables AI summaries for all users

> ⚠️ **Security Note:** If you set an API key, all users of your Space can trigger API calls on your account. Consider leaving it unset and letting users provide their own keys.

---

## Option 3: Streamlit Community Cloud

Free hosting for public repos.

### Steps

1. **Push your repo to GitHub** (must be public for free tier)

2. **Go to** [share.streamlit.io](https://share.streamlit.io)

3. **Click "New app"** and select:
   - Repository: `your-username/HaoLine`
   - Branch: `main`
   - Main file: `src/haoline/streamlit_app.py`

4. **Add secrets** (optional) in App Settings → Secrets:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```

5. **Deploy** — takes 2-3 minutes

### Limitations
- No GPU support on free tier
- 1GB memory limit
- Public repos only (for free)

---

## Option 4: Self-Hosted (Docker)

For private deployments or custom infrastructure.

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install haoline with all extras
RUN pip install --no-cache-dir haoline[full]

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the app
ENTRYPOINT ["haoline-web"]
```

### Build and Run

```bash
# Build
docker build -t haoline-web .

# Run
docker run -p 8501:8501 haoline-web

# With API key
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... haoline-web
```

### Docker Compose

```yaml
version: '3.8'
services:
  haoline:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    restart: unless-stopped
```

---

## Option 5: Cloud VMs

Deploy on AWS, GCP, Azure, or any cloud VM.

### Ubuntu/Debian Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+
sudo apt install python3.11 python3.11-venv -y

# Create virtual environment
python3.11 -m venv /opt/haoline
source /opt/haoline/bin/activate

# Install haoline
pip install haoline[full]

# Run (use screen/tmux for persistence)
haoline-web --server.port 8501 --server.address 0.0.0.0
```

### Systemd Service (Production)

Create `/etc/systemd/system/haoline.service`:

```ini
[Unit]
Description=HaoLine Web UI
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/haoline
Environment="PATH=/opt/haoline/bin"
ExecStart=/opt/haoline/bin/haoline-web --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable haoline
sudo systemctl start haoline
```

### Reverse Proxy (nginx)

```nginx
server {
    listen 80;
    server_name haoline.yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }
}
```

---

## Security Considerations

### API Keys

| Deployment | Recommendation |
|------------|----------------|
| **Public demo** | Don't set API key; let users provide their own |
| **Internal team** | Set shared API key via environment variable |
| **Enterprise** | Use API gateway with rate limiting |

### Model Privacy

HaoLine processes models **entirely in-browser/server memory**:
- Models are never sent to external services
- No telemetry or tracking
- LLM summaries only send architecture metadata (not weights)

### Network Security

- Use HTTPS in production (via reverse proxy or cloud load balancer)
- Restrict access via firewall rules or authentication proxy
- Consider [Streamlit authentication](https://docs.streamlit.io/knowledge-base/deploy/authentication-without-sso) for private deployments

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: haoline` | Run `pip install haoline[web]` |
| Port 8501 already in use | Use `--server.port 8502` |
| Out of memory on large models | Increase VM/container memory or use GPU instance |
| LLM summaries not working | Check `OPENAI_API_KEY` is set correctly |

### Logs

```bash
# Local
haoline-web 2>&1 | tee haoline.log

# Docker
docker logs -f <container_id>

# Systemd
journalctl -u haoline -f
```

---

## Platform Comparison

| Platform | Cost | GPU | Setup Time | Best For |
|----------|------|-----|------------|----------|
| **HF Spaces** | Free | Optional ($) | 5 min | Public demos |
| **Streamlit Cloud** | Free | No | 5 min | Quick sharing |
| **Docker** | Varies | Yes | 15 min | Private/enterprise |
| **Cloud VM** | $5-50/mo | Optional | 30 min | Full control |

---

## Need Help?

- [GitHub Issues](https://github.com/mdayku/HaoLine/issues)
- [Streamlit Docs](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
- [HF Spaces Docs](https://huggingface.co/docs/hub/spaces-sdks-streamlit)

