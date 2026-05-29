# NeuroScan AI FastAPI Deployment

## Local Windows

```powershell
cd D:\GIT\NeuroScan-AI
.\deploy\start.ps1 -HostName 127.0.0.1 -Port 8000
```

Open `http://127.0.0.1:8000`.

## Docker

```bash
cd D:/GIT/NeuroScan-AI
docker compose -f deploy/docker-compose.yml up --build
```

The Docker image does not bake in `.keras` model files. The compose file mounts the local `MODEL/` folder into the container at runtime.

## Important Environment Variables

- `GROQ_API_KEY`: enables Groq report generation. If absent, the API returns a local fallback report.
- `NEUROSCAN_CORS_ORIGINS`: comma-separated allowed origins, or `*` for local testing.
- `NEUROSCAN_MAX_UPLOAD_MB`: upload limit in MB. Default is `25`.
- `NEUROSCAN_MODELS_DIR`: optional custom model directory. Default is `MODEL/`.

## Endpoints

- `GET /health`
- `GET /ready`
- `POST /api/analyze`
- `GET /api/history`
- `GET /api/reports`
- `GET /api/reports/{filename}`
- `GET /api/model-info`
- `GET /docs`
